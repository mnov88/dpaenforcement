"""Phase 0 – Omni-scan feature expansion and global diagnostics."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from dataclasses import dataclass
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # Progress indicator for long CV loops
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm optional in CI
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

try:  # Optional tree-based models
    import lightgbm as lgb
except ImportError:  # pragma: no cover - dependency optional in CI
    lgb = None

try:  # Optional CatBoost models
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:  # pragma: no cover - dependency optional in CI
    CatBoostClassifier = None
    CatBoostRegressor = None

try:  # Optional community detection
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
except ImportError:  # pragma: no cover - dependency optional in CI
    nx = None
    louvain_communities = None

import shap
import statsmodels.api as sm

from .config import FACTS_CONFIG, EvennessPaths
from .data import load_wide_dataset


logger = logging.getLogger(__name__)


_GLM_MAX_ITER = 4000
_GLM_TOL = 1e-3
_GLM_CHECK_INTERVAL = 1000
_GLM_STAGNATION_EPS = 1e-4
_GLM_STAGNATION_PATIENCE = 2
_GLM_L1_GRID: tuple[float, ...] = (0.0, 0.1, 0.5)
_GLM_C_GRID: tuple[float, ...] = (0.25, 0.5, 1.0)


def safe_log_loss(y_true: Sequence[float], proba: np.ndarray) -> float:
    """Binary log loss tolerant to single-class folds."""

    try:
        return metrics.log_loss(y_true, proba, labels=[0, 1])
    except ValueError:
        return float("nan")


@dataclass(frozen=True)
class FeatureMetadata:
    """Descriptor for a single engineered feature in the universe."""

    feature: str
    source_column: str
    feature_type: str
    block: str


@dataclass(frozen=True)
class OmniScanOutputs:
    """Collection of paths produced by :func:`run_omniscan`."""

    feature_universe_json: str
    coverage_ledger_csv: str
    coverage_checklist_csv: str
    importance_heatmap_csv: str
    interaction_map_csv: str
    block_importance_csv: str
    shap_country_csv: str
    shap_dpa_csv: str
    sage_importance_csv: str
    specification_curve_csv: str
    stability_selection_csv: str
    knockoff_results_csv: str
    robust_driver_csv: str
    crt_results_csv: str
    jurisdiction_effects_csv: str
    heterogeneity_csv: str
    network_edges_csv: str
    community_summary_csv: str
    risk_band_parity_csv: str
    distribution_contrasts_csv: str


@dataclass
class _GlmPreprocessResult:
    X_train: sparse.csr_matrix
    X_test: sparse.csr_matrix
    feature_names: np.ndarray
    base_columns: list[str]
    features_in: int
    features_dropped: int
    imputer: SimpleImputer
    scaler: StandardScaler


def _prepare_glm_design(X_train: pd.DataFrame, X_test: pd.DataFrame) -> _GlmPreprocessResult:
    """Prepare sparse design matrices with missingness indicators preserved."""

    non_constant = X_train.notna().sum(axis=0) > 0
    columns = X_train.columns[non_constant]
    features_in = int(non_constant.sum())
    features_dropped = int(len(non_constant) - features_in)
    if features_in == 0:
        empty = sparse.csr_matrix((X_train.shape[0], 0))
        return _GlmPreprocessResult(
            X_train=empty,
            X_test=sparse.csr_matrix((X_test.shape[0], 0)),
            feature_names=np.array([], dtype=str),
            base_columns=[],
            features_in=0,
            features_dropped=features_dropped,
            imputer=SimpleImputer(strategy="mean", add_indicator=True),
            scaler=StandardScaler(with_mean=False),
        )
    trimmed_train = X_train.loc[:, columns]
    trimmed_test = X_test.loc[:, columns]
    imputer = SimpleImputer(strategy="mean", add_indicator=True)
    train_imputed = imputer.fit_transform(trimmed_train)
    test_imputed = imputer.transform(trimmed_test)
    feature_names = imputer.get_feature_names_out(columns)
    scaler = StandardScaler(with_mean=False)
    train_scaled = scaler.fit_transform(train_imputed)
    test_scaled = scaler.transform(test_imputed)
    train_sparse = sparse.csr_matrix(train_scaled)
    test_sparse = sparse.csr_matrix(test_scaled)
    return _GlmPreprocessResult(
        X_train=train_sparse,
        X_test=test_sparse,
        feature_names=np.asarray(feature_names, dtype=str),
        base_columns=list(columns),
        features_in=features_in,
        features_dropped=features_dropped,
        imputer=imputer,
        scaler=scaler,
    )


def _fit_logistic_glm(X_train: sparse.csr_matrix, y_train: pd.Series, C: float, l1_ratio: float) -> LogisticRegression | None:
    """Train a SAGA logistic regression with monitoring hooks."""

    classes = np.unique(y_train)
    if classes.size < 2:
        return None
    penalty = "elasticnet" if l1_ratio > 0 else "l2"
    params: dict[str, object] = {
        "penalty": penalty,
        "solver": "saga",
        "C": C,
        "tol": _GLM_TOL,
        "warm_start": True,
        "max_iter": min(_GLM_CHECK_INTERVAL, _GLM_MAX_ITER),
        "n_jobs": -1,
        "class_weight": "balanced",
        "fit_intercept": True,
    }
    if penalty == "elasticnet":
        params["l1_ratio"] = l1_ratio
    model = LogisticRegression(**params)
    total_iter = 0
    last_gap = np.nan
    stagnation_checks = 0
    dual_gap = np.nan
    soft_not_converged = False
    while total_iter < _GLM_MAX_ITER:
        remaining = _GLM_MAX_ITER - total_iter
        model.max_iter = min(_GLM_CHECK_INTERVAL, remaining)
        model.fit(X_train, y_train)
        n_iter = int(np.max(np.asarray(model.n_iter_, dtype=int)))
        total_iter += n_iter
        dual_gap = float(getattr(model, "dual_gap_", np.nan))
        if total_iter >= _GLM_CHECK_INTERVAL:
            if not np.isnan(dual_gap) and not np.isnan(last_gap):
                if abs(last_gap - dual_gap) < _GLM_STAGNATION_EPS:
                    stagnation_checks += 1
                else:
                    stagnation_checks = 0
                if stagnation_checks >= _GLM_STAGNATION_PATIENCE:
                    soft_not_converged = True
                    break
        last_gap = dual_gap
        if n_iter < model.max_iter:
            break
    model.total_iter_ = total_iter  # type: ignore[attr-defined]
    model.final_dual_gap_ = dual_gap  # type: ignore[attr-defined]
    model.soft_not_converged_ = soft_not_converged  # type: ignore[attr-defined]
    model.selected_l1_ratio_ = l1_ratio  # type: ignore[attr-defined]
    model.selected_C_ = C  # type: ignore[attr-defined]
    return model


def _glm_top_features(coef: np.ndarray, feature_names: np.ndarray, top_k: int = 10) -> list[str]:
    if coef.size == 0 or feature_names.size == 0:
        return []
    abs_coef = np.abs(coef)
    order = np.argsort(abs_coef)[::-1]
    labels: list[str] = []
    for idx in order[:top_k]:
        weight = abs_coef[idx]
        if weight <= 0:
            continue
        labels.append(f"{feature_names[idx]}:{weight:.4f}")
    return labels


def _record_glm_fold_metrics(
    *,
    outcome: str,
    spec: str,
    fold: int,
    preprocess: _GlmPreprocessResult,
    model: LogisticRegression,
    y_train: pd.Series,
    y_test: pd.Series,
    proba: np.ndarray,
    paths: EvennessPaths | None,
) -> None:
    iterations = int(getattr(model, "total_iter_", int(np.max(np.asarray(model.n_iter_, dtype=int)))))
    dual_gap = float(getattr(model, "final_dual_gap_", float("nan")))
    pct_non_zero = 0.0
    coef = np.asarray(model.coef_)
    if coef.ndim > 1:
        coef = coef[0]
    if coef.size:
        pct_non_zero = float(np.count_nonzero(np.abs(coef) > 1e-6) / coef.size * 100.0)
    top_features = _glm_top_features(coef, preprocess.feature_names)
    y_test_values = y_test.to_numpy()
    clipped = np.clip(proba, 1e-6, 1 - 1e-6)
    auc = float("nan")
    ap = float("nan")
    brier = float("nan")
    logloss = safe_log_loss(y_test_values, clipped)
    if len(np.unique(y_test_values)) > 1:
        try:
            auc = metrics.roc_auc_score(y_test_values, clipped)
        except ValueError:
            auc = float("nan")
    try:
        ap = metrics.average_precision_score(y_test_values, clipped)
    except ValueError:
        ap = float("nan")
    try:
        brier = metrics.brier_score_loss(y_test_values, clipped)
    except ValueError:
        brier = float("nan")
    record = {
        "outcome": outcome,
        "spec": spec,
        "fold": fold,
        "iterations": iterations,
        "duality_gap": dual_gap,
        "pct_non_zero": pct_non_zero,
        "top_features": "|".join(top_features),
        "features_in": preprocess.features_in,
        "features_dropped": preprocess.features_dropped,
        "auc": auc,
        "average_precision": ap,
        "brier": brier,
        "log_loss": logloss,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "soft_not_converged": bool(getattr(model, "soft_not_converged_", False)),
        "C": getattr(model, "selected_C_", float("nan")),
        "l1_ratio": getattr(model, "selected_l1_ratio_", float("nan")),
    }
    logger.info(
        "GLM fold %s spec=%s outcome=%s iter=%s gap=%.4g nz=%.2f auc=%s brier=%s",
        fold,
        spec,
        outcome,
        record["iterations"],
        record["duality_gap"],
        record["pct_non_zero"],
        f"{auc:.3f}" if not np.isnan(auc) else "nan",
        f"{brier:.3f}" if not np.isnan(brier) else "nan",
    )
    if paths is not None:
        path = Path(paths.fold_metrics_csv)
        df = pd.DataFrame([record])
        mode = "a" if path.exists() else "w"
        header = not path.exists()
        df.to_csv(path, mode=mode, header=header, index=False)


_COUNTRY_NORMALISATION = {
    "UK": "GB",
    "EL": "GR",
    "XK": "XK",
}

_BLOCK_PREFIXES: tuple[tuple[str, str], ...] = (
    ("q21_breach_types", "Breach Facts"),
    ("q25_sensitive_data", "Sensitive Data"),
    ("q46_vuln", "Vulnerable Groups"),
    ("q47_remedial", "Mitigations"),
    ("q53_powers", "Corrective Powers"),
    ("art33", "Article 33"),
    ("art34", "Article 34"),
    ("subjects_notified", "Article 34"),
    ("breach_case", "Breach Case"),
    ("organization_size", "Organisation Size"),
    ("organization_type", "Organisation Type"),
    ("case_origin", "Initiation"),
    ("isic", "Economic Sector"),
    ("decision_year", "Temporal"),
    ("decision_quarter", "Temporal"),
    ("decision_year_bucket", "Temporal"),
    ("days_since_gdpr", "Temporal"),
    ("country_code", "Jurisdiction"),
    ("dpa_name", "Jurisdiction"),
    ("cross_border", "Cross-Border"),
    ("n_principles", "Procedural"),
    ("n_corrective", "Procedural"),
    ("severity", "Corrective Powers"),
    ("remedy_only_case", "Mitigations"),
)


def _assign_block(source: str) -> str:
    lowered = source.lower()
    for prefix, block in _BLOCK_PREFIXES:
        if lowered.startswith(prefix):
            return block
    return "Other"


def _harmonise_country(series: pd.Series) -> pd.Series:
    values = series.fillna("UNKNOWN").astype(str).str.upper()
    return values.map(lambda x: _COUNTRY_NORMALISATION.get(x, x))


def _drop_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    conflict_cols = [c for c in df.columns if c.endswith("_exclusivity_conflict")]
    if not conflict_cols:
        return df
    mask = pd.Series(False, index=df.index)
    for col in conflict_cols:
        mask |= df[col].fillna(0).astype(int).eq(1)
    return df.loc[~mask].copy()


def _status_dummy_frames(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Mapping[str, float]]]:
    """Create dummy matrices for *_status style columns."""

    status_frames: dict[str, pd.DataFrame] = {}
    status_mix: dict[str, Mapping[str, float]] = {}
    status_cols = [c for c in df.columns if c.endswith("_status")]
    for status_col in status_cols:
        series = df[status_col].fillna("MISSING").astype(str).str.upper()
        prefix = status_col[:-7]
        indicator_cols = [
            c
            for c in df.columns
            if c.startswith(f"{prefix}_")
            and c
            not in {
                status_col,
                f"{prefix}_coverage_status",
                f"{prefix}_exclusivity_conflict",
                f"{prefix}_known",
                f"{prefix}_unknown",
            }
        ]
        discussed_mask = series.eq("DISCUSSED")
        for col in indicator_cols:
            df.loc[~discussed_mask, col] = np.nan
        dummies = pd.get_dummies(series, prefix=status_col, dummy_na=False)
        dummies = dummies.astype(float)
        status_frames[status_col] = dummies
        total = len(series)
        mix = {value: count / total for value, count in series.value_counts(dropna=False).items()}
        status_mix[prefix] = mix
    coverage_cols = [c for c in df.columns if c.endswith("_coverage_status")]
    for coverage_col in coverage_cols:
        series = df[coverage_col].fillna("MISSING").astype(str).str.upper()
        dummies = pd.get_dummies(series, prefix=coverage_col, dummy_na=False).astype(float)
        status_frames[coverage_col] = dummies
        prefix = coverage_col[:-16]
        if prefix not in status_mix:
            total = len(series)
            status_mix[prefix] = {value: count / total for value, count in series.value_counts(dropna=False).items()}
    return status_frames, status_mix


def _infer_feature_types(df: pd.DataFrame, exclude: Sequence[str]) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def _build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[FeatureMetadata], pd.DataFrame, pd.DataFrame]:
    """Expand structured facts into the omni-scan feature universe."""

    df = df.copy()
    df = _drop_conflicts(df)
    if "country_code" in df:
        df["country_code"] = _harmonise_country(df["country_code"])

    status_frames, status_mix = _status_dummy_frames(df)

    id_cols = {"decision_id"}
    outcome_cols = set(FACTS_CONFIG.outcome_columns)
    outcome_cols.update([c for c in df.columns if c.startswith("q53_powers_")])
    exclude_cols = set(outcome_cols)
    exclude_cols.update(c for c in df.columns if c.endswith("_exclusivity_conflict"))

    base_df = df.drop(columns=[c for c in df.columns if c in status_frames])

    numeric_cols, categorical_cols = _infer_feature_types(base_df, id_cols.union(exclude_cols))
    metadata: list[FeatureMetadata] = []
    parts: list[pd.DataFrame] = []

    numeric_cols = [col for col in numeric_cols if col not in id_cols]
    if numeric_cols:
        numeric = base_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.astype(float)
        parts.append(numeric)
        for col in numeric.columns:
            metadata.append(FeatureMetadata(feature=col, source_column=col, feature_type="numeric", block=_assign_block(col)))

    for col in categorical_cols:
        if col in id_cols or col in exclude_cols:
            continue
        categories = pd.get_dummies(base_df[col].fillna("MISSING"), prefix=col, dummy_na=False)
        categories = categories.astype(float)
        parts.append(categories)
        block = _assign_block(col)
        for feature in categories.columns:
            metadata.append(FeatureMetadata(feature=feature, source_column=col, feature_type="categorical", block=block))

    for status_col, frame in status_frames.items():
        block = _assign_block(status_col)
        parts.append(frame)
        for feature in frame.columns:
            metadata.append(FeatureMetadata(feature=feature, source_column=status_col, feature_type="status", block=block))

    if not parts:
        feature_matrix = pd.DataFrame(index=df.index)
    else:
        feature_matrix = pd.concat(parts, axis=1)
    feature_matrix.index = df.index
    feature_matrix = feature_matrix.loc[:, ~feature_matrix.columns.duplicated()]

    def sanitize(name: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", name)
        cleaned = cleaned.strip("_") or "feature"
        if cleaned[0].isdigit():
            cleaned = f"f_{cleaned}"
        return cleaned

    rename_map: dict[str, str] = {}
    seen: dict[str, int] = {}
    sanitized_metadata: list[FeatureMetadata] = []
    for meta in metadata:
        base = sanitize(meta.feature)
        count = seen.get(base, 0)
        if count:
            new_name = f"{base}_{count}"
        else:
            new_name = base
        seen[base] = count + 1
        rename_map[meta.feature] = new_name
        sanitized_metadata.append(
            FeatureMetadata(
                feature=new_name,
                source_column=meta.source_column,
                feature_type=meta.feature_type,
                block=meta.block,
            )
        )
    feature_matrix = feature_matrix.rename(columns=rename_map)
    metadata = sanitized_metadata

    coverage_records: list[dict[str, object]] = []
    for meta in metadata:
        series = feature_matrix[meta.feature]
        observed = float(series.notna().mean())
        variance = float(series.var(ddof=0)) if observed > 0 else 0.0
        record = {
            "feature": meta.feature,
            "source": meta.source_column,
            "block": meta.block,
            "feature_type": meta.feature_type,
            "observed_share": observed,
            "mean": float(series.mean(skipna=True)) if observed > 0 else 0.0,
            "variance": variance,
        }
        prefix = meta.source_column.split("_", 1)[0]
        mix = status_mix.get(prefix)
        if mix:
            for status_value, share in mix.items():
                record[f"status_{status_value.lower()}"] = float(share)
        coverage_records.append(record)
    coverage = pd.DataFrame(coverage_records)

    all_columns = [c for c in df.columns if c not in id_cols.union(outcome_cols)]
    checklist_records: list[dict[str, object]] = []
    feature_sources = {meta.source_column for meta in metadata}
    for col in sorted(all_columns):
        checklist_records.append({
            "column": col,
            "covered": col in feature_sources or any(meta.source_column.startswith(col) for meta in metadata),
        })
    checklist = pd.DataFrame(checklist_records)

    feature_matrix.insert(0, "decision_id", df.get("decision_id").astype(str))
    return feature_matrix, metadata, coverage, checklist


def _save_feature_universe(feature_matrix: pd.DataFrame, metadata: Sequence[FeatureMetadata], paths: EvennessPaths) -> None:
    paths.ensure()
    feature_matrix.to_parquet(paths.feature_cache, index=False)
    records = [meta.__dict__ for meta in metadata]
    with open(paths.feature_universe_json, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)


def _prepare_model_matrix(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    X = feature_matrix.set_index("decision_id")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace({np.inf: np.nan, -np.inf: np.nan})
    X = X.loc[:, X.notna().sum(axis=0) > 0]
    return X


def _outcome_columns(df: pd.DataFrame) -> list[str]:
    cols = list(FACTS_CONFIG.outcome_columns)
    power_cols = [c for c in df.columns if c.startswith("q53_powers_")]
    return cols + sorted(power_cols)


def _train_tree_model(
    X: pd.DataFrame,
    y: pd.Series,
    classification: bool,
    random_state: int = 42,
) -> tuple[object, np.ndarray, Mapping[str, object] | None]:
    """Fit a gradient boosted model with nested CV and return feature importances."""

    if classification:
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    else:
        splitter = KFold(n_splits=3, shuffle=True, random_state=random_state)

    base_params = {
        "feature_pre_filter": False,
        "min_gain_to_split": 0.0,
        "learning_rate": 0.05,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "max_depth": -1,
    }
    params_grid: list[dict[str, object]] = []
    for min_data in (5, 15):
        for num_leaves in (15, 31, 63):
            for n_estimators in (200, 400):
                for reg_lambda in (0.0, 1.0):
                    combo = base_params | {
                        "min_data_in_leaf": min_data,
                        "num_leaves": num_leaves,
                        "n_estimators": n_estimators,
                        "reg_lambda": reg_lambda,
                    }
                    params_grid.append(combo)

    best_score = float("inf")
    best_params: Mapping[str, object] | None = None
    for params in params_grid:
        fold_scores: list[float] = []
        for train_idx, test_idx in splitter.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            unique_train = np.unique(y_train)
            unique_test = np.unique(y_test)
            if classification and (unique_train.size < 2 or unique_test.size < 2):
                logger.warning(
                    "Skipping tree fold due to single-class target (train=%s, test=%s)",
                    unique_train,
                    unique_test,
                )
                fold_scores.append(float("nan"))
                continue
            try:
                if lgb is not None:
                    if classification:
                        model = lgb.LGBMClassifier(
                            objective="binary",
                            random_state=random_state,
                            class_weight="balanced",
                            n_jobs=-1,
                            **params,
                        )
                    else:
                        model = lgb.LGBMRegressor(
                            objective="regression",
                            random_state=random_state,
                            n_jobs=-1,
                            **params,
                        )
                elif classification and CatBoostClassifier is not None:
                    model = CatBoostClassifier(
                        verbose=False,
                        random_state=random_state,
                        iterations=params["n_estimators"],
                        learning_rate=params["learning_rate"],
                        depth=6,
                    )
                elif not classification and CatBoostRegressor is not None:
                    model = CatBoostRegressor(
                        verbose=False,
                        random_state=random_state,
                        iterations=params["n_estimators"],
                        learning_rate=params["learning_rate"],
                        depth=6,
                    )
                else:
                    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

                    if classification:
                        model = HistGradientBoostingClassifier(random_state=random_state)
                    else:
                        model = HistGradientBoostingRegressor(random_state=random_state)
                model.fit(X_train, y_train)
                if classification:
                    proba = model.predict_proba(X_test)[:, 1]
                    score = safe_log_loss(y_test, np.clip(proba, 1e-6, 1 - 1e-6))
                else:
                    pred = model.predict(X_test)
                    score = metrics.mean_squared_error(y_test, pred)
                fold_scores.append(score)
            except Exception as exc:
                logger.exception("Tree model fold failed with params %s", params, exc_info=exc)
                fold_scores.append(float("nan"))
        mean_score = float(np.nanmean(fold_scores)) if fold_scores else float("inf")
        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    if lgb is not None:
        if classification:
            model = lgb.LGBMClassifier(
                objective="binary",
                random_state=random_state,
                class_weight="balanced",
                n_jobs=-1,
                **(best_params or {}),
            )
        else:
            model = lgb.LGBMRegressor(
                objective="regression",
                random_state=random_state,
                n_jobs=-1,
                **(best_params or {}),
            )
    elif classification and CatBoostClassifier is not None:
        model = CatBoostClassifier(verbose=False, random_state=random_state, **(best_params or {}))
    elif not classification and CatBoostRegressor is not None:
        model = CatBoostRegressor(verbose=False, random_state=random_state, **(best_params or {}))
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        if classification:
            model = HistGradientBoostingClassifier(random_state=random_state)
        else:
            model = HistGradientBoostingRegressor(random_state=random_state)
    model.fit(X, y)

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importance = np.abs(np.asarray(model.coef_))
    else:
        importance = np.zeros(X.shape[1])
    return model, importance, best_params


def _count_effective_splits(model: object) -> int:
    if lgb is None or not hasattr(model, "booster_"):
        return -1
    try:
        dump = model.booster_.dump_model()
        splits = 0
        for tree in dump.get("tree_info", []):
            leaves = int(tree.get("num_leaves", 1))
            splits += max(0, leaves - 1)
        return splits
    except Exception:
        return -1


def _shap_summaries(model: object, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_matrix = shap_values
    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    shap_summary = pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
    shap_summary = shap_summary.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    if hasattr(explainer, "shap_interaction_values"):
        try:
            interactions = explainer.shap_interaction_values(X)
            if isinstance(interactions, list):
                interactions = interactions[1] if len(interactions) > 1 else interactions[0]
            upper = np.triu_indices_from(interactions[0], k=1)
            agg = []
            for idx in range(interactions.shape[0]):
                values = interactions[idx]
                agg.append(np.abs(values))
            mean_interactions = np.mean(np.stack(agg, axis=0), axis=0)
            records: list[dict[str, object]] = []
            for i, j in zip(*upper):
                weight = float(mean_interactions[i, j])
                if weight == 0:
                    continue
                records.append({
                    "feature_a": X.columns[i],
                    "feature_b": X.columns[j],
                    "weight": weight,
                })
            interactions_df = pd.DataFrame(records)
            interactions_df = interactions_df.sort_values("weight", ascending=False).reset_index(drop=True)
        except Exception:  # pragma: no cover - shap interaction fallback
            interactions_df = pd.DataFrame(columns=["feature_a", "feature_b", "weight"])
    else:
        interactions_df = pd.DataFrame(columns=["feature_a", "feature_b", "weight"])
    return shap_summary, interactions_df, pd.DataFrame(shap_matrix, columns=X.columns, index=X.index)


def _glm_linear_shap(
    model: LogisticRegression,
    preprocess: _GlmPreprocessResult,
    index: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    design = preprocess.X_train
    try:
        explainer = shap.LinearExplainer(model, design, feature_dependence="independent")
        shap_values = explainer.shap_values(design)
    except Exception:
        coef = np.asarray(model.coef_)
        if coef.ndim > 1:
            coef = coef[0]
        dense_design = design.toarray() if sparse.issparse(design) else np.asarray(design)
        shap_values = dense_design * coef
    if isinstance(shap_values, list):
        shap_matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_matrix = shap_values
    base_count = len(preprocess.base_columns)
    if base_count == 0:
        return (
            pd.DataFrame(columns=["feature", "mean_abs_shap"]),
            pd.DataFrame(columns=["feature_a", "feature_b", "weight"]),
            pd.DataFrame(index=index),
        )
    shap_matrix = np.asarray(shap_matrix)
    shap_matrix = shap_matrix[:, :base_count]
    shap_df = pd.DataFrame(shap_matrix, columns=preprocess.base_columns, index=index)
    mean_abs = shap_df.abs().mean(axis=0).reset_index()
    mean_abs.columns = ["feature", "mean_abs_shap"]
    mean_abs = mean_abs.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    interactions = pd.DataFrame(columns=["feature_a", "feature_b", "weight"])
    return mean_abs, interactions, shap_df


def _aggregate_block_importance(shap_summary: pd.DataFrame, metadata: Sequence[FeatureMetadata]) -> pd.DataFrame:
    meta_lookup = {meta.feature: meta.block for meta in metadata}
    shap_summary["block"] = shap_summary["feature"].map(meta_lookup).fillna("Other")
    block_summary = (
        shap_summary.groupby("block", as_index=False)["mean_abs_shap"].sum().sort_values("mean_abs_shap", ascending=False)
    )
    return block_summary


def _sage_importance(model: object, X: pd.DataFrame, y: pd.Series, classification: bool) -> pd.DataFrame:
    if classification and hasattr(model, "predict_proba"):
        baseline = model.predict_proba(X)[:, 1]
    else:
        baseline = model.predict(X)
        if classification and isinstance(baseline, np.ndarray) and baseline.ndim > 1:
            baseline = baseline[:, 1]
    loss_fn = safe_log_loss if classification else metrics.mean_squared_error
    base_loss = loss_fn(y, np.clip(baseline, 1e-6, 1 - 1e-6)) if classification else loss_fn(y, baseline)
    rng = np.random.default_rng(0)
    records: list[dict[str, float]] = []
    for col in X.columns:
        perturbed = X.copy()
        permuted = rng.permutation(perturbed[col].to_numpy())
        perturbed[col] = permuted
        if classification and hasattr(model, "predict_proba"):
            preds = model.predict_proba(perturbed)[:, 1]
        else:
            preds = model.predict(perturbed)
            if classification and isinstance(preds, np.ndarray) and preds.ndim > 1:
                preds = preds[:, 1]
        loss = loss_fn(y, np.clip(preds, 1e-6, 1 - 1e-6)) if classification else loss_fn(y, preds)
        records.append({"feature": col, "sage": float(loss - base_loss)})
    importance = pd.DataFrame(records).sort_values("sage", ascending=False).reset_index(drop=True)
    return importance


def _specification_matrix(
    X: pd.DataFrame,
    df: pd.DataFrame,
    metadata: Sequence[FeatureMetadata],
    outcome: str,
    classification: bool,
    paths: EvennessPaths | None = None,
) -> pd.DataFrame:
    meta_frame = pd.DataFrame([meta.__dict__ for meta in metadata])
    block_lookup = meta_frame.set_index("feature")["block"].to_dict()

    jurisdiction_features = [f for f, block in block_lookup.items() if block == "Jurisdiction"]
    sector_features = [f for f, block in block_lookup.items() if block == "Economic Sector"]
    temporal_features = [f for f, block in block_lookup.items() if block == "Temporal"]

    specs: list[dict[str, object]] = []
    toggles = [
        {"fe": fe, "sector": sector, "temporal": temporal, "model": model, "winsor": winsor}
        for fe in [True, False]
        for sector in [True, False]
        for temporal in [True, False]
        for model in ["lasso", "elasticnet"]
        for winsor in [None, 0.99]
    ]
    # Align X and y strictly on decision_id to avoid boolean indexer misalignment
    # df is already filtered upstream; we reconstruct y with decision_id as index
    if "decision_id" not in df.columns:
        return pd.DataFrame(columns=["outcome", "score"])  # defensive fallback
    decision_ids = df["decision_id"].astype(str)
    # Build y, indexed by decision_id
    y_map = df.set_index(decision_ids)[outcome]
    # Keep only rows present in X
    available_ids = X.index.intersection(y_map.index)
    if len(available_ids) == 0:
        return pd.DataFrame(columns=["outcome", "score"])  # nothing to evaluate
    X_use = X.loc[available_ids]
    y = y_map.loc[available_ids]
    for spec in toggles:
        drop_cols: list[str] = []
        if not spec["fe"]:
            drop_cols.extend(jurisdiction_features)
        if not spec["sector"]:
            drop_cols.extend(sector_features)
        if not spec["temporal"]:
            drop_cols.extend(temporal_features)
        drop_cols = [c for c in drop_cols if c in X_use.columns]
        X_spec = X_use.drop(columns=drop_cols) if drop_cols else X_use
        if X_spec.empty:
            continue
        if classification:
            splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        else:
            model = ElasticNet(alpha=0.1 if spec["model"] == "lasso" else 0.05, l1_ratio=0.5)
            pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler(with_mean=False)),
                    ("model", model),
                ]
            )
        X_values = X_spec
        if spec["winsor"] and not classification:
            upper = y.quantile(spec["winsor"])
            lower = y.quantile(1 - spec["winsor"])
            y_fit = y.clip(lower=lower, upper=upper)
        else:
            y_fit = y
        scores: list[float] = []
        if classification:
            spec_label = (
                f"fe={spec['fe']}|sector={spec['sector']}|temporal={spec['temporal']}|winsor={spec['winsor']}"
            )
            for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_values, y_fit), start=1):
                X_train = X_values.iloc[train_idx]
                X_test = X_values.iloc[test_idx]
                y_train = y_fit.iloc[train_idx]
                y_test = y_fit.iloc[test_idx]
                unique_train = np.unique(y_train.dropna())
                unique_test = np.unique(y_test.dropna())
                if unique_train.size < 2 or unique_test.size < 2:
                    logger.warning(
                        "Skipping fold %s for %s spec=%s: single-class target (train=%s, test=%s)",
                        fold_idx,
                        outcome,
                        spec_label,
                        unique_train,
                        unique_test,
                    )
                    continue
                preprocess = _prepare_glm_design(X_train, X_test)
                best_model: LogisticRegression | None = None
                best_proba: np.ndarray | None = None
                best_auc = float("nan")
                best_score_metric = float("-inf")
                for C in _GLM_C_GRID:
                    for l1_ratio in _GLM_L1_GRID:
                        model_candidate = _fit_logistic_glm(preprocess.X_train, y_train, C, l1_ratio)
                        if model_candidate is None:
                            continue
                        try:
                            proba_candidate = model_candidate.predict_proba(preprocess.X_test)[:, 1]
                        except Exception:
                            continue
                        auc_candidate = float("nan")
                        if unique_test.size > 1:
                            try:
                                auc_candidate = metrics.roc_auc_score(y_test, proba_candidate)
                            except ValueError:
                                auc_candidate = float("nan")
                        score_metric = auc_candidate
                        if np.isnan(score_metric):
                            loss = safe_log_loss(y_test, np.clip(proba_candidate, 1e-6, 1 - 1e-6))
                            score_metric = -loss if not np.isnan(loss) else float("-inf")
                        if best_model is None or score_metric > best_score_metric:
                            best_model = model_candidate
                            best_proba = proba_candidate
                            best_auc = auc_candidate
                            best_score_metric = score_metric
                if best_model is None or best_proba is None:
                    continue
                scores.append(best_auc)
                _record_glm_fold_metrics(
                    outcome=outcome,
                    spec=spec_label,
                    fold=fold_idx,
                    preprocess=preprocess,
                    model=best_model,
                    y_train=y_train,
                    y_test=y_test,
                    proba=best_proba,
                    paths=paths,
                )
        else:
            splitter = KFold(n_splits=3, shuffle=True, random_state=0)
            for train_idx, test_idx in splitter.split(X_values, y_fit):
                pipeline.fit(X_values.iloc[train_idx], y_fit.iloc[train_idx])
                pred = pipeline.predict(X_values.iloc[test_idx])
                score = metrics.r2_score(y_fit.iloc[test_idx], pred)
                scores.append(score)
        specs.append({
            "outcome": outcome,
            "fe": spec["fe"],
            "sector": spec["sector"],
            "temporal": spec["temporal"],
            "model": spec["model"],
            "winsor": spec["winsor"],
            "score": float(np.nanmean(scores)) if scores else float("nan"),
        })
    return pd.DataFrame(specs)


def _stability_selection(
    X: pd.DataFrame,
    y: pd.Series,
    classification: bool,
    iterations: int = 50,
    sample_frac: float = 0.75,
) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    counts = pd.Series(0, index=X.columns, dtype=float)
    total = pd.Series(0, index=X.columns, dtype=float)
    for _ in range(iterations):
        sample = rng.choice(X.index.to_numpy(), size=int(len(X) * sample_frac), replace=False)
        X_sub = X.loc[sample]
        y_sub = y.loc[sample]
        if classification:
            preprocess = _prepare_glm_design(X_sub, X_sub)
            model = _fit_logistic_glm(preprocess.X_train, y_sub, C=0.5, l1_ratio=0.1)
            if model is None or preprocess.features_in == 0:
                continue
        else:
            model = ElasticNet(alpha=0.1, l1_ratio=0.7)
            pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
                ("model", model),
            ]
        )
        if classification:
            coef = np.asarray(model.coef_)
            if coef.ndim > 1:
                coef = coef[0]
            base_count = len(preprocess.base_columns)
            coef = coef[:base_count]
            active_cols = [col for col in preprocess.base_columns if X_sub[col].notna().any()]
        else:
            pipeline.fit(X_sub, y_sub)
            coef = pipeline.named_steps["model"].coef_
            if coef.ndim > 1:
                coef = coef[0]
            coef = np.asarray(coef).flatten()
            active_mask = X_sub.notna().any(axis=0)
            active_cols = X_sub.columns[active_mask]
        if len(active_cols) == 0:
            continue
        use_len = min(len(coef), len(active_cols))
        coef = coef[:use_len]
        active_cols = active_cols[:use_len]
        selection = (np.abs(coef) > 1e-6).astype(float)
        sel_series = pd.Series(selection, index=active_cols, dtype=float)
        counts = counts.add(sel_series, fill_value=0.0)
        total.loc[active_cols] += 1
    probability = counts / total.replace(0, np.nan)
    return probability.reset_index().rename(columns={"index": "feature", 0: "selection_probability"})


def _knockoff_filter(X: pd.DataFrame, y: pd.Series, classification: bool, q: float = 0.1) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    knockoffs = X.apply(lambda col: rng.permutation(col.to_numpy()), axis=0, result_type="broadcast")
    knockoffs.columns = [f"{col}__knockoff" for col in X.columns]
    augmented = pd.concat([X, knockoffs], axis=1)
    if classification:
        preprocess = _prepare_glm_design(augmented, augmented)
        model = _fit_logistic_glm(preprocess.X_train, y, C=0.5, l1_ratio=0.1)
        if model is None or preprocess.features_in == 0:
            return pd.DataFrame(columns=["feature", "w_stat", "selected"])
    else:
        model = ElasticNet(alpha=0.05, l1_ratio=0.7)
    pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
            ("model", model),
        ]
    )
    if classification:
        pipeline.named_steps["model"] = model
        coef = np.asarray(model.coef_)
        if coef.ndim > 1:
            coef = coef[0]
        base_cols = preprocess.base_columns
        coef_series = pd.Series(coef[: len(base_cols)], index=base_cols, dtype=float)
        original_coef = coef_series.reindex(X.columns, fill_value=0.0).to_numpy()
        knockoff_cols = [f"{col}__knockoff" for col in X.columns]
        knockoff_coef = coef_series.reindex(knockoff_cols, fill_value=0.0).to_numpy()
    else:
        pipeline.fit(augmented, y)
        coef = pipeline.named_steps["model"].coef_
        if coef.ndim > 1:
            coef = coef[0]
        coef = np.asarray(coef).flatten()
        original_coef = coef[: X.shape[1]]
        knockoff_coef = coef[X.shape[1] :]
    w_stats = np.abs(original_coef) - np.abs(knockoff_coef)
    abs_values = np.sort(np.abs(w_stats))[::-1]
    threshold = np.inf
    for t in abs_values:
        if t == 0:
            continue
        numerator = 1 + np.sum(w_stats <= -t)
        denominator = max(1, np.sum(w_stats >= t))
        if numerator / denominator <= q:
            threshold = t
            break
    selected = [
        {"feature": feature, "w_stat": float(w), "selected": bool(abs(w) >= threshold if np.isfinite(threshold) else False)}
        for feature, w in zip(X.columns, w_stats)
    ]
    return pd.DataFrame(selected)


def _jurisdiction_effects(
    residuals: pd.Series,
    group: pd.Series,
    cluster: str,
) -> pd.DataFrame:
    # Build numeric design matrix and align with residuals
    design = pd.get_dummies(group.astype(str), prefix=cluster, drop_first=False).astype(float)
    y = pd.to_numeric(residuals, errors="coerce").astype(float)
    frame = pd.concat([y.rename("y"), design], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return pd.DataFrame(columns=[cluster, "effect", "ci_low", "ci_high", "pvalue"])  # nothing to estimate
    X = sm.add_constant(frame.drop(columns="y").to_numpy(dtype=float), has_constant="add")
    y_arr = frame["y"].to_numpy(dtype=float)
    model = sm.OLS(y_arr, X).fit()
    # Rebuild parameter index using column names
    param_index = ["const"] + frame.drop(columns="y").columns.tolist()
    params = pd.Series(model.params, index=param_index, copy=False).drop("const", errors="ignore")
    se = pd.Series(model.bse, index=param_index, copy=False).drop("const", errors="ignore")
    ci_low = params - 1.96 * se
    ci_high = params + 1.96 * se
    frame = pd.DataFrame(
        {
            cluster: pd.Index(params.index).str.replace(f"{cluster}_", "", regex=False),
            "effect": params.values,
            "ci_low": ci_low.values,
            "ci_high": ci_high.values,
            "pvalue": pd.Series(model.pvalues, index=param_index).drop("const", errors="ignore").values,
        }
    )
    return frame


def _crt_test(residuals: pd.Series, group: pd.Series, iterations: int = 200) -> float:
    rng = np.random.default_rng(2)
    # One-hot encode group and align with residuals
    design = pd.get_dummies(group.astype(str), drop_first=True).astype(float)
    y = pd.to_numeric(residuals, errors="coerce").astype(float)
    frame = pd.concat([y.rename("y"), design], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    # If not enough observations or no variation, return non-significant p-value to avoid crashing
    if frame.shape[0] < 10 or frame.shape[1] <= 1:
        return float("nan")
    X = sm.add_constant(frame.drop(columns="y").to_numpy(dtype=float), has_constant="add")
    y_arr = frame["y"].to_numpy(dtype=float)
    try:
        model = sm.OLS(y_arr, X).fit()
    except Exception:
        return float("nan")
    statistic = float(model.ssr)
    permuted_stats: list[float] = []
    for _ in range(iterations):
        shuffled = rng.permutation(y_arr)
        try:
            perm_model = sm.OLS(shuffled, X).fit()
            permuted_stats.append(float(perm_model.ssr))
        except Exception:
            continue
    permuted = np.asarray(permuted_stats)
    if permuted.size == 0:
        return float("nan")
    pvalue = float(np.mean(permuted <= statistic))
    return pvalue


def _risk_band_assignments(
    predictions: pd.Series,
    outcome: pd.Series,
    jurisdiction: pd.Series,
    bands: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Align indices to avoid constructor/union index errors
    common = predictions.index.intersection(outcome.index).intersection(jurisdiction.index)
    if len(common) == 0:
        empty = pd.DataFrame(columns=["band", "jurisdiction", "mean_outcome", "count"])
        return empty, pd.DataFrame(columns=["score", "outcome", "jurisdiction"]) 
    score = pd.to_numeric(predictions.loc[common], errors="coerce")
    y = pd.to_numeric(outcome.loc[common], errors="coerce")
    j = jurisdiction.loc[common].astype(str)
    df = pd.DataFrame({"score": score, "outcome": y, "jurisdiction": j}).replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty or df["score"].nunique() < 2:
        empty = pd.DataFrame(columns=["band", "jurisdiction", "mean_outcome", "count"])
        return empty, df
    try:
        df["band"] = pd.qcut(df["score"], q=np.linspace(0, 1, bands + 1), labels=False, duplicates="drop")
    except Exception:
        empty = pd.DataFrame(columns=["band", "jurisdiction", "mean_outcome", "count"])
        return empty, df
    summary = (
        df.groupby(["band", "jurisdiction"], as_index=False)
        .agg(mean_outcome=("outcome", "mean"), count=("outcome", "size"))
        .sort_values(["band", "jurisdiction"])
    )
    return summary, df


def _distribution_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    top_jurisdictions = df["jurisdiction"].value_counts().nlargest(5).index.tolist()
    for band, band_df in df.groupby("band"):
        jurisdictions = [g for g in top_jurisdictions if g in band_df["jurisdiction"].unique()]
        for i in range(len(jurisdictions)):
            for j in range(i + 1, len(jurisdictions)):
                a = band_df.loc[band_df["jurisdiction"] == jurisdictions[i], "outcome"]
                b = band_df.loc[band_df["jurisdiction"] == jurisdictions[j], "outcome"]
                if len(a) < 5 or len(b) < 5:
                    continue
                ks = stats.ks_2samp(a, b, alternative="two-sided").statistic
                try:
                    emd = stats.wasserstein_distance(a, b)
                except Exception:  # pragma: no cover - scipy fallback
                    emd = np.nan
                records.append(
                    {
                        "band": int(band),
                        "jurisdiction_a": jurisdictions[i],
                        "jurisdiction_b": jurisdictions[j],
                        "ks": float(ks),
                        "emd": float(emd),
                    }
                )
    return pd.DataFrame(records)


def _build_network(importance: pd.DataFrame, outcomes: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges: list[dict[str, object]] = []
    for outcome in tqdm(outcomes, desc="Omniscan outcomes"):
        subset = importance.loc[importance["outcome"] == outcome]
        for _, row in subset.iterrows():
            edges.append({"source": row["feature"], "target": outcome, "weight": row["importance"]})
    edge_df = pd.DataFrame(edges)
    if nx is None or edge_df.empty:
        return edge_df, pd.DataFrame(columns=["community", "members"])
    graph = nx.Graph()
    for _, row in edge_df.iterrows():
        graph.add_edge(row["source"], row["target"], weight=row["weight"])
    if louvain_communities is not None:
        communities = louvain_communities(graph, weight="weight")
    else:  # pragma: no cover - fallback if Louvain unavailable
        communities = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
    community_records = [
        {"community": idx, "members": ",".join(sorted(map(str, community)))}
        for idx, community in enumerate(communities)
    ]
    return edge_df, pd.DataFrame(community_records)


def run_omniscan(paths: EvennessPaths | None = None) -> OmniScanOutputs:
    """Execute the omni-scan workflow and persist artefacts to disk."""

    paths = paths or EvennessPaths()
    paths.ensure()
    Path(paths.fold_metrics_csv).unlink(missing_ok=True)

    wide_df = load_wide_dataset(paths.wide_csv)
    feature_matrix, metadata, coverage, checklist = _build_feature_matrix(wide_df)
    _save_feature_universe(feature_matrix, metadata, paths)
    coverage.to_csv(paths.coverage_ledger_csv, index=False)
    checklist.to_csv(paths.coverage_checklist_csv, index=False)

    X = _prepare_model_matrix(feature_matrix)
    outcomes = _outcome_columns(wide_df)

    importance_records: list[dict[str, object]] = []
    interaction_records: list[dict[str, object]] = []
    block_records: list[dict[str, object]] = []
    country_records: list[dict[str, object]] = []
    dpa_records: list[dict[str, object]] = []
    sage_records: list[dict[str, object]] = []
    spec_frames: list[pd.DataFrame] = []
    stability_frames: list[pd.DataFrame] = []
    knockoff_frames: list[pd.DataFrame] = []
    crt_records: list[dict[str, object]] = []
    jurisdiction_frames: list[pd.DataFrame] = []
    heterogeneity_frames: list[pd.DataFrame] = []
    risk_band_frames: list[pd.DataFrame] = []
    distribution_frames: list[pd.DataFrame] = []

    meta_lookup = {meta.feature: meta for meta in metadata}

    for outcome in tqdm(outcomes, desc="Omniscan outcomes"):
        if outcome not in wide_df.columns:
            continue
        y_raw = wide_df[outcome]
        mask = y_raw.notna()
        if mask.sum() < 50:
            continue
        decision_ids = wide_df.loc[mask, "decision_id"].astype(str)
        y = y_raw.loc[mask]
        y.index = decision_ids
        try:
            X_outcome = X.loc[decision_ids]
        except KeyError:
            available = X.index.intersection(decision_ids)
            y = y.loc[available]
            decision_ids = available
            X_outcome = X.loc[decision_ids]
        classification = y.dropna().isin({0, 1, True, False}).all()
        model, importance, _ = _train_tree_model(X_outcome, y, classification)
        fallback_reason: str | None = None
        if classification and lgb is not None and isinstance(model, lgb.LGBMClassifier):
            effective_splits = _count_effective_splits(model)
            train_auc = float("nan")
            if len(np.unique(y)) > 1:
                try:
                    train_auc = metrics.roc_auc_score(y, model.predict_proba(X_outcome)[:, 1])
                except Exception:
                    train_auc = float("nan")
            if 0 <= effective_splits < 5:
                fallback_reason = f"effective_splits={effective_splits}"
            elif not np.isnan(train_auc) and abs(train_auc - 0.5) <= 0.02:
                fallback_reason = f"train_auc={train_auc:.3f}"
        else:
            effective_splits = -1
            train_auc = float("nan")
        if fallback_reason:
            logger.warning("LightGBM fallback for %s due to %s", outcome, fallback_reason)
            preprocess_full = _prepare_glm_design(X_outcome, X_outcome)
            glm_model = _fit_logistic_glm(preprocess_full.X_train, y, C=0.5, l1_ratio=0.1)
            if glm_model is not None:
                shap_summary, interactions, shap_matrix = _glm_linear_shap(glm_model, preprocess_full, X_outcome.index)
            else:
                shap_summary, interactions, shap_matrix = _shap_summaries(model, X_outcome)
        else:
            shap_summary, interactions, shap_matrix = _shap_summaries(model, X_outcome)
        shap_summary["outcome"] = outcome
        interactions["outcome"] = outcome
        block_importance = _aggregate_block_importance(shap_summary, metadata)
        block_importance["outcome"] = outcome
        sage_importance = _sage_importance(model, X_outcome, y, classification)
        sage_importance["outcome"] = outcome

        for _, row in shap_summary.iterrows():
            importance_records.append({
                "outcome": outcome,
                "feature": row["feature"],
                "importance": float(row["mean_abs_shap"]),
            })
        for _, row in interactions.iterrows():
            interaction_records.append({
                "outcome": outcome,
                "feature_a": row["feature_a"],
                "feature_b": row["feature_b"],
                "importance": float(row["weight"]),
            })
        block_records.append(block_importance)
        sage_records.append(sage_importance)

        country = wide_df.loc[mask, "country_code"] if "country_code" in wide_df else pd.Series(index=y.index, dtype="string")
        dpa = wide_df.loc[mask, "dpa_name_canonical"] if "dpa_name_canonical" in wide_df else pd.Series(index=y.index, dtype="string")
        if not country.empty:
            grouped = shap_matrix.groupby(country).mean()
            grouped["jurisdiction"] = grouped.index
            grouped["outcome"] = outcome
            country_records.extend(
                grouped.reset_index(drop=True).melt(
                    id_vars=["jurisdiction", "outcome"], var_name="feature", value_name="mean_shap"
                ).to_dict("records")
            )
        if not dpa.empty:
            grouped = shap_matrix.groupby(dpa).mean()
            grouped["dpa"] = grouped.index
            grouped["outcome"] = outcome
            dpa_records.extend(
                grouped.reset_index(drop=True).melt(id_vars=["dpa", "outcome"], var_name="feature", value_name="mean_shap").to_dict("records")
            )

        spec_frames.append(
            _specification_matrix(X_outcome, wide_df.loc[mask], metadata, outcome, classification, paths)
        )
        stability = _stability_selection(X_outcome, y, classification)
        stability["outcome"] = outcome
        stability_frames.append(stability)
        knockoff = _knockoff_filter(X_outcome, y, classification)
        knockoff["outcome"] = outcome
        knockoff_frames.append(knockoff)

        # Determine robust drivers (selected features with probability >=0.6 & knockoff selected)
        stable_set = set(stability.loc[stability["selection_probability"] >= 0.6, "feature"])
        knockoff_set = set(knockoff.loc[knockoff["selected"], "feature"])
        robust = stable_set & knockoff_set
        if robust:
            for feature in sorted(robust):
                driver = meta_lookup.get(feature)
                block = driver.block if driver else "Other"
                heterogeneity_frames.append(
                    pd.DataFrame(
                        {
                            "outcome": [outcome],
                            "feature": [feature],
                            "block": [block],
                            "driver_type": ["robust"],
                        }
                    )
                )

        # DML residuals
        splitter = KFold(n_splits=3, shuffle=True, random_state=0)
        residuals = pd.Series(index=y.index, dtype=float)
        for train_idx, test_idx in splitter.split(X_outcome, y):
            model_fold, _, _ = _train_tree_model(X_outcome.iloc[train_idx], y.iloc[train_idx], classification)
            if classification and hasattr(model_fold, "predict_proba"):
                preds = model_fold.predict_proba(X_outcome.iloc[test_idx])[:, 1]
            else:
                preds = model_fold.predict(X_outcome.iloc[test_idx])
                if classification and isinstance(preds, np.ndarray) and preds.ndim > 1:
                    preds = preds[:, 1]
            residuals.iloc[test_idx] = y.iloc[test_idx] - preds

        if not country.empty:
            country_effects = _jurisdiction_effects(residuals, country, "country")
            country_effects["outcome"] = outcome
            jurisdiction_frames.append(country_effects)
            crt_records.append({
                "outcome": outcome,
                "group": "country",
                "pvalue": _crt_test(residuals, country),
            })
        if not dpa.empty:
            dpa_effects = _jurisdiction_effects(residuals, dpa, "dpa")
            dpa_effects["outcome"] = outcome
            jurisdiction_frames.append(dpa_effects)
            crt_records.append({
                "outcome": outcome,
                "group": "dpa",
                "pvalue": _crt_test(residuals, dpa),
            })

        risk_scores = model.predict(X_outcome)
        if classification and hasattr(model, "predict_proba"):
            risk_scores = model.predict_proba(X_outcome)[:, 1]
        elif classification and isinstance(risk_scores, np.ndarray) and risk_scores.ndim > 1:
            risk_scores = risk_scores[:, 1]
        band_summary, assignments = _risk_band_assignments(pd.Series(risk_scores, index=y.index), y, country)
        if not band_summary.empty:
            band_summary["outcome"] = outcome
            risk_band_frames.append(band_summary)
            distribution_frames.append(_distribution_contrasts(assignments))

    importance_df = pd.DataFrame(importance_records)
    importance_df.to_csv(paths.importance_heatmap_csv, index=False)
    pd.DataFrame(interaction_records).to_csv(paths.interaction_map_csv, index=False)
    if block_records:
        pd.concat(block_records, axis=0).to_csv(paths.block_importance_csv, index=False)
    else:
        pd.DataFrame(columns=["block", "mean_abs_shap", "outcome"]).to_csv(paths.block_importance_csv, index=False)
    pd.DataFrame(country_records).to_csv(paths.shap_country_csv, index=False)
    pd.DataFrame(dpa_records).to_csv(paths.shap_dpa_csv, index=False)
    if sage_records:
        pd.concat(sage_records, axis=0).to_csv(paths.sage_importance_csv, index=False)
    else:
        pd.DataFrame(columns=["feature", "sage", "outcome"]).to_csv(paths.sage_importance_csv, index=False)
    if spec_frames:
        pd.concat(spec_frames, axis=0).to_csv(paths.specification_curve_csv, index=False)
    else:
        pd.DataFrame(columns=["outcome", "score"]).to_csv(paths.specification_curve_csv, index=False)
    if stability_frames:
        pd.concat(stability_frames, axis=0).to_csv(paths.stability_selection_csv, index=False)
    else:
        pd.DataFrame(columns=["feature", "selection_probability", "outcome"]).to_csv(paths.stability_selection_csv, index=False)
    if knockoff_frames:
        knockoff_df = pd.concat(knockoff_frames, axis=0)
        knockoff_df.to_csv(paths.knockoff_results_csv, index=False)
        robust = knockoff_df.loc[knockoff_df["selected"]]
        robust.to_csv(paths.robust_driver_csv, index=False)
    else:
        pd.DataFrame(columns=["feature", "selected", "outcome"]).to_csv(paths.knockoff_results_csv, index=False)
        pd.DataFrame(columns=["feature", "selected", "outcome"]).to_csv(paths.robust_driver_csv, index=False)
    pd.DataFrame(crt_records).to_csv(paths.crt_results_csv, index=False)
    if jurisdiction_frames:
        pd.concat(jurisdiction_frames, axis=0).to_csv(paths.jurisdiction_effects_csv, index=False)
    else:
        pd.DataFrame(columns=["cluster", "effect"]).to_csv(paths.jurisdiction_effects_csv, index=False)
    if heterogeneity_frames:
        pd.concat(heterogeneity_frames, axis=0).to_csv(paths.heterogeneity_csv, index=False)
    else:
        pd.DataFrame(columns=["outcome", "feature", "block", "driver_type"]).to_csv(paths.heterogeneity_csv, index=False)
    if risk_band_frames:
        pd.concat(risk_band_frames, axis=0).to_csv(paths.risk_band_parity_csv, index=False)
    else:
        pd.DataFrame(columns=["band", "jurisdiction", "mean_outcome", "count", "outcome"]).to_csv(paths.risk_band_parity_csv, index=False)
    if distribution_frames:
        pd.concat(distribution_frames, axis=0).to_csv(paths.distribution_contrasts_csv, index=False)
    else:
        pd.DataFrame(columns=["band", "jurisdiction_a", "jurisdiction_b", "ks", "emd"]).to_csv(paths.distribution_contrasts_csv, index=False)

    network_edges, community_summary = _build_network(importance_df, outcomes)
    network_edges.to_csv(paths.network_edges_csv, index=False)
    community_summary.to_csv(paths.community_summary_csv, index=False)

    return OmniScanOutputs(
        feature_universe_json=str(paths.feature_universe_json),
        coverage_ledger_csv=str(paths.coverage_ledger_csv),
        coverage_checklist_csv=str(paths.coverage_checklist_csv),
        importance_heatmap_csv=str(paths.importance_heatmap_csv),
        interaction_map_csv=str(paths.interaction_map_csv),
        block_importance_csv=str(paths.block_importance_csv),
        shap_country_csv=str(paths.shap_country_csv),
        shap_dpa_csv=str(paths.shap_dpa_csv),
        sage_importance_csv=str(paths.sage_importance_csv),
        specification_curve_csv=str(paths.specification_curve_csv),
        stability_selection_csv=str(paths.stability_selection_csv),
        knockoff_results_csv=str(paths.knockoff_results_csv),
        robust_driver_csv=str(paths.robust_driver_csv),
        crt_results_csv=str(paths.crt_results_csv),
        jurisdiction_effects_csv=str(paths.jurisdiction_effects_csv),
        heterogeneity_csv=str(paths.heterogeneity_csv),
        network_edges_csv=str(paths.network_edges_csv),
        community_summary_csv=str(paths.community_summary_csv),
        risk_band_parity_csv=str(paths.risk_band_parity_csv),
        distribution_contrasts_csv=str(paths.distribution_contrasts_csv),
    )


__all__ = ["run_omniscan", "OmniScanOutputs", "FeatureMetadata"]
