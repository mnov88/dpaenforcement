"""Phase 0 Omni-Scan pipeline for GDPR DPA decisions."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import sage
import shap
from catboost import CatBoostClassifier, CatBoostRegressor
from knockpy import knockoff_filter
from knockpy.knockoffs import GaussianSampler
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


NUMERIC_TYPES = {"int64", "float64", "int32", "float32", "Int64"}


@dataclass
class OmniScanConfig:
    """Configuration holder for the omni-scan pipeline."""

    wide_csv: Path
    long_tables_dir: Path
    output_dir: Path
    id_column: str = "decision_id"
    outcomes: Sequence[str] = (
        "fine_positive",
        "fine_eur",
        "fine_log1p",
        "enforcement_severity_index",
    )
    minimum_records: int = 5
    risk_band_quantiles: int = 20
    random_state: int = 42
    specification_bootstrap: int = 100
    stability_threshold: float = 0.1
    crt_permutations: int = 500
    sage_samples: int = 32

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class FeatureMetadata:
    name: str
    source: str
    kind: str
    origin: str
    description: Optional[str] = None


class OmniScanRunner:
    """Builds the feature universe and executes the omni-scan analytics."""

    def __init__(self, config: OmniScanConfig) -> None:
        self.config = config
        self.config.ensure_output_dir()
        self.wide_df: Optional[pd.DataFrame] = None
        self.features_df: Optional[pd.DataFrame] = None
        self.features_meta: Dict[str, FeatureMetadata] = {}
        self.outcomes_present: List[str] = []
        self.importances: Dict[str, pd.DataFrame] = {}
        self.shap_interactions: Dict[str, pd.DataFrame] = {}
        self.block_importances: Dict[str, pd.DataFrame] = {}
        self.sage_values: Dict[str, pd.Series] = {}
        self.stability_results: Dict[str, pd.DataFrame] = {}
        self.knockoff_results: Dict[str, List[str]] = {}
        self.crt_results: Dict[str, pd.DataFrame] = {}
        self.leniency_map: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Data loading and feature engineering
    # ------------------------------------------------------------------
    def load_wide_csv(self) -> None:
        wide_df = pd.read_csv(self.config.wide_csv)
        if self.config.id_column not in wide_df.columns:
            raise ValueError(f"Expected id column {self.config.id_column} in wide CSV")
        self.wide_df = wide_df.set_index(self.config.id_column)

    @staticmethod
    def _is_textual_column(name: str) -> bool:
        text_markers = ("_text", "_tokens", "_lang", "_raw")
        return any(name.endswith(marker) for marker in text_markers)

    @staticmethod
    def _sanitize_token(token: str) -> str:
        cleaned = token.strip().replace(" ", "_").replace("-", "_")
        cleaned = cleaned.replace("/", "_").replace("%", "pct")
        cleaned = cleaned.replace("(", "").replace(")", "")
        cleaned = cleaned.replace(".", "_")
        return cleaned.upper()

    def _drop_conflict_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        conflict_cols = [c for c in df.columns if c.endswith("_exclusivity_conflict")]
        if not conflict_cols:
            return df
        mask = (df[conflict_cols] == 1).any(axis=1)
        return df.loc[~mask].copy()

    def _build_multi_select_features(self) -> Tuple[pd.DataFrame, List[FeatureMetadata]]:
        assert self.wide_df is not None
        frames: List[pd.DataFrame] = []
        metadata: List[FeatureMetadata] = []
        for path in sorted(self.config.long_tables_dir.glob("*.csv")):
            table = pd.read_csv(path)
            if {"decision_id", "option"} - set(table.columns):
                continue
            if "token_status" in table.columns:
                table = table[table["token_status"].isin(["KNOWN", "PRESENT", "YES"])]
            if table.empty:
                continue
            table["value"] = 1
            pivot = table.pivot_table(
                index="decision_id",
                columns="option",
                values="value",
                aggfunc="max",
                fill_value=0,
            )
            pivot.columns = [
                f"{path.stem}__{self._sanitize_token(col)}" for col in pivot.columns
            ]
            frames.append(pivot)
            for col in pivot.columns:
                metadata.append(
                    FeatureMetadata(
                        name=col,
                        source=f"long:{path.stem}",
                        kind="binary",
                        origin=path.stem,
                    )
                )
        if not frames:
            return pd.DataFrame(index=self.wide_df.index), metadata
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.join(frame, how="outer")
        merged = merged.fillna(0).astype(np.int8)
        return merged, metadata

    def build_feature_matrix(self) -> None:
        if self.wide_df is None:
            self.load_wide_csv()
        assert self.wide_df is not None
        wide_df = self._drop_conflict_rows(self.wide_df)
        numeric_cols = [c for c in wide_df.columns if str(wide_df[c].dtype) in NUMERIC_TYPES]
        bool_cols = [c for c in wide_df.columns if str(wide_df[c].dtype) == "bool"]
        object_cols = [
            c
            for c in wide_df.columns
            if wide_df[c].dtype == "object" and not self._is_textual_column(c)
        ]
        base_df = wide_df[numeric_cols + bool_cols].copy()
        for col in bool_cols:
            base_df[col] = base_df[col].astype(np.int8)
            self.features_meta[col] = FeatureMetadata(col, "wide", "binary", col)
        for col in numeric_cols:
            self.features_meta[col] = FeatureMetadata(col, "wide", "numeric", col)
        status_cols = [c for c in object_cols if c.endswith("_status")]
        categorical_cols = [c for c in object_cols if c not in status_cols]
        status_df = pd.get_dummies(wide_df[status_cols], prefix_sep="__", dummy_na=False)
        for col in status_df.columns:
            status_df[col] = status_df[col].astype(np.int8)
            base = col.split("__")[0]
            self.features_meta[col] = FeatureMetadata(col, "status", "binary", base, "Status indicator")
        base_df = base_df.join(status_df)
        cat_df = pd.get_dummies(wide_df[categorical_cols], prefix_sep="__", dummy_na=False)
        for col in cat_df.columns:
            cat_df[col] = cat_df[col].astype(np.int8)
            base = col.split("__")[0]
            self.features_meta[col] = FeatureMetadata(col, "wide_categorical", "binary", base)
        base_df = base_df.join(cat_df)
        long_df, long_meta = self._build_multi_select_features()
        for meta in long_meta:
            self.features_meta[meta.name] = meta
        base_df = base_df.join(long_df, how="left")
        base_df = base_df.fillna(0)
        self.features_df = base_df.loc[:, sorted(base_df.columns)]

    # ------------------------------------------------------------------
    # Coverage ledger and metadata outputs
    # ------------------------------------------------------------------
    def save_feature_universe(self) -> None:
        if self.features_df is None:
            raise RuntimeError("Feature matrix not built")
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "id_column": self.config.id_column,
            "n_cases": int(self.features_df.shape[0]),
            "n_features": int(self.features_df.shape[1]),
            "features": [
                {
                    "name": meta.name,
                    "source": meta.source,
                    "kind": meta.kind,
                    "origin": meta.origin,
                    **({"description": meta.description} if meta.description else {}),
                }
                for meta in sorted(self.features_meta.values(), key=lambda m: m.name)
            ],
        }
        out_path = self.config.output_dir / "features_universe.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def compute_coverage_ledger(self) -> pd.DataFrame:
        if self.features_df is None:
            raise RuntimeError("Feature matrix not built")
        records = []
        for col in self.features_df.columns:
            series = self.features_df[col]
            meta = self.features_meta.get(col, FeatureMetadata(col, "unknown", str(series.dtype), col))
            observed = float(series.notna().mean())
            nonzero = float((series != 0).mean()) if observed > 0 else 0.0
            variance = float(series.var()) if observed > 0 else 0.0
            records.append(
                {
                    "feature": col,
                    "source": meta.source,
                    "kind": meta.kind,
                    "origin": meta.origin,
                    "dtype": str(series.dtype),
                    "observed_rate": observed,
                    "nonzero_rate": nonzero,
                    "variance": variance,
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
            )
        ledger = pd.DataFrame(records).sort_values("feature")
        ledger.to_csv(self.config.output_dir / "coverage_ledger.csv", index=False)
        return ledger

    def save_no_feature_left_behind(self, ledger: pd.DataFrame) -> None:
        summary = ["# No-Feature-Left-Behind Checklist", ""]
        summary.append(f"Total features: {ledger.shape[0]}")
        summary.append("## Features by source")
        for source, count in sorted(ledger.groupby("source").size().items()):
            summary.append(f"- {source}: {count}")
        missing = ledger.loc[ledger["observed_rate"] == 0, "feature"].tolist()
        if missing:
            summary.append("\n## Features with zero coverage")
            summary.extend(f"- {feat}" for feat in missing)
        else:
            summary.append("\nAll features have at least some observed data.")
        (self.config.output_dir / "no_feature_left_behind.md").write_text("\n".join(summary), encoding="utf-8")

    # ------------------------------------------------------------------
    # Modeling helpers
    # ------------------------------------------------------------------
    def _prepare_outcomes(self) -> None:
        assert self.wide_df is not None
        available = [outcome for outcome in self.config.outcomes if outcome in self.wide_df.columns]
        if self.features_df is not None:
            power_cols: List[Tuple[str, float]] = []
            for col in self.features_df.columns:
                if not col.startswith("corrective_powers__"):
                    continue
                positives = float(self.features_df[col].sum())
                if positives >= self.config.minimum_records:
                    power_cols.append((col, positives))
            power_cols.sort(key=lambda item: item[1], reverse=True)
            for col, _ in power_cols[:3]:
                available.append(col)
        self.outcomes_present = sorted(set(available))

    def _build_model_matrix(self, outcome: str) -> Tuple[pd.DataFrame, pd.Series]:
        assert self.features_df is not None and self.wide_df is not None
        if outcome in self.wide_df.columns:
            target = self.wide_df[outcome]
        else:
            target = self.features_df[outcome]
        mask = target.notna()
        y = target.loc[mask]
        X = self.features_df.loc[mask].copy()
        if outcome in X.columns:
            X = X.drop(columns=[outcome])
        return X, y

    @staticmethod
    def _is_binary(series: pd.Series) -> bool:
        values = series.dropna().unique()
        if len(values) == 0:
            return False
        return set(values) <= {0, 1, True, False}

    def _nested_cv(
        self,
        estimator: BaseEstimator,
        param_grid: Dict[str, List],
        X: np.ndarray,
        y: np.ndarray,
        *,
        is_classification: bool,
    ) -> float:
        n_samples = X.shape[0]
        n_splits = min(5, n_samples) if n_samples > 5 else 3
        if n_splits < 3:
            n_splits = 3 if n_samples >= 3 else 2
        if is_classification:
            outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_state)
            scoring = "roc_auc"
        else:
            outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_state)
            scoring = "neg_mean_squared_error"
        single_grid = all(len(v) == 1 for v in param_grid.values())
        scores: List[float] = []
        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            if single_grid:
                params = {k: v[0] for k, v in param_grid.items()}
                estimator.set_params(**params)
                estimator.fit(X_train, y_train)
                preds = estimator.predict_proba(X_test)[:, 1] if is_classification else estimator.predict(X_test)
            else:
                inner_splits = 3 if len(np.unique(y_train)) > 1 else 2
                if is_classification:
                    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=self.config.random_state)
                else:
                    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=self.config.random_state)
                search = GridSearchCV(estimator, param_grid, cv=inner_cv, scoring=scoring, n_jobs=-1)
                search.fit(X_train, y_train)
                best = search.best_estimator_
                preds = best.predict_proba(X_test)[:, 1] if is_classification else best.predict(X_test)
            if is_classification:
                score = roc_auc_score(y_test, preds)
            else:
                score = -mean_squared_error(y_test, preds)
            scores.append(score)
        return float(np.mean(scores))

    def _fit_lightgbm(self, X: pd.DataFrame, y: pd.Series, *, is_classification: bool) -> Tuple[BaseEstimator, float]:
        estimator: BaseEstimator
        param_grid = {
            "num_leaves": [63],
            "learning_rate": [0.1],
            "n_estimators": [400],
        }
        if is_classification:
            estimator = LGBMClassifier(random_state=self.config.random_state, verbose=-1)
        else:
            estimator = LGBMRegressor(random_state=self.config.random_state, verbose=-1)
        nested_score = self._nested_cv(estimator, param_grid, X.values, y.values, is_classification=is_classification)
        if all(len(v) == 1 for v in param_grid.values()):
            estimator.set_params(**{k: v[0] for k, v in param_grid.items()})
            estimator.fit(X.values, y.values)
            return estimator, nested_score
        search = GridSearchCV(estimator, param_grid, cv=3)
        search.fit(X.values, y.values)
        return search.best_estimator_, nested_score

    def _fit_catboost(self, X: pd.DataFrame, y: pd.Series, *, is_classification: bool) -> Tuple[BaseEstimator, float]:
        params = {
            "depth": 6,
            "learning_rate": 0.1,
            "iterations": 500,
            "verbose": False,
            "random_seed": self.config.random_state,
        }
        if is_classification:
            model = CatBoostClassifier(loss_function="Logloss", **params)
            model.fit(X, y)
            preds = model.predict_proba(X)[:, 1]
            score = roc_auc_score(y, preds)
        else:
            model = CatBoostRegressor(loss_function="RMSE", **params)
            model.fit(X, y)
            preds = model.predict(X)
            score = -mean_squared_error(y, preds)
        return model, float(score)

    def _compute_shap(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        *,
        is_classification: bool,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if is_classification and isinstance(shap_values, list):
            shap_matrix = shap_values[1]
        else:
            shap_matrix = shap_values
        shap_df = pd.DataFrame(shap_matrix, columns=X.columns, index=X.index)
        importance_df = (
            shap_df.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})
        )
        interaction_df = pd.DataFrame(columns=["feature_1", "feature_2", "mean_abs_interaction"])
        try:
            if X.shape[1] <= 50:
                interactions = explainer.shap_interaction_values(X)
                if is_classification and isinstance(interactions, list):
                    interactions = interactions[1]
                tri = np.triu_indices_from(interactions, k=1)
                inter_records = []
                for i, j in zip(*tri):
                    val = np.abs(interactions[:, i, j]).mean()
                    if val == 0:
                        continue
                    inter_records.append(
                        {
                            "feature_1": X.columns[i],
                            "feature_2": X.columns[j],
                            "mean_abs_interaction": float(val),
                        }
                    )
                interaction_df = pd.DataFrame(inter_records).sort_values("mean_abs_interaction", ascending=False)
            else:
                subset = importance_df["feature"].head(10).tolist()
                sub_matrix = X[subset]
                interactions = explainer.shap_interaction_values(sub_matrix)
                if is_classification and isinstance(interactions, list):
                    interactions = interactions[1]
                tri = np.triu_indices_from(interactions, k=1)
                inter_records = []
                for i, j in zip(*tri):
                    val = np.abs(interactions[:, i, j]).mean()
                    if val == 0:
                        continue
                    inter_records.append(
                        {
                            "feature_1": subset[i],
                            "feature_2": subset[j],
                            "mean_abs_interaction": float(val),
                        }
                    )
                interaction_df = pd.DataFrame(inter_records).sort_values("mean_abs_interaction", ascending=False)
        except Exception:
            interaction_df = pd.DataFrame(columns=["feature_1", "feature_2", "mean_abs_interaction"])
        return importance_df, interaction_df

    def _compute_sage(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series, *, is_classification: bool) -> pd.Series:
        imputer = sage.impute.KernelImputer(X.values)
        if is_classification:
            loss = sage.losses.binary_crossentropy
            predictor = lambda data: model.predict_proba(data)[:, 1]
        else:
            loss = sage.losses.mse
            predictor = model.predict
        estimator = sage.PermutationEstimator(model=predictor, imputer=imputer, loss=loss)
        values = estimator(X.values, y.values, n_samples=self.config.sage_samples)
        return pd.Series(values, index=X.columns)

    def _group_block(self, feature: str) -> str:
        if feature.startswith("country_code__"):
            return "geo.country"
        if feature.startswith("dpa_name_canonical__"):
            return "geo.dpa"
        if feature.startswith("isic_"):
            return "sector"
        if feature.startswith("art33"):
            return "article_33"
        if feature.startswith("art34"):
            return "article_34"
        if feature.startswith("breach"):
            return "breach"
        if "rights" in feature:
            return "rights"
        if feature.startswith("corrective_powers"):
            return "powers"
        if feature.startswith("vulnerable") or "vuln" in feature:
            return "vulnerable_groups"
        if feature.startswith("mitig"):
            return "mitigations"
        if feature.startswith("aggrav"):
            return "aggravating"
        return feature.split("__")[0]

    def run_baseline_models(self) -> None:
        if self.features_df is None or self.wide_df is None:
            raise RuntimeError("Feature matrix not built")
        self._prepare_outcomes()
        records = []
        for outcome in self.outcomes_present:
            X, y = self._build_model_matrix(outcome)
            if len(y.dropna()) < self.config.minimum_records:
                continue
            print(f"[Omni-Scan] Baseline modeling for {outcome} ({len(y)} cases)")
            is_binary = self._is_binary(y)
            scaler = StandardScaler(with_mean=False)
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
            lgbm_model, lgbm_score = self._fit_lightgbm(X_scaled, y, is_classification=is_binary)
            cat_model, cat_score = self._fit_catboost(X_scaled, y, is_classification=is_binary)
            sample_size = min(200, X_scaled.shape[0])
            if sample_size < X_scaled.shape[0]:
                sampled_index = X_scaled.sample(n=sample_size, random_state=self.config.random_state).index
                X_for_shap = X_scaled.loc[sampled_index]
                y_for_sage = y.loc[sampled_index]
            else:
                X_for_shap = X_scaled
                y_for_sage = y
            importance_df, interaction_df = self._compute_shap(lgbm_model, X_for_shap, is_classification=is_binary)
            self.importances[outcome] = importance_df.assign(outcome=outcome)
            self.shap_interactions[outcome] = interaction_df.assign(outcome=outcome)
            top_features = importance_df["feature"].head(10).tolist()
            sage_matrix = X_for_shap[top_features]
            print(f"[Omni-Scan] Computing SAGE for {outcome} using {len(top_features)} features")
            sage_series = self._compute_sage(lgbm_model, sage_matrix, y_for_sage, is_classification=is_binary)
            if len(top_features) < X_scaled.shape[1]:
                filler = pd.Series(0.0, index=[c for c in X_scaled.columns if c not in top_features])
                sage_series = pd.concat([sage_series, filler])
            self.sage_values[outcome] = sage_series.reindex(X_scaled.columns)
            block_importance = (
                importance_df.assign(block=importance_df["feature"].map(self._group_block))
                .groupby("block")["mean_abs_shap"].sum()
                .sort_values(ascending=False)
            )
            self.block_importances[outcome] = block_importance.to_frame(name="mean_abs_shap")
            records.append({"outcome": outcome, "model": "lightgbm", "score": lgbm_score})
            records.append({"outcome": outcome, "model": "catboost", "score": cat_score})
            # Country-conditioned SHAP summaries
            shap_explainer = shap.TreeExplainer(lgbm_model)
            shap_values = shap_explainer.shap_values(X_scaled)
            if is_binary and isinstance(shap_values, list):
                shap_matrix = shap_values[1]
            else:
                shap_matrix = shap_values
            shap_df = pd.DataFrame(shap_matrix, columns=X_scaled.columns, index=X_scaled.index)
            if "country_code" in self.wide_df.columns:
                country_summary = (
                    shap_df.groupby(self.wide_df.loc[shap_df.index, "country_code"]).mean().abs().stack().reset_index()
                )
                country_summary = country_summary.rename(columns={"level_1": "feature", 0: "mean_shap"})
                country_summary["outcome"] = outcome
                country_summary.to_csv(
                    self.config.output_dir / f"country_shap_{outcome}.csv", index=False
                )
            if "dpa_name_canonical" in self.wide_df.columns:
                dpa_summary = (
                    shap_df.groupby(self.wide_df.loc[shap_df.index, "dpa_name_canonical"]).mean().abs().stack().reset_index()
                )
                dpa_summary = dpa_summary.rename(columns={"level_1": "feature", 0: "mean_shap"})
                dpa_summary["outcome"] = outcome
                dpa_summary.to_csv(
                    self.config.output_dir / f"dpa_shap_{outcome}.csv", index=False
                )
        metrics_df = pd.DataFrame(records)
        metrics_df.to_csv(self.config.output_dir / "baseline_metrics.csv", index=False)
        if self.importances:
            all_importances = pd.concat(self.importances.values(), ignore_index=True)
            all_importances.to_csv(self.config.output_dir / "shap_importances.csv", index=False)
        interaction_frames = [df for df in self.shap_interactions.values() if not df.empty]
        if interaction_frames:
            pd.concat(interaction_frames, ignore_index=True).to_csv(
                self.config.output_dir / "shap_interactions.csv", index=False
            )
        block_frames = []
        for outcome, df in self.block_importances.items():
            frame = df.reset_index().rename(columns={"index": "block"})
            frame["outcome"] = outcome
            block_frames.append(frame)
        if block_frames:
            pd.concat(block_frames, ignore_index=True).to_csv(
                self.config.output_dir / "block_importance.csv", index=False
            )
        sage_df = pd.concat(
            [
                values.rename("sage_importance").to_frame().assign(outcome=outcome)
                for outcome, values in self.sage_values.items()
            ],
            ignore_index=False,
        ).reset_index().rename(columns={"index": "feature"})
        sage_df.to_csv(self.config.output_dir / "sage_importance.csv", index=False)

    # ------------------------------------------------------------------
    # Specification curve & stability selection
    # ------------------------------------------------------------------
    def _select_feature_subset(
        self,
        X: pd.DataFrame,
        *,
        include_sector: bool,
        include_time: bool,
        include_country: bool,
    ) -> pd.DataFrame:
        cols = X.columns.tolist()
        if not include_sector:
            cols = [c for c in cols if not c.startswith("isic")]
        if not include_time:
            cols = [c for c in cols if "decision_year" not in c and "decision_quarter" not in c]
        if not include_country:
            cols = [c for c in cols if not c.startswith("country_code__") and c != "country_code"]
        return X[cols]

    def _winsorize(self, series: pd.Series, limits: Tuple[float, float] = (0.01, 0.99)) -> pd.Series:
        lower = series.quantile(limits[0])
        upper = series.quantile(limits[1])
        return series.clip(lower, upper)

    def run_specification_curve(self) -> None:
        if self.features_df is None or self.wide_df is None:
            raise RuntimeError("Feature matrix not built")
        records = []
        for outcome in self.outcomes_present:
            X, y = self._build_model_matrix(outcome)
            if len(y.dropna()) < self.config.minimum_records:
                continue
            is_binary = self._is_binary(y)
            for include_sector in [True, False]:
                for include_time in [True, False]:
                    for include_country in [True, False]:
                        for winsorize in [True, False]:
                            X_subset = self._select_feature_subset(
                                X,
                                include_sector=include_sector,
                                include_time=include_time,
                                include_country=include_country,
                            )
                            y_spec = self._winsorize(y) if winsorize and not is_binary else y
                            if is_binary:
                                model = LogisticRegression(
                                    penalty="elasticnet",
                                    l1_ratio=0.5,
                                    solver="saga",
                                    max_iter=2000,
                                )
                                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_state)
                                scores = []
                                for train_idx, test_idx in cv.split(X_subset, y_spec):
                                    model.fit(X_subset.iloc[train_idx], y_spec.iloc[train_idx])
                                    preds = model.predict_proba(X_subset.iloc[test_idx])[:, 1]
                                    scores.append(roc_auc_score(y_spec.iloc[test_idx], preds))
                            else:
                                model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000)
                                cv = KFold(n_splits=3, shuffle=True, random_state=self.config.random_state)
                                scores = []
                                for train_idx, test_idx in cv.split(X_subset, y_spec):
                                    model.fit(X_subset.iloc[train_idx], y_spec.iloc[train_idx])
                                    preds = model.predict(X_subset.iloc[test_idx])
                                    scores.append(r2_score(y_spec.iloc[test_idx], preds))
                            records.append(
                                {
                                    "outcome": outcome,
                                    "include_sector": include_sector,
                                    "include_time": include_time,
                                    "include_country": include_country,
                                    "winsorize": winsorize,
                                    "mean_score": float(np.mean(scores)),
                                }
                            )
        pd.DataFrame(records).to_csv(self.config.output_dir / "specification_curve.csv", index=False)

    def run_stability_selection(self) -> None:
        if self.features_df is None or self.wide_df is None:
            raise RuntimeError("Feature matrix not built")
        drivers = []
        for outcome in self.outcomes_present:
            X, y = self._build_model_matrix(outcome)
            if len(y.dropna()) < self.config.minimum_records:
                continue
            is_binary = self._is_binary(y)
            n_samples = len(y)
            selections = []
            for seed in range(self.config.specification_bootstrap):
                rng = np.random.default_rng(seed + self.config.random_state)
                sample_idx = rng.choice(n_samples, size=max(3, int(0.75 * n_samples)), replace=False)
                X_boot = X.iloc[sample_idx]
                y_boot = y.iloc[sample_idx]
                if is_binary:
                    model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)
                    model.fit(X_boot, y_boot)
                    coefs = model.coef_.ravel()
                else:
                    model = Lasso(alpha=0.1, max_iter=5000)
                    model.fit(X_boot, y_boot)
                    coefs = model.coef_
                selections.append((np.abs(coefs) > 1e-6).astype(int))
            if not selections:
                continue
            selection_matrix = np.vstack(selections)
            probs = selection_matrix.mean(axis=0)
            stability_df = pd.DataFrame({"feature": X.columns, "selection_probability": probs})
            stability_df = stability_df.sort_values("selection_probability", ascending=False)
            stability_df.to_csv(self.config.output_dir / f"stability_{outcome}.csv", index=False)
            self.stability_results[outcome] = stability_df
            selected = stability_df.loc[
                stability_df["selection_probability"] >= self.config.stability_threshold, "feature"
            ].tolist()
            drivers.append({"outcome": outcome, "features": selected})
        (self.config.output_dir / "robust_driver_list.json").write_text(
            json.dumps(drivers, indent=2),
            encoding="utf-8",
        )

    def run_knockoffs(self) -> None:
        if self.features_df is None or self.wide_df is None:
            raise RuntimeError("Feature matrix not built")
        results = {}
        for outcome in self.outcomes_present:
            X, y = self._build_model_matrix(outcome)
            if len(y.dropna()) < self.config.minimum_records:
                continue
            is_binary = self._is_binary(y)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            sampler = GaussianSampler(X_scaled)
            Xk = sampler.sample_knockoffs()
            filt = knockoff_filter.KnockoffFilter(fdr=0.1)
            if is_binary:
                z_stats = filt.forward_logistic(X_scaled, Xk, y.values)
            else:
                z_stats = filt.forward_lasso(X_scaled, Xk, y.values)
            selected = filt.select(z_stats)
            results[outcome] = [X.columns[i] for i in selected]
        self.knockoff_results = results
        (self.config.output_dir / "knockoff_selected.json").write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Fairness diagnostics
    # ------------------------------------------------------------------
    def run_crt(self) -> None:
        if self.features_df is None or self.wide_df is None:
            raise RuntimeError("Feature matrix not built")
        jurisdiction_cols = [
            c for c in self.features_df.columns if c.startswith("country_code__") or c.startswith("dpa_name_canonical__")
        ]
        if not jurisdiction_cols:
            return
        non_jurisdiction = [c for c in self.features_df.columns if c not in jurisdiction_cols]
        results = []
        leniency_frames = []
        for outcome in self.outcomes_present:
            if outcome in jurisdiction_cols:
                continue
            X, y = self._build_model_matrix(outcome)
            if len(y.dropna()) < self.config.minimum_records:
                continue
            is_binary = self._is_binary(y)
            X_nonjur = X[non_jurisdiction].fillna(0)
            X_jur = X[jurisdiction_cols].fillna(0)
            folds = KFold(n_splits=3, shuffle=True, random_state=self.config.random_state)
            residual_y = pd.Series(index=y.index, dtype=float)
            residual_t = pd.DataFrame(index=y.index, columns=jurisdiction_cols, dtype=float)
            for train_idx, test_idx in folds.split(X_nonjur, y):
                X_train, X_test = X_nonjur.iloc[train_idx], X_nonjur.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                if is_binary:
                    model_y = LGBMClassifier(random_state=self.config.random_state)
                    model_y.fit(X_train, y_train)
                    pred_y = model_y.predict_proba(X_test)[:, 1]
                else:
                    model_y = LGBMRegressor(random_state=self.config.random_state)
                    model_y.fit(X_train, y_train)
                    pred_y = model_y.predict(X_test)
                residual_y.iloc[test_idx] = y_test - pred_y
                for col in jurisdiction_cols:
                    clf = LogisticRegression(max_iter=1000)
                    clf.fit(X_train, X_jur.iloc[train_idx][col])
                    pred_t = clf.predict_proba(X_test)[:, 1]
                    residual_t.loc[y.iloc[test_idx].index, col] = X_jur.iloc[test_idx][col] - pred_t
            for col in jurisdiction_cols:
                ry = residual_y.values
                rt = residual_t[col].values
                corr = np.corrcoef(ry, rt)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                perm_values = []
                for seed in range(self.config.crt_permutations):
                    rng = np.random.default_rng(seed + self.config.random_state)
                    perm = rng.permutation(ry)
                    perm_corr = np.corrcoef(perm, rt)[0, 1]
                    if np.isnan(perm_corr):
                        perm_corr = 0.0
                    perm_values.append(abs(perm_corr))
                p_val = float((np.sum(np.abs(perm_values) >= abs(corr)) + 1) / (self.config.crt_permutations + 1))
                results.append(
                    {
                        "outcome": outcome,
                        "jurisdiction_feature": col,
                        "corr_stat": float(corr),
                        "p_value": p_val,
                    }
                )
            if "country_code" in self.wide_df.columns:
                groups = self.wide_df.loc[residual_y.index, "country_code"]
                summary = (
                    residual_y.groupby(groups)
                    .agg(["mean", "count", "std"])
                    .reset_index()
                    .rename(columns={"mean": "residual_mean", "count": "n", "std": "residual_std"})
                )
                summary["outcome"] = outcome
                summary["ci_lower"] = summary["residual_mean"] - 1.96 * summary["residual_std"] / np.sqrt(summary["n"].clip(lower=1))
                summary["ci_upper"] = summary["residual_mean"] + 1.96 * summary["residual_std"] / np.sqrt(summary["n"].clip(lower=1))
                leniency_frames.append(summary)
        if results:
            result_df = pd.DataFrame(results).sort_values("p_value")
            m = len(result_df)
            result_df["bh_q_value"] = [min(p * m / (i + 1), 1.0) for i, p in enumerate(result_df["p_value"])]
            result_df.to_csv(self.config.output_dir / "crt_results.csv", index=False)
            self.crt_results = {outcome: result_df[result_df["outcome"] == outcome] for outcome in result_df["outcome"].unique()}
        if leniency_frames:
            leniency_df = pd.concat(leniency_frames, ignore_index=True)
            leniency_df.to_csv(self.config.output_dir / "leniency_map.csv", index=False)
            self.leniency_map = leniency_df

    # ------------------------------------------------------------------
    # Network and risk-band analysis
    # ------------------------------------------------------------------
    def build_network(self) -> None:
        if not self.importances:
            return
        edges = []
        for outcome, df in self.importances.items():
            for _, row in df.iterrows():
                edges.append({"feature": row["feature"], "outcome": outcome, "weight": row["mean_abs_shap"]})
        edges_df = pd.DataFrame(edges)
        edges_df.to_csv(self.config.output_dir / "bipartite_edges.csv", index=False)
        graph = nx.Graph()
        for _, row in edges_df.iterrows():
            graph.add_node(row["feature"], bipartite=0)
            graph.add_node(row["outcome"], bipartite=1)
            graph.add_edge(row["feature"], row["outcome"], weight=row["weight"])
        communities = nx.algorithms.community.louvain_communities(graph, seed=self.config.random_state)
        community_records = []
        for cid, community in enumerate(communities):
            for node in community:
                community_records.append({"node": node, "community": cid})
        pd.DataFrame(community_records).to_csv(
            self.config.output_dir / "network_communities.csv", index=False
        )

    def risk_band_parity(self) -> None:
        if self.features_df is None or self.wide_df is None or "fine_positive" not in self.outcomes_present:
            return
        X, y = self._build_model_matrix("fine_positive")
        model = LGBMClassifier(random_state=self.config.random_state)
        model.fit(X, y)
        scores = model.predict_proba(X)[:, 1]
        quantiles = np.quantile(scores, np.linspace(0, 1, self.config.risk_band_quantiles + 1))
        bands = pd.cut(scores, bins=np.unique(quantiles), include_lowest=True, labels=False)
        self.wide_df.loc[y.index, "risk_band"] = bands
        records = []
        if "country_code" not in self.wide_df.columns:
            return
        for band in sorted(np.unique(bands[~pd.isna(bands)])):
            band_mask = bands == band
            band_outcome = y.loc[band_mask]
            countries = self.wide_df.loc[band_outcome.index, "country_code"]
            for country in countries.unique():
                values = band_outcome.loc[countries == country]
                records.append(
                    {
                        "band": int(band),
                        "country": country,
                        "mean_outcome": float(values.mean()),
                        "count": int(values.shape[0]),
                    }
                )
            unique_countries = countries.unique()
            if len(unique_countries) >= 2:
                base = unique_countries[0]
                base_values = band_outcome.loc[countries == base]
                for other in unique_countries[1:]:
                    other_values = band_outcome.loc[countries == other]
                    if len(base_values) < 2 or len(other_values) < 2:
                        continue
                    ks = ks_2samp(base_values, other_values)
                    records.append(
                        {
                            "band": int(band),
                            "country_pair": f"{base} vs {other}",
                            "ks_stat": float(ks.statistic),
                            "ks_pvalue": float(ks.pvalue),
                        }
                    )
        if records:
            pd.DataFrame(records).to_csv(self.config.output_dir / "risk_band_parity.csv", index=False)

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run(self) -> None:
        self.load_wide_csv()
        self.build_feature_matrix()
        self.save_feature_universe()
        ledger = self.compute_coverage_ledger()
        self.save_no_feature_left_behind(ledger)
        self.run_baseline_models()
        self.run_specification_curve()
        self.run_stability_selection()
        self.run_knockoffs()
        self.run_crt()
        self.build_network()
        self.risk_band_parity()
