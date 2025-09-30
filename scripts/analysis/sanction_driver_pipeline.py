from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score

from statsmodels.miscmodels.ordinal_model import OrderedModel

from .build_feature_matrix import META_SUFFIXES, POWER_TOKEN_COLUMNS

plt.switch_backend("Agg")

NUMERIC_BASE = (
    "breach_case",
    "n_principles_discussed",
    "n_principles_violated",
    "n_corrective_measures",
    "turnover_log1p",
    "decision_year_centered",
)

INDICATOR_FEATURES = (
    "art33_required_flag",
    "art33_submitted_flag",
    "art33_timely_flag",
    "art34_required_flag",
    "subjects_notified_flag",
    "q15_case_initiation_BREACH_NOTIFICATION",
    "q15_case_initiation_EX_OFFICIO_DPA_INITIATIVE",
    "q15_case_initiation_MEDIA_PUBLIC_ATTENTION",
    "q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY",
    "q25_sensitive_data_ARTICLE_10_CRIMINAL",
    "q28_mitigations_IMMEDIATE_CONTAINMENT",
    "q28_mitigations_SECURITY_IMPROVEMENTS",
    "q28_mitigations_COOPERATION_WITH_DPA",
    "q41_aggrav_PREVIOUS_INFRINGEMENTS",
    "q41_aggrav_INTENT_NEGLIGENCE",
    "q42_mitig_COOPERATION_WITH_AUTHORITY",
    "q47_remedial_BRING_PROCESSING_INTO_COMPLIANCE",
    "q47_remedial_RECTIFICATION_ERASURE",
    "q46_vuln_CHILDREN",
    "q46_vuln_EMPLOYEES",
    "q46_vuln_FINANCIALLY_VULNERABLE",
)

MULTI_PREFIXES = (
    "q21_breach_types",
    "q25_sensitive_data",
    "q28_mitigations",
    "q41_aggrav",
    "q42_mitig",
    "q46_vuln",
    "q47_remedial",
)

CATEGORY_FEATURES = ("country_group", "isic_section")

LOGIT_TARGETS = ("power_fine_flag", "power_warning_flag", "power_reprimand_flag", "power_none_flag")

ORDINAL_TARGET = "severity_rank"

SHAP_TARGETS = ("power_fine_flag", "power_warning_flag", "fine_log1p", "severity_rank")

CAUSAL_TREATMENTS = (
    "art33_timely_flag",
    "q15_case_initiation_BREACH_NOTIFICATION",
    "q15_case_initiation_EX_OFFICIO_DPA_INITIATIVE",
    "q28_mitigations_IMMEDIATE_CONTAINMENT",
    "q28_mitigations_COOPERATION_WITH_DPA",
    "q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY",
)

BASE_COLUMNS = (
    "decision_id",
    "country_code",
    "country_group",
    "isic_section",
    "dpa_name_canonical",
    "decision_year",
    "breach_case",
    "n_principles_discussed",
    "n_principles_violated",
    "n_corrective_measures",
    "turnover_log1p",
    "fine_log1p",
    "fine_eur",
    "power_fine_flag",
    "power_warning_flag",
    "power_reprimand_flag",
    "power_none_flag",
    "power_combined_flag",
    "power_any_flag",
    "art33_required_flag",
    "art33_submitted_flag",
    "art33_timely_flag",
    "art34_required_flag",
    "subjects_notified_flag",
    "art33_late_flag",
)

RANDOM_STATE = 42


@dataclass
class Config:
    feature_matrix: Path
    metadata_json: Path
    out_dir: Path
    latent_scores: Path | None = None
    shap_sample: int = 500


def parse_args(argv: Iterable[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description="Analyse sanction drivers in GDPR decisions")
    parser.add_argument(
        "--feature-matrix",
        type=Path,
        default=Path("outputs/analysis/feature_matrix.parquet"),
        help="Path to the feature matrix parquet",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=Path("outputs/analysis/feature_matrix_metadata.json"),
        help="Path to the feature matrix metadata JSON",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/analysis/sanction_drivers"),
        help="Directory to write outputs to",
    )
    parser.add_argument(
        "--latent-scores",
        type=Path,
        default=Path("outputs/analysis/interaction/latent_scores.parquet"),
        help="Optional latent scores parquet to merge",
    )
    parser.add_argument(
        "--shap-sample",
        type=int,
        default=500,
        help="Maximum number of rows to use for SHAP summaries",
    )
    args = parser.parse_args(argv)
    latent = args.latent_scores if args.latent_scores.exists() else None
    return Config(
        feature_matrix=args.feature_matrix,
        metadata_json=args.metadata_json,
        out_dir=args.out_dir,
        latent_scores=latent,
        shap_sample=args.shap_sample,
    )


def _load_metadata(metadata_path: Path) -> dict[str, Sequence[str]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata.get("column_groups", {})


def _load_dataframe(path: Path, latent: Path | None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if latent and latent.exists():
        latent_df = pd.read_parquet(latent)
        df = df.join(latent_df, how="left")
    return df


def _severity_from_powers(df: pd.DataFrame) -> pd.Series:
    from .hierarchical_severity_model import SEVERITY_ORDER, _severity_label

    if "severity_rank" in df.columns and df["severity_rank"].notna().any():
        return df["severity_rank"].fillna(0).astype(int)
    labels = df.apply(_severity_label, axis=1)
    return labels.map(SEVERITY_ORDER).fillna(0).astype(int)


def _indicator_columns(metadata: dict[str, Sequence[str]], key: str, df: pd.DataFrame) -> list[str]:
    cols = []
    for col in metadata.get(key, []):
        if any(col.endswith(sfx) for sfx in META_SUFFIXES):
            continue
        if col in POWER_TOKEN_COLUMNS:
            continue
        if col in df.columns:
            cols.append(col)
    return cols


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(float)


def build_analysis_frame(df: pd.DataFrame, metadata: dict[str, Sequence[str]]) -> pd.DataFrame:
    cols = [col for col in BASE_COLUMNS if col in df.columns]
    frame = df[cols].copy()
    frame["turnover_log1p"] = frame["turnover_log1p"].fillna(frame["turnover_log1p"].median())
    frame["decision_year_centered"] = frame["decision_year"].fillna(frame["decision_year"].median()) - frame["decision_year"].median()
    frame["severity_rank"] = _severity_from_powers(df)

    for prefix in MULTI_PREFIXES:
        columns = _indicator_columns(metadata, prefix, df)
        if not columns:
            continue
        values = df[columns].apply(_safe_numeric)
        frame[f"{prefix}_count"] = values.sum(axis=1)

    for column in INDICATOR_FEATURES:
        if column in df.columns:
            frame[column] = _safe_numeric(df[column])

    frame["power_fine_flag"] = _safe_numeric(frame.get("power_fine_flag", 0))
    frame["power_warning_flag"] = _safe_numeric(frame.get("power_warning_flag", 0))
    frame["power_reprimand_flag"] = _safe_numeric(frame.get("power_reprimand_flag", 0))
    frame["power_none_flag"] = _safe_numeric(frame.get("power_none_flag", 0))
    frame["power_combined_flag"] = _safe_numeric(frame.get("power_combined_flag", 0))

    for numeric_col in NUMERIC_BASE:
        if numeric_col in df.columns:
            frame[numeric_col] = _safe_numeric(df[numeric_col])

    for numeric_col in NUMERIC_BASE:
        if numeric_col in frame.columns:
            std = frame[numeric_col].std(ddof=0)
            if std and std > 0:
                frame[numeric_col] = (frame[numeric_col] - frame[numeric_col].mean()) / std

    for prefix in MULTI_PREFIXES:
        count_col = f"{prefix}_count"
        if count_col in frame.columns:
            std = frame[count_col].std(ddof=0)
            if std and std > 0:
                frame[count_col] = (frame[count_col] - frame[count_col].mean()) / std

    return frame


def design_matrix(frame: pd.DataFrame, numeric_features: Sequence[str], categorical_features: Sequence[str]) -> pd.DataFrame:
    cols = [col for col in numeric_features if col in frame.columns]
    X = frame[cols].copy().fillna(0).astype(float)
    cat_cols = [col for col in categorical_features if col in frame.columns]
    if cat_cols:
        dummies = pd.get_dummies(frame[cat_cols].fillna("UNKNOWN"), drop_first=True, dtype=float)
        X = pd.concat([X, dummies], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    return X


def fit_logistic_models(frame: pd.DataFrame, config: Config, features: Sequence[str]) -> tuple[pd.DataFrame, dict[str, sm.GLMResultsWrapper]]:
    outputs: list[pd.DataFrame] = []
    models: dict[str, sm.GLMResultsWrapper] = {}
    X_common = design_matrix(frame, features, CATEGORY_FEATURES)
    X_common = sm.add_constant(X_common, has_constant="add")
    for target in LOGIT_TARGETS:
        if target not in frame.columns:
            continue
        subset = frame[[target]].join(X_common)
        subset = subset.dropna(subset=[target])
        y = subset[target]
        if y.nunique() < 2 or len(subset) < 100:
            continue
        model = sm.GLM(y, subset.drop(columns=[target]), family=sm.families.Binomial())
        result = model.fit()
        models[target] = result
        coef = result.summary2().tables[1].reset_index().rename(columns={"index": "feature"})
        coef["target"] = target
        try:
            margeff = result.get_margeff()
            me_frame = margeff.summary_frame()
            me_frame = me_frame.reset_index().rename(columns={"index": "feature"})
            me_frame["target"] = target
            me_frame.to_csv(config.out_dir / f"{target}_marginal_effects.csv", index=False)
        except Exception:
            pass
        coef.to_csv(config.out_dir / f"{target}_logit_coefficients.csv", index=False)
        with (config.out_dir / f"{target}_logit_summary.txt").open("w", encoding="utf-8") as fout:
            fout.write(result.summary2().as_text())
        outputs.append(coef)
    combined = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    if not combined.empty:
        combined.to_csv(config.out_dir / "logit_coefficients_all.csv", index=False)
    return combined, models


def evaluate_calibration(frame: pd.DataFrame, model: sm.GLMResultsWrapper, target: str, config: Config) -> pd.DataFrame:
    X = sm.add_constant(design_matrix(frame, NUMERIC_BASE + INDICATOR_FEATURES + tuple(f"{p}_count" for p in MULTI_PREFIXES), CATEGORY_FEATURES), has_constant="add")
    mask = frame[target].notna()
    preds = model.predict(X.loc[mask])
    actual = frame.loc[mask, target]
    metrics = {
        "brier": brier_score_loss(actual, preds),
        "auc": roc_auc_score(actual, preds) if actual.nunique() > 1 else float("nan"),
    }
    with (config.out_dir / "calibration_metrics.json").open("w", encoding="utf-8") as fout:
        json.dump(metrics, fout, indent=2)
    bins = pd.cut(preds, bins=np.linspace(0, 1, 11), include_lowest=True)
    calibration = pd.DataFrame({"pred_bin": bins, "pred": preds, "actual": actual})
    calib_summary = calibration.groupby("pred_bin").agg(mean_pred=("pred", "mean"), mean_actual=("actual", "mean"), cases=("pred", "count")).reset_index()
    plt.figure(figsize=(5, 5))
    plt.plot(calib_summary["mean_pred"], calib_summary["mean_actual"], marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Ideal")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed share")
    plt.title(f"Calibration – {target}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.out_dir / f"calibration_{target}.png", dpi=200)
    plt.close()
    return calib_summary


def fairness_diagnostics(frame: pd.DataFrame, model: sm.GLMResultsWrapper, target: str, config: Config) -> pd.DataFrame:
    X = sm.add_constant(design_matrix(frame, NUMERIC_BASE + INDICATOR_FEATURES + tuple(f"{p}_count" for p in MULTI_PREFIXES), CATEGORY_FEATURES), has_constant="add")
    mask = frame[target].notna()
    preds = model.predict(X.loc[mask])
    compare = frame.loc[mask, ["country_group", "isic_section", target]].copy()
    compare["pred"] = preds
    rows = []
    for group_col in ("country_group", "isic_section"):
        if group_col not in compare.columns:
            continue
        grouped = compare.groupby(group_col).agg(
            predicted_mean=("pred", "mean"),
            actual_mean=(target, "mean"),
            cases=(target, "count"),
        ).reset_index()
        grouped = grouped.rename(columns={group_col: "group_value"})
        grouped.insert(0, "group_axis", group_col)
        rows.append(grouped)
    fairness_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not fairness_df.empty:
        fairness_df.to_csv(config.out_dir / "fairness_summary.csv", index=False)
    return fairness_df


def fit_ordinal_model(frame: pd.DataFrame, features: Sequence[str], config: Config) -> tuple[pd.DataFrame, OrderedModel]:
    if ORDINAL_TARGET not in frame.columns:
        return pd.DataFrame(), None
    X = design_matrix(frame, features, CATEGORY_FEATURES)
    if X.empty:
        return pd.DataFrame(), None
    y = frame[ORDINAL_TARGET]
    mask = y.notna()
    if mask.sum() < 200:
        return pd.DataFrame(), None
    model = OrderedModel(y[mask], X.loc[mask], distr="logit")
    result = model.fit(method="bfgs", maxiter=200, disp=False)
    with (config.out_dir / "ordinal_model_summary.txt").open("w", encoding="utf-8") as fout:
        fout.write(result.summary().as_text())
    params = result.params
    conf_int = result.conf_int()
    ord_df = pd.DataFrame(
        {
            "feature": params.index,
            "coef": params.values,
            "std_err": result.bse,
            "p_value": result.pvalues,
            "conf_low": conf_int[0].values,
            "conf_high": conf_int[1].values,
        }
    )
    ord_df.to_csv(config.out_dir / "ordinal_coefficients.csv", index=False)
    return ord_df, model


def fit_conditional_fine_model(frame: pd.DataFrame, features: Sequence[str], config: Config) -> pd.DataFrame:
    mask = (frame.get("power_fine_flag", 0) == 1) & frame["fine_log1p"].notna()
    if mask.sum() < 200:
        return pd.DataFrame()
    X = design_matrix(frame.loc[mask], features, CATEGORY_FEATURES)
    X = sm.add_constant(X, has_constant="add")
    y = frame.loc[mask, "fine_log1p"]
    model = sm.OLS(y, X).fit()
    with (config.out_dir / "conditional_fine_summary.txt").open("w", encoding="utf-8") as fout:
        fout.write(model.summary().as_text())
    coef = model.summary2().tables[1].reset_index().rename(columns={"index": "feature"})
    coef.to_csv(config.out_dir / "conditional_fine_coefficients.csv", index=False)
    return coef


def train_gradient_boosting(frame: pd.DataFrame, features: Sequence[str], config: Config) -> dict[str, dict[str, object]]:
    X = design_matrix(frame, features, CATEGORY_FEATURES)
    models: dict[str, dict[str, object]] = {}
    for target in SHAP_TARGETS:
        if target not in frame.columns:
            continue
        mask = frame[target].notna()
        if mask.sum() < 300:
            continue
        X_target = X.loc[mask]
        y = frame.loc[mask, target]
        if y.nunique() <= 1:
            continue
        if y.dtype.kind in {"f", "i", "u"} and y.nunique() == 2:
            model = GradientBoostingClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            )
            model.fit(X_target, y)
            y_pred = model.predict_proba(X_target)[:, 1]
            auc = roc_auc_score(y, y_pred)
        else:
            model = GradientBoostingRegressor(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            )
            model.fit(X_target, y)
            auc = float("nan")
        models[target] = {
            "model": model,
            "X": X_target,
            "y": y,
            "auc": auc,
        }
        with (config.out_dir / f"gb_{target}_metrics.json").open("w", encoding="utf-8") as fout:
            json.dump({"auc": auc}, fout, indent=2)
    return models


def compute_shap(models: dict[str, dict[str, object]], config: Config) -> None:
    shap_dir = config.out_dir / "shap"
    shap_dir.mkdir(exist_ok=True)
    for target, payload in models.items():
        model = payload["model"]
        X = payload["X"]
        sample = X.sample(n=min(config.shap_sample, len(X)), random_state=RANDOM_STATE) if len(X) > config.shap_sample else X
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_array = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_array = shap_values
        shap_df = pd.DataFrame(shap_array, columns=sample.columns)
        shap_summary = shap_df.abs().mean().sort_values(ascending=False).to_frame(name="mean_abs_shap")
        shap_summary = shap_summary.reset_index().rename(columns={"index": "feature"})
        shap_summary.to_csv(shap_dir / f"{target}_shap_importance.csv", index=False)
        shap.summary_plot(shap_array, sample, show=False)
        plt.tight_layout()
        plt.savefig(shap_dir / f"{target}_shap_summary.png", dpi=200)
        plt.close()
        shap_df.to_parquet(shap_dir / f"{target}_shap_values.parquet")


def causal_forest(frame: pd.DataFrame, features: Sequence[str], config: Config) -> pd.DataFrame:
    X = design_matrix(frame, features, CATEGORY_FEATURES)
    outcome = frame.get("power_fine_flag")
    if outcome is None or outcome.nunique() < 2:
        return pd.DataFrame()
    rows = []
    for treatment in CAUSAL_TREATMENTS:
        if treatment not in frame.columns:
            continue
        mask = frame[treatment].notna() & outcome.notna()
        subset = frame.loc[mask]
        X_treat = X.loc[mask]
        treated = subset[treatment] == 1
        if treated.sum() < 100 or (~treated).sum() < 100:
            continue
        model_t = RandomForestRegressor(n_estimators=400, min_samples_leaf=20, random_state=RANDOM_STATE)
        model_c = RandomForestRegressor(n_estimators=400, min_samples_leaf=20, random_state=RANDOM_STATE + 1)
        model_t.fit(X_treat[treated], outcome[mask][treated])
        model_c.fit(X_treat[~treated], outcome[mask][~treated])
        cate = model_t.predict(X_treat) - model_c.predict(X_treat)
        rows.append(
            {
                "treatment": treatment,
                "mean_effect": float(np.mean(cate)),
                "iqr": float(np.percentile(cate, 75) - np.percentile(cate, 25)),
                "treated_n": int(treated.sum()),
                "control_n": int((~treated).sum()),
            }
        )
    cf_df = pd.DataFrame(rows)
    if not cf_df.empty:
        cf_df.to_csv(config.out_dir / "causal_forest_summary.csv", index=False)
    return cf_df


def policy_scenarios(frame: pd.DataFrame, model: sm.GLMResultsWrapper, features: Sequence[str], config: Config) -> pd.DataFrame:
    if model is None:
        return pd.DataFrame()
    X = design_matrix(frame, features, CATEGORY_FEATURES)
    X = sm.add_constant(X, has_constant="add")
    baseline_idx = (frame["severity_rank"] - frame["severity_rank"].median()).abs().idxmin()
    baseline = X.loc[[baseline_idx]].copy()
    scenarios = {"baseline": baseline.iloc[0].copy()}
    modifications = {
        "late_notification": {"art33_timely_flag": 0, "art33_late_flag": 1},
        "timely_notification": {"art33_timely_flag": 1, "art33_late_flag": 0},
        "ex_officio": {"q15_case_initiation_EX_OFFICIO_DPA_INITIATIVE": 1},
        "breach_notification": {"q15_case_initiation_BREACH_NOTIFICATION": 1},
        "mitigation_heavy": {"q28_mitigations_IMMEDIATE_CONTAINMENT": 1, "q28_mitigations_SECURITY_IMPROVEMENTS": 1},
        "sensitive_data": {"q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY": 1},
        "aggravating": {"q41_aggrav_PREVIOUS_INFRINGEMENTS": 1, "q41_aggrav_INTENT_NEGLIGENCE": 1},
    }
    rows = []
    for name, change in modifications.items():
        scenario = scenarios["baseline"].copy()
        for key, value in change.items():
            if key in scenario.index:
                scenario[key] = value
        prob = float(model.predict(pd.DataFrame([scenario]))[0])
        rows.append({"scenario": name, "predicted_prob": prob})
    scenario_df = pd.DataFrame(rows)
    scenario_df.to_csv(config.out_dir / "policy_scenarios.csv", index=False)
    return scenario_df


def main(argv: Iterable[str] | None = None) -> None:
    config = parse_args(argv)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = _load_metadata(config.metadata_json)
    df = _load_dataframe(config.feature_matrix, config.latent_scores)
    analysis_frame = build_analysis_frame(df, metadata)

    feature_columns = list(NUMERIC_BASE) + list(INDICATOR_FEATURES) + [f"{p}_count" for p in MULTI_PREFIXES]
    logit_table, models = fit_logistic_models(analysis_frame, config, feature_columns)

    power_model = models.get("power_fine_flag")
    if power_model is not None:
        calibration_df = evaluate_calibration(analysis_frame, power_model, "power_fine_flag", config)
        fairness_df = fairness_diagnostics(analysis_frame, power_model, "power_fine_flag", config)
    else:
        calibration_df = pd.DataFrame()
        fairness_df = pd.DataFrame()

    ordinal_df, ordinal_model = fit_ordinal_model(analysis_frame, feature_columns, config)
    fine_df = fit_conditional_fine_model(analysis_frame, feature_columns, config)

    gb_models = train_gradient_boosting(analysis_frame, feature_columns, config)
    if gb_models:
        compute_shap(gb_models, config)

    cf_df = causal_forest(analysis_frame, feature_columns, config)
    scenarios = policy_scenarios(analysis_frame, power_model, feature_columns, config)

    snapshot = {
        "logit_targets": list(models.keys()),
        "ordinal_available": bool(ordinal_model),
        "conditional_fine_available": not fine_df.empty,
        "gradient_boosting_targets": list(gb_models.keys()),
        "causal_forest_treatments": cf_df["treatment"].tolist() if not cf_df.empty else [],
        "fairness_axes": ["country_group", "isic_section"] if not fairness_df.empty else [],
        "scenarios": scenarios.to_dict(orient="records") if not scenarios.empty else [],
    }
    with (config.out_dir / "pipeline_snapshot.json").open("w", encoding="utf-8") as fout:
        json.dump(snapshot, fout, indent=2)


if __name__ == "__main__":  # pragma: no cover
    main()
