"""Command line interface for the evenness analysis toolkit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_MATCHING_CROSS,
    DEFAULT_MATCHING_WITHIN,
    EvennessPaths,
    FACTS_CONFIG,
    LENIENCY_RANDOM_SLOPE_DRIVERS,
    ROBUSTNESS_SCENARIOS,
)
from .data import build_fact_matrix
from .decomposition import run_oaxaca_blinder
from .foundation import run_phase_one
from .interaction import interaction_scan
from .leniency import compute_leniency_index
from .matching import perform_matching
from .modeling import fit_logistic, fit_mixed_effects, fit_ols, model_to_dict
from .plots import plot_balance, plot_icc_bars, plot_leniency_map, plot_shap_summary
from .predictive import gradient_boosting_diagnostics
from .robustness import run_robustness_suite
from .variance import variance_summary


def _load_fact_matrix(args: argparse.Namespace) -> pd.DataFrame:
    return build_fact_matrix(args.wide_csv, discussed_only=getattr(args, "discussed_only", False))


def _indicator_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for prefix in FACTS_CONFIG.multi_value_prefixes:
        indicator_cols = [
            c
            for c in df.columns
            if c.startswith(f"{prefix}_")
            and c
            not in {
                f"{prefix}_coverage_status",
                f"{prefix}_status",
                f"{prefix}_exclusivity_conflict",
                f"{prefix}_known",
                f"{prefix}_unknown",
            }
        ]
        cols.extend(indicator_cols)
    return cols


def _categorical_columns() -> list[str]:
    return ["breach_case", "organization_size_tier", "organization_type", "case_origin"]


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    base = ["n_principles_violated", "n_corrective_measures", "days_since_gdpr"]
    return [col for col in base if col in df.columns]


def _build_formula(outcome: str, df: pd.DataFrame) -> str:
    indicators = _indicator_columns(df)
    numeric = _numeric_columns(df)
    categorical = _categorical_columns()
    terms: list[str] = []
    terms.extend(indicators)
    terms.extend(numeric)
    terms.extend([f"C({col})" for col in categorical if col in df.columns])
    terms.append("C(country_code)")
    terms.append("C(dpa_name_canonical)")
    rhs = " + ".join(dict.fromkeys(terms))
    return f"{outcome} ~ {rhs}"


def _compute_balance(df: pd.DataFrame, matches: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(columns=["feature", "smd", "match_type"])
    records: list[dict[str, object]] = []
    index_df = df.set_index("decision_id")
    for match_type, group in matches.groupby("match_type"):
        for feature in features:
            if feature not in index_df.columns:
                continue
            source = pd.to_numeric(index_df.loc[group["source_id"], feature], errors="coerce")
            target = pd.to_numeric(index_df.loc[group["target_id"], feature], errors="coerce")
            weights = group["weight"].to_numpy()
            if source.isna().all() or target.isna().all():
                continue
            mean_s = np.average(source.fillna(source.mean()), weights=weights)
            mean_t = np.average(target.fillna(target.mean()), weights=weights)
            var_s = np.average((source - mean_s) ** 2, weights=weights)
            var_t = np.average((target - mean_t) ** 2, weights=weights)
            denom = np.sqrt((var_s + var_t) / 2)
            if denom == 0:
                continue
            smd = (mean_s - mean_t) / denom
            records.append({"feature": feature, "smd": smd, "match_type": match_type})
    return pd.DataFrame(records)


def cmd_prepare(args: argparse.Namespace) -> None:
    df = build_fact_matrix(args.wide_csv, discussed_only=args.discussed_only)
    out_path = Path(args.out or EvennessPaths().feature_cache)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    print(f"Wrote fact matrix to {out_path}")


def cmd_match(args: argparse.Namespace) -> None:
    df = _load_fact_matrix(args)
    within = perform_matching(df, DEFAULT_MATCHING_WITHIN, within_country=True, match_type="within")
    cross = perform_matching(df, DEFAULT_MATCHING_CROSS, within_country=False, match_type="cross")
    paths = EvennessPaths()
    paths.ensure()
    within.to_csv(args.within_out or paths.match_within_csv, index=False)
    cross.to_csv(args.cross_out or paths.match_cross_csv, index=False)
    features = DEFAULT_MATCHING_WITHIN.gower_numeric
    balance = pd.concat([
        _compute_balance(df, within, features),
        _compute_balance(df, cross, features),
    ])
    if args.balance_plot:
        plot_balance(balance, Path(args.balance_plot))
    print("Generated matching edges and balance diagnostics")


def cmd_models(args: argparse.Namespace) -> None:
    df = _load_fact_matrix(args)
    logistic_formula = _build_formula("fine_positive", df)
    linear_formula = _build_formula("fine_log1p", df)
    logistic = fit_logistic(df, logistic_formula, cluster_col="dpa_name_canonical")
    linear = fit_ols(df, linear_formula, cluster_col="dpa_name_canonical")
    random_terms = [term for term in ["q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY", "q46_vuln_CHILDREN"] if term in df.columns]
    re_formula = "1"
    if random_terms:
        re_formula = "1 + " + " + ".join(random_terms)
    mixed = fit_mixed_effects(
        df,
        linear_formula,
        group_col="dpa_name_canonical",
        re_formula=re_formula,
        vc_formula={"country": "0 + C(country_code)"},
    )
    out_dir = Path(args.model_dir or EvennessPaths().model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logistic.json").write_text(json.dumps(model_to_dict(logistic), indent=2))
    (out_dir / "linear.json").write_text(json.dumps(model_to_dict(linear), indent=2))
    (out_dir / "mixed.json").write_text(json.dumps(model_to_dict(mixed), indent=2))
    print(f"Saved model summaries to {out_dir}")


def cmd_leniency(args: argparse.Namespace) -> None:
    df = _load_fact_matrix(args)
    result = compute_leniency_index(df, random_slope_terms=[t for t in LENIENCY_RANDOM_SLOPE_DRIVERS if t in df.columns])
    paths = EvennessPaths()
    paths.ensure()
    result.frame.to_csv(args.out_csv or paths.leniency_csv, index=False)
    plot_leniency_map(result.frame, Path(args.out_plot or paths.leniency_plot))
    variance = variance_summary({"dpa": result.dpa_model, "country": result.country_model})
    if args.icc_plot:
        plot_icc_bars(variance, Path(args.icc_plot))
    print("Computed leniency index and plots")


def cmd_variance(args: argparse.Namespace) -> None:
    df = _load_fact_matrix(args)
    result = compute_leniency_index(df)
    variance = variance_summary({"dpa": result.dpa_model, "country": result.country_model})
    out_path = Path(args.out or EvennessPaths().variance_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    variance.to_csv(out_path, index=False)
    print(f"Wrote variance components to {out_path}")


def cmd_decompose(args: argparse.Namespace) -> None:
    df = _load_fact_matrix(args)
    features = _indicator_columns(df) + _numeric_columns(df)
    result = run_oaxaca_blinder(
        df,
        outcome=args.outcome,
        group_col=args.group_col,
        group_a=args.group_a,
        group_b=args.group_b,
        features=features,
    )
    out_dir = Path(args.out_dir or EvennessPaths().decomposition_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"oaxaca_{args.group_a}_vs_{args.group_b}_{args.outcome}.csv"
    result.to_csv(out_path, index=False)
    print(f"Wrote decomposition results to {out_path}")


def cmd_robustness(args: argparse.Namespace) -> None:
    df = _load_fact_matrix(args)
    logistic_formula = _build_formula("fine_positive", df)
    linear_formula = _build_formula("fine_log1p", df)
    outputs = run_robustness_suite(
        df,
        ROBUSTNESS_SCENARIOS,
        logistic_formula,
        linear_formula,
        fact_features=_indicator_columns(df) + _numeric_columns(df),
    )
    out_dir = Path(args.out_dir or EvennessPaths().robustness_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2))
    print(f"Stored robustness results in {out_dir}")


def cmd_interactions(args: argparse.Namespace) -> None:
    df = _load_fact_matrix(args)
    base_formula = _build_formula(args.outcome, df)
    result = interaction_scan(
        df,
        outcome=args.outcome,
        base_formula=base_formula,
        interaction_terms=_indicator_columns(df),
    )
    out_path = Path(args.out or EvennessPaths().model_dir / "interaction_scan.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Wrote interaction scan to {out_path}")


def cmd_predictive(args: argparse.Namespace) -> None:
    df = _load_fact_matrix(args)
    features = _indicator_columns(df) + _numeric_columns(df)
    result = gradient_boosting_diagnostics(
        df,
        outcome=args.outcome,
        feature_cols=features,
        classification=args.outcome == "fine_positive",
    )
    out_dir = Path(args.out_dir or EvennessPaths().model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"gb_{args.outcome}.json").write_text(
        json.dumps({"metrics": result["metrics"]}, indent=2)
    )
    plot_shap_summary(result["shap_summary"], out_dir / f"shap_{args.outcome}.png")
    print(f"Saved predictive diagnostics under {out_dir}")


def cmd_phase_one(args: argparse.Namespace) -> None:
    df = build_fact_matrix(args.wide_csv, discussed_only=args.discussed_only)
    outputs = run_phase_one(df)
    summary = {
        "X_full": len(outputs.X_full),
        "X_timeobs": len(outputs.X_timeobs),
        "CEM strata": outputs.twins_cem["stratum_id"].nunique() if not outputs.twins_cem.empty else 0,
        "Gower within matches": len(outputs.twins_gower_within),
        "Gower cross matches": len(outputs.twins_gower_cross),
        "Risk bands": outputs.twins_riskbands["risk_ventile"].nunique() if not outputs.twins_riskbands.empty else 0,
    }
    print("Phase 1 artefacts generated:")
    for key, value in summary.items():
        print(f"  - {key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GDPR evenness analysis toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p_phase1 = sub.add_parser("phase-one", help="Run Phase 1 foundation workflow")
    p_phase1.add_argument("--wide-csv", default=EvennessPaths().wide_csv)
    p_phase1.add_argument("--discussed-only", action="store_true")
    p_phase1.set_defaults(func=cmd_phase_one)

    p_prepare = sub.add_parser("prepare-data", help="Build fact matrix")
    p_prepare.add_argument("--wide-csv", default=EvennessPaths().wide_csv)
    p_prepare.add_argument("--out")
    p_prepare.add_argument("--discussed-only", action="store_true")
    p_prepare.set_defaults(func=cmd_prepare)

    p_match = sub.add_parser("build-matches", help="Construct matching edges")
    p_match.add_argument("--wide-csv", default=EvennessPaths().wide_csv)
    p_match.add_argument("--within-out")
    p_match.add_argument("--cross-out")
    p_match.add_argument("--balance-plot")
    p_match.set_defaults(func=cmd_match)

    p_models = sub.add_parser("fit-models", help="Run conditional disparity models")
    p_models.add_argument("--wide-csv", default=EvennessPaths().wide_csv)
    p_models.add_argument("--model-dir")
    p_models.set_defaults(func=cmd_models)

    p_leniency = sub.add_parser("leniency-index", help="Compute leniency/severity map")
    p_leniency.add_argument("--wide-csv", default=EvennessPaths().wide_csv)
    p_leniency.add_argument("--out-csv")
    p_leniency.add_argument("--out-plot")
    p_leniency.add_argument("--icc-plot")
    p_leniency.set_defaults(func=cmd_leniency)

    p_variance = sub.add_parser("variance", help="Export variance components")
    p_variance.add_argument("--wide-csv", default=EvennessPaths().wide_csv)
    p_variance.add_argument("--out")
    p_variance.set_defaults(func=cmd_variance)

    p_decomp = sub.add_parser("decompose", help="Run Oaxaca-Blinder decomposition")
    p_decomp.add_argument("--wide-csv", default=EvennessPaths().wide_csv)
    p_decomp.add_argument("--outcome", default="fine_log1p")
    p_decomp.add_argument("--group-col", default="country_code")
    p_decomp.add_argument("--group-a", required=True)
    p_decomp.add_argument("--group-b", required=True)
    p_decomp.add_argument("--out-dir")
    p_decomp.set_defaults(func=cmd_decompose)

    p_robust = sub.add_parser("robustness", help="Run robustness suite")
    p_robust.add_argument("--wide-csv", default=EvennessPaths().wide_csv)
    p_robust.add_argument("--out-dir")
    p_robust.set_defaults(func=cmd_robustness)

    p_interact = sub.add_parser("interaction-scan", help="Jurisdiction-driver interactions")
    p_interact.add_argument("--wide-csv", default=EvennessPaths().wide_csv)
    p_interact.add_argument("--outcome", default="fine_log1p")
    p_interact.add_argument("--out")
    p_interact.set_defaults(func=cmd_interactions)

    p_pred = sub.add_parser("predictive", help="Gradient boosting validation")
    p_pred.add_argument("--wide-csv", default=EvennessPaths().wide_csv)
    p_pred.add_argument("--outcome", default="fine_log1p")
    p_pred.add_argument("--out-dir")
    p_pred.set_defaults(func=cmd_predictive)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
