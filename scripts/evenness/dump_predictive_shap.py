"""Export predictive SHAP rankings to CSV for quick inspection."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import build_fact_matrix
from .cli import _indicator_columns, _numeric_columns, _categorical_columns
from .predictive import gradient_boosting_diagnostics


def main() -> int:
    df = build_fact_matrix()
    base_features = _indicator_columns(df) + _numeric_columns(df)
    # Add categorical controls and jurisdictional signals for a richer SHAP cut
    extra_cats = _categorical_columns()
    jurisdiction = ["country_code", "dpa_name_canonical", "decision_year_bucket"]
    candidates = extra_cats + jurisdiction
    present = [c for c in candidates if c in df.columns]
    features = base_features + present
    out_dir = Path("outputs/evenness/models")
    out_dir.mkdir(parents=True, exist_ok=True)

    for outcome, classification in [("fine_positive", True), ("fine_log1p", False)]:
        result = gradient_boosting_diagnostics(
            df,
            outcome=outcome,
            feature_cols=features,
            classification=classification,
        )
        shap_summary: pd.DataFrame = result["shap_summary"]
        shap_summary.to_csv(out_dir / f"shap_{outcome}.csv", index=False)
        shap_summary.head(25).to_csv(out_dir / f"shap_{outcome}_top25.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


