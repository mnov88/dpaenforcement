from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .hierarchical_severity_model import (
    SEVERITY_ORDER,
    BASIC_FEATURES,
    CATEGORICAL_FEATURES,
    _load_inputs,
    _severity_label,
    _build_multi_features,
    _dpa_shrinkage_features,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regularised multinomial severity classifier")
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
        help="Path to metadata JSON emitted alongside the feature matrix",
    )
    parser.add_argument(
        "--latent-scores",
        type=Path,
        default=Path("outputs/analysis/interaction/latent_scores.parquet"),
        help="Optional latent component scores parquet to merge",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/analysis/hierarchical_severity_regularized"),
        help="Directory to write model artefacts",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularisation strength for the multinomial model (L2)",
    )
    return parser.parse_args(argv)


def _prepare_dataset(
    df: pd.DataFrame,
    column_groups: dict[str, Sequence[str]],
    latent_scores: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    latent_columns: list[str] = []
    if latent_scores.exists():
        latent = pd.read_parquet(latent_scores)
        df = df.join(latent, how="left")
        latent_columns = list(latent.columns)

    df = df.copy()
    df["severity_label"] = df.apply(_severity_label, axis=1)
    df["severity_rank"] = df["severity_label"].map(SEVERITY_ORDER).astype(int)

    multi = _build_multi_features(df, column_groups)
    shrink = _dpa_shrinkage_features(df, df["severity_rank"])

    base = df[list(BASIC_FEATURES)].fillna(0).astype(float)
    cat = pd.get_dummies(
        df[list(CATEGORICAL_FEATURES)].fillna("UNKNOWN"),
        prefix=CATEGORICAL_FEATURES,
        drop_first=True,
        dtype=float,
    )
    latent = df[latent_columns].fillna(0) if latent_columns else pd.DataFrame(index=df.index)

    design = pd.concat([base, multi, shrink, cat, latent], axis=1)
    design = design.loc[:, ~design.columns.duplicated()]
    design = design.fillna(0).astype(float)

    return design, df["severity_rank"], df[["decision_id", "dpa_name_canonical", "severity_label"]]


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    df, column_groups = _load_inputs(args.feature_matrix, args.metadata_json)
    design, target, meta = _prepare_dataset(df, column_groups, args.latent_scores)

    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(design)

    model = LogisticRegression(
        penalty="l2",
        C=args.c,
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        n_jobs=None,
    )
    model.fit(X, target)

    probas = model.predict_proba(X)
    expected = probas @ model.classes_

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    coef_rows = []
    for cls_idx, cls in enumerate(model.classes_):
        coefs = model.coef_[cls_idx]
        order = np.argsort(np.abs(coefs))[::-1]
        for idx in order[:40]:
            coef_rows.append(
                {
                    "class": int(cls),
                    "feature": design.columns[idx],
                    "coefficient": float(coefs[idx]),
                }
            )
    pd.DataFrame(coef_rows).to_csv(out_dir / "top_coefficients.csv", index=False)

    diagnostics = {
        "classes": [int(cls) for cls in model.classes_],
        "C": args.c,
        "n_features": design.shape[1],
        "n_samples": design.shape[0],
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    output = meta.copy()
    output["severity_rank"] = target
    output["severity_expected"] = expected
    output.to_csv(out_dir / "severity_predictions.csv", index=False)


if __name__ == "__main__":
    main()
