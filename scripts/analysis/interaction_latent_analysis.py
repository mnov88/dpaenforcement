from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .build_feature_matrix import META_SUFFIXES

TOKEN_SETS: dict[str, tuple[str, ...]] = {
    "breach_types": ("q21_breach_types",),
    "vulnerabilities": ("q46_vuln",),
    "remedial_actions": ("q47_remedial",),
    "powers": ("q53_powers",),
}

LATENT_SETS: dict[str, tuple[str, ...]] = {
    "rights_discussed": ("q56_rights_discussed",),
    "rights_violated": ("q57_rights_violated",),
    "access_issues": ("q58_access_issues",),
    "adm_issues": ("q59_adm_issues",),
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interaction networks and latent structure analysis")
    parser.add_argument(
        "--feature-matrix",
        type=Path,
        default=Path("outputs/analysis/feature_matrix.parquet"),
        help="Path to feature matrix parquet",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=Path("outputs/analysis/feature_matrix_metadata.json"),
        help="Path to metadata JSON produced by build_feature_matrix",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/analysis/interaction"),
        help="Directory to store artefacts",
    )
    parser.add_argument(
        "--top-cooccurrence",
        type=int,
        default=40,
        help="Number of edges to retain per token set",
    )
    parser.add_argument(
        "--latent-components",
        type=int,
        default=3,
        help="Number of principal components to compute per latent set",
    )
    parser.add_argument(
        "--latent-scores-path",
        type=Path,
        default=Path("outputs/analysis/interaction/latent_scores.parquet"),
        help="Destination parquet for per-record latent component scores",
    )
    return parser.parse_args(argv)


def _load_inputs(feature_matrix: Path, metadata_json: Path) -> tuple[pd.DataFrame, dict[str, Sequence[str]]]:
    df = pd.read_parquet(feature_matrix)
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    return df, metadata.get("column_groups", {})


def _collect_columns(prefixes: tuple[str, ...], metadata: dict[str, Sequence[str]]) -> list[str]:
    columns: list[str] = []
    for prefix in prefixes:
        for col in metadata.get(prefix, []):
            if any(col.endswith(suffix) for suffix in META_SUFFIXES):
                continue
            columns.append(col)
    return columns


def _cooccurrence_matrix(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, list[str]]:
    if not columns:
        return np.zeros((0, 0)), []
    matrix = df[columns].fillna(0).to_numpy(dtype=float)
    matrix = (matrix > 0).astype(float)
    co_matrix = matrix.T @ matrix
    np.fill_diagonal(co_matrix, 0)
    return co_matrix, columns


def _top_edges(matrix: np.ndarray, labels: list[str], limit: int) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    if matrix.size == 0:
        return edges
    triu_indices = np.triu_indices_from(matrix, k=1)
    weights = matrix[triu_indices]
    order = np.argsort(weights)[::-1]
    count = 0
    for idx in order:
        weight = int(weights[idx])
        if weight <= 0:
            break
        i = triu_indices[0][idx]
        j = triu_indices[1][idx]
        edges.append({"source": labels[i], "target": labels[j], "weight": weight})
        count += 1
        if count >= limit:
            break
    return edges


def _principal_components(
    df: pd.DataFrame, columns: list[str], n_components: int
) -> tuple[dict[str, object], pd.DataFrame]:
    if not columns:
        return ({"components": [], "explained_variance_ratio": []}, pd.DataFrame(index=df.index))
    matrix = df[columns].fillna(0).to_numpy(dtype=float)
    if matrix.shape[0] < 2:
        return ({"components": [], "explained_variance_ratio": []}, pd.DataFrame(index=df.index))
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    rank = min(n_components, vh.shape[0])
    components = vh[:rank]
    variances = (s ** 2) / (matrix.shape[0] - 1)
    explained_ratio = variances[:rank] / variances.sum()
    scores = matrix @ components.T
    score_columns = {f"pc{idx+1}": scores[:, idx] for idx in range(rank)}
    scores_df = pd.DataFrame(score_columns, index=df.index)
    loadings = []
    for idx in range(rank):
        comp = components[idx]
        component = {
            "component": idx + 1,
            "loadings": {
                columns[j]: float(comp[j]) for j in np.argsort(np.abs(comp))[::-1][:10]
            },
        }
        loadings.append(component)
    return (
        {
            "components": loadings,
            "explained_variance_ratio": explained_ratio.tolist(),
        },
        scores_df,
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    df, metadata = _load_inputs(args.feature_matrix, args.metadata_json)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    interaction_payload = {}
    for label, prefixes in TOKEN_SETS.items():
        columns = _collect_columns(prefixes, metadata)
        matrix, names = _cooccurrence_matrix(df, columns)
        edges = _top_edges(matrix, names, args.top_cooccurrence)
        interaction_payload[label] = {
            "columns": names,
            "edge_count": len(edges),
            "edges": edges,
        }
    (args.out_dir / "cooccurrence_networks.json").write_text(
        json.dumps(interaction_payload, indent=2), encoding="utf-8"
    )

    latent_payload = {}
    latent_score_frames: list[pd.DataFrame] = []
    for label, prefixes in LATENT_SETS.items():
        columns = _collect_columns(prefixes, metadata)
        summary, scores = _principal_components(df, columns, args.latent_components)
        latent_payload[label] = summary
        if summary["components"]:
            rows = []
            for component in summary["components"]:
                comp_idx = component["component"]
                for token, loading in component["loadings"].items():
                    rows.append({"component": comp_idx, "token": token, "loading": loading})
            out_path = args.out_dir / f"{label}_loadings.csv"
            pd.DataFrame(rows).to_csv(out_path, index=False)
        if not scores.empty:
            renamed = scores.rename(columns=lambda c: f"{label}_{c}")
            latent_score_frames.append(renamed)
    (args.out_dir / "latent_components.json").write_text(
        json.dumps(latent_payload, indent=2), encoding="utf-8"
    )

    if latent_score_frames:
        combined = pd.concat(latent_score_frames, axis=1)
        combined.index = df.index
        args.latent_scores_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(args.latent_scores_path)


if __name__ == "__main__":
    main()
