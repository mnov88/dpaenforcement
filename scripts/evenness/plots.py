"""Plotting utilities for evenness analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_style("whitegrid")


def plot_leniency_map(frame: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(12, 8))
    order = frame.loc[frame["jurisdiction_level"] == "DPA"].sort_values("effect")["jurisdiction"]
    subset = frame.loc[frame["jurisdiction_level"] == "DPA"]
    sns.pointplot(
        data=subset,
        y="jurisdiction",
        x="effect",
        join=False,
        errorbar=None,
        order=order,
        color="#1f77b4",
    )
    for _, row in subset.iterrows():
        plt.plot([row["lower"], row["upper"]], [row["jurisdiction"], row["jurisdiction"]], color="#1f77b4")
    plt.axvline(0, color="black", linewidth=1, linestyle="--")
    plt.title("Leniency / Severity residual index (DPA-level)")
    plt.xlabel("Residual (higher = more severe)")
    plt.ylabel("DPA")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_icc_bars(frame: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.barplot(data=frame, x="component", y="icc", color="#ff7f0e")
    plt.ylabel("Intraclass Correlation")
    plt.xlabel("Component")
    plt.ylim(0, 1)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_balance(frame: pd.DataFrame, out_path: Path, threshold: float = 0.1) -> None:
    plt.figure(figsize=(8, 6))
    sns.barplot(data=frame, x="feature", y="smd", hue="match_type")
    plt.axhline(threshold, color="black", linestyle="--", linewidth=1)
    plt.axhline(-threshold, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Standardized Mean Difference")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_shap_summary(shap_summary: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    plt.figure(figsize=(10, 6))
    subset = shap_summary.head(top_n)
    sns.barplot(data=subset, x="mean_abs_shap", y="feature", color="#2ca02c")
    plt.xlabel("Mean |SHAP|")
    plt.ylabel("Feature")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


__all__ = ["plot_leniency_map", "plot_icc_bars", "plot_balance", "plot_shap_summary"]
