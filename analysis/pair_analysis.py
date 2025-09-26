#!/usr/bin/env python3
"""Generate curated bivariate summaries and effect-size diagnostics for GDPR DPA fines."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

try:
    import scikit_posthocs as sp
except ImportError as exc:  # pragma: no cover - guarded import for runtime execution
    raise SystemExit(
        "scikit-posthocs is required for Dunn's test. Install it via `pip install scikit-posthocs`."
    ) from exc


DATA_PATH = Path("raw_data/responses/parsed_responses_min.csv")
OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_COL = "fine_amount_eur"
TARGET_WINSORIZED = "fine_amount_eur_w"
TARGET_LOG = "fine_log1p"
TURNOVER_COL = "annual_turnover_eur"
TURNOVER_WINSORIZED = "annual_turnover_eur_w"
TURNOVER_LOG = "turnover_log1p"

MISSING_TOKENS = {
    "",
    "NOT_APPLICABLE",
    "NOT_DISCUSSED",
    "NOT_MENTIONED",
    "NOT_DETERMINED",
    "NOT_SPECIFIED",
    "NOT_AVAILABLE",
    "NOT_ESTABLISHED",
    "NOT_IDENTIFIED",
    "NOT_PROVIDED",
    "NOT_REQUIRED",
    "NOT_DOCUMENTED",
    "NOT_REPORTED",
    "STRING",
    "ENUM",
    "YYYY-MM-DD",
    "YYYY-MM-DD,NOT_DISCUSSED",
}

TOP_K_DEFAULT = 10
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99
BOOTSTRAP_SAMPLES = 1000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


@dataclass
class PairSpec:
    pair_id: str
    left_var: str
    right_var: str
    pair_type: str  # cat_num, num_num, time_num
    description: str
    rationale: str
    top_k: Optional[int] = TOP_K_DEFAULT
    notes: Optional[str] = None

    @property
    def left_label(self) -> str:
        return self.left_var

    @property
    def right_label(self) -> str:
        return self.right_var


def winsorize_series(series: pd.Series, lower: float = WINSOR_LOWER, upper: float = WINSOR_UPPER) -> pd.Series:
    """Winsorize a numeric series by clipping to the given quantiles."""
    if series.dropna().empty:
        return series
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    if pd.isna(lower_bound) or pd.isna(upper_bound):
        return series
    if lower_bound > upper_bound:
        lower_bound, upper_bound = upper_bound, lower_bound
    return series.clip(lower=lower_bound, upper=upper_bound)


def apply_top_k(series: pd.Series, k: Optional[int]) -> pd.Series:
    if k is None:
        return series
    counts = series.value_counts(dropna=False)
    top_levels = [lvl for lvl in counts.head(k).index if not pd.isna(lvl)]
    top_set = set(top_levels)

    def mapper(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        return value if value in top_set else "OTHER"

    return series.map(mapper)


def tidy_markdown_table(df: pd.DataFrame) -> str:
    def escape(value: Any) -> str:
        text = "" if pd.isna(value) else str(value)
        return text.replace("|", "\\|")

    headers = [escape(col) for col in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = ["| " + " | ".join(escape(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header_line, separator, *rows]) + "\n"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Normalize string placeholders
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        series = df[col].astype(str).str.strip()
        series = series.replace({"nan": pd.NA, "None": pd.NA})
        series = series.replace(MISSING_TOKENS, pd.NA)
        series = series.replace({"": pd.NA})
        df[col] = series

    # Parse issue date and derive month bucket
    issue = df["issue_date"].astype(str)
    invalid_mask = issue.str.contains("NOT_DISCUSSED", case=False, na=False) | issue.str.contains("YYYY", na=False)
    issue = issue.mask(invalid_mask, other=np.nan)
    df["issue_date"] = pd.to_datetime(issue, errors="coerce", utc=True)
    df["issue_month"] = df["issue_date"].dt.to_period("M").astype(str)
    df.loc[df["issue_month"] == "NaT", "issue_month"] = pd.NA

    # Winsorize numeric targets
    df[TARGET_WINSORIZED] = winsorize_series(df[TARGET_COL])
    df[TURNOVER_WINSORIZED] = winsorize_series(df[TURNOVER_COL])
    df[TARGET_LOG] = np.log1p(df[TARGET_WINSORIZED])
    df[TURNOVER_LOG] = np.log1p(df[TURNOVER_WINSORIZED])
    df["fine_positive"] = (df[TARGET_COL] > 0).astype(float)
    return df


def pair_specs() -> List[PairSpec]:
    return [
        PairSpec(
            pair_id="fine_x_country",
            left_var="country_code",
            right_var=TARGET_WINSORIZED,
            pair_type="cat_num",
            description="Fine amounts by respondent country code",
            rationale="Country captures legal culture and enforcement priorities; consolidating top issuers prevents sparse tails.",
            notes="Fine metric winsorized at 1% tails; medians compared to overall baseline.",
        ),
        PairSpec(
            pair_id="fine_x_dpa",
            left_var="dpa_name",
            right_var=TARGET_WINSORIZED,
            pair_type="cat_num",
            description="Fine amounts by issuing authority",
            rationale="Authority-level comparisons highlight institutional sanctioning patterns beyond geography.",
            notes="Top ten DPAs by volume retained; others grouped as OTHER.",
        ),
        PairSpec(
            pair_id="fine_x_role",
            left_var="defendant_role",
            right_var=TARGET_WINSORIZED,
            pair_type="cat_num",
            description="Fine amounts by procedural role (controller/processor)",
            rationale="Role signals statutory obligations and may shape penalty ceilings.",
        ),
        PairSpec(
            pair_id="fine_x_jurisdiction",
            left_var="jurisdiction_complexity",
            right_var=TARGET_WINSORIZED,
            pair_type="cat_num",
            description="Fine amounts by jurisdictional complexity",
            rationale="Cross-border coordination has been linked to escalated sanctions under the one-stop-shop regime.",
        ),
        PairSpec(
            pair_id="fine_x_crossborder",
            left_var="cross_border",
            right_var=TARGET_WINSORIZED,
            pair_type="cat_num",
            description="Fine amounts by cross-border flag",
            rationale="Explicit cross-border markings corroborate jurisdictional complexity results and check coding consistency.",
        ),
        PairSpec(
            pair_id="fine_x_cooperation",
            left_var="cooperation_level",
            right_var=TARGET_WINSORIZED,
            pair_type="cat_num",
            description="Fine amounts by cooperation level",
            rationale="Cooperation mitigates sanctions; we quantify effect magnitude and tail behaviour.",
        ),
        PairSpec(
            pair_id="fine_x_firsttime",
            left_var="first_time_violation",
            right_var=TARGET_WINSORIZED,
            pair_type="cat_num",
            description="Fine amounts by repeat-offender status",
            rationale="Recidivism expectations (Art. 83) motivate differential sanctioning; tests assess materiality.",
        ),
        PairSpec(
            pair_id="fine_x_breachtype",
            left_var="breach_type",
            right_var=TARGET_WINSORIZED,
            pair_type="cat_num",
            description="Fine amounts by breach typology",
            rationale="Breach archetypes (human/technical/organizational) potentially correlate with harm severity and fines.",
            notes="Placeholder-only responses removed before top-K truncation.",
        ),
        PairSpec(
            pair_id="fine_x_subjects",
            left_var="subjects_notified",
            right_var=TARGET_WINSORIZED,
            pair_type="cat_num",
            description="Fine amounts by data subject notification status",
            rationale="Notification behaviour proxies remediation diligence, a key mitigating factor.",
        ),
        PairSpec(
            pair_id="fine_x_legalbasis",
            left_var="legal_bases_relied",
            right_var=TARGET_WINSORIZED,
            pair_type="cat_num",
            description="Fine amounts by legal basis invoked",
            rationale="Legal basis disputes underpin proportionality debates; grouping highlights sanction differentials.",
        ),
        PairSpec(
            pair_id="fine_x_issuemonth",
            left_var="issue_month",
            right_var=TARGET_WINSORIZED,
            pair_type="time_num",
            description="Fine amounts by decision month",
            rationale="Temporal trends capture enforcement ramp-up and seasonal clustering; months with sparse data bundled.",
        ),
        PairSpec(
            pair_id="fine_x_turnover",
            left_var=TURNOVER_WINSORIZED,
            right_var=TARGET_WINSORIZED,
            pair_type="num_num",
            description="Fine amounts versus reported annual turnover",
            rationale="Benchmarking fines against turnover tests proportionality and Article 83 scaling.",
            notes="Both axes winsorized at 1% tails; log-log diagnostics reported separately.",
        ),
    ]


def compute_missing_counts(df: pd.DataFrame, left_col: str, target_col: str) -> Tuple[int, int, int]:
    left_missing = df[left_col].isna()
    target_missing = df[target_col].isna()
    missing_both = int((left_missing & target_missing).sum())
    missing_left_only = int((left_missing & ~target_missing).sum())
    missing_right_only = int((~left_missing & target_missing).sum())
    return missing_both, missing_left_only, missing_right_only


def group_summary(
    df: pd.DataFrame,
    pair: PairSpec,
    group_col: str,
    target_col: str,
    baseline_median: float,
    missing_counts: Tuple[int, int, int],
    overall_n: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    groups = df.groupby(group_col, dropna=True)

    def ratio(value: float) -> Optional[float]:
        if baseline_median == 0 or pd.isna(baseline_median):
            return np.nan
        return value / baseline_median if not pd.isna(value) else np.nan

    # Overall row
    overall_stats = {
        "pair_id": pair.pair_id,
        "left_var": pair.left_var,
        "right_var": pair.right_var,
        "pair_type": pair.pair_type,
        "level": "__overall__",
        "level_order": 0,
        "n_available": overall_n,
        "group_n": overall_n,
        "group_pct": 1.0,
        "median": df[target_col].median(),
        "mean": df[target_col].mean(),
        "median_log": df[TARGET_LOG].median(),
        "mean_log": df[TARGET_LOG].mean(),
        "p25": df[target_col].quantile(0.25),
        "p75": df[target_col].quantile(0.75),
        "ratio_to_overall_median": 1.0,
        "zero_pct": (df[TARGET_COL] == 0).mean(),
        "missing_both": missing_counts[0],
        "missing_left_only": missing_counts[1],
        "missing_right_only": missing_counts[2],
    }
    records.append(overall_stats)

    for order, (level, subset) in enumerate(
        sorted(groups, key=lambda kv: (-kv[1][target_col].median(), kv[0]))
    ):
        group_n = int(len(subset))
        record = {
            "pair_id": pair.pair_id,
            "left_var": pair.left_var,
            "right_var": pair.right_var,
            "pair_type": pair.pair_type,
            "level": level,
            "level_order": order + 1,
            "n_available": overall_n,
            "group_n": group_n,
            "group_pct": group_n / overall_n if overall_n else np.nan,
            "median": subset[target_col].median(),
            "mean": subset[target_col].mean(),
            "median_log": subset[TARGET_LOG].median(),
            "mean_log": subset[TARGET_LOG].mean(),
            "p25": subset[target_col].quantile(0.25),
            "p75": subset[target_col].quantile(0.75),
            "ratio_to_overall_median": ratio(subset[target_col].median()),
            "zero_pct": (subset[TARGET_COL] == 0).mean(),
            "missing_both": missing_counts[0],
            "missing_left_only": missing_counts[1],
            "missing_right_only": missing_counts[2],
        }
        records.append(record)
    return records


def summarize_pairs(df: pd.DataFrame, pairs: Sequence[PairSpec]) -> pd.DataFrame:
    summary_records: List[Dict[str, Any]] = []

    for pair in pairs:
        if pair.pair_type == "num_num":
            base = df[[pair.left_var, pair.right_var, TARGET_COL, TARGET_LOG]].copy()
            missing_counts = compute_missing_counts(base, pair.left_var, pair.right_var)
            available = base.dropna(subset=[pair.left_var, pair.right_var])
            if available.empty:
                continue
            # Bin the numeric predictor into quartiles (min 3 unique bins)
            unique_vals = available[pair.left_var].nunique()
            q = min(4, unique_vals)
            if q < 2:
                continue
            available = available.assign(
                group=pd.qcut(
                    available[pair.left_var],
                    q=q,
                    duplicates="drop",
                    precision=3,
                ).astype(str)
            )
            grouped = available[["group", pair.right_var, TARGET_COL, TARGET_LOG]].rename(
                columns={"group": pair.left_var, pair.right_var: pair.right_var}
            )
            baseline_median = available[pair.right_var].median()
            summary_records.extend(
                group_summary(
                    grouped,
                    pair,
                    group_col=pair.left_var,
                    target_col=pair.right_var,
                    baseline_median=baseline_median,
                    missing_counts=missing_counts,
                    overall_n=len(available),
                )
            )
        else:
            base = df[[pair.left_var, pair.right_var, TARGET_COL, TARGET_LOG]].copy()
            missing_counts = compute_missing_counts(base, pair.left_var, pair.right_var)
            available = base.dropna(subset=[pair.left_var, pair.right_var])
            if available.empty:
                continue
            available.loc[:, pair.left_var] = apply_top_k(available[pair.left_var], pair.top_k)
            available = available.dropna(subset=[pair.left_var, pair.right_var])
            if available.empty:
                continue
            baseline_median = available[pair.right_var].median()
            summary_records.extend(
                group_summary(
                    available,
                    pair,
                    group_col=pair.left_var,
                    target_col=pair.right_var,
                    baseline_median=baseline_median,
                    missing_counts=missing_counts,
                    overall_n=len(available),
                )
            )

    return pd.DataFrame(summary_records)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    n_x = x.size
    n_y = y.size
    if n_x == 0 or n_y == 0:
        return np.nan
    greater = 0
    lesser = 0
    for value in x:
        greater += np.sum(value > y)
        lesser += np.sum(value < y)
    return (greater - lesser) / (n_x * n_y)


def bootstrap_ci(
    func,
    data: Sequence[np.ndarray],
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = RANDOM_SEED,
) -> Tuple[Optional[float], Optional[float]]:
    rng = np.random.default_rng(seed)
    estimates: List[float] = []
    for _ in range(samples):
        resampled = [arr[rng.integers(0, len(arr), len(arr))] for arr in data]
        try:
            estimates.append(func(resampled))
        except ValueError:
            continue
    if not estimates:
        return (None, None)
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def epsilon_squared(statistic: float, n_total: int, n_groups: int) -> float:
    if n_total <= n_groups:
        return np.nan
    value = (statistic - n_groups + 1) / (n_total - n_groups)
    return float(max(0.0, value))


def analyze_cat_num_effect(data: pd.DataFrame, pair: PairSpec) -> Dict[str, Any]:
    groups = [grp[pair.right_var].values for _, grp in data.groupby(pair.left_var)]
    group_labels = [name for name, _ in data.groupby(pair.left_var)]
    n_groups = len(groups)
    n_obs = int(sum(len(g) for g in groups))
    if n_groups < 2:
        raise ValueError("Need at least two groups for analysis")

    result: Dict[str, Any] = {
        "pair_id": pair.pair_id,
        "pair_type": pair.pair_type,
        "effect_size_name": None,
        "effect_size": None,
        "ci_lower": None,
        "ci_upper": None,
        "test_name": None,
        "test_statistic": None,
        "p_value": None,
        "extra": None,
        "n_obs": n_obs,
        "n_groups": n_groups,
    }

    if n_groups == 2:
        x, y = groups
        delta = cliffs_delta(x, y)
        ci_low, ci_high = bootstrap_ci(lambda samples: cliffs_delta(samples[0], samples[1]), [x, y])
        u_stat, p_value = stats.mannwhitneyu(x, y, alternative="two-sided")
        result.update(
            {
                "effect_size_name": "cliffs_delta",
                "effect_size": float(delta),
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "test_name": "mannwhitneyu",
                "test_statistic": float(u_stat),
                "p_value": float(p_value),
                "extra": f"groups={group_labels}",
            }
        )
    else:
        h_stat, p_value = stats.kruskal(*groups)
        eps_sq = epsilon_squared(h_stat, n_obs, n_groups)
        ci_low, ci_high = bootstrap_ci(
            lambda samples: epsilon_squared(stats.kruskal(*samples)[0], n_obs, n_groups),
            groups,
        )
        dunn = sp.posthoc_dunn(
            data[[pair.left_var, pair.right_var]],
            val_col=pair.right_var,
            group_col=pair.left_var,
            p_adjust="holm",
        )
        np.fill_diagonal(dunn.values, np.nan)
        mask = np.isfinite(dunn.values)
        if mask.any():
            min_idx = np.nanargmin(dunn.values)
            row_idx, col_idx = divmod(int(min_idx), dunn.shape[1])
            min_p = float(dunn.values[row_idx, col_idx])
            min_pair = f"{dunn.index[row_idx]} vs {dunn.columns[col_idx]}"
        else:
            min_p = np.nan
            min_pair = None
        result.update(
            {
                "effect_size_name": "epsilon_squared",
                "effect_size": float(eps_sq) if not pd.isna(eps_sq) else None,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "test_name": "kruskal",
                "test_statistic": float(h_stat),
                "p_value": float(p_value),
                "extra": f"strongest_pair={min_pair}; dunn_p={min_p:.3g}" if min_pair else None,
            }
        )
    return result


def analyze_num_num_effect(data: pd.DataFrame, pair: PairSpec) -> Dict[str, Any]:
    available = data.dropna(subset=[pair.left_var, pair.right_var])
    if available.empty:
        raise ValueError("Insufficient data for numeric pair")
    x = available[pair.left_var].to_numpy()
    y = available[pair.right_var].to_numpy()
    rho, p_value = stats.spearmanr(x, y)
    rho_log, _ = stats.spearmanr(available[TURNOVER_LOG], available[TARGET_LOG])
    rng = np.random.default_rng(RANDOM_SEED)
    boot_estimates: List[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample_idx = rng.integers(0, len(available), len(available))
        sample = available.iloc[sample_idx]
        boot_estimates.append(stats.spearmanr(sample[pair.left_var], sample[pair.right_var])[0])
    ci_low, ci_high = (
        (float(np.percentile(boot_estimates, 2.5)), float(np.percentile(boot_estimates, 97.5)))
        if boot_estimates
        else (None, None)
    )
    slope, intercept, lower, upper = stats.theilslopes(y, x, 0.95)
    slope_log, intercept_log, lower_log, upper_log = stats.theilslopes(
        available[TARGET_LOG], available[TURNOVER_LOG], 0.95
    )
    return {
        "pair_id": pair.pair_id,
        "pair_type": pair.pair_type,
        "effect_size_name": "spearman_rho",
        "effect_size": float(rho),
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "test_name": "spearman",
        "test_statistic": float(rho),
        "p_value": float(p_value),
        "extra": (
            f"log_rho={rho_log:.3f}; theil_slope={slope:.3g}"
            f" ({lower:.3g},{upper:.3g}); log_slope={slope_log:.3g} ({lower_log:.3g},{upper_log:.3g})"
        ),
        "n_obs": int(len(available)),
        "n_groups": 1,
    }


def analyze_time_num_effect(data: pd.DataFrame, pair: PairSpec) -> Dict[str, Any]:
    # Treat as categorical for main test
    return analyze_cat_num_effect(data, pair)


def effect_sizes(df: pd.DataFrame, pairs: Sequence[PairSpec]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for pair in pairs:
        if pair.pair_type == "num_num":
            base = df[[pair.left_var, pair.right_var, TURNOVER_LOG, TARGET_LOG]].copy()
            base = base.dropna(subset=[pair.left_var, pair.right_var])
            if base.empty:
                continue
            records.append(analyze_num_num_effect(base, pair))
            continue

        base = df[[pair.left_var, pair.right_var]].copy()
        base = base.dropna(subset=[pair.left_var, pair.right_var])
        if base.empty:
            continue
        base.loc[:, pair.left_var] = apply_top_k(base[pair.left_var], pair.top_k)
        base = base.dropna(subset=[pair.left_var, pair.right_var])
        if base.empty:
            continue
        if pair.pair_type == "time_num":
            record = analyze_time_num_effect(base, pair)
        else:
            record = analyze_cat_num_effect(base, pair)
        records.append(record)

    effect_df = pd.DataFrame(records)
    if effect_df.empty:
        return effect_df

    adj = multipletests(effect_df["p_value"].fillna(1.0).to_numpy(), method="holm")
    effect_df["p_value_holm"] = adj[1]
    return effect_df


def build_pairing_map(pairs: Sequence[PairSpec]) -> pd.DataFrame:
    records = []
    for pair in pairs:
        record = asdict(pair)
        record.update(
            {
                "analysis_target": TARGET_WINSORIZED,
                "log_metric": TARGET_LOG,
                "top_k_applied": pair.top_k,
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def write_outputs(
    pair_map: pd.DataFrame,
    summary_book: pd.DataFrame,
    effects: pd.DataFrame,
) -> None:
    pair_map.to_csv(OUTPUT_DIR / "pairing_map.csv", index=False)
    summary_book.to_csv(OUTPUT_DIR / "bivariate_summary_book.csv", index=False)
    effects.to_csv(OUTPUT_DIR / "effect_sizes.csv", index=False)

    (OUTPUT_DIR / "pairing_map.md").write_text(tidy_markdown_table(pair_map), encoding="utf-8")
    # Summary book is large; provide Markdown pointer with key columns
    summary_cols = [
        "pair_id",
        "level",
        "group_n",
        "median",
        "mean",
        "median_log",
        "p25",
        "p75",
        "ratio_to_overall_median",
        "zero_pct",
    ]
    summary_preview = summary_book[summary_cols].copy()
    (OUTPUT_DIR / "bivariate_summary_book.md").write_text(
        tidy_markdown_table(summary_preview.head(60)),
        encoding="utf-8",
    )
    if not effects.empty:
        (OUTPUT_DIR / "effect_sizes.md").write_text(tidy_markdown_table(effects), encoding="utf-8")


def main() -> None:
    df = load_data()
    pairs = pair_specs()
    pair_map = build_pairing_map(pairs)
    summary_book = summarize_pairs(df, pairs)
    effects = effect_sizes(df, pairs)
    write_outputs(pair_map, summary_book, effects)


if __name__ == "__main__":
    main()
