#!/usr/bin/env python3
"""
Private vs Public Sector Hypothesis Testing
==========================================

This script compares outcomes between private (BUSINESS) and public (PUBLIC_AUTHORITY)
sectors using the cleaned GDPR enforcement dataset. It computes:
- Fine amount differences (Mann–Whitney U, Cliff's delta, bootstrap CIs)
- Fine likelihood differences (Fisher's exact, odds ratios with CIs)
- Sanction mix differences across `Sanction_*` binaries (FDR-adjusted)
- Optional key binary/context differences (FDR-adjusted)

Outputs a JSON summary with effect sizes, p-values, and q-values (BH-FDR).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import Table2x2
import argparse


def load_latest_cleaned_data() -> pd.DataFrame:
    files = list(Path('.').glob('dataNorway_cleaned_*.csv'))
    if not files:
        raise FileNotFoundError("No cleaned data files found.")
    latest = sorted(files)[-1]
    return pd.read_csv(latest)


def prepare_sector(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = df['A15_DefendantCategory'].isin(['BUSINESS', 'PUBLIC_AUTHORITY'])
    df = df.loc[mask].copy()
    df['sector'] = df['A15_DefendantCategory'].map({'BUSINESS': 'private', 'PUBLIC_AUTHORITY': 'public'})
    return df


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta (x vs y)."""
    # Efficient approximation using ranks
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    # Pairwise comparison (vectorized via sort)
    xy = np.concatenate([x, y])
    ranks = pd.Series(xy).rank(method='average').values
    rx = ranks[:nx]
    ry = ranks[nx:]
    # Use relation with Mann-Whitney U: delta = 2*U/(nx*ny) - 1
    U, _ = mannwhitneyu(x, y, alternative='two-sided')
    return (2 * U / (nx * ny)) - 1


def bootstrap_ci_stat(x: np.ndarray, y: np.ndarray, stat_fn, n_boot: int = 2000, alpha: float = 0.05,
                      random_state: int = 42) -> Tuple[float, Tuple[float, float]]:
    rng = np.random.default_rng(random_state)
    x = np.asarray(x)
    y = np.asarray(y)
    boots = []
    for _ in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        boots.append(stat_fn(xb, yb))
    boots = np.array(boots)
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    point = stat_fn(x, y)
    return point, (float(lo), float(hi))


def median_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.median(a) - np.median(b))


def to_binary_series(s: pd.Series) -> pd.Series:
    vals = s.copy()
    if vals.dtype in [np.int64, np.int32, np.float64, np.float32, 'int64', 'float64']:
        return (vals.fillna(0).astype(float) > 0.5).astype(int)
    # Treat strings
    sval = vals.astype(str).str.upper().str.strip()
    return sval.isin(['Y', 'YES', 'TRUE', '1']).astype(int)


def fisher_stats(ct: np.ndarray) -> Dict[str, Any]:
    t = Table2x2(ct)
    or_est = float(t.oddsratio)
    lo, hi = t.oddsratio_confint()
    res = t.test_nominal_association()  # Fisher exact p-value
    return {
        'odds_ratio': or_est,
        'ci95': [float(lo), float(hi)],
        'p_value': float(res.pvalue)
    }


def analyze_fine_amount(df: pd.DataFrame) -> Dict[str, Any]:
    result: Dict[str, Any] = {'outcome': 'A46_FineAmount_EUR'}
    if 'A46_FineAmount_EUR' not in df.columns:
        result['note'] = 'A46_FineAmount_EUR missing'
        return result

    private = df.loc[df['sector'] == 'private', 'A46_FineAmount_EUR'].dropna().astype(float)
    public = df.loc[df['sector'] == 'public', 'A46_FineAmount_EUR'].dropna().astype(float)

    if len(private) == 0 or len(public) == 0:
        result['note'] = 'Insufficient data in one or both groups'
        return result

    stat, p = mannwhitneyu(private, public, alternative='two-sided')
    delta = cliffs_delta(private.values, public.values)
    delta_pt, delta_ci = bootstrap_ci_stat(private.values, public.values, 
                                           lambda a, b: cliffs_delta(a, b))
    med_diff_pt, med_diff_ci = bootstrap_ci_stat(private.values, public.values, median_diff)

    result.update({
        'test': 'Mann-Whitney U',
        'u_stat': float(stat),
        'p_value': float(p),
        'effect_cliffs_delta': float(delta_pt),
        'effect_cliffs_delta_ci95': [float(delta_ci[0]), float(delta_ci[1])],
        'median_private': float(np.median(private)),
        'median_public': float(np.median(public)),
        'median_diff_private_minus_public': float(med_diff_pt),
        'median_diff_ci95': [float(med_diff_ci[0]), float(med_diff_ci[1])],
        'n_private': int(len(private)),
        'n_public': int(len(public))
    })

    # Also report zero-fine sensitivity
    pz = private[private > 0]
    gz = public[public > 0]
    if len(pz) > 0 and len(gz) > 0:
        _, p_nonzero = mannwhitneyu(pz, gz, alternative='two-sided')
        result['p_value_excluding_zero'] = float(p_nonzero)

    return result


def analyze_fine_likelihood(df: pd.DataFrame) -> Dict[str, Any]:
    result: Dict[str, Any] = {'outcome': 'Sanction_Fine'}
    if 'Sanction_Fine' not in df.columns:
        result['note'] = 'Sanction_Fine not present (expected from cleaning)'
        return result

    # Build 2x2 table
    sec = df['sector']
    fine = (df['Sanction_Fine'].fillna(0).astype(int) > 0).astype(int)
    ct = pd.crosstab(sec, fine).reindex(index=['private', 'public'], columns=[0, 1], fill_value=0).values

    if ct.shape != (2, 2):
        result['note'] = 'Unexpected contingency shape'
        return result

    stats = fisher_stats(ct)
    result.update(stats)
    result['table'] = {'private': {'no_fine': int(ct[0,0]), 'fine': int(ct[0,1])},
                       'public': {'no_fine': int(ct[1,0]), 'fine': int(ct[1,1])}}
    return result


def analyze_sanction_mix(df: pd.DataFrame) -> Dict[str, Any]:
    results: Dict[str, Any] = {'family': 'sanction_mix', 'tests': []}
    sanction_cols = [c for c in df.columns if c.startswith('Sanction_')]
    pvals: List[float] = []

    for col in sorted(sanction_cols):
        vals = (df[col].fillna(0).astype(int) > 0).astype(int)
        ct = pd.crosstab(df['sector'], vals).reindex(index=['private', 'public'], columns=[0, 1], fill_value=0)
        if ct.shape != (2, 2):
            continue
        stats = fisher_stats(ct.values)
        out = {'variable': col, **stats, 'table': ct.to_dict()}
        results['tests'].append(out)
        pvals.append(stats['p_value'])

    if pvals:
        reject, qvals, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        for i, (rej, q) in enumerate(zip(reject, qvals)):
            results['tests'][i]['q_value'] = float(q)
            results['tests'][i]['significant_fdr_05'] = bool(rej)

    return results


def analyze_key_binaries(df: pd.DataFrame) -> Dict[str, Any]:
    results: Dict[str, Any] = {'family': 'key_binaries', 'tests': []}
    candidates = ['A5_CrossBorder', 'A16_SensitiveData', 'A21_DataTransfers']
    existing = [c for c in candidates if c in df.columns]
    pvals: List[float] = []

    for col in existing:
        vals = to_binary_series(df[col])
        ct = pd.crosstab(df['sector'], vals).reindex(index=['private', 'public'], columns=[0, 1], fill_value=0)
        if ct.shape != (2, 2):
            continue
        stats = fisher_stats(ct.values)
        out = {'variable': col, **stats, 'table': ct.to_dict()}
        results['tests'].append(out)
        pvals.append(stats['p_value'])

    if pvals:
        reject, qvals, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        for i, (rej, q) in enumerate(zip(reject, qvals)):
            results['tests'][i]['q_value'] = float(q)
            results['tests'][i]['significant_fdr_05'] = bool(rej)

    return results


def main():
    print("Sector Analysis: Private vs Public")
    print("=" * 50)

    parser = argparse.ArgumentParser(description="Sector Analysis: Private vs Public")
    parser.add_argument('--data', '-d', dest='data_path', help='Path to cleaned CSV; defaults to latest dataNorway_cleaned_*.csv')
    args = parser.parse_args()

    if args.data_path:
        df = pd.read_csv(args.data_path)
        data_file_used = args.data_path
    else:
        df = load_latest_cleaned_data()
        data_file_used = str(sorted(Path('.').glob('dataNorway_cleaned_*.csv'))[-1])

    print(f"Loaded cleaned data: {len(df)} rows, {len(df.columns)} columns")

    df = prepare_sector(df)
    print(f"Subset to sector groups: {df['sector'].value_counts().to_dict()}")

    results: Dict[str, Any] = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_private': int((df['sector'] == 'private').sum()),
            'n_public': int((df['sector'] == 'public').sum()),
            'file_used': data_file_used
        }
    }

    # Primary outcomes
    results['fine_amount'] = analyze_fine_amount(df)
    results['fine_likelihood'] = analyze_fine_likelihood(df)

    # Secondary families
    results['sanction_mix'] = analyze_sanction_mix(df)
    results['key_binaries'] = analyze_key_binaries(df)

    # Save results
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'sector_analysis_summary_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✅ Sector analysis complete")
    print(f"📁 Results saved to: {out_path}")

    # Brief console summary
    fa = results['fine_amount']
    if 'p_value' in fa:
        print(f"Fine amount p={fa['p_value']:.4f}, Cliff's delta={fa['effect_cliffs_delta']:.3f}, med Δ={fa['median_diff_private_minus_public']:.0f}")
    fl = results['fine_likelihood']
    if 'p_value' in fl:
        print(f"Fine likelihood OR={fl['odds_ratio']:.2f} (p={fl['p_value']:.4f})")


if __name__ == '__main__':
    main()


