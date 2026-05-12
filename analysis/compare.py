"""
Analysis Module — Comparison & Statistical Tests (FASE 8)
==========================================================
Computes performance metrics, paired inferential tests, confidence intervals,
and comparison tables for controller-oriented reproducibility reporting.
"""

import math

import numpy as np
import pandas as pd
from scipy import stats

from src.simulation import SimulationEngine


DEFAULT_METRIC_COLS = [
    "iwu_mm",
    "mean_r_pct",
    "mse",
    "target_pct",
    "stress_days",
    "n_irrigation_events",
    "yield_dry_t_ha",
    "iwue_kg_ha_per_mm",
    "aq_transpiration_ratio",
]


METRIC_ROLE_MAP = {
    "iwu_mm": "headline_control",
    "mse": "headline_control",
    "target_pct": "headline_control",
    "mean_r_pct": "attribution_behavior",
    "stress_days": "attribution_behavior",
    "n_irrigation_events": "attribution_behavior",
    "irrigation_frequency_pct": "attribution_behavior",
    "mean_depth_per_event_mm": "attribution_behavior",
    "mean_interval_between_events_days": "attribution_behavior",
    "max_interval_without_irrigation_days": "attribution_behavior",
    "yield_dry_t_ha": "robustness_only",
    "iwue_kg_ha_per_mm": "robustness_only",
    "aq_transpiration_ratio": "robustness_only",
    "aq_stress_days_tr": "robustness_only",
    "median_depth_per_event_mm": "robustness_only",
    "cv_event_depth": "robustness_only",
    "median_interval_between_events_days": "robustness_only",
    "max_interval_between_events_days": "robustness_only",
}


def _paired_ci(diff: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Return t-based CI for the paired mean difference."""
    n = len(diff)
    mean_diff = float(np.mean(diff))
    if n < 2:
        return mean_diff, mean_diff
    sd = float(np.std(diff, ddof=1))
    if math.isclose(sd, 0.0):
        return mean_diff, mean_diff
    alpha = 1.0 - confidence
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    margin = t_crit * sd / math.sqrt(n)
    return mean_diff - margin, mean_diff + margin


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * float(p_values[idx])
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def _bh_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running_min = 1.0
    for reverse_rank, idx in enumerate(order[::-1], start=1):
        rank = n - reverse_rank + 1
        adj = float(p_values[idx]) * n / rank
        running_min = min(running_min, adj)
        adjusted[idx] = min(1.0, running_min)
    return adjusted.tolist()


def apply_multiple_comparison_corrections(
    stats_df: pd.DataFrame,
    group_cols: list[str] | None = None,
    p_col: str = "p_value",
) -> pd.DataFrame:
    """Append Holm and BH corrected p-values within the chosen groups."""
    if stats_df is None or stats_df.empty or p_col not in stats_df.columns:
        return stats_df

    out = stats_df.copy()
    out["p_value_holm"] = np.nan
    out["p_value_bh"] = np.nan
    out["significant_holm"] = False
    out["significant_bh"] = False

    group_cols = group_cols or []
    if not group_cols:
        groups = [(None, out.index.tolist())]
    else:
        groups = [(key, idx.tolist()) for key, idx in out.groupby(group_cols, dropna=False).groups.items()]

    for _, idxs in groups:
        valid_idxs = [idx for idx in idxs if pd.notna(out.at[idx, p_col])]
        if not valid_idxs:
            continue
        p_values = [float(out.at[idx, p_col]) for idx in valid_idxs]
        holm_vals = _holm_adjust(p_values)
        bh_vals = _bh_adjust(p_values)
        for row_idx, holm_val, bh_val in zip(valid_idxs, holm_vals, bh_vals):
            out.at[row_idx, "p_value_holm"] = holm_val
            out.at[row_idx, "p_value_bh"] = bh_val
            out.at[row_idx, "significant_holm"] = holm_val < 0.05
            out.at[row_idx, "significant_bh"] = bh_val < 0.05
    return out


def _nonparametric_confirmation(diff: np.ndarray) -> dict:
    """Run Wilcoxon confirmation when the paired sample is small enough."""
    n = len(diff)
    nonzero = diff[~np.isclose(diff, 0.0)]
    if n < 2:
        return {
            "nonparametric_test": "not_run",
            "wilcoxon_statistic": np.nan,
            "wilcoxon_p_value": np.nan,
            "wilcoxon_note": "insufficient_pairs",
        }
    if len(nonzero) == 0:
        return {
            "nonparametric_test": "wilcoxon_signed_rank",
            "wilcoxon_statistic": 0.0,
            "wilcoxon_p_value": 1.0,
            "wilcoxon_note": "all_differences_zero",
        }
    try:
        stat, p_value = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="auto")
        return {
            "nonparametric_test": "wilcoxon_signed_rank",
            "wilcoxon_statistic": float(stat),
            "wilcoxon_p_value": float(p_value),
            "wilcoxon_note": "paired_confirmation",
        }
    except ValueError:
        return {
            "nonparametric_test": "wilcoxon_signed_rank",
            "wilcoxon_statistic": np.nan,
            "wilcoxon_p_value": np.nan,
            "wilcoxon_note": "not_applicable",
        }


def compute_paired_statistics(
    reference_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    metric_cols: list | None = None,
    alpha: float = 0.05,
    group_label: str = "overall",
    nonparametric_threshold: int = 12,
) -> pd.DataFrame:
    """
    Run paired tests with CI and optional Wilcoxon confirmation.

    The difference is defined as `reference - baseline`.
    """
    if metric_cols is None:
        metric_cols = list(DEFAULT_METRIC_COLS)

    merged = pd.merge(
        reference_metrics,
        baseline_metrics,
        on=["year", "season"],
        how="inner",
        suffixes=("_reference", "_baseline"),
    )
    if merged.empty:
        return pd.DataFrame()

    results = []
    for col in metric_cols:
        ref_col = f"{col}_reference"
        base_col = f"{col}_baseline"
        if ref_col not in merged.columns or base_col not in merged.columns:
            continue

        ref_vals = merged[ref_col].values
        base_vals = merged[base_col].values
        valid = ~(pd.isna(ref_vals) | pd.isna(base_vals))
        ref_vals = ref_vals[valid]
        base_vals = base_vals[valid]
        n = len(ref_vals)
        if n < 2:
            continue

        diff = ref_vals - base_vals
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
        if np.allclose(diff, 0.0):
            t_stat, p_value = 0.0, 1.0
        else:
            t_stat, p_value = stats.ttest_rel(ref_vals, base_vals)
        cohens_d = mean_diff / std_diff if not math.isclose(std_diff, 0.0) else 0.0
        ci_low, ci_high = _paired_ci(diff)

        np_result = {
            "nonparametric_test": "not_run",
            "wilcoxon_statistic": np.nan,
            "wilcoxon_p_value": np.nan,
            "wilcoxon_note": "n_above_threshold",
        }
        if n <= nonparametric_threshold:
            np_result = _nonparametric_confirmation(diff)

        results.append({
            "analysis_scope": group_label,
            "metric": col,
            "metric_role": METRIC_ROLE_MAP.get(col, "robustness_only"),
            "reference_mean": float(np.mean(ref_vals)),
            "reference_std": float(np.std(ref_vals, ddof=1)) if n > 1 else 0.0,
            "baseline_mean": float(np.mean(base_vals)),
            "baseline_std": float(np.std(base_vals, ddof=1)) if n > 1 else 0.0,
            "difference": mean_diff,
            "difference_std": std_diff,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "ci_method": "paired_t_interval",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "significant": float(p_value) < alpha,
            "n_pairs": int(n),
            **np_result,
        })
    return pd.DataFrame(results)


def paired_stats_by_season(
    reference_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    metric_cols: list | None = None,
    alpha: float = 0.05,
    nonparametric_threshold: int = 12,
) -> pd.DataFrame:
    """Run paired statistics separately for each season."""
    if reference_metrics is None or baseline_metrics is None:
        return pd.DataFrame()
    seasons = sorted(
        set(reference_metrics.get("season", pd.Series(dtype=str)).astype(str).tolist()) &
        set(baseline_metrics.get("season", pd.Series(dtype=str)).astype(str).tolist())
    )
    frames = []
    for season in seasons:
        ref_season = reference_metrics[reference_metrics["season"].astype(str) == season]
        base_season = baseline_metrics[baseline_metrics["season"].astype(str) == season]
        if ref_season.empty or base_season.empty:
            continue
        season_df = compute_paired_statistics(
            ref_season,
            base_season,
            metric_cols=metric_cols,
            alpha=alpha,
            group_label="seasonal",
            nonparametric_threshold=nonparametric_threshold,
        )
        if season_df.empty:
            continue
        season_df.insert(1, "season", season)
        frames.append(season_df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_scenario_summary(all_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-scenario (year × season) summary metrics.
    
    Args:
        all_results_df: Full simulation results DataFrame
        
    Returns:
        Summary DataFrame with one row per scenario
    """
    summaries = []
    
    for (year, season), group in all_results_df.groupby(['year', 'season']):
        metrics = SimulationEngine.compute_season_metrics(group)
        metrics['year'] = year
        metrics['season'] = season
        summaries.append(metrics)
    
    return pd.DataFrame(summaries)


def paired_t_test(fuzzy_metrics: pd.DataFrame, reactive_metrics: pd.DataFrame,
                  metric_cols: list = None, alpha: float = 0.05) -> pd.DataFrame:
    """
    Run paired t-test comparing fuzzy vs reactive controller.
    
    Args:
        fuzzy_metrics: Summary metrics from fuzzy controller
        reactive_metrics: Summary metrics from reactive controller
        metric_cols: Columns to test (default: iwu_mm, mean_r_pct, mse, target_pct)
        alpha: Significance level
        
    Returns:
        DataFrame with t-statistic, p-value, and significance for each metric
    """
    stats_df = compute_paired_statistics(
        fuzzy_metrics,
        reactive_metrics,
        metric_cols=metric_cols,
        alpha=alpha,
        group_label="overall",
        nonparametric_threshold=12,
    )
    if stats_df.empty:
        return stats_df

    out = stats_df.rename(
        columns={
            "reference_mean": "fuzzy_mean",
            "reference_std": "fuzzy_std",
            "baseline_mean": "reactive_mean",
            "baseline_std": "reactive_std",
        }
    )
    return apply_multiple_comparison_corrections(out, group_cols=["analysis_scope"])


def generate_comparison_report(fuzzy_summary: pd.DataFrame,
                               reactive_summary: pd.DataFrame,
                               save_path: str = None) -> str:
    """
    Generate a text comparison report.
    
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 70)
    report.append("COMPARISON REPORT: Fuzzy-Phenology vs Reactive Controller")
    report.append("=" * 70)
    
    # Overall means
    report.append("\n--- Overall Averages ---")
    report.append(f"{'Metric':<25s} {'Fuzzy':>12s} {'Reactive':>12s} {'Diff':>12s}")
    report.append("-" * 61)
    
    for col in [
        'iwu_mm', 'mean_r_pct', 'mse', 'target_pct',
        'stress_days', 'n_irrigation_events',
        'yield_dry_t_ha', 'iwue_kg_ha_per_mm', 'aq_transpiration_ratio'
    ]:
        if col in fuzzy_summary.columns and col in reactive_summary.columns:
            f_mean = fuzzy_summary[col].mean()
            r_mean = reactive_summary[col].mean()
            diff = f_mean - r_mean
            report.append(f"{col:<25s} {f_mean:12.2f} {r_mean:12.2f} {diff:+12.2f}")
    
    # Paired t-test
    report.append("\n--- Paired t-test Results (α = 0.05) ---")
    ttest_results = paired_t_test(fuzzy_summary, reactive_summary)
    
    if not ttest_results.empty:
        report.append(f"{'Metric':<25s} {'t-stat':>10s} {'p-value':>10s} {'Significant':>12s}")
        report.append("-" * 57)
        for _, row in ttest_results.iterrows():
            sig = "YES ***" if row['significant'] else "no"
            report.append(f"{row['metric']:<25s} {row['t_statistic']:10.3f} "
                         f"{row['p_value']:10.4f} {sig:>12s}")
    
    # Per-season breakdown
    report.append("\n--- Per-Season Breakdown ---")
    seasons = sorted(
        set(fuzzy_summary.get('season', pd.Series(dtype=str)).astype(str).tolist()) |
        set(reactive_summary.get('season', pd.Series(dtype=str)).astype(str).tolist()),
        key=lambda s: (0 if 'Penghujan' in s else 1 if 'Peralihan' in s else 2 if 'Kemarau' in s else 99, s),
    )
    for season in seasons:
        f_season = fuzzy_summary[fuzzy_summary['season'].astype(str) == season]
        r_season = reactive_summary[reactive_summary['season'].astype(str) == season]
        
        if f_season.empty or r_season.empty:
            continue
        
        report.append(f"\n  {season}:")
        report.append(f"    Fuzzy    — IWU: {f_season['iwu_mm'].mean():.1f}mm, "
                     f"SM: {f_season['mean_r_pct'].mean():.1f}%FC, "
                     f"Target%: {f_season['target_pct'].mean():.1f}%")
        report.append(f"    Reactive — IWU: {r_season['iwu_mm'].mean():.1f}mm, "
                     f"SM: {r_season['mean_r_pct'].mean():.1f}%FC, "
                     f"Target%: {r_season['target_pct'].mean():.1f}%")
    
    report.append("\n" + "=" * 70)
    
    report_str = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_str)
    
    return report_str
