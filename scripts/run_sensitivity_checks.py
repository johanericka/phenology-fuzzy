#!/usr/bin/env python3
"""
Standalone Sensitivity runner.

This script intentionally keeps robustness checks outside
`main.py` and `src/`. Variants are robustness branches around the frozen
Fuzzy-Phenology controller, not a replacement core model and not a retuning
protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.compare import compute_scenario_summary
from src.aquacrop_bridge import AquaCropRunConfig, run_aquacrop_for_all_scenarios
from src.config import (
    CN,
    INITIAL_MOISTURE_FC,
    IRRIGATION_EFFICIENCY,
    KEMARAU_TARGET_BOOSTS,
    PHENOLOGY_OVERLAP_DAYS,
    PLANTING_SEASONS,
    SIMULATION_YEARS,
    TOTAL_DAYS,
    WEATHER_COVERAGE_MIN_FRACTION,
    YIELD_FOCUS_MODE,
    get_output_dir,
)
from src.data_cleansing import cleanse_weather_data
from src.fuzzy_controller import FuzzyIrrigationController
from src.phenology import (
    compute_dynamic_targets,
    get_dominant_phase,
    get_dominant_subphase,
    get_kc_for_hst,
)
from src.water_balance import WaterBalanceModel


DEFAULT_PROFILE = {
    "profile_name": "iwue_lean_2",
    "upper_ref_margin_frac": 0.18,
    "overshoot_tol_frac": 0.06,
    "max_irrigation_daily_mm": 26.0,
    "vegetative_weight": 0.30,
    "reproductive_weight": 0.95,
    "maturation_weight": 0.10,
}

DEFAULT_OUTPUT_DIR = (
    Path(get_output_dir()) / "reproducibility_2015_2024" / "sensitivity"
)


class SensitivityWaterBalanceModel(WaterBalanceModel):
    """Water-balance variant with local runoff controls for sensitivity."""

    def __init__(
        self,
        initial_r: float | None = None,
        cn_override: float | None = None,
        runoff_multiplier: float = 1.0,
    ):
        super().__init__(initial_r=initial_r)
        if cn_override is not None:
            self.cn = float(cn_override)
        self.runoff_multiplier = float(runoff_multiplier)

    def calculate_runoff(self, precipitation: float) -> float:
        runoff = super().calculate_runoff(precipitation)
        return float(max(0.0, runoff * self.runoff_multiplier))


@dataclass(frozen=True)
class BatchPolicy:
    """Fixed operational policy applied after the controller recommendation."""

    policy_id: str
    min_interval_days: int = 1
    release_threshold_mm: float | None = None


class SensitivityEngine:
    """Closed-loop runner with optional runoff, batching, and phenology-time adapters."""

    def __init__(
        self,
        controller: FuzzyIrrigationController,
        weather_df: pd.DataFrame,
        simulation_years: list[int],
        planting_seasons: dict,
        total_days: int = TOTAL_DAYS,
        initial_r: float = INITIAL_MOISTURE_FC,
        overlap_days: int | None = PHENOLOGY_OVERLAP_DAYS,
        yield_focus_mode: bool = YIELD_FOCUS_MODE,
        target_boosts: dict | None = None,
        wb_factory: Callable[[float], WaterBalanceModel] | None = None,
        batch_policy: BatchPolicy | None = None,
        phenology_mode: str = "dap",
        gdd_base_temp_c: float = 10.0,
    ):
        self.controller = controller
        self.weather_df = weather_df.copy()
        self.simulation_years = list(simulation_years)
        self.planting_seasons = dict(planting_seasons)
        self.total_days = int(total_days)
        self.initial_r = float(initial_r)
        self.overlap_days = overlap_days
        self.yield_focus_mode = bool(yield_focus_mode)
        self.target_boosts = dict(KEMARAU_TARGET_BOOSTS if target_boosts is None else target_boosts)
        self.wb_factory = wb_factory or (lambda initial_r: WaterBalanceModel(initial_r=initial_r))
        self.batch_policy = batch_policy
        self.phenology_mode = str(phenology_mode)
        self.gdd_base_temp_c = float(gdd_base_temp_c)

        if not pd.api.types.is_datetime64_any_dtype(self.weather_df["Date"]):
            self.weather_df["Date"] = pd.to_datetime(self.weather_df["Date"])

    def run_all_scenarios(self, verbose: bool = False) -> pd.DataFrame:
        all_results = []
        for year in self.simulation_years:
            for season_name, season_info in self.planting_seasons.items():
                try:
                    planting_date = datetime(year, season_info["start_month"], season_info["start_day"])
                except ValueError:
                    continue

                end_date = planting_date + timedelta(days=self.total_days - 1)
                weather_range = self.weather_df[
                    (self.weather_df["Date"] >= pd.Timestamp(planting_date))
                    & (self.weather_df["Date"] <= pd.Timestamp(end_date))
                ]
                if len(weather_range) < self.total_days * WEATHER_COVERAGE_MIN_FRACTION:
                    if verbose:
                        print(
                            f"Skipping {year} {season_name}: insufficient weather "
                            f"({len(weather_range)}/{self.total_days} days)"
                        )
                    continue

                season_df = self.run_season(planting_date, season_name=season_name, year=year)
                if not season_df.empty:
                    all_results.append(season_df)

        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    def run_season(self, planting_date: datetime, season_name: str, year: int) -> pd.DataFrame:
        wb = self.wb_factory(self.initial_r)
        results = []
        pending_irrigation = 0.0
        last_applied_hst = -10_000
        phenology_coords = self._build_phenology_coordinates(planting_date)

        for hst in range(self.total_days):
            current_date = planting_date + timedelta(days=hst)
            weather_row = self.weather_df[self.weather_df["Date"] == pd.Timestamp(current_date)]
            if weather_row.empty:
                continue

            weather = weather_row.iloc[0]
            et0 = float(weather["Et0"])
            precipitation = float(weather["Prcp"])
            phenology_hst = float(phenology_coords[hst])

            phase = get_dominant_phase(phenology_hst, overlap_days=self.overlap_days)
            subphase = get_dominant_subphase(phenology_hst, overlap_days=self.overlap_days)
            targets = self._adjust_targets_for_season(
                compute_dynamic_targets(phenology_hst, overlap_days=self.overlap_days),
                season_name=season_name,
                phase=phase,
            )
            kc = get_kc_for_hst(phenology_hst, overlap_days=self.overlap_days)

            irrig_result = self.controller.compute_irrigation(
                r_prev=wb.r,
                hst=phenology_hst,
                rU=targets["rU"],
                rL=targets["rL"],
                rWP=targets["rWP"],
            )
            recommended_irrigation = float(irrig_result["irrigation_mm"])
            irrigation_mm, pending_irrigation, last_applied_hst = self._apply_batch_policy(
                hst=hst,
                recommended_irrigation=recommended_irrigation,
                pending_irrigation=pending_irrigation,
                last_applied_hst=last_applied_hst,
            )

            wb_result = wb.update(
                precipitation=precipitation,
                irrigation=irrigation_mm,
                et0=et0,
                kc=kc,
            )

            results.append(
                {
                    "date": current_date,
                    "year": year,
                    "season": season_name,
                    "hst": hst,
                    "phenology_hst": phenology_hst,
                    "phenology_mode": self.phenology_mode,
                    "phase": phase,
                    "subphase": subphase,
                    "kc": kc,
                    "precipitation": precipitation,
                    "et0": et0,
                    "tmin": float(weather.get("Tmin", 0.0)),
                    "tmax": float(weather.get("Tmax", 0.0)),
                    "theta_prev": wb_result["theta_prev"],
                    "theta": wb_result["theta"],
                    "r_prev": wb_result["r_prev"],
                    "r": wb_result["r"],
                    "r_pct": wb_result["r"] * 100.0,
                    "runoff": wb_result["runoff"],
                    "etc": wb_result["etc"],
                    "deep_percolation": wb_result["deep_percolation"],
                    "drought_indicator": irrig_result["drought_indicator"],
                    "phase_sensitivity": irrig_result["phase_sensitivity"],
                    "phase_critical": irrig_result.get(
                        "phase_critical",
                        int(irrig_result.get("phase_sensitivity", 0.0) >= 0.5),
                    ),
                    "urgency": irrig_result["urgency"],
                    "recommended_irrigation_mm": recommended_irrigation,
                    "irrigation_mm": irrigation_mm,
                    "pending_irrigation_mm": pending_irrigation,
                    "sm_ref": irrig_result.get("sm_ref", np.nan),
                    "sm_cap": irrig_result.get("sm_cap", np.nan),
                    "refill_deficit": irrig_result.get("refill_deficit", np.nan),
                    "rL": targets["rL"],
                    "rU": targets["rU"],
                    "rWP": targets["rWP"],
                    "rL_pct": targets["rL"] * 100.0,
                    "rU_pct": targets["rU"] * 100.0,
                    "rWP_pct": targets["rWP"] * 100.0,
                }
            )

        return pd.DataFrame(results)

    def _apply_batch_policy(
        self,
        hst: int,
        recommended_irrigation: float,
        pending_irrigation: float,
        last_applied_hst: int,
    ) -> tuple[float, float, int]:
        if self.batch_policy is None:
            return recommended_irrigation, 0.0, hst if recommended_irrigation > 0 else last_applied_hst

        pending = pending_irrigation + recommended_irrigation
        if pending <= 1e-9:
            return 0.0, 0.0, last_applied_hst

        interval_ready = (hst - last_applied_hst) >= self.batch_policy.min_interval_days
        threshold = self.batch_policy.release_threshold_mm
        threshold_ready = threshold is not None and pending >= float(threshold)
        end_flush = hst == self.total_days - 1

        if interval_ready and (threshold is None or threshold_ready or end_flush):
            return float(pending), 0.0, hst
        return 0.0, float(pending), last_applied_hst

    def _build_phenology_coordinates(self, planting_date: datetime) -> np.ndarray:
        if self.phenology_mode != "gdd":
            return np.arange(self.total_days, dtype=float)

        rows = []
        for hst in range(self.total_days):
            current_date = planting_date + timedelta(days=hst)
            weather_row = self.weather_df[self.weather_df["Date"] == pd.Timestamp(current_date)]
            if weather_row.empty:
                rows.append(0.0)
                continue
            weather = weather_row.iloc[0]
            tmean = (float(weather["Tmin"]) + float(weather["Tmax"])) / 2.0
            rows.append(max(0.0, tmean - self.gdd_base_temp_c))

        daily_gdd = np.asarray(rows, dtype=float)
        total_gdd = float(daily_gdd.sum())
        if total_gdd <= 1e-9:
            return np.arange(self.total_days, dtype=float)

        # Previous cumulative GDD keeps day 0 aligned with HST 0.
        prev_cum = np.concatenate([[0.0], np.cumsum(daily_gdd)[:-1]])
        coords = prev_cum / total_gdd * float(self.total_days - 1)
        return np.clip(coords, 0.0, float(self.total_days - 1))

    def _adjust_targets_for_season(self, targets: dict, season_name: str, phase: str) -> dict:
        t = {k: float(v) for k, v in targets.items()}
        if not self.yield_focus_mode:
            return t
        if "kemarau" not in str(season_name).strip().lower():
            return t

        boost = self.target_boosts.get(phase, {"rL": 0.0, "rU": 0.0})
        t["rL"] = min(1.0, t["rL"] + float(boost.get("rL", 0.0)))
        t["rU"] = min(1.0, t["rU"] + float(boost.get("rU", 0.0)))
        t["rL"] = max(t["rL"], t["rWP"] + 0.10)
        t["rU"] = max(t["rU"], t["rL"] + 0.05)
        t["rU"] = min(1.0, t["rU"])
        t["rL"] = min(t["rL"], t["rU"] - 0.05)
        return t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone Phase 3 sensitivity checks."
    )
    parser.add_argument("--weather", default=str(PROJECT_ROOT / "data" / "cuaca-complete.txt"))
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--season", choices=["MT-1", "MT-2", "MT-3"], default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--runtime-config",
        default=str(PROJECT_ROOT / "output" / "reproducibility_2015_2024" / "selected_runtime_config.json"),
    )
    parser.add_argument("--with-aquacrop", action="store_true", help="Add AquaCrop yield/IWUE metrics.")
    parser.add_argument("--write-raw", action="store_true", help="Persist daily raw result CSV files.")
    parser.add_argument("--dry-run", action="store_true", help="Write notes only; do not execute simulations.")
    return parser.parse_args()


def load_runtime_config(path: str) -> tuple[float, dict, str]:
    config_path = Path(path)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return (
            float(payload.get("eta_star", IRRIGATION_EFFICIENCY)),
            dict(payload.get("fuzzy_profile_selected", DEFAULT_PROFILE)),
            f"loaded from {config_path}",
        )
    return IRRIGATION_EFFICIENCY, dict(DEFAULT_PROFILE), "fallback DEFAULT_PROFILE; no retuning performed"


def resolve_planting_seasons(season_alias: str | None) -> dict:
    if season_alias is None:
        return dict(PLANTING_SEASONS)
    mapping = {
        "MT-1": "MT-1_Penghujan",
        "MT-2": "MT-2_Peralihan",
        "MT-3": "MT-3_Kemarau",
    }
    return {mapping[season_alias]: PLANTING_SEASONS[mapping[season_alias]]}


def build_controller(eta: float, profile: dict) -> FuzzyIrrigationController:
    return FuzzyIrrigationController(efficiency=eta, **dict(profile))


def enrich_with_aquacrop_if_requested(
    all_results: pd.DataFrame,
    summary: pd.DataFrame,
    weather_df: pd.DataFrame,
    eta: float,
    with_aquacrop: bool,
) -> pd.DataFrame:
    if not with_aquacrop or all_results.empty or summary.empty:
        return summary
    aq_metrics = run_aquacrop_for_all_scenarios(
        all_results_df=all_results,
        weather_df=weather_df,
        cfg=AquaCropRunConfig(app_efficiency=eta),
        verbose=False,
    )
    if aq_metrics.empty:
        return summary
    return pd.merge(summary, aq_metrics, on=["year", "season"], how="left")


def summarize_variant(
    family: str,
    variant: str,
    all_results: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: dict,
) -> dict:
    row = {
        "family": family,
        "variant": variant,
        "n_scenarios": int(len(summary)),
        "mean_iwu_mm": float(summary["iwu_mm"].mean()) if "iwu_mm" in summary else np.nan,
        "mean_events": float(summary["n_irrigation_events"].mean())
        if "n_irrigation_events" in summary
        else np.nan,
        "mean_event_depth_mm": float(summary["mean_depth_per_event_mm"].mean())
        if "mean_depth_per_event_mm" in summary
        else np.nan,
        "mean_mse": float(summary["mse"].mean()) if "mse" in summary else np.nan,
        "mean_target_pct": float(summary["target_pct"].mean()) if "target_pct" in summary else np.nan,
        "mean_runoff_mm": float(all_results.groupby(["year", "season"])["runoff"].sum().mean())
        if "runoff" in all_results
        else np.nan,
        "mean_deep_percolation_mm": float(
            all_results.groupby(["year", "season"])["deep_percolation"].sum().mean()
        )
        if "deep_percolation" in all_results
        else np.nan,
        "mean_yield_dry_t_ha": float(summary["yield_dry_t_ha"].mean())
        if "yield_dry_t_ha" in summary
        else np.nan,
        "mean_iwue_kg_ha_per_mm": float(summary["iwue_kg_ha_per_mm"].mean())
        if "iwue_kg_ha_per_mm" in summary
        else np.nan,
    }
    row.update(metadata)
    return row


def run_variant(
    family: str,
    variant: str,
    weather_df: pd.DataFrame,
    simulation_years: list[int],
    planting_seasons: dict,
    eta: float,
    profile: dict,
    output_dir: Path,
    with_aquacrop: bool,
    write_raw: bool,
    metadata: dict | None = None,
    wb_factory: Callable[[float], WaterBalanceModel] | None = None,
    batch_policy: BatchPolicy | None = None,
    phenology_mode: str = "dap",
) -> tuple[pd.DataFrame, dict]:
    controller = build_controller(eta, profile)
    engine = SensitivityEngine(
        controller=controller,
        weather_df=weather_df,
        simulation_years=simulation_years,
        planting_seasons=planting_seasons,
        wb_factory=wb_factory,
        batch_policy=batch_policy,
        phenology_mode=phenology_mode,
    )
    all_results = engine.run_all_scenarios(verbose=False)
    if all_results.empty:
        return pd.DataFrame(), {"family": family, "variant": variant, "n_scenarios": 0}

    summary = compute_scenario_summary(all_results)
    summary = enrich_with_aquacrop_if_requested(all_results, summary, weather_df, eta, with_aquacrop)
    meta = dict(metadata or {})
    row = summarize_variant(family, variant, all_results, summary, meta)

    if write_raw:
        raw_path = output_dir / f"{family}_{variant}_daily_results.csv"
        scenario_path = output_dir / f"{family}_{variant}_scenario_summary.csv"
        all_results.to_csv(raw_path, index=False)
        summary.to_csv(scenario_path, index=False)

    return summary, row


def write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_notes(
    output_dir: Path,
    args: argparse.Namespace,
    eta: float,
    profile: dict,
    runtime_source: str,
    generated_files: list[str],
) -> None:
    lines = [
        "# Sensitivity notes",
        "",
        f"Prepared by: standalone runner `{Path(__file__).name}`",
        f"Runtime config: eta={eta:.3f}, profile={profile.get('profile_name', 'unknown')} ({runtime_source})",
        f"Years: {args.start_year}-{args.end_year}",
        f"Season filter: {args.season or 'all seasons'}",
        f"AquaCrop yield/IWUE enrichment: {'enabled' if args.with_aquacrop else 'disabled'}",
        "",
        "## Dynamic-control guardrails",
        "",
        "- These variants are robustness checks around the same Fuzzy-Phenology controller.",
        "- The script does not retune eta, fuzzy memberships, target bands, or phase weights.",
        "- Outputs are isolated under `output/reproducibility_2015_2024/sensitivity/` by default.",
        "- Runoff variants are labeled as runoff sensitivity only; no pond-storage proxy is implemented here.",
        "- Batching variants are closed-loop reruns using fixed operational policies after daily controller recommendations.",
        "- GDD timing is a transparent phenology-coordinate perturbation, not cultivar-calibrated phenology.",
        "- ET0 variants are deterministic perturbation sensitivity, not stochastic robustness.",
        "- Soil-moisture sensor noise is not implemented because the main engine does not separate observed and true state.",
        "- AWD-like baselines and full ponded-field hydraulics remain outside Phase 3 scope.",
        "",
        "## Generated outputs",
        "",
    ]
    if generated_files:
        lines.extend(f"- `{path}`" for path in generated_files)
    else:
        lines.append("- No simulation CSV was generated.")
    lines.append("")
    (output_dir / "sensitivity_notes.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eta, profile, runtime_source = load_runtime_config(args.runtime_config)
    generated_files: list[str] = []

    if args.dry_run:
        write_notes(output_dir, args, eta, profile, runtime_source, generated_files)
        print(f"Dry run complete. Notes written to {output_dir}")
        return 0

    simulation_years = list(range(args.start_year, args.end_year + 1))
    planting_seasons = resolve_planting_seasons(args.season)
    weather_df = cleanse_weather_data(
        args.weather,
        start_year=min(simulation_years),
        end_year=max(simulation_years) + 1,
        verbose=False,
    )

    runoff_rows = []
    runoff_variants = [
        {
            "variant": "base_cn85",
            "cn_override": CN,
            "runoff_multiplier": 1.0,
            "label": "base SCS-CN runoff",
        },
        {
            "variant": "reduced_cn70",
            "cn_override": 70.0,
            "runoff_multiplier": 1.0,
            "label": "reduced runoff through lower CN",
        },
        {
            "variant": "half_runoff_cn85",
            "cn_override": CN,
            "runoff_multiplier": 0.5,
            "label": "reduced runoff multiplier",
        },
        {
            "variant": "no_runoff",
            "cn_override": CN,
            "runoff_multiplier": 0.0,
            "label": "no-runoff sensitivity",
        },
    ]
    for spec in runoff_variants:
        _, row = run_variant(
            family="runoff_sensitivity",
            variant=spec["variant"],
            weather_df=weather_df,
            simulation_years=simulation_years,
            planting_seasons=planting_seasons,
            eta=eta,
            profile=profile,
            output_dir=output_dir,
            with_aquacrop=args.with_aquacrop,
            write_raw=args.write_raw,
            metadata={
                "cn": float(spec["cn_override"]),
                "runoff_multiplier": float(spec["runoff_multiplier"]),
                "variant_label": spec["label"],
            },
            wb_factory=lambda initial_r, s=spec: SensitivityWaterBalanceModel(
                initial_r=initial_r,
                cn_override=s["cn_override"],
                runoff_multiplier=s["runoff_multiplier"],
            ),
        )
        runoff_rows.append(row)
    runoff_path = output_dir / "runoff_or_pond_proxy_summary.csv"
    write_summary(runoff_path, runoff_rows)
    generated_files.append(display_path(runoff_path))

    batching_rows = []
    batching_variants = [
        ("base_no_batching", None, "base closed-loop daily application"),
        ("min_interval_2d", BatchPolicy("min_interval_2d", min_interval_days=2), "fixed 2-day minimum interval"),
        ("min_interval_3d", BatchPolicy("min_interval_3d", min_interval_days=3), "fixed 3-day minimum interval"),
        (
            "min_interval_2d_release_5mm",
            BatchPolicy("min_interval_2d_release_5mm", min_interval_days=2, release_threshold_mm=5.0),
            "2-day minimum interval plus 5 mm release threshold",
        ),
    ]
    for variant, policy, label in batching_variants:
        _, row = run_variant(
            family="batching",
            variant=variant,
            weather_df=weather_df,
            simulation_years=simulation_years,
            planting_seasons=planting_seasons,
            eta=eta,
            profile=profile,
            output_dir=output_dir,
            with_aquacrop=args.with_aquacrop,
            write_raw=args.write_raw,
            metadata={
                "batch_policy": "none" if policy is None else policy.policy_id,
                "min_interval_days": np.nan if policy is None else policy.min_interval_days,
                "release_threshold_mm": np.nan if policy is None else policy.release_threshold_mm,
                "variant_label": label,
            },
            batch_policy=policy,
        )
        batching_rows.append(row)
    batching_path = output_dir / "batching_summary.csv"
    write_summary(batching_path, batching_rows)
    generated_files.append(display_path(batching_path))

    gdd_rows = []
    for variant, phenology_mode in [("base_dap", "dap"), ("gdd_base10c", "gdd")]:
        _, row = run_variant(
            family="gdd_phenology",
            variant=variant,
            weather_df=weather_df,
            simulation_years=simulation_years,
            planting_seasons=planting_seasons,
            eta=eta,
            profile=profile,
            output_dir=output_dir,
            with_aquacrop=args.with_aquacrop,
            write_raw=args.write_raw,
            metadata={
                "phenology_mode": phenology_mode,
                "gdd_base_temp_c": 10.0 if phenology_mode == "gdd" else np.nan,
            },
            phenology_mode=phenology_mode,
        )
        gdd_rows.append(row)
    gdd_path = output_dir / "gdd_phenology_summary.csv"
    write_summary(gdd_path, gdd_rows)
    generated_files.append(display_path(gdd_path))

    noise_rows = []
    for variant, factor in [
        ("et0_base", 1.00),
        ("et0_minus_5pct", 0.95),
        ("et0_plus_5pct", 1.05),
        ("et0_minus_10pct", 0.90),
        ("et0_plus_10pct", 1.10),
    ]:
        weather_variant = weather_df.copy()
        weather_variant["Et0"] = weather_variant["Et0"].astype(float) * factor
        _, row = run_variant(
            family="et0_perturbation",
            variant=variant,
            weather_df=weather_variant,
            simulation_years=simulation_years,
            planting_seasons=planting_seasons,
            eta=eta,
            profile=profile,
            output_dir=output_dir,
            with_aquacrop=args.with_aquacrop,
            write_raw=args.write_raw,
            metadata={
                "et0_multiplier": factor,
                "sensor_noise": "not_implemented_observed_true_state_not_split",
            },
        )
        noise_rows.append(row)
    noise_path = output_dir / "noise_robustness_summary.csv"
    write_summary(noise_path, noise_rows)
    generated_files.append(display_path(noise_path))

    notes_path = output_dir / "sensitivity_notes.md"
    generated_files.append(display_path(notes_path))
    write_notes(output_dir, args, eta, profile, runtime_source, generated_files)

    print("Sensitivity runner complete.")
    for path in generated_files:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
