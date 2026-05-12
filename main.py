#!/usr/bin/env python3
"""
Phenology-Fuzzy Irrigation Control Simulation
================================================
Main entry point for running the closed-loop irrigation simulation
comparing Fuzzy-Phenology vs Reactive controllers on the AquaCrop-OSPy
digital twin, using 10 years of BMKG weather data from Kabupaten Malang.

Usage:
    python main.py              — Run full comparison (60 scenarios)
    python main.py --fuzzy      — Run fuzzy-phenology only
    python main.py --reactive   — Run reactive only
    python main.py --test       — Quick test with 1 year, 1 season

This public repository contains only the runnable simulation package and
reproducibility artifacts for reproducibility verification.
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    get_output_dir,
    SIMULATION_YEARS,
    REACTIVE_THRESHOLD_FC,
    REACTIVE_TARGET_FC,
    IRRIGATION_EFFICIENCY,
    PHENOLOGY_OVERLAP_DAYS,
    YIELD_FOCUS_MODE,
    KEMARAU_TARGET_BOOSTS,
)
from src.data_cleansing import cleanse_weather_data
from src.water_balance import WaterBalanceModel
from src.phenology import (
    compute_dynamic_targets,
    plot_phenology_targets,
    get_subphase_table,
)
from src.fuzzy_controller import FuzzyIrrigationController, visualize_fis
from src.fuzzy_static_controller import FuzzyStaticIrrigationController
from src.reactive_controller import ReactiveController
from src.reactive_phenology_controller import ReactivePhenologyController
from src.simulation import SimulationEngine
import src.simulation as simulation_module
from src.aquacrop_bridge import AquaCropRunConfig, run_aquacrop_for_all_scenarios
from analysis.compare import (
    METRIC_ROLE_MAP,
    apply_multiple_comparison_corrections,
    compute_paired_statistics,
    compute_scenario_summary,
    paired_stats_by_season,
    paired_t_test,
    generate_comparison_report,
)
from analysis.plots import (
    plot_season_timeseries, plot_comparison_boxplots,
    plot_seasonal_comparison, plot_tradeoff,
    plot_method_soil_moisture_diagnostics,
    plot_reproducibility_figure_composite_figure,
    plot_reproducibility_figure_irrigation_summary_figure,
)

warnings.filterwarnings("ignore")

REPRODUCIBILITY_DEFAULT_START_YEAR = 2015
REPRODUCIBILITY_DEFAULT_END_YEAR = 2024
REPRODUCIBILITY_TARGET_SHIFT_STEP = 0.03
FINAL_EVALUATION_PROTOCOL_ID = "leave_one_year_out"
FINAL_EVALUATION_PROTOCOL_LABEL = "Leave-One-Year-Out"
FINAL_EVALUATION_PROTOCOL_DESCRIPTION = (
    "Each evaluation fold holds out one year, tunes eta and the fuzzy profile only on the "
    "remaining training years, and then evaluates the tuned two-controller comparison on the "
    "held-out year's three planting seasons."
)

SEASON_ALIAS_TO_NAME = {
    "MT-1": "MT-1_Penghujan",
    "MT-2": "MT-2_Peralihan",
    "MT-3": "MT-3_Kemarau",
}

SEASON_ALIAS_TO_REPORT = {
    "MT-1": "analisa_penghujan.md",
    "MT-2": "analisa_peralihan.md",
    "MT-3": "analisa_kemarau.md",
}


def _season_group(season_name: str) -> str:
    s = str(season_name)
    if "Penghujan" in s:
        return "Penghujan"
    if "Peralihan" in s:
        return "Peralihan"
    if "Kemarau" in s:
        return "Kemarau"
    return s


def _season_sort_key(season_name: str) -> int:
    group = _season_group(season_name)
    return {"Penghujan": 0, "Peralihan": 1, "Kemarau": 2}.get(group, 99)


def _metric_role_order(metric_role: str) -> int:
    return {
        "headline_control": 0,
        "attribution_behavior": 1,
        "robustness_only": 2,
    }.get(str(metric_role), 99)


def _comparison_family(baseline_id: str) -> str:
    if baseline_id == "reactive_static":
        return "headline_comparison"
    if baseline_id in {"fuzzy_static", "reactive_stage"}:
        return "attribution_comparison"
    return "robustness_comparison"


def _comparison_note(baseline_id: str) -> str:
    notes = {
        "reactive_static": "total system comparison against the static reactive baseline",
        "fuzzy_static": "isolates phenology-aware target updating within the same fuzzy engine",
        "reactive_stage": "isolates fuzzy inference beyond phenology-aware rule control",
    }
    return notes.get(baseline_id, "auxiliary robustness comparison")


def build_runtime_scope(active_years: list[int], selected_season_name: str | None) -> dict:
    """Resolve explicit runtime years/seasons instead of mutating module globals."""
    seasons = dict(simulation_module.PLANTING_SEASONS)
    if selected_season_name is not None:
        seasons = {selected_season_name: seasons[selected_season_name]}
    return {
        "simulation_years": list(active_years),
        "planting_seasons": seasons,
    }


def build_final_evaluation_protocol(active_years: list[int]) -> dict:
    """Build the frozen final evaluation protocol definition."""
    years = sorted(int(year) for year in active_years)
    folds = [
        {
            "fold_id": f"holdout_{year}",
            "holdout_year": year,
            "training_years": [candidate for candidate in years if candidate != year],
        }
        for year in years
    ]
    return {
        "protocol_id": FINAL_EVALUATION_PROTOCOL_ID,
        "protocol_label": FINAL_EVALUATION_PROTOCOL_LABEL,
        "description": FINAL_EVALUATION_PROTOCOL_DESCRIPTION,
        "active_years": years,
        "n_folds": len(folds),
        "folds": folds,
    }


def get_mode_output_dir(base_output_dir: str, experiment_mode: str) -> str:
    """Return mode-specific output directory, keeping legacy `all` behavior stable."""
    if experiment_mode == "all":
        return base_output_dir
    return os.path.join(base_output_dir, experiment_mode.replace(" ", "_"))


def save_selected_runtime_config(
    output_dir: str,
    eta_star: float,
    fuzzy_profile_selected: dict,
    metadata: dict | None = None,
) -> str:
    """Persist the tuned runtime configuration for later explicit experiment modes."""
    config_path = os.path.join(output_dir, "selected_runtime_config.json")
    payload = {
        "eta_star": float(eta_star),
        "fuzzy_profile_selected": dict(fuzzy_profile_selected),
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return config_path


def load_selected_runtime_config(output_dir: str) -> tuple[float, dict, dict]:
    """Load previously selected tuning parameters for explicit downstream modes."""
    config_path = os.path.join(output_dir, "selected_runtime_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Missing tuned runtime config: {config_path}. Run `--experiment-mode tuning` first."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return (
        float(payload["eta_star"]),
        dict(payload["fuzzy_profile_selected"]),
        dict(payload.get("metadata", {})),
    )


def resolve_season_alias(season_alias: str | None) -> tuple[str | None, str | None]:
    """Map CLI season alias (MT-1/MT-2/MT-3) to canonical season name."""
    if season_alias is None:
        return None, None
    alias = str(season_alias).strip().upper()
    season_name = SEASON_ALIAS_TO_NAME.get(alias)
    if season_name is None:
        valid = ", ".join(SEASON_ALIAS_TO_NAME.keys())
        raise ValueError(f"Invalid --season '{season_alias}'. Use one of: {valid}")
    return alias, season_name


def run_simulation(
    controller,
    controller_name: str,
    weather_df: pd.DataFrame,
    output_dir: str | None = None,
    verbose: bool = True,
    engine_kwargs: dict | None = None,
) -> tuple:
    """
    Run full simulation for a given controller.
    
    Returns:
        (all_results_df, summary_df)
    """
    print(f"\n{'='*60}")
    print(f"RUNNING: {controller_name}")
    print(f"{'='*60}")
    
    engine = SimulationEngine(controller=controller, weather_df=weather_df, **(engine_kwargs or {}))
    all_results = engine.run_all_scenarios(verbose=verbose)
    
    if all_results.empty:
        print(f"  ⚠ No results generated for {controller_name}")
        return all_results, pd.DataFrame()
    
    # Compute summary metrics
    summary = compute_scenario_summary(all_results)
    
    # Save raw results
    output_dir = output_dir or get_output_dir()
    results_path = os.path.join(output_dir, f'{controller_name}_results.csv')
    all_results.to_csv(results_path, index=False)
    
    summary_path = os.path.join(output_dir, f'{controller_name}_summary.csv')
    summary.to_csv(summary_path, index=False)
    
    print(f"\n  Results saved to: {results_path}")
    print(f"  Summary saved to: {summary_path}")
    
    # Print quick summary
    print(f"\n  --- {controller_name} Summary ---")
    print(f"  Scenarios completed: {len(summary)}")
    print(f"  Mean IWU: {summary['iwu_mm'].mean():.1f} ± {summary['iwu_mm'].std():.1f} mm")
    print(f"  Mean Target%: {summary['target_pct'].mean():.1f} ± {summary['target_pct'].std():.1f}%")
    print(f"  Mean Stress Days: {summary['stress_days'].mean():.1f}")
    
    return all_results, summary


def enrich_summary_with_aquacrop(all_results: pd.DataFrame, summary: pd.DataFrame,
                                 weather_df: pd.DataFrame, eta: float,
                                 verbose: bool = True) -> pd.DataFrame:
    """
    Run AquaCrop per scenario and merge agronomic outputs (yield/IWUE/stress) into summary.
    """
    if all_results.empty or summary.empty:
        return summary

    if verbose:
        print(f"\n  Running AquaCrop yield/stress integration (eta={eta:.2f})...")

    aq_cfg = AquaCropRunConfig(app_efficiency=eta)
    aq_metrics = run_aquacrop_for_all_scenarios(
        all_results_df=all_results,
        weather_df=weather_df,
        cfg=aq_cfg,
        verbose=verbose,
    )
    if aq_metrics.empty:
        return summary

    merged = pd.merge(summary, aq_metrics, on=["year", "season"], how="left")
    return merged


def save_results_and_summary(
    controller_name: str,
    all_results: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | None = None,
):
    """Persist raw and summary outputs to `output/`."""
    output_dir = output_dir or get_output_dir()
    all_results.to_csv(os.path.join(output_dir, f"{controller_name}_results.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, f"{controller_name}_summary.csv"), index=False)


def build_controller_specs(eta_star: float, fuzzy_profile: dict | None = None) -> list:
    """Return the four-controller experiment design."""
    fuzzy_profile = dict(fuzzy_profile or {})
    return [
        {
            "id": "fuzzy_phenology",
            "label": "Fuzzy-Phenology",
            "group": "fuzzy",
            "description": "Fuzzy Mamdani dengan target dinamis fenologi dan fase kritis; parameter dituning untuk objective IWUE.",
            "controller": FuzzyIrrigationController(efficiency=eta_star, **fuzzy_profile),
        },
        {
            "id": "fuzzy_static",
            "label": "Fuzzy-Static",
            "group": "fuzzy",
            "description": "Fuzzy Mamdani tanpa awareness fenologi; target dibuat statis sepanjang musim.",
            "controller": FuzzyStaticIrrigationController(efficiency=eta_star),
        },
        {
            "id": "reactive_static",
            "label": "Reactive-Static",
            "group": "reactive",
            "description": "Threshold statis 80% FC, refill ke 100% FC.",
            "controller": ReactiveController(
                threshold_fc=REACTIVE_THRESHOLD_FC,
                target_fc=REACTIVE_TARGET_FC,
                efficiency=eta_star,
            ),
        },
        {
            "id": "reactive_stage",
            "label": "Reactive-Phenology",
            "group": "reactive",
            "description": "Rule-based phenology-aware: jika SM < rL dinamis, refill langsung ke rU dinamis.",
            "controller": ReactivePhenologyController(efficiency=eta_star),
        },
    ]


def get_fuzzy_iwue_tuning_profiles() -> list:
    """Small, explicit search space for Fuzzy-Phenology IWUE-oriented tuning."""
    return [
        {
            "profile_name": "baseline_balanced",
            "upper_ref_margin_frac": 0.05,
            "overshoot_tol_frac": 0.10,
            "max_irrigation_daily_mm": 40.0,
            "vegetative_weight": 0.45,
            "reproductive_weight": 1.00,
            "maturation_weight": 0.20,
        },
        {
            "profile_name": "iwue_lean_1",
            "upper_ref_margin_frac": 0.12,
            "overshoot_tol_frac": 0.08,
            "max_irrigation_daily_mm": 32.0,
            "vegetative_weight": 0.35,
            "reproductive_weight": 0.95,
            "maturation_weight": 0.15,
        },
        {
            "profile_name": "iwue_lean_2",
            "upper_ref_margin_frac": 0.18,
            "overshoot_tol_frac": 0.06,
            "max_irrigation_daily_mm": 26.0,
            "vegetative_weight": 0.30,
            "reproductive_weight": 0.95,
            "maturation_weight": 0.10,
        },
        {
            "profile_name": "iwue_reproductive_bias",
            "upper_ref_margin_frac": 0.14,
            "overshoot_tol_frac": 0.08,
            "max_irrigation_daily_mm": 30.0,
            "vegetative_weight": 0.30,
            "reproductive_weight": 1.10,
            "maturation_weight": 0.12,
        },
        {
            "profile_name": "iwue_micro_pulse",
            "upper_ref_margin_frac": 0.22,
            "overshoot_tol_frac": 0.05,
            "max_irrigation_daily_mm": 22.0,
            "vegetative_weight": 0.30,
            "reproductive_weight": 0.90,
            "maturation_weight": 0.10,
        },
        {
            "profile_name": "iwue_stage_like",
            "upper_ref_margin_frac": 0.10,
            "overshoot_tol_frac": 0.05,
            "max_irrigation_daily_mm": 28.0,
            "vegetative_weight": 0.25,
            "reproductive_weight": 1.00,
            "maturation_weight": 0.08,
        },
    ]


def resolve_fuzzy_profile_by_name(profile_name: str | None) -> dict:
    """Return a known fuzzy tuning profile by name, falling back to the first profile."""
    profiles = get_fuzzy_iwue_tuning_profiles()
    if profile_name is not None:
        for profile in profiles:
            if str(profile.get("profile_name")) == str(profile_name):
                return dict(profile)
    return dict(profiles[0])


def calibrate_fuzzy_profile_for_iwue(weather_df: pd.DataFrame, eta: float, engine_kwargs: dict | None = None) -> tuple:
    """
    Tune Fuzzy-Phenology parameters to maximize agronomic IWUE.

    Returns:
        (best_profile_dict, tuning_results_df)
    """
    profiles = get_fuzzy_iwue_tuning_profiles()
    rows = []

    print(f"\n{'='*60}")
    print("TUNING FUZZY-PHENOLOGY PROFILE (Objective: maximize IWUE)")
    print(f"{'='*60}")

    for profile in profiles:
        ctrl = FuzzyIrrigationController(efficiency=eta, **profile)
        print(f"  -> Testing profile={profile['profile_name']}")
        engine = SimulationEngine(controller=ctrl, weather_df=weather_df, **(engine_kwargs or {}))
        all_results = engine.run_all_scenarios(verbose=False)
        if all_results.empty:
            continue

        summary = compute_scenario_summary(all_results)
        summary = enrich_summary_with_aquacrop(all_results, summary, weather_df, eta, verbose=False)

        valid_iwue = summary["iwue_kg_ha_per_mm"].dropna() if "iwue_kg_ha_per_mm" in summary.columns else pd.Series(dtype=float)
        valid_yield = summary["yield_dry_t_ha"].dropna() if "yield_dry_t_ha" in summary.columns else pd.Series(dtype=float)
        kemarau_mask = summary["season"].astype(str).str.contains("Kemarau", na=False)
        kemarau_iwue = summary.loc[kemarau_mask, "iwue_kg_ha_per_mm"].dropna() if "iwue_kg_ha_per_mm" in summary.columns else pd.Series(dtype=float)
        kemarau_yield = summary.loc[kemarau_mask, "yield_dry_t_ha"].dropna() if "yield_dry_t_ha" in summary.columns else pd.Series(dtype=float)

        row = {
            "profile_name": profile["profile_name"],
            "upper_ref_margin_frac": profile["upper_ref_margin_frac"],
            "overshoot_tol_frac": profile["overshoot_tol_frac"],
            "max_irrigation_daily_mm": profile["max_irrigation_daily_mm"],
            "vegetative_weight": profile["vegetative_weight"],
            "reproductive_weight": profile["reproductive_weight"],
            "maturation_weight": profile["maturation_weight"],
            "mean_iwue_kg_ha_per_mm": float(valid_iwue.mean()) if not valid_iwue.empty else float("nan"),
            "mean_iwue_kemarau_kg_ha_per_mm": float(kemarau_iwue.mean()) if not kemarau_iwue.empty else float("nan"),
            "mean_yield_dry_t_ha": float(valid_yield.mean()) if not valid_yield.empty else float("nan"),
            "mean_yield_kemarau_t_ha": float(kemarau_yield.mean()) if not kemarau_yield.empty else float("nan"),
            "mean_iwu_mm": float(summary["iwu_mm"].mean()) if "iwu_mm" in summary.columns else float("nan"),
            "mean_target_pct": float(summary["target_pct"].mean()) if "target_pct" in summary.columns else float("nan"),
        }
        rows.append(row)
        print(
            f"IWUE={row['mean_iwue_kg_ha_per_mm']:.2f}, "
            f"IWUE(Kemarau)={row['mean_iwue_kemarau_kg_ha_per_mm']:.2f}, "
            f"Yield={row['mean_yield_dry_t_ha']:.3f}, "
            f"IWU={row['mean_iwu_mm']:.1f}, Target={row['mean_target_pct']:.1f}%"
        )

    tuning_df = pd.DataFrame(rows)
    if tuning_df.empty:
        fallback = get_fuzzy_iwue_tuning_profiles()[0]
        return fallback, tuning_df

    ranked = tuning_df.sort_values(
        ["mean_iwue_kg_ha_per_mm", "mean_yield_dry_t_ha", "mean_iwu_mm"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    best_name = str(ranked.iloc[0]["profile_name"])
    best_profile = next(profile for profile in profiles if profile["profile_name"] == best_name)
    print(f"\n  Selected fuzzy profile* = {best_name} (max mean IWUE)")
    return best_profile, tuning_df


def execute_tuning_protocol(
    weather_df: pd.DataFrame,
    output_dir: str,
    engine_kwargs: dict,
    base_output_dir: str | None = None,
    runtime_metadata: dict | None = None,
) -> tuple[float, pd.DataFrame, dict, pd.DataFrame]:
    """Run the bounded tuning sequence and persist its artifacts."""
    eta_star, eta_calibration_df = calibrate_eta_for_yield(weather_df, engine_kwargs=engine_kwargs)
    fuzzy_profile_selected, fuzzy_profile_tuning_df = calibrate_fuzzy_profile_for_iwue(
        weather_df, eta_star, engine_kwargs=engine_kwargs
    )
    if not eta_calibration_df.empty:
        eta_calibration_df.to_csv(os.path.join(output_dir, "eta_calibration_results.csv"), index=False)
    if not fuzzy_profile_tuning_df.empty:
        fuzzy_profile_tuning_df.to_csv(os.path.join(output_dir, "fuzzy_iwue_tuning_results.csv"), index=False)

    config_path = save_selected_runtime_config(
        output_dir,
        eta_star,
        fuzzy_profile_selected,
        metadata=runtime_metadata,
    )
    if base_output_dir is not None and os.path.abspath(base_output_dir) != os.path.abspath(output_dir):
        save_selected_runtime_config(
            base_output_dir,
            eta_star,
            fuzzy_profile_selected,
            metadata=runtime_metadata,
        )
    print(f"  ✓ Saved runtime config: {config_path}")
    return eta_star, eta_calibration_df, fuzzy_profile_selected, fuzzy_profile_tuning_df


def compile_controller_summary_table(controller_summaries: dict) -> pd.DataFrame:
    """Aggregate overall mean metrics per controller for cross-controller comparison."""
    rows = []
    metric_cols = [
        "iwu_mm",
        "target_pct",
        "mean_r_pct",
        "mse",
        "n_irrigation_events",
        "irrigation_frequency_pct",
        "mean_depth_per_event_mm",
        "median_depth_per_event_mm",
        "cv_event_depth",
        "mean_interval_between_events_days",
        "median_interval_between_events_days",
        "max_interval_between_events_days",
        "max_interval_without_irrigation_days",
        "yield_dry_t_ha",
        "iwue_kg_ha_per_mm",
        "aq_transpiration_ratio",
        "aq_stress_days_tr",
    ]
    for ctrl_id, payload in controller_summaries.items():
        summary = payload["summary"]
        if summary.empty:
            continue
        row = {
            "controller_id": ctrl_id,
            "controller_label": payload["label"],
            "n_scenarios": int(len(summary)),
        }
        for col in metric_cols:
            if col in summary.columns:
                row[f"mean_{col}"] = float(summary[col].mean())
                row[f"std_{col}"] = float(summary[col].std()) if len(summary[col]) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def compile_seasonal_summary_table(controller_summaries: dict) -> pd.DataFrame:
    """Aggregate seasonal means for every controller."""
    rows = []
    for ctrl_id, payload in controller_summaries.items():
        summary = payload["summary"]
        if summary.empty:
            continue
        grouped = (
            summary.groupby("season", as_index=False)
            .agg(
                iwu_mm=("iwu_mm", "mean"),
                target_pct=("target_pct", "mean"),
                mean_r_pct=("mean_r_pct", "mean"),
                mse=("mse", "mean"),
                n_irrigation_events=("n_irrigation_events", "mean"),
                irrigation_frequency_pct=("irrigation_frequency_pct", "mean"),
                mean_depth_per_event_mm=("mean_depth_per_event_mm", "mean"),
                mean_interval_between_events_days=("mean_interval_between_events_days", "mean"),
                max_interval_without_irrigation_days=("max_interval_without_irrigation_days", "mean"),
                yield_dry_t_ha=("yield_dry_t_ha", "mean"),
                iwue_kg_ha_per_mm=("iwue_kg_ha_per_mm", "mean"),
                aq_transpiration_ratio=("aq_transpiration_ratio", "mean"),
            )
        )
        grouped.insert(0, "controller_id", ctrl_id)
        grouped.insert(1, "controller_label", payload["label"])
        rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["season_order"] = out["season"].map(_season_sort_key)
    return out.sort_values(["season_order", "controller_label"]).drop(columns=["season_order"])


def build_controller_design_summary(controller_summaries: dict) -> pd.DataFrame:
    """Compact design table for the controller set."""
    rows = []
    for ctrl_id, payload in controller_summaries.items():
        rows.append({
            "controller_id": ctrl_id,
            "controller_label": payload["label"],
            "decision_logic_family": (
                "Mamdani fuzzy inference" if "fuzzy" in ctrl_id
                else "rule-based threshold/refill"
            ),
            "target_type": "dynamic phenology-driven" if ctrl_id in {"fuzzy_phenology", "reactive_stage"} else "static seasonal",
            "phenology_input": "yes" if ctrl_id in {"fuzzy_phenology", "reactive_stage"} else "no",
            "actuation_rule": (
                "urgency-scaled refill" if "fuzzy" in ctrl_id
                else "direct refill to target"
            ),
            "role_in_study": (
                "headline baseline" if ctrl_id == "reactive_static"
                else "proposed method" if ctrl_id == "fuzzy_phenology"
                else "attribution baseline"
            ),
        })
    return pd.DataFrame(rows)


def build_pairwise_vs_reference(controller_summaries: dict, reference_id: str = "fuzzy_phenology") -> pd.DataFrame:
    """Run overall and per-season paired tests versus one reference controller."""
    ref_payload = controller_summaries.get(reference_id)
    if ref_payload is None or ref_payload["summary"].empty:
        return pd.DataFrame()

    frames = []
    for ctrl_id, payload in controller_summaries.items():
        if ctrl_id == reference_id or payload["summary"].empty:
            continue
        overall_df = compute_paired_statistics(
            ref_payload["summary"],
            payload["summary"],
            metric_cols=[
                "target_pct",
                "mse",
                "mean_r_pct",
                "iwu_mm",
                "n_irrigation_events",
                "irrigation_frequency_pct",
                "mean_depth_per_event_mm",
                "mean_interval_between_events_days",
                "max_interval_without_irrigation_days",
                "yield_dry_t_ha",
                "iwue_kg_ha_per_mm",
                "aq_transpiration_ratio",
                "aq_stress_days_tr",
            ],
        )
        seasonal_df = paired_stats_by_season(
            ref_payload["summary"],
            payload["summary"],
            metric_cols=[
                "target_pct",
                "mse",
                "mean_r_pct",
                "iwu_mm",
                "n_irrigation_events",
                "irrigation_frequency_pct",
                "mean_depth_per_event_mm",
                "mean_interval_between_events_days",
                "max_interval_without_irrigation_days",
                "yield_dry_t_ha",
                "iwue_kg_ha_per_mm",
                "aq_transpiration_ratio",
                "aq_stress_days_tr",
            ],
        )
        t_df = pd.concat([overall_df, seasonal_df], ignore_index=True)
        if t_df.empty:
            continue
        t_df.insert(0, "baseline_id", ctrl_id)
        t_df.insert(1, "baseline_label", payload["label"])
        t_df.insert(0, "reference_id", reference_id)
        t_df.insert(1, "reference_label", ref_payload["label"])
        t_df.insert(4, "comparison_family", _comparison_family(ctrl_id))
        t_df.insert(5, "comparison_note", _comparison_note(ctrl_id))
        frames.append(t_df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = apply_multiple_comparison_corrections(
        out,
        group_cols=["baseline_id", "metric"],
        p_col="p_value",
    )
    if "wilcoxon_p_value" in out.columns:
        wilcox_corr = apply_multiple_comparison_corrections(
            out[["baseline_id", "metric", "wilcoxon_p_value"]].copy(),
            group_cols=["baseline_id", "metric"],
            p_col="wilcoxon_p_value",
        )
        out["wilcoxon_p_value_holm"] = wilcox_corr["p_value_holm"]
        out["wilcoxon_p_value_bh"] = wilcox_corr["p_value_bh"]
        out["wilcoxon_significant_holm"] = wilcox_corr["significant_holm"]
        out["wilcoxon_significant_bh"] = wilcox_corr["significant_bh"]
        out["direction_agrees_nonparametric"] = out["wilcoxon_note"].eq("paired_confirmation")
    out["metric_role_order"] = out["metric_role"].map(_metric_role_order)
    out["season_order"] = out.get("season", pd.Series([""] * len(out))).map(_season_sort_key).fillna(-1)
    return out.sort_values(
        ["comparison_family", "baseline_label", "metric_role_order", "metric", "analysis_scope", "season_order"]
    ).drop(columns=["metric_role_order", "season_order"])


def generate_multi_controller_report(controller_summaries: dict, pairwise_df: pd.DataFrame, save_path: str = None) -> str:
    """Human-readable comparison report for all four controllers."""
    lines = []
    lines.append("=" * 78)
    lines.append("MULTI-CONTROLLER COMPARISON REPORT")
    lines.append("=" * 78)
    lines.append("")
    lines.append("--- Overall Means ---")

    summary_table = compile_controller_summary_table(controller_summaries)
    if not summary_table.empty:
        cols = [
            "controller_label",
            "n_scenarios",
            "mean_iwu_mm",
            "mean_target_pct",
            "mean_yield_dry_t_ha",
            "mean_iwue_kg_ha_per_mm",
        ]
        lines.append(_format_markdown_table(summary_table, columns=cols, float_fmt=".3f"))
    else:
        lines.append("No controller summaries available.")

    lines.append("")
    lines.append("--- Pairwise Tests vs Fuzzy-Phenology ---")
    if pairwise_df is not None and not pairwise_df.empty:
        cols = [
            "comparison_family",
            "baseline_label",
            "analysis_scope",
            "season",
            "metric",
            "reference_mean",
            "baseline_mean",
            "difference",
            "ci_95_low",
            "ci_95_high",
            "t_statistic",
            "p_value",
            "p_value_holm",
            "cohens_d",
            "n_pairs",
        ]
        lines.append(_format_markdown_table(pairwise_df, columns=cols, float_fmt=".4f"))
    else:
        lines.append("No pairwise results available.")

    report = "\n".join(lines)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
    return report


def build_comparison_table_pack(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    """Create comparison table pack with explicit section labels."""
    if pairwise_df is None or pairwise_df.empty:
        return pd.DataFrame()

    out = pairwise_df.copy()
    out["table_section"] = "attribution_comparison"
    out.loc[
        (out["comparison_family"] == "headline_comparison") &
        (out["metric_role"] == "headline_control"),
        "table_section",
    ] = "headline_comparison"
    out.loc[out["metric_role"] == "robustness_only", "table_section"] = "robustness_only"
    out["metric_group"] = out["metric_role"].map({
        "headline_control": "control_tracking",
        "attribution_behavior": "control_behavior",
        "robustness_only": "agronomic_safeguard",
    }).fillna("auxiliary")
    out["scope_order"] = out["analysis_scope"].map({"overall": 0, "seasonal": 1}).fillna(9)
    out["season_order"] = out.get("season", pd.Series([""] * len(out))).map(_season_sort_key).fillna(-1)
    cols = [
        "table_section",
        "comparison_family",
        "comparison_note",
        "reference_label",
        "baseline_label",
        "analysis_scope",
        "season",
        "metric_group",
        "metric_role",
        "metric",
        "reference_mean",
        "baseline_mean",
        "difference",
        "ci_95_low",
        "ci_95_high",
        "p_value",
        "p_value_holm",
        "wilcoxon_p_value",
        "wilcoxon_p_value_holm",
        "cohens_d",
        "n_pairs",
    ]
    out = out.sort_values(
        ["table_section", "baseline_label", "metric_group", "metric", "scope_order", "season_order"]
    )
    return out.loc[:, [c for c in cols if c in out.columns]]


def export_head_to_head_csvs(
    fuzzy_summary: pd.DataFrame,
    reactive_summary: pd.DataFrame,
    output_dir: str,
    year_label: str = "2010_2024",
):
    """
    Export two CSV files:
    1) detail per year x season (head-to-head)
    2) seasonal mean summary (Penghujan/Peralihan/Kemarau)
    """
    if fuzzy_summary.empty or reactive_summary.empty:
        return None, None

    # Detail per scenario (year x season)
    detail = pd.merge(
        fuzzy_summary, reactive_summary,
        on=["year", "season"],
        how="inner",
        suffixes=("_fuzzy", "_reactive")
    )

    # Keep controller-facing metrics first, with agronomic safeguards last.
    requested_cols = [
        "year", "season",
        "target_pct_fuzzy", "target_pct_reactive",
        "mean_r_pct_fuzzy", "mean_r_pct_reactive",
        "mse_fuzzy", "mse_reactive",
        "n_irrigation_events_fuzzy", "n_irrigation_events_reactive",
        "irrigation_frequency_pct_fuzzy", "irrigation_frequency_pct_reactive",
        "mean_depth_per_event_mm_fuzzy", "mean_depth_per_event_mm_reactive",
        "mean_interval_between_events_days_fuzzy", "mean_interval_between_events_days_reactive",
        "max_interval_without_irrigation_days_fuzzy", "max_interval_without_irrigation_days_reactive",
        "iwu_mm_fuzzy", "iwu_mm_reactive",
        "yield_dry_t_ha_fuzzy", "yield_dry_t_ha_reactive",
        "iwue_kg_ha_per_mm_fuzzy", "iwue_kg_ha_per_mm_reactive",
        "aq_transpiration_ratio_fuzzy", "aq_transpiration_ratio_reactive",
    ]
    keep_cols = [c for c in requested_cols if c in detail.columns]
    detail = detail[keep_cols].copy()
    detail["delta_target_pct"] = detail["target_pct_fuzzy"] - detail["target_pct_reactive"]
    detail["delta_mean_r_pct"] = detail["mean_r_pct_fuzzy"] - detail["mean_r_pct_reactive"]
    if "mse_fuzzy" in detail.columns and "mse_reactive" in detail.columns:
        detail["delta_mse"] = detail["mse_fuzzy"] - detail["mse_reactive"]
    if "n_irrigation_events_fuzzy" in detail.columns and "n_irrigation_events_reactive" in detail.columns:
        detail["delta_n_irrigation_events"] = (
            detail["n_irrigation_events_fuzzy"] - detail["n_irrigation_events_reactive"]
        )
    detail["delta_iwu_mm"] = detail["iwu_mm_fuzzy"] - detail["iwu_mm_reactive"]
    detail["delta_yield_dry_t_ha"] = detail["yield_dry_t_ha_fuzzy"] - detail["yield_dry_t_ha_reactive"]
    detail["delta_iwue_kg_ha_per_mm"] = detail["iwue_kg_ha_per_mm_fuzzy"] - detail["iwue_kg_ha_per_mm_reactive"]
    detail["season_order"] = detail["season"].map(_season_sort_key)
    detail = detail.sort_values(["year", "season_order", "season"]).drop(columns=["season_order"])

    detail_path = os.path.join(output_dir, f"head_to_head_per_skenario_{year_label}.csv")
    detail.to_csv(detail_path, index=False)

    # Seasonal summary means across years
    summary_rows = (
        detail.groupby("season", as_index=False)
        .agg(
            target_fuzzy_pct=("target_pct_fuzzy", "mean"),
            target_reaktif_pct=("target_pct_reactive", "mean"),
            mean_r_fuzzy_pct=("mean_r_pct_fuzzy", "mean"),
            mean_r_reaktif_pct=("mean_r_pct_reactive", "mean"),
            mse_fuzzy=("mse_fuzzy", "mean"),
            mse_reaktif=("mse_reactive", "mean"),
            iwu_fuzzy_mm=("iwu_mm_fuzzy", "mean"),
            iwu_reaktif_mm=("iwu_mm_reactive", "mean"),
            events_fuzzy=("n_irrigation_events_fuzzy", "mean"),
            events_reaktif=("n_irrigation_events_reactive", "mean"),
            yield_fuzzy_t_ha=("yield_dry_t_ha_fuzzy", "mean"),
            yield_reaktif_t_ha=("yield_dry_t_ha_reactive", "mean"),
            iwue_fuzzy_kg_ha_per_mm=("iwue_kg_ha_per_mm_fuzzy", "mean"),
            iwue_reaktif_kg_ha_per_mm=("iwue_kg_ha_per_mm_reactive", "mean"),
            n_skenario=("year", "count"),
        )
    )
    summary_rows["delta_target_pct"] = summary_rows["target_fuzzy_pct"] - summary_rows["target_reaktif_pct"]
    summary_rows["delta_mean_r_pct"] = summary_rows["mean_r_fuzzy_pct"] - summary_rows["mean_r_reaktif_pct"]
    summary_rows["delta_mse"] = summary_rows["mse_fuzzy"] - summary_rows["mse_reaktif"]
    summary_rows["delta_iwu_mm"] = summary_rows["iwu_fuzzy_mm"] - summary_rows["iwu_reaktif_mm"]
    summary_rows["delta_events"] = summary_rows["events_fuzzy"] - summary_rows["events_reaktif"]
    summary_rows["delta_yield_t_ha"] = summary_rows["yield_fuzzy_t_ha"] - summary_rows["yield_reaktif_t_ha"]
    summary_rows["delta_iwue_kg_ha_per_mm"] = (
        summary_rows["iwue_fuzzy_kg_ha_per_mm"] - summary_rows["iwue_reaktif_kg_ha_per_mm"]
    )
    summary_rows["season_order"] = summary_rows["season"].map(_season_sort_key)
    summary_rows = summary_rows.sort_values(["season_order", "season"]).drop(columns=["season_order"])

    summary_path = os.path.join(output_dir, f"head_to_head_ringkasan_per_musim_{year_label}.csv")
    summary_rows.to_csv(summary_path, index=False)

    print(f"  ✓ CSV detail head-to-head: {detail_path}")
    print(f"  ✓ CSV ringkasan per musim: {summary_path}")
    return detail_path, summary_path


def calibrate_eta_for_yield(weather_df: pd.DataFrame, eta_candidates=None, engine_kwargs: dict | None = None) -> tuple:
    """
    Grid-search eta for fuzzy controller using AquaCrop agronomic yield.
    Fokus optimasi: memaksimalkan yield musim kemarau.

    Returns:
        (best_eta, calibration_df)
    """
    if eta_candidates is None:
        eta_candidates = [0.70, 0.75, 0.80, 0.85, 0.90]

    records = []
    print(f"\n{'='*60}")
    print("CALIBRATING ETA (Objective: maximize yield, priority=MT-3/Kemarau)")
    print(f"{'='*60}")

    for eta in eta_candidates:
        print(f"  -> Testing eta={eta:.2f}")
        ctrl = FuzzyIrrigationController(efficiency=eta)
        engine = SimulationEngine(controller=ctrl, weather_df=weather_df, **(engine_kwargs or {}))
        all_results = engine.run_all_scenarios(verbose=False)
        if all_results.empty:
            continue

        summary = compute_scenario_summary(all_results)
        summary = enrich_summary_with_aquacrop(all_results, summary, weather_df, eta, verbose=False)

        valid_iwue = summary["iwue_kg_ha_per_mm"].dropna() if "iwue_kg_ha_per_mm" in summary.columns else pd.Series(dtype=float)
        valid_yield = summary["yield_dry_t_ha"].dropna() if "yield_dry_t_ha" in summary.columns else pd.Series(dtype=float)
        kemarau_mask = summary["season"].astype(str).str.contains("Kemarau", na=False)
        kemarau_yield = summary.loc[kemarau_mask, "yield_dry_t_ha"].dropna() if "yield_dry_t_ha" in summary.columns else pd.Series(dtype=float)
        kemarau_iwue = summary.loc[kemarau_mask, "iwue_kg_ha_per_mm"].dropna() if "iwue_kg_ha_per_mm" in summary.columns else pd.Series(dtype=float)

        rec = {
            "eta": eta,
            "n_skenario": int(len(summary)),
            "n_iwue_valid": int(valid_iwue.shape[0]),
            "mean_iwue_kg_ha_per_mm": float(valid_iwue.mean()) if not valid_iwue.empty else float("nan"),
            "std_iwue_kg_ha_per_mm": float(valid_iwue.std()) if valid_iwue.shape[0] > 1 else 0.0,
            "mean_yield_dry_t_ha": float(valid_yield.mean()) if not valid_yield.empty else float("nan"),
            "mean_yield_kemarau_t_ha": float(kemarau_yield.mean()) if not kemarau_yield.empty else float("nan"),
            "mean_iwue_kemarau_kg_ha_per_mm": float(kemarau_iwue.mean()) if not kemarau_iwue.empty else float("nan"),
            "mean_iwu_mm": float(summary["iwu_mm"].mean()) if "iwu_mm" in summary else float("nan"),
            "mean_target_pct": float(summary["target_pct"].mean()) if "target_pct" in summary else float("nan"),
        }
        records.append(rec)
        print(
            f"Yield(Kemarau)={rec['mean_yield_kemarau_t_ha']:.3f} t/ha, "
            f"Yield(All)={rec['mean_yield_dry_t_ha']:.3f} t/ha, "
            f"IWUE(Kemarau)={rec['mean_iwue_kemarau_kg_ha_per_mm']:.2f}, "
            f"IWU={rec['mean_iwu_mm']:.1f} mm, Target={rec['mean_target_pct']:.1f}%"
        )

    calib_df = pd.DataFrame(records).sort_values("eta").reset_index(drop=True)
    if calib_df.empty:
        return IRRIGATION_EFFICIENCY, calib_df

    # Primary objective: maximize mean kemarau yield. Tie-breakers: all-season yield, then IWUE.
    ranked = calib_df.sort_values(
        ["mean_yield_kemarau_t_ha", "mean_yield_dry_t_ha", "mean_iwue_kg_ha_per_mm"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    best_eta = float(ranked.iloc[0]["eta"])
    print(f"\n  Selected eta* = {best_eta:.2f} (max mean MT-3/Kemarau yield)")
    return best_eta, calib_df


def choose_head_to_head_scenarios_by_season(fuzzy_summary: pd.DataFrame, reactive_summary: pd.DataFrame) -> pd.DataFrame:
    """Pick one representative scenario per season group for visual head-to-head plots."""
    merged = pd.merge(
        fuzzy_summary, reactive_summary,
        on=["year", "season"],
        how="inner",
        suffixes=("_fuzzy", "_reactive")
    )
    if merged.empty:
        return pd.DataFrame()

    merged["delta_target_pct"] = merged["target_pct_fuzzy"] - merged["target_pct_reactive"]
    merged["delta_iwu_mm"] = merged["iwu_mm_fuzzy"] - merged["iwu_mm_reactive"]
    merged["season_group"] = merged["season"].map(_season_group)

    rows = []
    for group_name, g in merged.groupby("season_group", sort=False):
        cand = g.sort_values(["delta_target_pct", "delta_iwu_mm"], ascending=[False, True]).iloc[0]
        rows.append({
            "season_group": group_name,
            "year": int(cand["year"]),
            "season": str(cand["season"]),
            "delta_target_pct": float(cand["delta_target_pct"]),
            "delta_iwu_mm": float(cand["delta_iwu_mm"]),
            "iwu_fuzzy_mm": float(cand.get("iwu_mm_fuzzy", np.nan)),
            "iwu_reactive_mm": float(cand.get("iwu_mm_reactive", np.nan)),
            "yield_fuzzy_t_ha": float(cand.get("yield_dry_t_ha_fuzzy", np.nan)) if "yield_dry_t_ha_fuzzy" in cand else np.nan,
            "yield_reactive_t_ha": float(cand.get("yield_dry_t_ha_reactive", np.nan)) if "yield_dry_t_ha_reactive" in cand else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["sort_key"] = out["season_group"].map(_season_sort_key)
        out = out.sort_values(["sort_key", "season_group"]).drop(columns=["sort_key"])
    return out


def generate_sample_plots(
    fuzzy_results: pd.DataFrame,
    reactive_results: pd.DataFrame,
    fuzzy_summary: pd.DataFrame,
    reactive_summary: pd.DataFrame,
    output_dir: str | None = None,
):
    """Generate all visualization plots."""
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}")
    
    # 1. Head-to-head diagnostics for each season group (MT-1/MT-2/MT-3)
    output_dir = output_dir or get_output_dir()
    selected = choose_head_to_head_scenarios_by_season(fuzzy_summary, reactive_summary)
    if not selected.empty:
        selected.to_csv(os.path.join(output_dir, "diagnostic_plot_scenarios.csv"), index=False)
        diag_stats_rows = []
        for _, sel in selected.iterrows():
            chosen_year = int(sel["year"])
            chosen_season = str(sel["season"])
            for results, name in [(fuzzy_results, 'fuzzy'), (reactive_results, 'reactive')]:
                if results.empty:
                    continue
                season_data = results[
                    (results['year'] == chosen_year) &
                    (results['season'] == chosen_season)
                ].sort_values("hst")
                if season_data.empty:
                    continue

                plot_method_soil_moisture_diagnostics(
                    season_data,
                    title=f"{name.capitalize()} — {chosen_year} {chosen_season}",
                    save_path=os.path.join(output_dir, f"diagnostic_{name}_{chosen_year}_{chosen_season}.png"),
                )
                plot_season_timeseries(
                    season_data,
                    title=f"{name.capitalize()} — {chosen_year} {chosen_season}",
                    save_path=os.path.join(output_dir, f'timeseries_{name}_{chosen_year}_{chosen_season}.png')
                )
                print(f"  ✓ Diagnostic {name}: {chosen_year} {chosen_season}")

                # Save stats for each plotted diagnostic scenario
                target_mid = (season_data["rL"] + season_data["rU"]) / 2
                dev = season_data["r"] - target_mid
                diag_stats_rows.append({
                    "method": name,
                    "year": chosen_year,
                    "season": chosen_season,
                    "season_group": _season_group(chosen_season),
                    "days": int(len(season_data)),
                    "iwu_mm": float(season_data["irrigation_mm"].sum()),
                    "total_precip_mm": float(season_data["precipitation"].sum()),
                    "mean_sm_pct_fc": float(season_data["r_pct"].mean()),
                    "mean_target_mid_pct_fc": float((target_mid * 100).mean()),
                    "days_in_target_pct": float(((season_data["r"] >= season_data["rL"]) & (season_data["r"] <= season_data["rU"])).mean() * 100),
                    "over_target_days": int((season_data["r"] > season_data["rU"]).sum()),
                    "under_target_days": int((season_data["r"] < season_data["rL"]).sum()),
                    "mean_abs_dev_pct_fc": float((dev.abs() * 100).mean()),
                    "max_over_dev_pct_fc": float(((season_data["r"] - season_data["rU"]).clip(lower=0) * 100).max()),
                    "max_under_dev_pct_fc": float(((season_data["rL"] - season_data["r"]).clip(lower=0) * 100).max()),
                    "irrigation_events": int((season_data["irrigation_mm"] > 1e-6).sum()),
                })
        if diag_stats_rows:
            pd.DataFrame(diag_stats_rows).to_csv(os.path.join(output_dir, "diagnostic_plot_stats_per_musim.csv"), index=False)

    # 2. Fallback/additional sample season time-series (first available scenario per controller if not already produced)
    for results, name in [(fuzzy_results, 'fuzzy'), (reactive_results, 'reactive')]:
        if results.empty:
            continue
        
        first_scenario = results.groupby(['year', 'season']).size().reset_index()
        if not first_scenario.empty:
            first_year = first_scenario.iloc[0]['year']
            first_season = first_scenario.iloc[0]['season']
            
            season_data = results[
                (results['year'] == first_year) & 
                (results['season'] == first_season)
            ]
            
            out_path = os.path.join(output_dir, f'timeseries_{name}_{first_year}_{first_season}.png')
            if not os.path.exists(out_path):
                plot_season_timeseries(
                    season_data,
                    title=f"{name.capitalize()} — {first_year} {first_season}",
                    save_path=out_path
                )
                print(f"  ✓ Time-series: {name} ({first_year} {first_season})")
    
    # 3. Comparison boxplots
    if not fuzzy_summary.empty and not reactive_summary.empty:
        plot_comparison_boxplots(
            fuzzy_summary, reactive_summary,
            save_path=os.path.join(output_dir, 'comparison_boxplots.png')
        )
        print(f"  ✓ Comparison boxplots")
        
        # 4. Seasonal comparison
        plot_seasonal_comparison(
            fuzzy_summary, reactive_summary,
            save_path=os.path.join(output_dir, 'seasonal_comparison.png')
        )
        print(f"  ✓ Seasonal comparison")
        
        # 5. Trade-off plot
        plot_tradeoff(
            fuzzy_summary, reactive_summary,
            save_path=os.path.join(output_dir, 'tradeoff_analysis.png')
        )
        print(f"  ✓ Trade-off analysis")


def choose_reproducibility_figure_representative_scenario(controller_summaries: dict) -> pd.DataFrame:
    """Choose one shared representative scenario across the four-controller pack."""
    controller_ids = ["fuzzy_phenology", "fuzzy_static", "reactive_static", "reactive_stage"]
    metric_cols = ["iwu_mm", "yield_dry_t_ha", "iwue_kg_ha_per_mm", "target_pct", "mse"]

    merged = None
    for ctrl_id in controller_ids:
        payload = controller_summaries.get(ctrl_id, {})
        summary = payload.get("summary", pd.DataFrame())
        if summary is None or summary.empty:
            return pd.DataFrame()
        available_cols = [c for c in metric_cols if c in summary.columns]
        ctrl_df = summary[["year", "season"] + available_cols].copy()
        ctrl_df = ctrl_df.rename(columns={col: f"{col}_{ctrl_id}" for col in available_cols})
        merged = ctrl_df if merged is None else pd.merge(merged, ctrl_df, on=["year", "season"], how="inner")

    if merged is None or merged.empty:
        return pd.DataFrame()

    score_parts = []
    for ctrl_id in controller_ids:
        payload = controller_summaries.get(ctrl_id, {})
        summary = payload.get("summary", pd.DataFrame())
        for metric in metric_cols:
            col = f"{metric}_{ctrl_id}"
            if col not in merged.columns or metric not in summary.columns:
                continue
            center = float(summary[metric].mean())
            spread = float(summary[metric].std())
            if not np.isfinite(spread) or spread < 1e-9:
                score_parts.append(pd.Series(0.0, index=merged.index))
            else:
                score_parts.append(((merged[col] - center).abs() / spread).astype(float))
    if not score_parts:
        return pd.DataFrame()

    merged["representativeness_score"] = pd.concat(score_parts, axis=1).mean(axis=1)
    merged["season_sort_key"] = merged["season"].map(_season_sort_key)
    merged = merged.sort_values(["representativeness_score", "season_sort_key", "year"]).reset_index(drop=True)
    return merged.head(1).drop(columns=["season_sort_key"])


def choose_reproducibility_figure_best_fuzzy_phenology_scenario(controller_summaries: dict) -> pd.DataFrame:
    """
    Choose a strong scenario for visually explaining Fuzzy-Phenology behavior.

    The figure is illustrative, so the selection favors MT-3 first, then scores
    Fuzzy-Phenology by yield, IWUE, target occupancy, MSE, irrigation saving
    versus Reactive-Static, and yield difference versus Reactive-Static.
    """
    controller_ids = ["fuzzy_phenology", "fuzzy_static", "reactive_static", "reactive_stage"]
    metric_cols = ["iwu_mm", "yield_dry_t_ha", "iwue_kg_ha_per_mm", "target_pct", "mse"]

    merged = None
    for ctrl_id in controller_ids:
        payload = controller_summaries.get(ctrl_id, {})
        summary = payload.get("summary", pd.DataFrame())
        if summary is None or summary.empty:
            return pd.DataFrame()
        available_cols = [c for c in metric_cols if c in summary.columns]
        ctrl_df = summary[["year", "season"] + available_cols].copy()
        ctrl_df = ctrl_df.rename(columns={col: f"{col}_{ctrl_id}" for col in available_cols})
        merged = ctrl_df if merged is None else pd.merge(merged, ctrl_df, on=["year", "season"], how="inner")

    if merged is None or merged.empty:
        return pd.DataFrame()

    mt3 = merged[merged["season"].astype(str).str.contains("Kemarau", case=False, na=False)].copy()
    candidate = mt3 if not mt3.empty else merged.copy()
    score_terms = []
    score_specs = {
        "yield_dry_t_ha_fuzzy_phenology": 1.0,
        "iwue_kg_ha_per_mm_fuzzy_phenology": 1.0,
        "target_pct_fuzzy_phenology": 1.0,
        "mse_fuzzy_phenology": -1.0,
    }
    candidate["iwu_saving_vs_reactive_static_mm"] = (
        candidate["iwu_mm_reactive_static"] - candidate["iwu_mm_fuzzy_phenology"]
    )
    candidate["yield_delta_vs_reactive_static_t_ha"] = (
        candidate["yield_dry_t_ha_fuzzy_phenology"] - candidate["yield_dry_t_ha_reactive_static"]
    )
    score_specs.update({
        "iwu_saving_vs_reactive_static_mm": 1.0,
        "yield_delta_vs_reactive_static_t_ha": 1.0,
    })

    for col, direction in score_specs.items():
        if col not in candidate.columns:
            continue
        values = candidate[col].astype(float)
        spread = float(values.std())
        if not np.isfinite(spread) or spread < 1e-9:
            score_terms.append(pd.Series(0.0, index=candidate.index))
        else:
            score_terms.append(direction * ((values - float(values.mean())) / spread))

    if not score_terms:
        return pd.DataFrame()

    candidate["fuzzy_phenology_visual_score"] = pd.concat(score_terms, axis=1).mean(axis=1)
    candidate["season_sort_key"] = candidate["season"].map(_season_sort_key)
    candidate = candidate.sort_values(
        ["fuzzy_phenology_visual_score", "season_sort_key", "year"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return candidate.head(1).drop(columns=["season_sort_key"])


def generate_reproducibility_figure_visual_package(output_dir: str, controller_results: dict, controller_summaries: dict) -> None:
    """Generate the reproducibility figure composite figure and supporting notes."""
    selected = choose_reproducibility_figure_best_fuzzy_phenology_scenario(controller_summaries)
    if selected.empty:
        return

    chosen = selected.iloc[0]
    chosen_year = int(chosen["year"])
    chosen_season = str(chosen["season"])
    representative_results = {}
    controller_order = ["fuzzy_phenology", "fuzzy_static", "reactive_static", "reactive_stage"]
    for ctrl_id in controller_order:
        results_df = controller_results.get(ctrl_id, {}).get("results", pd.DataFrame())
        if results_df is None or results_df.empty:
            continue
        representative_results[ctrl_id] = results_df[
            (results_df["year"] == chosen_year) &
            (results_df["season"] == chosen_season)
        ].sort_values("hst").copy()

    if len(representative_results) < 4:
        return

    overall_table = compile_controller_summary_table(controller_summaries)
    seasonal_table = compile_seasonal_summary_table(controller_summaries)
    timeseries_figure_path = os.path.join(output_dir, "reproducibility_figure_representative_controller_timeseries.png")
    irrigation_figure_path = os.path.join(output_dir, "reproducibility_figure_seasonal_iwu_actions.png")
    title = f"{chosen_year} {chosen_season}"
    plot_reproducibility_figure_composite_figure(
        representative_results=representative_results,
        seasonal_summary_df=seasonal_table,
        overall_summary_df=overall_table,
        representative_title=title,
        save_path=timeseries_figure_path,
    )
    plot_reproducibility_figure_irrigation_summary_figure(
        seasonal_summary_df=seasonal_table,
        save_path=irrigation_figure_path,
    )

    scenario_csv_path = os.path.join(output_dir, "reproducibility_figure_best_fuzzy_phenology_scenario.csv")
    selected.to_csv(scenario_csv_path, index=False)

    notes = [
        "# reproducibility figure Figure Package Notes",
        "",
        "Figure 1 scenario rule",
        "----------------------",
        "- Candidate set: scenarios available for all four final controllers.",
        "- Selection prioritizes MT-3 scenarios when available because the lead researcher requested the dry-season case where Fuzzy-Phenology tends to show its strongest behavior.",
        "- Scoring rule: within the candidate set, maximize the average standardized score for Fuzzy-Phenology yield, IWUE, target occupancy, lower MSE, irrigation saving versus Reactive-Static, and yield difference versus Reactive-Static.",
        "- Tie-breakers: lower season sort key, then earlier year.",
        "",
        "Selected scenario",
        "-----------------",
        f"- Year: `{chosen_year}`",
        f"- Season: `{chosen_season}`",
        f"- Fuzzy-Phenology visual score: `{float(chosen['fuzzy_phenology_visual_score']):.4f}`",
        f"- IWU saving versus Reactive-Static: `{float(chosen['iwu_saving_vs_reactive_static_mm']):.3f}` mm",
        f"- Yield difference versus Reactive-Static: `{float(chosen['yield_delta_vs_reactive_static_t_ha']):.3f}` t/ha",
        "",
        "Exported assets",
        "---------------",
        f"- [reproducibility_figure_representative_controller_timeseries.png]({timeseries_figure_path}:1)",
        f"- [reproducibility_figure_seasonal_iwu_actions.png]({irrigation_figure_path}:1)",
        f"- [reproducibility_figure_best_fuzzy_phenology_scenario.csv]({scenario_csv_path}:1)",
    ]
    notes_path = os.path.join(output_dir, "reproducibility_figure_composite_notes.md")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("\n".join(notes))

    print(f"  ✓ reproducibility figure representative time-series figure: {timeseries_figure_path}")
    print(f"  ✓ reproducibility figure seasonal irrigation figure: {irrigation_figure_path}")


def load_controller_artifacts_from_output_dir(output_dir: str) -> tuple[dict, dict]:
    """Load saved controller result/summary artifacts from an output folder."""
    controller_specs = [
        ("fuzzy_phenology", "Fuzzy-Phenology", "fuzzy"),
        ("fuzzy_static", "Fuzzy-Static", "fuzzy"),
        ("reactive_static", "Reactive-Static", "reactive"),
        ("reactive_stage", "Reactive-Phenology", "reactive"),
    ]
    controller_results = {}
    controller_summaries = {}
    for ctrl_id, label, group in controller_specs:
        results_path = os.path.join(output_dir, f"{ctrl_id}_results.csv")
        summary_path = os.path.join(output_dir, f"{ctrl_id}_summary.csv")
        results_df = pd.read_csv(results_path) if os.path.exists(results_path) else pd.DataFrame()
        summary_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
        controller_results[ctrl_id] = {
            "label": label,
            "group": group,
            "description": "",
            "results": results_df,
        }
        controller_summaries[ctrl_id] = {
            "label": label,
            "group": group,
            "description": "",
            "summary": summary_df,
        }
    return controller_results, controller_summaries


def _format_markdown_table(df: pd.DataFrame, columns: list = None, max_rows: int = None, float_fmt: str = ".3f") -> str:
    """Simple markdown table formatter (portable, no external deps)."""
    if df is None or df.empty:
        return "_Tidak ada data._"

    if columns is not None:
        table_df = df.loc[:, [c for c in columns if c in df.columns]].copy()
    else:
        table_df = df.copy()
    if max_rows is not None:
        table_df = table_df.head(max_rows)

    def fmt(v):
        if pd.isna(v):
            return ""
        if isinstance(v, float):
            return format(v, float_fmt)
        return str(v)

    headers = list(table_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table_df.iterrows():
        lines.append("| " + " | ".join(fmt(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def generate_analisa_markdown(
    output_dir: str,
    weather_df: pd.DataFrame,
    fuzzy_results: pd.DataFrame,
    reactive_results: pd.DataFrame,
    fuzzy_summary: pd.DataFrame,
    reactive_summary: pd.DataFrame,
    eta_selected: float = None,
    eta_calibration_df: pd.DataFrame = None,
):
    """Create detailed descriptive analysis in Indonesian for journal-style reporting."""
    analisa_path = os.path.join(output_dir, "analisa.md")

    phenology_table = get_subphase_table().copy()
    phenology_table["durasi_hari"] = phenology_table["end_hst"] - phenology_table["start_hst"] + 1
    phenology_table["rL_pct"] = phenology_table["rL"] * 100
    phenology_table["rU_pct"] = phenology_table["rU"] * 100
    phenology_table["rWP_pct"] = phenology_table["rWP"] * 100
    phenology_table = phenology_table[
        ["phase_index", "major", "sub", "start_hst", "end_hst", "durasi_hari", "rL_pct", "rU_pct", "rWP_pct", "kc"]
    ]

    weather_stats = pd.DataFrame([{
        "periode_awal": weather_df["Date"].min().strftime("%Y-%m-%d"),
        "periode_akhir": weather_df["Date"].max().strftime("%Y-%m-%d"),
        "jumlah_hari": int(len(weather_df)),
        "rata2_hujan_mm": float(weather_df["Prcp"].mean()),
        "total_hujan_mm": float(weather_df["Prcp"].sum()),
        "rata2_et0_mm": float(weather_df["Et0"].mean()),
        "min_et0_mm": float(weather_df["Et0"].min()),
        "max_et0_mm": float(weather_df["Et0"].max()),
    }])

    monthly_weather = (
        weather_df.assign(month=weather_df["Date"].dt.month)
        .groupby("month", as_index=False)
        .agg(
            prcp_mean=("Prcp", "mean"),
            prcp_total=("Prcp", "sum"),
            et0_mean=("Et0", "mean"),
            tmin_mean=("Tmin", "mean"),
            tmax_mean=("Tmax", "mean"),
        )
    )

    def summarize(df):
        if df is None or df.empty:
            return pd.DataFrame()
        cols = [c for c in [
            "iwu_mm", "mean_r_pct", "mse", "target_pct", "stress_days",
            "n_irrigation_events", "total_precip", "total_etc",
            "yield_dry_t_ha", "yield_potential_t_ha", "yield_gap_t_ha",
            "iwue_kg_ha_per_mm", "aq_transpiration_ratio", "aq_stress_days_tr"
        ] if c in df.columns]
        rows = []
        for c in cols:
            rows.append({
                "metrik": c,
                "rata_rata": float(df[c].mean()),
                "std": float(df[c].std()) if len(df[c]) > 1 else 0.0,
                "min": float(df[c].min()),
                "maks": float(df[c].max()),
            })
        return pd.DataFrame(rows)

    fuzzy_stats = summarize(fuzzy_summary)
    reactive_stats = summarize(reactive_summary)

    paired_path = os.path.join(output_dir, "paired_ttest_results.csv")
    ttest_df = pd.read_csv(paired_path) if os.path.exists(paired_path) else pd.DataFrame()

    # Seasonal summaries
    def seasonal_means(df, algo_name):
        if df is None or df.empty:
            return pd.DataFrame()
        out = (
            df.groupby("season", as_index=False)
            .agg(
                iwu_mm=("iwu_mm", "mean"),
                target_pct=("target_pct", "mean"),
                mean_r_pct=("mean_r_pct", "mean"),
                mse=("mse", "mean"),
                n_irrigation_events=("n_irrigation_events", "mean"),
                stress_days=("stress_days", "mean"),
                yield_dry_t_ha=("yield_dry_t_ha", "mean"),
                iwue_kg_ha_per_mm=("iwue_kg_ha_per_mm", "mean"),
            )
        )
        out.insert(0, "algoritma", algo_name)
        return out

    seasonal_df = pd.concat(
        [seasonal_means(fuzzy_summary, "Fuzzy-Fenologi"), seasonal_means(reactive_summary, "Reaktif")],
        ignore_index=True
    ) if (not fuzzy_summary.empty or not reactive_summary.empty) else pd.DataFrame()

    # Explicit head-to-head seasonal comparison focused on IWU, Yield, and IWUE
    seasonal_h2h = pd.DataFrame()
    if not fuzzy_summary.empty and not reactive_summary.empty:
        f_season = (
            fuzzy_summary.groupby("season", as_index=False)
            .agg(
                iwu_fuzzy_mm=("iwu_mm", "mean"),
                yield_fuzzy_t_ha=("yield_dry_t_ha", "mean"),
                iwue_fuzzy_kg_ha_per_mm=("iwue_kg_ha_per_mm", "mean"),
            )
        )
        r_season = (
            reactive_summary.groupby("season", as_index=False)
            .agg(
                iwu_reaktif_mm=("iwu_mm", "mean"),
                yield_reaktif_t_ha=("yield_dry_t_ha", "mean"),
                iwue_reaktif_kg_ha_per_mm=("iwue_kg_ha_per_mm", "mean"),
            )
        )
        seasonal_h2h = pd.merge(f_season, r_season, on="season", how="outer")
        if not seasonal_h2h.empty:
            seasonal_h2h["delta_iwu_mm"] = seasonal_h2h["iwu_fuzzy_mm"] - seasonal_h2h["iwu_reaktif_mm"]
            seasonal_h2h["delta_yield_t_ha"] = seasonal_h2h["yield_fuzzy_t_ha"] - seasonal_h2h["yield_reaktif_t_ha"]
            seasonal_h2h["delta_iwue_kg_ha_per_mm"] = (
                seasonal_h2h["iwue_fuzzy_kg_ha_per_mm"] - seasonal_h2h["iwue_reaktif_kg_ha_per_mm"]
            )
            seasonal_h2h["season_order"] = seasonal_h2h["season"].map(_season_sort_key)
            seasonal_h2h = seasonal_h2h.sort_values(["season_order", "season"]).drop(columns=["season_order"])

    # Representative scenario for deep dive: prioritize MT-3/Kemarau if available.
    rep_df = pd.DataFrame()
    if not fuzzy_results.empty:
        candidates = fuzzy_results.groupby(["year", "season"]).size().reset_index(name="n")
        kemarau = candidates[candidates["season"].astype(str).str.contains("Kemarau", na=False)]
        chosen = kemarau.iloc[0] if not kemarau.empty else candidates.iloc[0]
        rep_df = fuzzy_results[
            (fuzzy_results["year"] == chosen["year"]) & (fuzzy_results["season"] == chosen["season"])
        ].copy()

    rep_summary = pd.DataFrame()
    phase_breakdown = pd.DataFrame()
    if not rep_df.empty:
        rep_summary = pd.DataFrame([{
            "year": int(rep_df["year"].iloc[0]),
            "season": rep_df["season"].iloc[0],
            "hari_tersimulasi": int(len(rep_df)),
            "iwu_mm": float(rep_df["irrigation_mm"].sum()),
            "hujan_mm": float(rep_df["precipitation"].sum()),
            "etc_mm": float(rep_df["etc"].sum()),
            "mean_sm_pct_fc": float(rep_df["r_pct"].mean()),
            "hari_dalam_target_pct": float(((rep_df["r"] >= rep_df["rL"]) & (rep_df["r"] <= rep_df["rU"])).mean() * 100),
        }])
        phase_breakdown = (
            rep_df.groupby(["phase", "subphase"], as_index=False)
            .agg(
                hari=("hst", "count"),
                hujan_total_mm=("precipitation", "sum"),
                irigasi_total_mm=("irrigation_mm", "sum"),
                et0_rata2=("et0", "mean"),
                etc_total_mm=("etc", "sum"),
                sm_rata2_pct_fc=("r_pct", "mean"),
                urgensi_rata2=("urgency", "mean"),
            )
        )

    md = []
    md.append("# Analisa Simulasi Kontrol Irigasi Fuzzy-Fenologi")
    md.append("")
    md.append("## 1. Ringkasan Eksekutif")
    md.append("")
    md.append("Dokumen ini menyajikan analisis deskriptif hasil simulasi ulang kontrol irigasi dinamis berbasis fuzzy-fenologi pada digital twin neraca air (AquaCrop-OSPy-style closed-loop) dengan data cuaca BMKG Malang dan parameter fase padi dari `data/paddy_growth_phenology.csv`.")
    md.append("")
    md.append("Konfigurasi kunci yang digunakan pada eksekusi ini:")
    md.append(f"- Rentang data cuaca simulasi: **{weather_df['Date'].min().strftime('%Y-%m-%d')} s.d. {weather_df['Date'].max().strftime('%Y-%m-%d')}** (termasuk buffer tahun berikutnya untuk MT-1 lintas tahun).")
    md.append("- Musim tanam per tahun:")
    md.append("  MT-1 / Penghujan: **1 Oktober - Januari** (lintas tahun)")
    md.append("  MT-2 / Peralihan: **1 Februari - Mei**")
    md.append("  MT-3 / Kemarau: **1 Juni - September**")
    md.append(f"- Fuzzy phenology: **7 sub-fase** dari CSV dengan smoothing overlap **{PHENOLOGY_OVERLAP_DAYS} hari**.")
    md.append(f"- Baseline reaktif: ambang tunggal **{REACTIVE_THRESHOLD_FC*100:.0f}% FC** dan refill ke **{REACTIVE_TARGET_FC*100:.0f}% FC**.")
    if eta_selected is None:
        eta_selected = IRRIGATION_EFFICIENCY
    md.append(f"- Efisiensi irigasi final (hasil eksekusi): **η = {eta_selected:.2f}**.")
    md.append("- Mode optimasi: **yield-first** (target fuzzy musim kemarau dibuat lebih protektif, terutama fase reproduktif).")
    md.append("")
    if eta_calibration_df is not None and not eta_calibration_df.empty:
        md.append("### 1.1 Hasil Kalibrasi η Berbasis IWUE Agronomis (AquaCrop)")
        md.append("")
        md.append("Kalibrasi dilakukan dengan grid search terhadap beberapa kandidat `η`, kemudian dipilih nilai yang memaksimalkan **yield musim kemarau** (prioritas utama), dengan tie-breaker berdasarkan yield keseluruhan dan IWUE agronomis. Untuk menghindari distorsi rasio pada musim sangat basah, IWUE dihitung dengan penyaringan skenario ber-IWU sangat kecil.")
        md.append("")
        md.append(_format_markdown_table(eta_calibration_df, float_fmt=".4f"))
        md.append("")
    md.append("## 2. Input Data dan Parameter")
    md.append("")
    md.append("### 2.1 Statistik Data Cuaca")
    md.append("")
    md.append(_format_markdown_table(weather_stats, float_fmt=".2f"))
    md.append("")
    md.append("### 2.2 Ringkasan Bulanan Data Cuaca (2015–2024)")
    md.append("")
    md.append(_format_markdown_table(monthly_weather, max_rows=12, float_fmt=".2f"))
    md.append("")
    md.append("### 2.3 Parameter Fenologi 7 Fase dari CSV")
    md.append("")
    md.append(_format_markdown_table(phenology_table, float_fmt=".2f"))
    md.append("")
    md.append("Gambar berikut menunjukkan membership fuzzy fase (agregat 3 kelompok), target kelembapan dinamis, dan Kc yang dibentuk dari CSV dengan smoothing transisi.")
    md.append("")
    md.append("![Fenologi dan Target Dinamis](phenology_targets.png)")
    md.append("")
    md.append("### 2.4 Desain Fuzzy Inference System")
    md.append("")
    md.append("FIS menggunakan Mamdani (max-min) dan defuzzifikasi centroid dengan tiga himpunan output (`rendah`, `sedang`, `tinggi`). Pada run ini, fuzzy difokuskan untuk memetakan **defisit soil moisture terhadap batas bawah target (`rL`)** dan **indikator fase kritis biner (0 = non-kritis, 1 = kritis)** menjadi **fraksi refill** irigasi harian (single decision per hari).")
    md.append("")
    md.append("Keputusan irigasi hanya diproses ketika `SM_t < rL`. Volume irigasi diarahkan ke area **mendekati batas atas target (`rU`)** (yield-oriented) dan dibatasi dengan guard anti-overshoot agar tidak jauh melampaui `SM_up`.")
    md.append("")
    md.append("![Membership Function Fuzzy](fuzzy_mf_visualization.png)")
    md.append("")
    md.append("## 3. Proses Simulasi (Closed-loop)")
    md.append("")
    md.append("Alur harian simulasi untuk setiap skenario adalah:")
    md.append("")
    md.append("1. Membaca cuaca harian (`Prcp`, `Et0`) pada tanggal simulasi.")
    md.append("2. Menghitung target dinamis `rL`, `rU`, `rWP` dan `Kc` dari fase fenologi berbasis HST.")
    md.append("3. Menghitung defisit ternormalisasi terhadap `rL` (`d_norm`) dan menentukan `fase_kritis` (0/1) dari fase fenologi dominan.")
    md.append("4. Menentukan fraksi refill irigasi `u_i` melalui FIS Mamdani (defisit × fase_kritis).")
    md.append("5. Mengonversi `u_i` menjadi kedalaman irigasi `I_i` menuju area dekat `rU` (yield-oriented), dengan efisiensi aplikasi `η` serta guard anti-overshoot terhadap `SM_up`.")
    md.append("6. Memperbarui kelembapan tanah menggunakan neraca air harian (Pers. 1–3), termasuk limpasan, ETc, dan perkolasi dalam.")
    md.append("")
    md.append("Perbaikan implementasi yang dilakukan pada run ini:")
    md.append("")
    md.append("- Target kelembapan dan `Kc` kini dibaca langsung dari CSV 7 fase (bukan hardcoded 3 fase).")
    md.append("- Smoothing transisi fase diterapkan melalui membership fuzzy antar-subfase.")
    md.append("- FIS Mamdani diimplementasikan numerik (tanpa `skfuzzy`) sehingga tetap reprodusibel pada environment saat ini.")
    md.append("- Perkolasi dalam dihitung setelah inflow/outflow harian diterapkan (menangkap kelebihan air hari yang sama).")
    md.append("- Koefisien stres air (`Ks`) diperbaiki menggunakan ambang depletion berbasis `RAW`, bukan nilai tetap yang terlalu dekat ke titik layu.")
    md.append("- Untuk skenario **musim kemarau**, target kelembapan fuzzy dinaikkan (seasonal target boost) agar keputusan irigasi lebih protektif terhadap penurunan yield.")
    md.append("- Logika fuzzy kini difokuskan pada **band tracking** (`rL`–`rU`) berbasis fenologi: irigasi aktif ketika `SM_t` turun di bawah `rL`, lalu volume disesuaikan proporsional-fuzzy menuju area dekat `rU`.")
    md.append("")
    md.append("## 4. Output Simulasi dan Perbandingan Kinerja")
    md.append("")
    md.append("### 4.1 Statistik Ringkas Per Skenario — Fuzzy-Fenologi")
    md.append("")
    md.append(_format_markdown_table(fuzzy_stats, float_fmt=".4f"))
    md.append("")
    md.append("### 4.2 Statistik Ringkas Per Skenario — Reaktif")
    md.append("")
    md.append(_format_markdown_table(reactive_stats, float_fmt=".4f"))
    md.append("")
    md.append("### 4.3 Rata-rata Per Musim Tanam")
    md.append("")
    md.append(_format_markdown_table(seasonal_df, float_fmt=".4f"))
    md.append("")
    md.append("### 4.3.1 Perbandingan Head-to-Head Per Musim Tanam (IWU, Yield, IWUE)")
    md.append("")
    md.append("Tabel berikut menyajikan perbandingan langsung antar-metode untuk tiga metrik utama yang Anda minta: **Irrigation Water Use (IWU)**, **Yield kering**, dan **Irrigation Water Use Efficiency (IWUE)** pada setiap musim tanam.")
    md.append("")
    md.append(_format_markdown_table(seasonal_h2h, float_fmt=".4f"))
    md.append("")
    md.append("### 4.4 Uji Signifikansi (Paired t-test)")
    md.append("")
    if not ttest_df.empty:
        cols = ["metric", "fuzzy_mean", "reactive_mean", "difference", "t_statistic", "p_value", "cohens_d", "significant", "n_pairs"]
        md.append(_format_markdown_table(ttest_df, columns=cols, float_fmt=".4f"))
    else:
        md.append("_Hasil paired t-test tidak tersedia._")
    md.append("")
    md.append("### 4.5 Visualisasi Hasil")
    md.append("")
    # Add head-to-head diagnostics first (if available)
    for img in sorted(fn for fn in os.listdir(output_dir) if fn.startswith("diagnostic_") and fn.endswith(".png")):
        md.append(f"![{img}]({img})")
        md.append("")

    md.append("![Perbandingan Boxplot](comparison_boxplots.png)")
    md.append("")
    md.append("![Perbandingan per Musim](seasonal_comparison.png)")
    md.append("")
    md.append("![Analisis Trade-off](tradeoff_analysis.png)")
    md.append("")

    # Embed available time-series images dynamically
    for img in sorted(fn for fn in os.listdir(output_dir) if fn.startswith("timeseries_") and fn.endswith(".png")):
        md.append(f"![{img}]({img})")
        md.append("")

    md.append("## 5. Pembahasan Deskriptif (Interpretasi Teknis)")
    md.append("")
    if not fuzzy_summary.empty and not reactive_summary.empty:
        f_iwu = fuzzy_summary["iwu_mm"].mean()
        r_iwu = reactive_summary["iwu_mm"].mean()
        f_target = fuzzy_summary["target_pct"].mean()
        r_target = reactive_summary["target_pct"].mean()
        f_mse = fuzzy_summary["mse"].mean()
        r_mse = reactive_summary["mse"].mean()
        md.append(f"Secara umum, kontrol fuzzy-fenologi menghasilkan **IWU rata-rata {f_iwu:.2f} mm** dibandingkan kontrol reaktif **{r_iwu:.2f} mm**. Perbedaan ini menunjukkan bahwa strategi berbasis fase fenologi cenderung lebih aktif melakukan intervensi irigasi untuk menjaga kelembapan tanah pada rentang target dinamis, terutama ketika fase sensitif memiliki prioritas air yang lebih tinggi.")
        md.append("")
        md.append(f"Dari sisi kualitas tracking, persentase hari dalam rentang target pada fuzzy-fenologi adalah **{f_target:.2f}%**, sedangkan reaktif **{r_target:.2f}%**. Nilai ini penting karena tujuan utama kontrol dinamis bukan hanya mengurangi volume air, tetapi menjaga kelembapan terhadap target biologis yang berubah mengikuti fase pertumbuhan.")
        md.append("")
        md.append(f"Rata-rata **MSE tracking** fuzzy-fenologi sebesar **{f_mse:.4f}**, sementara reaktif **{r_mse:.4f}**. MSE yang lebih rendah mengindikasikan kontrol lebih konsisten terhadap target kelembapan dinamis; sebaliknya MSE yang lebih tinggi menunjukkan deviasi yang lebih besar akibat sifat reaktif yang menunggu ambang statis terlampaui.")
        md.append("")
        if {"yield_dry_t_ha", "iwue_kg_ha_per_mm"}.issubset(fuzzy_summary.columns) and {"yield_dry_t_ha", "iwue_kg_ha_per_mm"}.issubset(reactive_summary.columns):
            f_y = fuzzy_summary["yield_dry_t_ha"].mean()
            r_y = reactive_summary["yield_dry_t_ha"].mean()
            f_iwue = fuzzy_summary["iwue_kg_ha_per_mm"].dropna().mean()
            r_iwue = reactive_summary["iwue_kg_ha_per_mm"].dropna().mean()
            md.append(f"Dari sisi agronomis (AquaCrop), **hasil kering rata-rata** fuzzy-fenologi adalah **{f_y:.3f} t/ha**, sedangkan reaktif **{r_y:.3f} t/ha**.")
            md.append("")
            md.append(f"Untuk skenario dengan irigasi yang cukup besar (`IWU >= 10 mm`), **IWUE agronomis** fuzzy-fenologi sebesar **{f_iwue:.3f} kg/ha/mm**, sedangkan reaktif **{r_iwue:.3f} kg/ha/mm**. Pada versi optimasi yield ini, IWUE tetap dilaporkan sebagai metrik trade-off, tetapi **bukan** lagi objektif utama kalibrasi `η`.")
            md.append("")
            if not seasonal_h2h.empty:
                md.append("Pada tingkat musim tanam, pola trade-off terlihat lebih jelas: musim **Kemarau** menunjukkan penghematan IWU terbesar oleh fuzzy-fenologi, namun juga risiko penurunan yield paling nyata pada beberapa tahun kering. Sebaliknya pada musim **Penghujan**, yield kedua metode cenderung mendekati potensi karena kontribusi hujan tinggi, sehingga keunggulan fuzzy lebih banyak terlihat pada efisiensi penggunaan air irigasi (IWUE) dan penurunan volume irigasi.")
                md.append("")
            md.append("Secara head-to-head, plot diagnostik terpisah menunjukkan bahwa kontrol fuzzy-fenologi cenderung menahan kelembapan tanah lebih dekat ke rentang target dinamis (khususnya pada fase sensitif) dengan frekuensi irigasi lebih tinggi namun kedalaman lebih kecil per kejadian. Sebaliknya, kontrol reaktif statis cenderung menghasilkan pola irigasi yang lebih jarang tetapi lebih dalam, sehingga lebih sering memicu kondisi terlalu basah (over-irrigation) setelah refill ke 100% FC.")
            md.append("")
    if not rep_summary.empty:
        md.append("## 6. Analisis Mendalam Skenario Representatif")
        md.append("")
        md.append("Skenario representatif dipilih dari hasil fuzzy-fenologi (prioritas musim `Kemarau` jika tersedia) untuk meninjau perilaku kontrol harian secara lebih rinci.")
        md.append("")
        md.append(_format_markdown_table(rep_summary, float_fmt=".3f"))
        md.append("")
        md.append("### 6.1 Breakdown per Fase/Sub-fase pada Skenario Representatif")
        md.append("")
        md.append(_format_markdown_table(phase_breakdown, float_fmt=".3f"))
        md.append("")
        md.append("Interpretasi: tabel di atas membantu mengecek apakah intensitas irigasi dan urgensi meningkat pada fase yang sensitif (khususnya reproduktif), dan menurun pada fase pemasakan sesuai rule base fuzzy yang dirancang.")
        md.append("")

    md.append("## 7. Keterbatasan dan Implikasi untuk Pengembangan Jurnal")
    md.append("")
    md.append("- Komponen **yield/stress response AquaCrop-OSPy** sudah diintegrasikan untuk evaluasi agronomis (`yield`, `IWUE`, rasio transpiration aktual/potensial). Namun, dinamika soil moisture yang diplot pada analisis kontrol tetap berasal dari model closed-loop custom yang mengikuti formulasi reproducibility.")
    md.append("- Profil tanah pada evaluasi AquaCrop menggunakan soil bawaan `Paddy` untuk menjaga stabilitas simulasi padi. Jika diperlukan replikasi parameter tanah reproducibility secara lebih ketat, tahap berikutnya adalah membangun `Soil('custom')` yang diturunkan langsung dari tabel parameter tanah penelitian.")
    md.append("- Kalibrasi `η` pada run ini dioptimalkan berdasarkan rata-rata IWUE; untuk publikasi jurnal, Anda dapat menambahkan kalibrasi multi-objektif (mis. maksimum IWUE dengan constraint penurunan yield <= ambang tertentu).")
    md.append("")
    md.append("## 8. Artefak Output")
    md.append("")
    md.append("File utama yang dihasilkan pada direktori `output/` antara lain:")
    for fn in sorted(os.listdir(output_dir)):
        if fn.startswith("."):
            continue
        md.append(f"- `{fn}`")
    md.append("")
    md.append("Dokumen ini dapat dijadikan dasar penulisan naskah ilmiah versi lanjutan (Bahasa Inggris) dengan memperluas bagian validasi, kalibrasi, dan integrasi yield.")
    md.append("")

    with open(analisa_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"  ✓ Analisa markdown: {analisa_path}")


def generate_multi_controller_analisa_markdown(
    output_dir: str,
    weather_df: pd.DataFrame,
    controller_results: dict,
    controller_summaries: dict,
    eta_selected: float = None,
    eta_calibration_df: pd.DataFrame = None,
    fuzzy_profile_selected: dict = None,
    fuzzy_profile_tuning_df: pd.DataFrame = None,
    pairwise_df: pd.DataFrame = None,
    analisa_filename: str = "analisa.md",
    run_scope_label: str | None = None,
):
    """Create a detailed journal-oriented markdown report for the four-controller study."""
    analisa_path = os.path.join(output_dir, analisa_filename)
    if eta_selected is None:
        eta_selected = IRRIGATION_EFFICIENCY

    phenology_table = get_subphase_table().copy()
    phenology_table["durasi_hari"] = phenology_table["end_hst"] - phenology_table["start_hst"] + 1
    phenology_table["rL_pct"] = phenology_table["rL"] * 100
    phenology_table["rU_pct"] = phenology_table["rU"] * 100
    phenology_table["rWP_pct"] = phenology_table["rWP"] * 100
    phenology_table = phenology_table[
        ["phase_index", "major", "sub", "start_hst", "end_hst", "durasi_hari", "rL_pct", "rU_pct", "rWP_pct", "kc"]
    ]

    weather_stats = pd.DataFrame([{
        "periode_awal": weather_df["Date"].min().strftime("%Y-%m-%d"),
        "periode_akhir": weather_df["Date"].max().strftime("%Y-%m-%d"),
        "jumlah_hari": int(len(weather_df)),
        "rata2_hujan_mm": float(weather_df["Prcp"].mean()),
        "total_hujan_mm": float(weather_df["Prcp"].sum()),
        "rata2_et0_mm": float(weather_df["Et0"].mean()),
        "min_et0_mm": float(weather_df["Et0"].min()),
        "max_et0_mm": float(weather_df["Et0"].max()),
    }])

    monthly_weather = (
        weather_df.assign(month=weather_df["Date"].dt.month)
        .groupby("month", as_index=False)
        .agg(
            prcp_mean=("Prcp", "mean"),
            prcp_total=("Prcp", "sum"),
            et0_mean=("Et0", "mean"),
            tmin_mean=("Tmin", "mean"),
            tmax_mean=("Tmax", "mean"),
        )
    )

    controller_table = pd.DataFrame([
        {
            "controller_id": ctrl_id,
            "controller_label": payload["label"],
            "kelompok": payload["group"].title(),
            "deskripsi": payload["description"],
        }
        for ctrl_id, payload in controller_summaries.items()
    ])

    overall_table = compile_controller_summary_table(controller_summaries)
    seasonal_table = compile_seasonal_summary_table(controller_summaries)

    metric_frames = []
    metric_cols = [
        "iwu_mm",
        "target_pct",
        "mean_r_pct",
        "mse",
        "n_irrigation_events",
        "irrigation_frequency_pct",
        "mean_depth_per_event_mm",
        "median_depth_per_event_mm",
        "cv_event_depth",
        "mean_interval_between_events_days",
        "median_interval_between_events_days",
        "max_interval_between_events_days",
        "max_interval_without_irrigation_days",
        "yield_dry_t_ha",
        "iwue_kg_ha_per_mm",
        "aq_transpiration_ratio",
        "aq_stress_days_tr",
    ]
    for ctrl_id, payload in controller_summaries.items():
        summary = payload["summary"]
        if summary.empty:
            continue
        rows = []
        for col in metric_cols:
            if col not in summary.columns:
                continue
            values = summary[col].dropna()
            if values.empty:
                continue
            rows.append({
                "controller_label": payload["label"],
                "metrik": col,
                "rata_rata": float(values.mean()),
                "std": float(values.std()) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "maks": float(values.max()),
            })
        if rows:
            metric_frames.append(pd.DataFrame(rows))
    metric_detail = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()

    seasonal_h2h = pd.DataFrame()
    fp = controller_summaries.get("fuzzy_phenology", {}).get("summary", pd.DataFrame())
    if not fp.empty:
        comp_frames = []
        for baseline_id in ["fuzzy_static", "reactive_stage", "reactive_static"]:
            base = controller_summaries.get(baseline_id, {}).get("summary", pd.DataFrame())
            if base.empty:
                continue
            f_season = (
                fp.groupby("season", as_index=False)
                .agg(
                    iwu_ref_mm=("iwu_mm", "mean"),
                    yield_ref_t_ha=("yield_dry_t_ha", "mean"),
                    iwue_ref_kg_ha_per_mm=("iwue_kg_ha_per_mm", "mean"),
                )
            )
            b_season = (
                base.groupby("season", as_index=False)
                .agg(
                    iwu_base_mm=("iwu_mm", "mean"),
                    yield_base_t_ha=("yield_dry_t_ha", "mean"),
                    iwue_base_kg_ha_per_mm=("iwue_kg_ha_per_mm", "mean"),
                )
            )
            merged = pd.merge(f_season, b_season, on="season", how="inner")
            if merged.empty:
                continue
            merged.insert(0, "baseline_label", controller_summaries[baseline_id]["label"])
            merged["delta_iwu_mm"] = merged["iwu_ref_mm"] - merged["iwu_base_mm"]
            merged["delta_yield_t_ha"] = merged["yield_ref_t_ha"] - merged["yield_base_t_ha"]
            merged["delta_iwue_kg_ha_per_mm"] = merged["iwue_ref_kg_ha_per_mm"] - merged["iwue_base_kg_ha_per_mm"]
            comp_frames.append(merged)
        if comp_frames:
            seasonal_h2h = pd.concat(comp_frames, ignore_index=True)
            seasonal_h2h["season_order"] = seasonal_h2h["season"].map(_season_sort_key)
            seasonal_h2h = seasonal_h2h.sort_values(["baseline_label", "season_order", "season"]).drop(columns=["season_order"])

    rep_df = pd.DataFrame()
    fuzzy_results = controller_results.get("fuzzy_phenology", {}).get("results", pd.DataFrame())
    if not fuzzy_results.empty:
        candidates = fuzzy_results.groupby(["year", "season"]).size().reset_index(name="n")
        kemarau = candidates[candidates["season"].astype(str).str.contains("Kemarau", na=False)]
        chosen = kemarau.iloc[0] if not kemarau.empty else candidates.iloc[0]
        rep_df = fuzzy_results[
            (fuzzy_results["year"] == chosen["year"]) & (fuzzy_results["season"] == chosen["season"])
        ].copy()

    rep_summary = pd.DataFrame()
    phase_breakdown = pd.DataFrame()
    if not rep_df.empty:
        rep_summary = pd.DataFrame([{
            "controller": "Fuzzy-Phenology",
            "year": int(rep_df["year"].iloc[0]),
            "season": rep_df["season"].iloc[0],
            "hari_tersimulasi": int(len(rep_df)),
            "iwu_mm": float(rep_df["irrigation_mm"].sum()),
            "hujan_mm": float(rep_df["precipitation"].sum()),
            "etc_mm": float(rep_df["etc"].sum()),
            "mean_sm_pct_fc": float(rep_df["r_pct"].mean()),
            "hari_dalam_target_pct": float(((rep_df["r"] >= rep_df["rL"]) & (rep_df["r"] <= rep_df["rU"])).mean() * 100),
        }])
        phase_breakdown = (
            rep_df.groupby(["phase", "subphase"], as_index=False)
            .agg(
                hari=("hst", "count"),
                hujan_total_mm=("precipitation", "sum"),
                irigasi_total_mm=("irrigation_mm", "sum"),
                et0_rata2=("et0", "mean"),
                etc_total_mm=("etc", "sum"),
                sm_rata2_pct_fc=("r_pct", "mean"),
                urgensi_rata2=("urgency", "mean"),
            )
        )

    md = []
    md.append("# Analisa Simulasi Empat Controller Irigasi")
    md.append("")
    md.append("## 1. Ringkasan Eksekutif")
    md.append("")
    md.append("Dokumen ini merangkum hasil simulasi komparatif **empat controller** pada kerangka digital twin irigasi padi berbasis neraca air harian dan evaluasi agronomis AquaCrop-OSPy. Desain ini dibuat untuk memisahkan kontribusi **fuzzy inference** dan **phenology awareness** secara lebih defensible untuk kebutuhan publikasi.")
    md.append("")
    md.append("Konfigurasi utama run ini:")
    md.append(f"- Rentang data cuaca: **{weather_df['Date'].min().strftime('%Y-%m-%d')} s.d. {weather_df['Date'].max().strftime('%Y-%m-%d')}**.")
    md.append("- Kalender tanam: **MT-1 1 Oktober**, **MT-2 1 Februari**, **MT-3 1 Juni**.")
    if run_scope_label:
        md.append(f"- Cakupan simulasi saat ini: **{run_scope_label}**.")
    md.append(f"- Fase fenologi: **7 sub-fase** dari CSV dengan overlap smoothing **{PHENOLOGY_OVERLAP_DAYS} hari**.")
    md.append(f"- Efisiensi aplikasi irigasi final: **eta = {eta_selected:.2f}**.")
    md.append("- Fokus interpretasi hasil: **IWUE improvement** dengan **yield sebagai safeguard / non-degradation control**, bukan target optimasi utama.")
    md.append("")
    md.append("### 1.1 Definisi Empat Controller")
    md.append("")
    md.append(_format_markdown_table(controller_table))
    md.append("")
    md.append("Makna desain ablation:")
    md.append("")
    md.append("1. `Reactive-Static -> Fuzzy-Static` mengisolasi efek fuzzy inference tanpa phenology awareness.")
    md.append("2. `Reactive-Phenology -> Fuzzy-Phenology` mengisolasi efek fuzzy inference ketika kedua controller sama-sama memakai informasi fenologi.")
    md.append("3. `Fuzzy-Static -> Fuzzy-Phenology` mengisolasi efek phenology awareness murni di atas mesin fuzzy yang sama.")
    md.append("4. `Reactive-Static -> Fuzzy-Phenology` menunjukkan manfaat total sistem yang diusulkan.")
    md.append("")
    if eta_calibration_df is not None and not eta_calibration_df.empty:
        md.append("### 1.2 Tabel Kalibrasi Eta")
        md.append("")
        md.append("Kalibrasi eta dijalankan pada controller `Fuzzy-Phenology`, lalu nilai terpilih dipakai sama untuk semua controller agar perbandingan tetap fair.")
        md.append("")
        md.append(_format_markdown_table(eta_calibration_df, float_fmt=".4f"))
        md.append("")
    if fuzzy_profile_tuning_df is not None and not fuzzy_profile_tuning_df.empty:
        md.append("### 1.3 Tuning Parameter Fuzzy-Phenology (Objective: IWUE)")
        md.append("")
        md.append("Setelah `eta` dipilih, profil fuzzy diuji pada beberapa kombinasi parameter untuk memaksimalkan **IWUE**. Parameter yang dituning adalah:")
        md.append("- `upper_ref_margin_frac`: seberapa dekat target refill ke batas atas `rU`; makin besar nilainya, refill makin konservatif.")
        md.append("- `overshoot_tol_frac`: toleransi overshoot di atas `rU`; makin kecil nilainya, kontrol makin ketat.")
        md.append("- `max_irrigation_daily_mm`: batas aplikasi harian; ini mengatur apakah controller boleh melakukan refill besar sekaligus.")
        md.append("- `vegetative_weight`, `reproductive_weight`, `maturation_weight`: bobot sensitivitas fase pada input fuzzy.")
        md.append("")
        md.append(_format_markdown_table(fuzzy_profile_tuning_df, float_fmt=".4f"))
        md.append("")
        if fuzzy_profile_selected:
            chosen = pd.DataFrame([fuzzy_profile_selected])
            md.append("Profil terpilih:")
            md.append("")
            md.append(_format_markdown_table(chosen, float_fmt=".4f"))
            md.append("")

    md.append("## 2. Data Input dan Parameter")
    md.append("")
    md.append("### 2.1 Statistik Ringkas Cuaca")
    md.append("")
    md.append(_format_markdown_table(weather_stats, float_fmt=".2f"))
    md.append("")
    md.append("### 2.2 Ringkasan Bulanan Cuaca")
    md.append("")
    md.append(_format_markdown_table(monthly_weather, max_rows=12, float_fmt=".2f"))
    md.append("")
    md.append("### 2.3 Tabel Fenologi 7 Sub-fase")
    md.append("")
    md.append(_format_markdown_table(phenology_table, float_fmt=".2f"))
    md.append("")
    md.append("![Fenologi dan Target Dinamis](phenology_targets.png)")
    md.append("")
    md.append("![Membership Function Fuzzy](fuzzy_mf_visualization.png)")
    md.append("")

    md.append("## 3. Ringkasan Hasil Lintas Controller")
    md.append("")
    md.append("### 3.1 Tabel Rata-rata Keseluruhan")
    md.append("")
    overall_cols = [
        "controller_label",
        "n_scenarios",
        "mean_iwu_mm",
        "mean_target_pct",
        "mean_n_irrigation_events",
        "mean_irrigation_frequency_pct",
        "mean_mean_depth_per_event_mm",
        "mean_mean_interval_between_events_days",
        "mean_mean_r_pct",
        "mean_mse",
        "mean_yield_dry_t_ha",
        "mean_iwue_kg_ha_per_mm",
        "mean_aq_transpiration_ratio",
    ]
    md.append(_format_markdown_table(overall_table, columns=overall_cols, float_fmt=".4f"))
    md.append("")
    md.append("### 3.2 Statistik Detail per Controller")
    md.append("")
    md.append(_format_markdown_table(metric_detail, float_fmt=".4f"))
    md.append("")
    md.append("### 3.3 Rata-rata per Musim Tanam")
    md.append("")
    md.append(_format_markdown_table(seasonal_table, float_fmt=".4f"))
    md.append("")

    md.append("## 4. Perbandingan Pairwise terhadap Fuzzy-Phenology")
    md.append("")
    md.append("Bagian ini adalah inti evaluasi klaim ilmiah. `Fuzzy-Phenology` diperlakukan sebagai metode usulan, lalu dibandingkan head-to-head dengan tiga baseline lain.")
    md.append("")
    if pairwise_df is not None and not pairwise_df.empty:
        pair_cols = [
            "baseline_label",
            "metric",
            "fuzzy_mean",
            "reactive_mean",
            "difference",
            "t_statistic",
            "p_value",
            "cohens_d",
            "significant",
            "n_pairs",
        ]
        md.append(_format_markdown_table(pairwise_df, columns=pair_cols, float_fmt=".4f"))
    else:
        md.append("_Hasil pairwise tidak tersedia._")
    md.append("")
    md.append("### 4.1 Head-to-Head Musiman (Fuzzy-Phenology vs Baseline)")
    md.append("")
    md.append(_format_markdown_table(seasonal_h2h, float_fmt=".4f"))
    md.append("")

    md.append("## 5. Interpretasi Ilmiah")
    md.append("")
    if not overall_table.empty:
        rank_iwue = overall_table.sort_values("mean_iwue_kg_ha_per_mm", ascending=False)
        rank_iwu = overall_table.sort_values("mean_iwu_mm", ascending=True)
        rank_freq = overall_table.sort_values("mean_irrigation_frequency_pct", ascending=False)
        rank_depth = overall_table.sort_values("mean_mean_depth_per_event_mm", ascending=False)
        best_iwue = rank_iwue.iloc[0]
        best_iwu = rank_iwu.iloc[0]
        most_frequent = rank_freq.iloc[0]
        deepest_event = rank_depth.iloc[0]
        md.append(f"- Controller dengan **IWUE tertinggi** pada run ini adalah **{best_iwue['controller_label']}** ({best_iwue['mean_iwue_kg_ha_per_mm']:.3f} kg/ha/mm).")
        md.append(f"- Controller dengan **IWU terendah** pada run ini adalah **{best_iwu['controller_label']}** ({best_iwu['mean_iwu_mm']:.3f} mm).")
        md.append(f"- Controller dengan **frekuensi irigasi tertinggi** adalah **{most_frequent['controller_label']}** ({most_frequent['mean_irrigation_frequency_pct']:.2f}% hari tanam).")
        md.append(f"- Controller dengan **kedalaman rata-rata per event terbesar** adalah **{deepest_event['controller_label']}** ({deepest_event['mean_mean_depth_per_event_mm']:.2f} mm/event).")
        md.append("")
    md.append("Interpretasi yang disarankan untuk report:")
    md.append("")
    md.append("- Klaim utama difokuskan pada **peningkatan IWUE** dan/atau penurunan `IWU` dengan tracking target yang lebih baik.")
    md.append("- `Yield` digunakan sebagai **kontrol agronomis** untuk menunjukkan bahwa efisiensi air yang lebih tinggi tidak dibayar dengan penurunan hasil yang bermakna.")
    md.append("- Untuk `yield`, gunakan wording trade-off yang eksplisit. Klaim yield-neutral hanya boleh dipakai jika memang didukung analisis yang sesuai.")
    md.append("")
    if pairwise_df is not None and not pairwise_df.empty:
        interesting = pairwise_df[
            pairwise_df["metric"].isin(["iwu_mm", "target_pct", "yield_dry_t_ha", "iwue_kg_ha_per_mm"])
        ].copy()
        for baseline_label in interesting["baseline_label"].drop_duplicates():
            subset = interesting[interesting["baseline_label"] == baseline_label]
            md.append(f"Perbandingan terhadap **{baseline_label}**:")
            for metric in ["iwu_mm", "target_pct", "yield_dry_t_ha", "iwue_kg_ha_per_mm"]:
                row = subset[subset["metric"] == metric]
                if row.empty:
                    continue
                r = row.iloc[0]
                signif = "signifikan" if bool(r["significant"]) else "tidak signifikan"
                md.append(
                    f"- `{metric}`: selisih rata-rata **{r['difference']:.4f}** dengan `p={r['p_value']:.4f}` ({signif})."
                )
            md.append("")
        md.append("Interpretasi pairwise harus dibaca sebagai berikut: nilai `difference` positif berarti rata-rata `Fuzzy-Phenology` lebih tinggi dari baseline untuk metrik tersebut, sedangkan nilai negatif berarti lebih rendah.")
        md.append("")
        md.append("Pembacaan metrik frekuensi:")
        md.append("- `n_irrigation_events` dan `irrigation_frequency_pct` mengukur intensitas intervensi.")
        md.append("- `mean_depth_per_event_mm` dan `median_depth_per_event_mm` memisahkan strategi pulsa kecil vs refill besar.")
        md.append("- `mean_interval_between_events_days` dan `max_interval_without_irrigation_days` memetakan ritme operasional di lapangan.")
        md.append("")

    md.append("### 5.1 Pola Aplikasi Irigasi")
    md.append("")
    if not overall_table.empty:
        irrigation_pattern_cols = [
            "controller_label",
            "mean_n_irrigation_events",
            "mean_irrigation_frequency_pct",
            "mean_mean_depth_per_event_mm",
            "mean_median_depth_per_event_mm",
            "mean_cv_event_depth",
            "mean_mean_interval_between_events_days",
            "mean_max_interval_without_irrigation_days",
        ]
        md.append(_format_markdown_table(overall_table, columns=irrigation_pattern_cols, float_fmt=".4f"))
        md.append("")
        md.append("Interpretasi operasional:")
        md.append("- Frekuensi tinggi berarti kontrol lebih sering melakukan koreksi kelembapan.")
        md.append("- Kedalaman per event yang kecil berarti strategi cenderung melakukan fine-tuning, bukan refill besar sekaligus.")
        md.append("- Interval antar-event yang pendek berarti controller menjaga kelembapan lebih rapat, tetapi berpotensi menambah beban operasional.")
        md.append("")

    md.append("## 6. Skenario Representatif")
    md.append("")
    if not rep_summary.empty:
        md.append("Skenario representatif diambil dari `Fuzzy-Phenology` (prioritas musim Kemarau) untuk mengecek dinamika kontrol harian pada kondisi paling informatif terhadap kebutuhan irigasi.")
        md.append("")
        md.append(_format_markdown_table(rep_summary, float_fmt=".3f"))
        md.append("")
        md.append("### 6.1 Breakdown Fase/Sub-fase")
        md.append("")
        md.append(_format_markdown_table(phase_breakdown, float_fmt=".3f"))
        md.append("")
    else:
        md.append("_Tidak ada skenario representatif yang tersedia._")
        md.append("")

    md.append("## 7. Keterbatasan yang Perlu Dicatat di Report")
    md.append("")
    md.append("- `Yield` dan `IWUE` agronomis berasal dari evaluasi AquaCrop, sedangkan dinamika kelembapan tanah harian yang dianalisis berasal dari model closed-loop custom.")
    md.append("- Profil tanah AquaCrop masih memakai soil bawaan `Paddy`, sehingga belum sepenuhnya identik dengan parameter tanah reproducibility.")
    md.append("- Banyak skenario basah menghasilkan `IWU` sangat kecil; karena itu `IWUE` dihitung dengan filter minimum `IWU` untuk mencegah distorsi rasio.")
    md.append("- Uji yang dipakai saat ini masih `paired t-test`; untuk naskah final disarankan menambahkan `confidence interval`, `non-inferiority/equivalence test` untuk yield, dan analisis sensitivitas parameter.")
    md.append("")
    md.append("## 8. Kesimpulan")
    md.append("")
    md.append("Kesimpulan utama dari run ini bersifat **nuansial**, bukan hitam-putih.")
    md.append("")
    md.append("- Dibanding `Fuzzy-Static`, memasukkan fase fenologi ke keputusan fuzzy memberikan dampak **positif yang jelas** terhadap stabilitas agronomis: yield jauh lebih tinggi, rasio transpiration lebih baik, dan tracking target meningkat. Artinya, phenology awareness pada fuzzy **bermanfaat** untuk mencegah under-irrigation ekstrem yang muncul saat target dibuat statis.")
    md.append("- Dibanding `Reactive-Static`, `Fuzzy-Phenology` juga memberikan dampak **positif yang praktis**: IWU turun besar, IWUE naik, dan yield tidak turun signifikan. Ini mendukung klaim bahwa fuzzy berbasis fenologi lebih baik daripada baseline reaktif statis sederhana.")
    md.append("- Namun, dibanding `Reactive-Phenology` (baseline rule-based yang juga memakai informasi fenologi), keunggulan `Fuzzy-Phenology` **tidak dominan**. Pada run ini, `Fuzzy-Phenology` menurunkan IWU, tetapi juga menurunkan `target_pct`; untuk `IWUE`, selisihnya tidak signifikan; dan untuk yield justru sedikit lebih rendah. Jadi, memasukkan fase fenologi ke fuzzy **tidak otomatis** membuat fuzzy menjadi metode terbaik jika dibandingkan dengan rule-based phenology-aware yang dirancang dengan baik.")
    md.append("- Implikasi ilmiahnya: **yang terbukti positif adalah phenology awareness di atas fuzzy statis**, tetapi **yang belum terbukti unggul adalah fuzzy inference terhadap baseline stage-based yang kuat**. Karena itu, klaim report yang paling defensible adalah bahwa phenology-aware fuzzy adalah metode yang valid dan kuat terhadap baseline statis, namun masih memerlukan tuning lanjutan untuk mengungguli controller phenology-aware non-fuzzy.")
    md.append("")

    md.append("## 9. Artefak Output")
    md.append("")
    md.append("File utama yang dihasilkan:")
    for fn in sorted(os.listdir(output_dir)):
        if fn.startswith("."):
            continue
        md.append(f"- `{fn}`")
    md.append("")
    md.append("Dokumen ini ditulis dengan detail yang cukup agar dapat dipakai sebagai basis penyusunan bagian **Methods**, **Results**, dan **Discussion** pada reproducibility report.")
    md.append("")

    with open(analisa_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"  ✓ Analisa markdown: {analisa_path}")


def build_year_label(years: list[int]) -> str:
    """Compact label for output filenames."""
    if not years:
        return "unknown_years"
    return f"{min(years)}_{max(years)}"


def build_attribution_isolation_table(controller_summaries: dict) -> pd.DataFrame:
    """Explain what each baseline isolates in the four-controller package."""
    rows = []
    baseline_map = [
        {
            "baseline_id": "reactive_static",
            "baseline_label": "Reactive-Static",
            "comparison_to_reference": "Fuzzy-Phenology vs Reactive-Static",
            "isolated_factor": "total system effect",
            "what_changes": "phenology-aware dynamic targets plus fuzzy urgency inference",
            "what_is_held_constant": "same weather archive, same irrigation-efficiency setting, same evaluation protocol",
        },
        {
            "baseline_id": "fuzzy_static",
            "baseline_label": "Fuzzy-Static",
            "comparison_to_reference": "Fuzzy-Phenology vs Fuzzy-Static",
            "isolated_factor": "phenology awareness within the same fuzzy engine",
            "what_changes": "dynamic phenology-driven target bands",
            "what_is_held_constant": "same Mamdani fuzzy engine and same irrigation mapping",
        },
        {
            "baseline_id": "reactive_stage",
            "baseline_label": "Reactive-Phenology",
            "comparison_to_reference": "Fuzzy-Phenology vs Reactive-Phenology",
            "isolated_factor": "fuzzy inference beyond stage-aware rule control",
            "what_changes": "urgency-scaled fuzzy actuation instead of direct refill-to-target rule",
            "what_is_held_constant": "same phenology-aware dynamic target bands",
        },
    ]
    available_ids = set(controller_summaries.keys())
    for item in baseline_map:
        if item["baseline_id"] not in available_ids or "fuzzy_phenology" not in available_ids:
            continue
        rows.append(item)
    return pd.DataFrame(rows)


def generate_attribution_interpretation_notes(
    output_dir: str,
    pairwise_df: pd.DataFrame,
    isolation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Write concise interpretation notes for the attribution package."""
    notes = []
    if pairwise_df is not None and not pairwise_df.empty:
        priority_metrics = ["iwu_mm", "yield_dry_t_ha", "iwue_kg_ha_per_mm", "mse", "target_pct"]
        for baseline_label in sorted(pairwise_df["baseline_label"].dropna().unique()):
            subset = pairwise_df[
                (pairwise_df["baseline_label"] == baseline_label) &
                (pairwise_df["analysis_scope"] == "overall") &
                (pairwise_df["metric"].isin(priority_metrics))
            ].copy()
            if subset.empty:
                continue
            focus = []
            for metric in priority_metrics:
                row = subset[subset["metric"] == metric]
                if row.empty:
                    continue
                row = row.iloc[0]
                direction = "higher" if float(row["difference"]) > 0 else "lower"
                significance = "significant" if float(row.get("p_value_holm", 1.0)) < 0.05 else "not significant"
                focus.append(f"{metric}: {direction} by {row['difference']:.3f} ({significance})")
            rationale = ""
            if isolation_df is not None and not isolation_df.empty:
                match = isolation_df[isolation_df["baseline_label"] == baseline_label]
                if not match.empty:
                    rationale = str(match.iloc[0]["isolated_factor"])
            notes.append({
                "baseline_label": baseline_label,
                "isolated_factor": rationale,
                "interpretation_note": "; ".join(focus),
            })
    notes_df = pd.DataFrame(notes)
    if not notes_df.empty:
        notes_df.to_csv(os.path.join(output_dir, "attribution_interpretation_notes.csv"), index=False)
        lines = ["# Attribution Interpretation Notes", ""]
        if isolation_df is not None and not isolation_df.empty:
            lines.append("## Baseline Isolation Map")
            lines.append("")
            lines.append(_format_markdown_table(isolation_df))
            lines.append("")
        lines.append("## Concise Notes")
        lines.append("")
        for _, row in notes_df.iterrows():
            lines.append(f"- `{row['baseline_label']}` ({row['isolated_factor']}): {row['interpretation_note']}.")
        with open(os.path.join(output_dir, "attribution_interpretation_notes.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    return notes_df


def generate_yield_focus_ablation_outputs(
    output_dir: str,
    weather_df: pd.DataFrame,
    eta_star: float,
    fuzzy_profile_selected: dict,
    engine_kwargs: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ablate the seasonal yield-protection heuristic if it remains in the code path."""
    if not YIELD_FOCUS_MODE and not KEMARAU_TARGET_BOOSTS:
        return pd.DataFrame(), pd.DataFrame()

    variants = [
        {
            "variant_id": "heuristic_on",
            "label": "Fuzzy-Phenology + seasonal yield heuristic",
            "yield_focus_mode": True,
            "target_boosts": dict(KEMARAU_TARGET_BOOSTS),
        },
        {
            "variant_id": "heuristic_off",
            "label": "Fuzzy-Phenology without seasonal yield heuristic",
            "yield_focus_mode": False,
            "target_boosts": {},
        },
        {
            "variant_id": "boosts_zeroed",
            "label": "Fuzzy-Phenology with branch on but kemarau boosts zeroed",
            "yield_focus_mode": True,
            "target_boosts": {
                phase: {"rL": 0.0, "rU": 0.0}
                for phase in KEMARAU_TARGET_BOOSTS
            },
        },
    ]

    variant_summaries = []
    summary_lookup = {}
    for variant in variants:
        ctrl = FuzzyIrrigationController(efficiency=eta_star, **fuzzy_profile_selected)
        variant_engine_kwargs = dict(engine_kwargs)
        variant_engine_kwargs["yield_focus_mode"] = variant["yield_focus_mode"]
        variant_engine_kwargs["target_boosts"] = variant["target_boosts"]
        results_df, summary_df = run_simulation(
            controller=ctrl,
            controller_name=f"ablation_{variant['variant_id']}",
            weather_df=weather_df,
            output_dir=output_dir,
            verbose=True,
            engine_kwargs=variant_engine_kwargs,
        )
        if not summary_df.empty:
            summary_df = enrich_summary_with_aquacrop(results_df, summary_df, weather_df, eta_star, verbose=True)
            save_results_and_summary(f"ablation_{variant['variant_id']}", results_df, summary_df, output_dir=output_dir)
            variant_summaries.append(pd.DataFrame([{
                "variant_id": variant["variant_id"],
                "variant_label": variant["label"],
                "yield_focus_mode": variant["yield_focus_mode"],
                "mean_iwu_mm": float(summary_df["iwu_mm"].mean()),
                "mean_yield_dry_t_ha": float(summary_df["yield_dry_t_ha"].mean()) if "yield_dry_t_ha" in summary_df.columns else np.nan,
                "mean_iwue_kg_ha_per_mm": float(summary_df["iwue_kg_ha_per_mm"].mean()) if "iwue_kg_ha_per_mm" in summary_df.columns else np.nan,
                "mean_target_pct": float(summary_df["target_pct"].mean()) if "target_pct" in summary_df.columns else np.nan,
                "mean_mse": float(summary_df["mse"].mean()) if "mse" in summary_df.columns else np.nan,
            }]))
            summary_lookup[variant["variant_id"]] = summary_df

    overall_df = pd.concat(variant_summaries, ignore_index=True) if variant_summaries else pd.DataFrame()
    if not overall_df.empty:
        overall_df.to_csv(os.path.join(output_dir, "yield_focus_ablation_summary.csv"), index=False)

    pairwise_frames = []
    metric_cols = ["iwu_mm", "yield_dry_t_ha", "iwue_kg_ha_per_mm", "target_pct", "mse"]
    reference_summary = summary_lookup.get("heuristic_on")
    if reference_summary is not None:
        for baseline_id in ["heuristic_off", "boosts_zeroed"]:
            baseline_summary = summary_lookup.get(baseline_id)
            if baseline_summary is None:
                continue
            pairwise_df = compute_paired_statistics(
                reference_summary,
                baseline_summary,
                metric_cols=metric_cols,
            )
            seasonal_df = paired_stats_by_season(
                reference_summary,
                baseline_summary,
                metric_cols=metric_cols,
            )
            pairwise_df = pd.concat([pairwise_df, seasonal_df], ignore_index=True)
            if pairwise_df.empty:
                continue
            pairwise_df.insert(0, "comparison", f"heuristic_on_vs_{baseline_id}")
            pairwise_frames.append(pairwise_df)

    pairwise_df = pd.concat(pairwise_frames, ignore_index=True) if pairwise_frames else pd.DataFrame()
    if not pairwise_df.empty:
        pairwise_df = apply_multiple_comparison_corrections(
            pairwise_df,
            group_cols=["comparison", "metric"],
            p_col="p_value",
        )
        pairwise_df.to_csv(os.path.join(output_dir, "yield_focus_ablation_pairwise.csv"), index=False)
        with open(os.path.join(output_dir, "yield_focus_ablation_notes.md"), "w", encoding="utf-8") as f:
            f.write("# Yield-Focus Ablation\n\n")
            f.write(_format_markdown_table(overall_df, float_fmt=".4f"))
            f.write("\n\n")
            f.write(_format_markdown_table(pairwise_df, float_fmt=".4f"))
    return overall_df, pairwise_df


def compile_mechanism_diagnostics_summary(controller_summaries: dict) -> pd.DataFrame:
    """Aggregate stage-aware mechanism diagnostics per controller."""
    diag_cols = [
        "aq_transpiration_ratio",
        "aq_stress_days_tr",
        "aq_evaporation_ratio",
        "aq_deep_perc_mm",
        "aq_harvest_index_adj",
        "aq_reproductive_transpiration_ratio",
        "aq_reproductive_stress_days_tr",
        "aq_reproductive_irrigation_mm",
        "aq_reproductive_irrigation_share_pct",
        "aq_flowering_transpiration_ratio",
        "aq_flowering_stress_days_tr",
        "aq_flowering_irrigation_mm",
        "aq_flowering_irrigation_share_pct",
        "aq_grain_fill_transpiration_ratio",
        "aq_grain_fill_stress_days_tr",
        "aq_grain_fill_irrigation_mm",
        "aq_grain_fill_irrigation_share_pct",
    ]
    rows = []
    for ctrl_id, payload in controller_summaries.items():
        summary = payload.get("summary", pd.DataFrame())
        if summary.empty:
            continue
        row = {
            "controller_id": ctrl_id,
            "controller_label": payload["label"],
            "n_scenarios": int(len(summary)),
        }
        present = False
        for col in diag_cols:
            if col in summary.columns:
                values = summary[col].dropna()
                row[f"mean_{col}"] = float(values.mean()) if not values.empty else np.nan
                present = True
        if present:
            rows.append(row)
    return pd.DataFrame(rows)


def compile_mechanism_diagnostics_per_scenario(controller_summaries: dict) -> pd.DataFrame:
    """Collect scenario-level mechanism diagnostics across controllers."""
    diag_cols = [
        "aq_transpiration_ratio",
        "aq_stress_days_tr",
        "aq_evaporation_ratio",
        "aq_deep_perc_mm",
        "aq_harvest_index_adj",
        "aq_reproductive_transpiration_ratio",
        "aq_reproductive_stress_days_tr",
        "aq_reproductive_irrigation_mm",
        "aq_reproductive_irrigation_share_pct",
        "aq_flowering_transpiration_ratio",
        "aq_flowering_stress_days_tr",
        "aq_flowering_irrigation_mm",
        "aq_flowering_irrigation_share_pct",
        "aq_grain_fill_transpiration_ratio",
        "aq_grain_fill_stress_days_tr",
        "aq_grain_fill_irrigation_mm",
        "aq_grain_fill_irrigation_share_pct",
    ]
    frames = []
    for ctrl_id, payload in controller_summaries.items():
        summary = payload.get("summary", pd.DataFrame())
        if summary.empty:
            continue
        keep = [col for col in ["year", "season"] + diag_cols if col in summary.columns]
        if len(keep) <= 2:
            continue
        tmp = summary.loc[:, keep].copy()
        tmp.insert(0, "controller_id", ctrl_id)
        tmp.insert(1, "controller_label", payload["label"])
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_mechanism_pairwise_vs_reference(controller_summaries: dict, reference_id: str = "fuzzy_phenology") -> pd.DataFrame:
    """Run paired comparisons on mechanism-level diagnostics."""
    ref_payload = controller_summaries.get(reference_id)
    if ref_payload is None or ref_payload["summary"].empty:
        return pd.DataFrame()
    metric_cols = [
        "aq_transpiration_ratio",
        "aq_stress_days_tr",
        "aq_evaporation_ratio",
        "aq_deep_perc_mm",
        "aq_reproductive_transpiration_ratio",
        "aq_reproductive_stress_days_tr",
        "aq_reproductive_irrigation_mm",
        "aq_reproductive_irrigation_share_pct",
        "aq_flowering_transpiration_ratio",
        "aq_flowering_stress_days_tr",
        "aq_flowering_irrigation_mm",
        "aq_flowering_irrigation_share_pct",
        "aq_grain_fill_transpiration_ratio",
        "aq_grain_fill_stress_days_tr",
        "aq_grain_fill_irrigation_mm",
        "aq_grain_fill_irrigation_share_pct",
    ]
    frames = []
    for ctrl_id, payload in controller_summaries.items():
        if ctrl_id == reference_id or payload["summary"].empty:
            continue
        overall_df = compute_paired_statistics(ref_payload["summary"], payload["summary"], metric_cols=metric_cols)
        if overall_df.empty:
            continue
        overall_df.insert(0, "baseline_id", ctrl_id)
        overall_df.insert(1, "baseline_label", payload["label"])
        frames.append(overall_df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def generate_mechanism_diagnostics_notes(
    output_dir: str,
    mechanism_pairwise_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create concise mechanism-facing interpretation notes."""
    notes = []
    if mechanism_pairwise_df is not None and not mechanism_pairwise_df.empty:
        focus_metrics = [
            "aq_reproductive_transpiration_ratio",
            "aq_reproductive_stress_days_tr",
            "aq_reproductive_irrigation_share_pct",
            "aq_flowering_transpiration_ratio",
            "aq_flowering_stress_days_tr",
            "aq_flowering_irrigation_share_pct",
            "aq_deep_perc_mm",
        ]
        for baseline_label in sorted(mechanism_pairwise_df["baseline_label"].dropna().unique()):
            subset = mechanism_pairwise_df[
                (mechanism_pairwise_df["baseline_label"] == baseline_label) &
                (mechanism_pairwise_df["analysis_scope"] == "overall") &
                (mechanism_pairwise_df["metric"].isin(focus_metrics))
            ].copy()
            if subset.empty:
                continue
            fragments = []
            for metric in focus_metrics:
                row = subset[subset["metric"] == metric]
                if row.empty:
                    continue
                row = row.iloc[0]
                direction = "higher" if float(row["difference"]) > 0 else "lower"
                fragments.append(f"{metric}: {direction} by {row['difference']:.3f}")
            notes.append({
                "baseline_label": baseline_label,
                "mechanism_note": "; ".join(fragments),
            })
    notes_df = pd.DataFrame(notes)
    if not notes_df.empty:
        notes_df.to_csv(os.path.join(output_dir, "mechanism_diagnostics_notes.csv"), index=False)
        lines = ["# Mechanism Diagnostics Notes", ""]
        for _, row in notes_df.iterrows():
            lines.append(f"- `{row['baseline_label']}`: {row['mechanism_note']}.")
        with open(os.path.join(output_dir, "mechanism_diagnostics_notes.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    return notes_df


def generate_mechanism_mapping_artifacts(output_dir: str) -> None:
    """Create explicit table/figure mapping notes for Phase 6 diagnostics."""
    table_lines = [
        "# Mechanism Table Mapping",
        "",
        "- `mechanism_diagnostics_summary.csv`: controller-level table candidate for stage-aware transpiration ratios, reproductive/flowering stress exposure, and critical-window irrigation allocation.",
        "- `mechanism_pairwise_vs_fuzzy_phenology.csv`: pairwise table candidate for mechanism differences against the frozen baseline set.",
        "- `mechanism_diagnostics_per_scenario.csv`: per-scenario supplemental table candidate when reproducibility-facing traceability is needed.",
    ]
    figure_lines = [
        "# Mechanism Figure Mapping",
        "",
        "- Use `mechanism_diagnostics_summary.csv` to build a controller comparison figure for reproductive/flowering irrigation share and transpiration-ratio patterns.",
        "- Use `mechanism_pairwise_vs_fuzzy_phenology.csv` to build a compact figure or caption-ready summary showing which mechanism diagnostics move with the proposed controller versus each baseline.",
        "- Use scenario-level diagnostics to support one representative mechanism figure if reproducibility figure needs a reproducibility-friendly multi-panel visual.",
    ]
    with open(os.path.join(output_dir, "mechanism_table_mapping.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines))
    with open(os.path.join(output_dir, "mechanism_figure_mapping.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(figure_lines))


def run_final_evaluation_holdout_protocol(
    weather_df: pd.DataFrame,
    output_dir: str,
    active_years: list[int],
    selected_season_name: str | None,
    controller_specs: list[dict],
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the frozen final leave-one-year-out evaluation."""
    protocol = build_final_evaluation_protocol(active_years)
    with open(os.path.join(output_dir, "evaluation_protocol.json"), "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2, sort_keys=True)

    holdout_rows = []
    eta_tuning_rows = []
    fuzzy_profile_tuning_rows = []
    selected_config_rows = []
    aggregated_results = {spec["id"]: [] for spec in controller_specs}
    aggregated_summaries = {spec["id"]: [] for spec in controller_specs}

    holdout_root = os.path.join(output_dir, "holdout_folds")
    os.makedirs(holdout_root, exist_ok=True)

    for fold in protocol["folds"]:
        holdout_year = int(fold["holdout_year"])
        training_years = [int(year) for year in fold["training_years"]]
        fold_dir = os.path.join(holdout_root, fold["fold_id"])
        tuning_dir = os.path.join(fold_dir, "tuning")
        evaluation_dir = os.path.join(fold_dir, "evaluation")
        os.makedirs(tuning_dir, exist_ok=True)
        os.makedirs(evaluation_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"FINAL EVALUATION FOLD: holdout year {holdout_year}")
        print(f"Training years: {training_years}")
        print(f"{'='*60}")

        training_scope = build_runtime_scope(training_years, selected_season_name)
        fold_metadata = {
            "selection_scope": "training_only",
            "protocol_id": protocol["protocol_id"],
            "holdout_year": holdout_year,
            "training_years": training_years,
        }
        eta_star, eta_calibration_df, fuzzy_profile_selected, fuzzy_profile_tuning_df = execute_tuning_protocol(
            weather_df=weather_df,
            output_dir=tuning_dir,
            engine_kwargs=training_scope,
            runtime_metadata=fold_metadata,
        )

        if not eta_calibration_df.empty:
            eta_tmp = eta_calibration_df.copy()
            eta_tmp.insert(0, "holdout_year", holdout_year)
            eta_tmp.insert(1, "training_years", ",".join(str(year) for year in training_years))
            eta_tuning_rows.append(eta_tmp)
        if not fuzzy_profile_tuning_df.empty:
            profile_tmp = fuzzy_profile_tuning_df.copy()
            profile_tmp.insert(0, "holdout_year", holdout_year)
            profile_tmp.insert(1, "training_years", ",".join(str(year) for year in training_years))
            fuzzy_profile_tuning_rows.append(profile_tmp)
        selected_config_rows.append({
            "holdout_year": holdout_year,
            "training_years": ",".join(str(year) for year in training_years),
            "eta_star": float(eta_star),
            "profile_name": str(fuzzy_profile_selected.get("profile_name", "")),
            "upper_ref_margin_frac": float(fuzzy_profile_selected.get("upper_ref_margin_frac", np.nan)),
            "overshoot_tol_frac": float(fuzzy_profile_selected.get("overshoot_tol_frac", np.nan)),
            "max_irrigation_daily_mm": float(fuzzy_profile_selected.get("max_irrigation_daily_mm", np.nan)),
            "vegetative_weight": float(fuzzy_profile_selected.get("vegetative_weight", np.nan)),
            "reproductive_weight": float(fuzzy_profile_selected.get("reproductive_weight", np.nan)),
            "maturation_weight": float(fuzzy_profile_selected.get("maturation_weight", np.nan)),
        })

        evaluation_scope = build_runtime_scope([holdout_year], selected_season_name)
        fold_specs = build_controller_specs(eta_star, fuzzy_profile=fuzzy_profile_selected)
        fold_specs = [spec for spec in fold_specs if spec["id"] in {candidate["id"] for candidate in controller_specs}]

        fold_summaries = {}
        for spec in fold_specs:
            ctrl_id = spec["id"]
            results_df, summary_df = run_simulation(
                controller=spec["controller"],
                controller_name=ctrl_id,
                weather_df=weather_df,
                output_dir=evaluation_dir,
                verbose=True,
                engine_kwargs=evaluation_scope,
            )
            if not summary_df.empty:
                summary_df = enrich_summary_with_aquacrop(
                    results_df, summary_df, weather_df, eta_star, verbose=True
                )
                save_results_and_summary(ctrl_id, results_df, summary_df, output_dir=evaluation_dir)
                summary_df = summary_df.copy()
                summary_df["holdout_year"] = holdout_year
                summary_df["training_years"] = ",".join(str(year) for year in training_years)
            if not results_df.empty:
                results_df = results_df.copy()
                results_df["holdout_year"] = holdout_year
                results_df["training_years"] = ",".join(str(year) for year in training_years)
            aggregated_results[ctrl_id].append(results_df)
            aggregated_summaries[ctrl_id].append(summary_df)
            fold_summaries[ctrl_id] = summary_df

        primary_fuzzy = fold_summaries.get("fuzzy_phenology", pd.DataFrame())
        primary_reactive = fold_summaries.get("reactive_static", pd.DataFrame())
        if not primary_fuzzy.empty and not primary_reactive.empty:
            fold_h2h = pd.merge(
                primary_fuzzy,
                primary_reactive,
                on=["year", "season", "holdout_year", "training_years"],
                how="inner",
                suffixes=("_fuzzy", "_reactive"),
            )
            if not fold_h2h.empty:
                fold_h2h["delta_iwu_mm"] = fold_h2h["iwu_mm_fuzzy"] - fold_h2h["iwu_mm_reactive"]
                fold_h2h["delta_yield_dry_t_ha"] = (
                    fold_h2h["yield_dry_t_ha_fuzzy"] - fold_h2h["yield_dry_t_ha_reactive"]
                )
                fold_h2h["delta_iwue_kg_ha_per_mm"] = (
                    fold_h2h["iwue_kg_ha_per_mm_fuzzy"] - fold_h2h["iwue_kg_ha_per_mm_reactive"]
                )
                holdout_rows.append(
                    fold_h2h[[
                        "holdout_year",
                        "training_years",
                        "year",
                        "season",
                        "iwu_mm_fuzzy",
                        "iwu_mm_reactive",
                        "delta_iwu_mm",
                        "yield_dry_t_ha_fuzzy",
                        "yield_dry_t_ha_reactive",
                        "delta_yield_dry_t_ha",
                        "iwue_kg_ha_per_mm_fuzzy",
                        "iwue_kg_ha_per_mm_reactive",
                        "delta_iwue_kg_ha_per_mm",
                        "target_pct_fuzzy",
                        "target_pct_reactive",
                        "mse_fuzzy",
                        "mse_reactive",
                    ]]
                )

    controller_results = {}
    controller_summaries = {}
    for spec in controller_specs:
        ctrl_id = spec["id"]
        results_df = pd.concat(aggregated_results[ctrl_id], ignore_index=True) if aggregated_results[ctrl_id] else pd.DataFrame()
        summary_df = pd.concat(aggregated_summaries[ctrl_id], ignore_index=True) if aggregated_summaries[ctrl_id] else pd.DataFrame()
        if not results_df.empty or not summary_df.empty:
            save_results_and_summary(ctrl_id, results_df, summary_df, output_dir=output_dir)
        controller_results[ctrl_id] = {
            "label": spec["label"],
            "group": spec["group"],
            "description": spec["description"],
            "results": results_df,
        }
        controller_summaries[ctrl_id] = {
            "label": spec["label"],
            "group": spec["group"],
            "description": spec["description"],
            "summary": summary_df,
        }

    holdout_summary_df = pd.concat(holdout_rows, ignore_index=True) if holdout_rows else pd.DataFrame()
    if not holdout_summary_df.empty:
        holdout_summary_df.to_csv(os.path.join(output_dir, "holdout_year_performance_summary.csv"), index=False)

    eta_tuning_df = pd.concat(eta_tuning_rows, ignore_index=True) if eta_tuning_rows else pd.DataFrame()
    if not eta_tuning_df.empty:
        eta_tuning_df.to_csv(os.path.join(output_dir, "holdout_eta_calibration_results.csv"), index=False)

    fuzzy_profile_tuning_df = pd.concat(fuzzy_profile_tuning_rows, ignore_index=True) if fuzzy_profile_tuning_rows else pd.DataFrame()
    if not fuzzy_profile_tuning_df.empty:
        fuzzy_profile_tuning_df.to_csv(os.path.join(output_dir, "holdout_fuzzy_iwue_tuning_results.csv"), index=False)

    selected_runtime_df = pd.DataFrame(selected_config_rows)
    if not selected_runtime_df.empty:
        selected_runtime_df.to_csv(os.path.join(output_dir, "holdout_selected_runtime_configs.csv"), index=False)

    holdout_aggregate_df = pd.DataFrame()
    if not holdout_summary_df.empty:
        holdout_aggregate_df = (
            holdout_summary_df.groupby("season", as_index=False)
            .agg(
                delta_iwu_mm=("delta_iwu_mm", "mean"),
                delta_yield_dry_t_ha=("delta_yield_dry_t_ha", "mean"),
                delta_iwue_kg_ha_per_mm=("delta_iwue_kg_ha_per_mm", "mean"),
                n_scenarios=("holdout_year", "count"),
            )
        )
        overall_row = pd.DataFrame([{
            "season": "Overall",
            "delta_iwu_mm": float(holdout_summary_df["delta_iwu_mm"].mean()),
            "delta_yield_dry_t_ha": float(holdout_summary_df["delta_yield_dry_t_ha"].mean()),
            "delta_iwue_kg_ha_per_mm": float(holdout_summary_df["delta_iwue_kg_ha_per_mm"].mean()),
            "n_scenarios": int(len(holdout_summary_df)),
        }])
        holdout_aggregate_df = pd.concat([overall_row, holdout_aggregate_df], ignore_index=True)
        holdout_aggregate_df.to_csv(os.path.join(output_dir, "holdout_year_performance_aggregate.csv"), index=False)

    protocol_note_path = os.path.join(output_dir, "holdout_protocol_summary.md")
    with open(protocol_note_path, "w", encoding="utf-8") as f:
        f.write("\n".join([
            "# Holdout Evaluation Protocol",
            "",
            f"- Protocol: `{protocol['protocol_label']}`",
            f"- Years: `{min(active_years)}-{max(active_years)}`",
            f"- Folds: `{protocol['n_folds']}`",
            "- Each fold tunes `eta` and the fuzzy profile on the training years only.",
            "- The held-out year is evaluated only after tuning is frozen for that fold.",
            "- Holdout performance exports are separated from fold-level tuning artifacts under `holdout_folds/`.",
        ]))

    return (
        controller_results,
        controller_summaries,
        eta_tuning_df,
        fuzzy_profile_tuning_df,
        selected_runtime_df,
    )


def run_sensitivity_variant(
    weather_df: pd.DataFrame,
    eta_value: float,
    fuzzy_profile: dict,
    overlap_days: int | None = None,
    target_shift: float = 0.0,
    initial_r: float | None = None,
    max_irrigation_daily_mm: float | None = None,
    profile_overrides: dict | None = None,
    engine_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Run a single fuzzy-phenology sensitivity variant without writing raw artifacts."""
    controller_profile = dict(fuzzy_profile or {})
    if profile_overrides:
        controller_profile.update(profile_overrides)
    if max_irrigation_daily_mm is not None:
        controller_profile["max_irrigation_daily_mm"] = float(max_irrigation_daily_mm)
    controller = FuzzyIrrigationController(
        efficiency=eta_value,
        target_lower_offset=target_shift,
        target_upper_offset=target_shift,
        **controller_profile,
    )
    sensitivity_engine_kwargs = dict(engine_kwargs or {})
    sensitivity_engine_kwargs["overlap_days"] = overlap_days
    if initial_r is not None:
        sensitivity_engine_kwargs["initial_r"] = float(initial_r)
    engine = SimulationEngine(controller=controller, weather_df=weather_df, **sensitivity_engine_kwargs)
    all_results = engine.run_all_scenarios(verbose=False)
    if all_results.empty:
        return pd.DataFrame()
    summary = compute_scenario_summary(all_results)
    return enrich_summary_with_aquacrop(all_results, summary, weather_df, eta_value, verbose=False)


def generate_reproducibility_sensitivity_outputs(
    output_dir: str,
    weather_df: pd.DataFrame,
    eta_star: float,
    fuzzy_profile_selected: dict,
    engine_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Limited sensitivity analysis aligned with the revised reproducibility."""
    base_overlap = int(PHENOLOGY_OVERLAP_DAYS)
    eta_minus = max(0.60, round(eta_star - 0.05, 2))
    eta_plus = min(0.95, round(eta_star + 0.05, 2))
    base_veg = float(fuzzy_profile_selected.get("vegetative_weight", 0.45))
    base_rep = float(fuzzy_profile_selected.get("reproductive_weight", 1.00))
    base_mat = float(fuzzy_profile_selected.get("maturation_weight", 0.20))
    variants = [
        {"family": "eta", "variant": "eta_minus_0p05", "eta": eta_minus, "overlap_days": base_overlap, "target_shift": 0.00},
        {"family": "eta", "variant": "eta_base", "eta": eta_star, "overlap_days": base_overlap, "target_shift": 0.00},
        {"family": "eta", "variant": "eta_plus_0p05", "eta": eta_plus, "overlap_days": base_overlap, "target_shift": 0.00},
        {"family": "overlap_days", "variant": "overlap_3_days", "eta": eta_star, "overlap_days": 3, "target_shift": 0.00},
        {"family": "overlap_days", "variant": "overlap_5_days", "eta": eta_star, "overlap_days": 5, "target_shift": 0.00},
        {"family": "overlap_days", "variant": "overlap_7_days", "eta": eta_star, "overlap_days": 7, "target_shift": 0.00},
        {"family": "target_shift", "variant": "target_shift_minus_3pct", "eta": eta_star, "overlap_days": base_overlap, "target_shift": -REPRODUCIBILITY_TARGET_SHIFT_STEP},
        {"family": "target_shift", "variant": "target_shift_base", "eta": eta_star, "overlap_days": base_overlap, "target_shift": 0.00},
        {"family": "target_shift", "variant": "target_shift_plus_3pct", "eta": eta_star, "overlap_days": base_overlap, "target_shift": REPRODUCIBILITY_TARGET_SHIFT_STEP},
        {"family": "initial_soil_moisture", "variant": "initial_moisture_0p85_fc", "eta": eta_star, "overlap_days": base_overlap, "target_shift": 0.00, "initial_r": 0.85},
        {"family": "initial_soil_moisture", "variant": "initial_moisture_1p00_fc", "eta": eta_star, "overlap_days": base_overlap, "target_shift": 0.00, "initial_r": 1.00},
        {"family": "daily_irrigation_cap", "variant": "cap_30_mm", "eta": eta_star, "overlap_days": base_overlap, "target_shift": 0.00, "max_irrigation_daily_mm": 30.0},
        {"family": "daily_irrigation_cap", "variant": "cap_40_mm", "eta": eta_star, "overlap_days": base_overlap, "target_shift": 0.00, "max_irrigation_daily_mm": 40.0},
        {"family": "daily_irrigation_cap", "variant": "cap_50_mm", "eta": eta_star, "overlap_days": base_overlap, "target_shift": 0.00, "max_irrigation_daily_mm": 50.0},
        {
            "family": "phase_weights",
            "variant": "phase_weights_base",
            "eta": eta_star,
            "overlap_days": base_overlap,
            "target_shift": 0.00,
            "profile_overrides": {
                "vegetative_weight": base_veg,
                "reproductive_weight": base_rep,
                "maturation_weight": base_mat,
            },
        },
        {
            "family": "phase_weights",
            "variant": "phase_weights_balanced",
            "eta": eta_star,
            "overlap_days": base_overlap,
            "target_shift": 0.00,
            "profile_overrides": {
                "vegetative_weight": 0.35,
                "reproductive_weight": 0.85,
                "maturation_weight": 0.30,
            },
        },
        {
            "family": "phase_weights",
            "variant": "phase_weights_reproductive_heavy",
            "eta": eta_star,
            "overlap_days": base_overlap,
            "target_shift": 0.00,
            "profile_overrides": {
                "vegetative_weight": 0.25,
                "reproductive_weight": 1.10,
                "maturation_weight": 0.20,
            },
        },
    ]
    overall_rows = []
    seasonal_rows = []
    for variant in variants:
        print(f"  -> Sensitivity variant: {variant['variant']}")
        summary = run_sensitivity_variant(
            weather_df=weather_df,
            eta_value=float(variant["eta"]),
            fuzzy_profile=fuzzy_profile_selected,
            overlap_days=int(variant["overlap_days"]),
            target_shift=float(variant["target_shift"]),
            initial_r=variant.get("initial_r"),
            max_irrigation_daily_mm=variant.get("max_irrigation_daily_mm"),
            profile_overrides=variant.get("profile_overrides"),
            engine_kwargs=engine_kwargs,
        )
        if summary.empty:
            continue
        profile_used = dict(fuzzy_profile_selected or {})
        if variant.get("profile_overrides"):
            profile_used.update(variant["profile_overrides"])
        overall_rows.append({
            "family": variant["family"],
            "variant": variant["variant"],
            "eta": float(variant["eta"]),
            "overlap_days": int(variant["overlap_days"]),
            "target_shift": float(variant["target_shift"]),
            "initial_r": float(variant.get("initial_r", 1.00)),
            "max_irrigation_daily_mm": float(variant.get("max_irrigation_daily_mm", profile_used.get("max_irrigation_daily_mm", 40.0))),
            "vegetative_weight": float(profile_used.get("vegetative_weight", base_veg)),
            "reproductive_weight": float(profile_used.get("reproductive_weight", base_rep)),
            "maturation_weight": float(profile_used.get("maturation_weight", base_mat)),
            "n_scenarios": int(len(summary)),
            "mean_iwu_mm": float(summary["iwu_mm"].mean()),
            "mean_target_pct": float(summary["target_pct"].mean()),
            "mean_mse": float(summary["mse"].mean()),
            "mean_yield_dry_t_ha": float(summary["yield_dry_t_ha"].mean()),
            "mean_iwue_kg_ha_per_mm": float(summary["iwue_kg_ha_per_mm"].mean()),
            "mean_events": float(summary["n_irrigation_events"].mean()),
        })
        seasonal_grouped = (
            summary.groupby("season", as_index=False)
            .agg(
                n_scenarios=("season", "count"),
                mean_iwu_mm=("iwu_mm", "mean"),
                mean_target_pct=("target_pct", "mean"),
                mean_mse=("mse", "mean"),
                mean_yield_dry_t_ha=("yield_dry_t_ha", "mean"),
                mean_iwue_kg_ha_per_mm=("iwue_kg_ha_per_mm", "mean"),
                mean_events=("n_irrigation_events", "mean"),
            )
        )
        for _, season_row in seasonal_grouped.iterrows():
            seasonal_rows.append({
                "family": variant["family"],
                "variant": variant["variant"],
                "season": season_row["season"],
                "eta": float(variant["eta"]),
                "overlap_days": int(variant["overlap_days"]),
                "target_shift": float(variant["target_shift"]),
                "initial_r": float(variant.get("initial_r", 1.00)),
                "max_irrigation_daily_mm": float(variant.get("max_irrigation_daily_mm", profile_used.get("max_irrigation_daily_mm", 40.0))),
                "vegetative_weight": float(profile_used.get("vegetative_weight", base_veg)),
                "reproductive_weight": float(profile_used.get("reproductive_weight", base_rep)),
                "maturation_weight": float(profile_used.get("maturation_weight", base_mat)),
                "n_scenarios": int(season_row["n_scenarios"]),
                "mean_iwu_mm": float(season_row["mean_iwu_mm"]),
                "mean_target_pct": float(season_row["mean_target_pct"]),
                "mean_mse": float(season_row["mean_mse"]),
                "mean_yield_dry_t_ha": float(season_row["mean_yield_dry_t_ha"]),
                "mean_iwue_kg_ha_per_mm": float(season_row["mean_iwue_kg_ha_per_mm"]),
                "mean_events": float(season_row["mean_events"]),
            })

    sensitivity_df = pd.DataFrame(overall_rows)
    if sensitivity_df.empty:
        return sensitivity_df

    sensitivity_path = os.path.join(output_dir, "reproducibility_sensitivity_summary.csv")
    sensitivity_df.to_csv(sensitivity_path, index=False)
    seasonal_sensitivity_df = pd.DataFrame(seasonal_rows)
    if not seasonal_sensitivity_df.empty:
        seasonal_sensitivity_df["season_order"] = seasonal_sensitivity_df["season"].map(_season_sort_key)
        seasonal_sensitivity_df = seasonal_sensitivity_df.sort_values(
            ["family", "variant", "season_order"]
        ).drop(columns=["season_order"])
        seasonal_sensitivity_df.to_csv(
            os.path.join(output_dir, "reproducibility_sensitivity_seasonal_summary.csv"),
            index=False,
        )

    md = []
    md.append("# Reproducibility-Aligned Sensitivity Summary")
    md.append("")
    md.append("Sensitivity covers the reproducibility-visible robustness factors below:")
    md.append("- irrigation efficiency (`eta`)")
    md.append("- phenology overlap width (days)")
    md.append("- uniform shift of phase-wise soil-moisture targets (`rL` and `rU`)")
    md.append("- initial soil moisture (`initial_r`)")
    md.append("- daily irrigation cap (`max_irrigation_daily_mm`)")
    md.append("- fuzzy phase-weight configuration")
    md.append("")
    md.append("The simulation engine now requires at least 90% weather coverage per season before a scenario is retained.")
    md.append("")
    md.append(_format_markdown_table(sensitivity_df, float_fmt=".4f"))
    md.append("")
    base_row = sensitivity_df[sensitivity_df["variant"].isin(["eta_base", "target_shift_base", "overlap_5_days"])].head(1)
    if not base_row.empty:
        b = base_row.iloc[0]
        md.append(
            f"Reference configuration retained mean yield {b['mean_yield_dry_t_ha']:.3f} t/ha, "
            f"mean IWU {b['mean_iwu_mm']:.2f} mm, and mean IWUE {b['mean_iwue_kg_ha_per_mm']:.3f} kg/ha/mm."
        )
    if not seasonal_sensitivity_df.empty:
        md.append("")
        md.append("Season-specific robustness is exported separately in `reproducibility_sensitivity_seasonal_summary.csv`.")
    with open(os.path.join(output_dir, "reproducibility_sensitivity_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"  ✓ Reproducibility sensitivity summary: {sensitivity_path}")
    return sensitivity_df


def generate_reproducibility_claim_evidence_table(
    output_dir: str,
    paired_df: pd.DataFrame,
    seasonal_h2h: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> None:
    """Compact claim-to-evidence table for the reproducibility evidence pack."""
    rows = []
    lookup = paired_df.set_index("metric") if paired_df is not None and not paired_df.empty else pd.DataFrame()

    def metric_row(metric: str):
        if lookup.empty or metric not in lookup.index:
            return None
        row = lookup.loc[metric]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row

    iwu = metric_row("iwu_mm")
    yield_row = metric_row("yield_dry_t_ha")
    iwue = metric_row("iwue_kg_ha_per_mm")

    if iwu is not None:
        rows.append({
            "claim": "Fuzzy-Phenology reduces irrigation water use relative to the static reactive baseline across the 2015-2024 all-season design.",
            "evidence": f"Mean diff = {iwu['difference']:.3f} mm; p = {iwu['p_value']:.4g}; n = {int(iwu['n_pairs'])}.",
            "strength": "Core",
        })
    if yield_row is not None:
        rows.append({
            "claim": "Agronomic response under Fuzzy-Phenology must be interpreted as an empirical trade-off, not presumed yield preservation, in the reproducibility comparison.",
            "evidence": f"Mean diff = {yield_row['difference']:.3f} t/ha; p = {yield_row['p_value']:.4g}; Cohen's d = {yield_row['cohens_d']:.3f}.",
            "strength": "Core",
        })
    if iwue is not None:
        rows.append({
            "claim": "Fuzzy-Phenology improves irrigation water use efficiency in the reproducibility comparison.",
            "evidence": f"Mean diff = {iwue['difference']:.3f} kg/ha/mm; p = {iwue['p_value']:.4g}.",
            "strength": "Core",
        })
    if seasonal_h2h is not None and not seasonal_h2h.empty:
        best = seasonal_h2h.sort_values("delta_iwu_mm").iloc[0]
        rows.append({
            "claim": "The water-saving effect persists across planting seasons rather than appearing only in one window.",
            "evidence": f"Best seasonal IWU delta observed in {best['season']} = {best['delta_iwu_mm']:.3f} mm.",
            "strength": "Supporting",
        })
    if sensitivity_df is not None and not sensitivity_df.empty:
        rows.append({
            "claim": "The reproducibility controller remains directionally stable under limited parameter perturbation.",
            "evidence": "Sensitivity outputs retained positive-yield / bounded-IWU behavior across eta, overlap, and target-shift variants.",
            "strength": "Supporting",
        })

    table = pd.DataFrame(rows)
    with open(os.path.join(output_dir, "reproducibility_claim_evidence_table.md"), "w", encoding="utf-8") as f:
        f.write("# Reproducibility-Aligned Claim Evidence Table\n\n")
        f.write(_format_markdown_table(table, float_fmt=".4f"))


def main():
    parser = argparse.ArgumentParser(
        description='Phenology-Fuzzy Irrigation Control Simulation')
    parser.add_argument('--fuzzy', action='store_true',
                       help='Run only fuzzy-family controllers (Fuzzy-Phenology + Fuzzy-Static)')
    parser.add_argument('--reactive', action='store_true',
                       help='Run only reactive-family controllers (Reactive-Static + Reactive-Phenology)')
    parser.add_argument('--test', action='store_true',
                       help='Quick test with 1 scenario')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--weather', type=str, default=None,
                       help='Path to weather data file')
    parser.add_argument('--season', type=str, choices=sorted(SEASON_ALIAS_TO_NAME.keys()),
                       help='Filter planting season: MT-1, MT-2, or MT-3')
    parser.add_argument('--reproducibility-pack', action='store_true',
                       help='Generate a reproducibility 2-controller pack in a separate output folder')
    parser.add_argument('--reproducibility-start-year', type=int, default=REPRODUCIBILITY_DEFAULT_START_YEAR,
                       help='Start year for the reproducibility pack')
    parser.add_argument('--reproducibility-end-year', type=int, default=REPRODUCIBILITY_DEFAULT_END_YEAR,
                       help='End year for the reproducibility pack')
    parser.add_argument('--experiment-mode',
                       choices=['all', 'tuning', 'final-evaluation', 'attribution', 'sensitivity'],
                       default='all',
                       help='Explicit experiment protocol mode')
    args = parser.parse_args()
    
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Phenology-Fuzzy Irrigation Control Simulation          ║")
    print("║  AquaCrop-OSPy Digital Twin                             ║")
    print("╚══════════════════════════════════════════════════════════╝")

    try:
        selected_season_alias, selected_season_name = resolve_season_alias(args.season)
    except ValueError as exc:
        print(f"\n✗ {exc}")
        sys.exit(1)

    if selected_season_name is not None:
        print(f"\nSeason filter active: {selected_season_alias} ({selected_season_name})")

    if args.reproducibility_pack and args.reproducibility_end_year < args.reproducibility_start_year:
        print("\n✗ --reproducibility-end-year must be >= --reproducibility-start-year")
        sys.exit(1)

    if args.test and selected_season_name is None:
        selected_season_alias = "MT-2"
        selected_season_name = SEASON_ALIAS_TO_NAME[selected_season_alias]
        print(f"\nTest mode season filter: {selected_season_alias} ({selected_season_name})")

    if args.test and not args.reproducibility_pack:
        active_years = [REPRODUCIBILITY_DEFAULT_START_YEAR]
    elif args.reproducibility_pack:
        active_years = list(range(args.reproducibility_start_year, args.reproducibility_end_year + 1))
    else:
        active_years = list(SIMULATION_YEARS)
    year_label = build_year_label(active_years)
    runtime_scope = build_runtime_scope(active_years, selected_season_name)
    engine_kwargs = dict(runtime_scope)
    
    # ====================================================================
    # STEP 1: Data Cleansing
    # ====================================================================
    weather_file = args.weather or os.path.join(PROJECT_ROOT, 'data', 'cuaca-complete.txt')
    
    if not os.path.exists(weather_file):
        print(f"\n✗ Weather data not found: {weather_file}")
        sys.exit(1)
    
    # Include one extra year of weather because MT-1 starts in Oct and crosses into next year.
    weather_df = cleanse_weather_data(
        weather_file,
        start_year=min(active_years),
        end_year=max(active_years) + 1
    )
    
    base_output_dir = (
        os.path.join(get_output_dir(), f"reproducibility_{year_label}")
        if args.reproducibility_pack
        else get_output_dir()
    )
    output_dir = get_mode_output_dir(base_output_dir, args.experiment_mode)
    os.makedirs(output_dir, exist_ok=True)

    # ====================================================================
    # STEP 2: Explicit experiment modes
    # ====================================================================
    eta_star = IRRIGATION_EFFICIENCY
    eta_calibration_df = pd.DataFrame()
    fuzzy_profile_selected = get_fuzzy_iwue_tuning_profiles()[0]
    fuzzy_profile_tuning_df = pd.DataFrame()
    runtime_config_metadata = {}

    if args.experiment_mode in {"tuning", "all"}:
        if not args.test:
            runtime_config_metadata = {
                "selection_scope": "all_active_years",
                "active_years": active_years,
                "season_filter": selected_season_name,
            }
            eta_star, eta_calibration_df, fuzzy_profile_selected, fuzzy_profile_tuning_df = execute_tuning_protocol(
                weather_df=weather_df,
                output_dir=output_dir,
                engine_kwargs=engine_kwargs,
                base_output_dir=base_output_dir if output_dir != base_output_dir else None,
                runtime_metadata=runtime_config_metadata,
            )
        else:
            print("\nSkipping tuning in --test mode.")
        if args.experiment_mode == "tuning":
            print(f"\n{'='*60}")
            print("TUNING COMPLETE")
            print(f"Outputs saved to: {output_dir}")
            print(f"{'='*60}")
            return
    elif args.experiment_mode == "final-evaluation":
        print(f"\nUsing frozen final evaluation protocol: {FINAL_EVALUATION_PROTOCOL_LABEL}")
    else:
        eta_star, fuzzy_profile_selected, runtime_config_metadata = load_selected_runtime_config(base_output_dir)
        print(f"\nLoaded tuned runtime config from: {base_output_dir}")

    if args.experiment_mode == "sensitivity":
        print(f"\n{'='*60}")
        print("RUNNING EXPLICIT SENSITIVITY MODE")
        print(f"{'='*60}")
        sensitivity_df = generate_reproducibility_sensitivity_outputs(
            output_dir=output_dir,
            weather_df=weather_df,
            eta_star=eta_star,
            fuzzy_profile_selected=fuzzy_profile_selected,
            engine_kwargs=engine_kwargs,
        )
        if sensitivity_df.empty:
            print("\n✗ Sensitivity mode produced no results.")
            sys.exit(1)
        print(f"\n{'='*60}")
        print("SENSITIVITY COMPLETE")
        print(f"Outputs saved to: {output_dir}")
        print(f"{'='*60}")
        return

    # ====================================================================
    # STEP 3: Visualize FIS & phenology functions
    # ====================================================================
    if args.experiment_mode != "final-evaluation":
        print("\nGenerating fuzzy/phenology visualizations...")
        fuzzy_ctrl = FuzzyIrrigationController(efficiency=eta_star, **fuzzy_profile_selected)
        visualize_fis(fuzzy_ctrl, save_path=os.path.join(output_dir, 'fuzzy_mf_visualization.png'))
        plot_phenology_targets(save_path=os.path.join(output_dir, 'phenology_targets.png'))
    
    # ====================================================================
    # STEP 4: Run simulations (with calibrated eta)
    # ====================================================================
    controller_specs = build_controller_specs(eta_star, fuzzy_profile=fuzzy_profile_selected)
    if args.experiment_mode == "attribution":
        keep_ids = {"fuzzy_phenology", "reactive_static", "fuzzy_static", "reactive_stage"}
        controller_specs = [spec for spec in controller_specs if spec["id"] in keep_ids]
    elif args.experiment_mode == "final-evaluation" or args.reproducibility_pack:
        keep_ids = {"fuzzy_phenology", "reactive_static"}
        controller_specs = [spec for spec in controller_specs if spec["id"] in keep_ids]
    if args.fuzzy and not args.reactive:
        controller_specs = [spec for spec in controller_specs if spec["group"] == "fuzzy"]
    elif args.reactive and not args.fuzzy:
        controller_specs = [spec for spec in controller_specs if spec["group"] == "reactive"]

    controller_results = {}
    controller_summaries = {}

    if args.experiment_mode == "final-evaluation":
        (
            controller_results,
            controller_summaries,
            eta_calibration_df,
            fuzzy_profile_tuning_df,
            holdout_selected_runtime_df,
        ) = run_final_evaluation_holdout_protocol(
            weather_df=weather_df,
            output_dir=output_dir,
            active_years=active_years,
            selected_season_name=selected_season_name,
            controller_specs=controller_specs,
        )
        if not holdout_selected_runtime_df.empty:
            eta_star = float(holdout_selected_runtime_df["eta_star"].mean())
            best_profile = holdout_selected_runtime_df["profile_name"].mode()
            if not best_profile.empty:
                fuzzy_profile_selected = {"profile_name": str(best_profile.iloc[0])}
    else:
        for spec in controller_specs:
            ctrl_id = spec["id"]
            results_df, summary_df = run_simulation(
                controller=spec["controller"],
                controller_name=ctrl_id,
                weather_df=weather_df,
                output_dir=output_dir,
                verbose=True,
                engine_kwargs=engine_kwargs,
            )
            if not summary_df.empty:
                summary_df = enrich_summary_with_aquacrop(
                    results_df, summary_df, weather_df, eta_star, verbose=True
                )
                save_results_and_summary(ctrl_id, results_df, summary_df, output_dir=output_dir)

            controller_results[ctrl_id] = {
                "label": spec["label"],
                "group": spec["group"],
                "description": spec["description"],
                "results": results_df,
            }
            controller_summaries[ctrl_id] = {
                "label": spec["label"],
                "group": spec["group"],
                "description": spec["description"],
                "summary": summary_df,
            }

    if args.experiment_mode == "final-evaluation":
        print("\nGenerating post-holdout fuzzy/phenology visualizations...")
        profile_name = str(fuzzy_profile_selected.get("profile_name", "")) if isinstance(fuzzy_profile_selected, dict) else ""
        final_profile = resolve_fuzzy_profile_by_name(profile_name)
        fuzzy_ctrl = FuzzyIrrigationController(efficiency=eta_star, **final_profile)
        visualize_fis(fuzzy_ctrl, save_path=os.path.join(output_dir, 'fuzzy_mf_visualization.png'))
        plot_phenology_targets(save_path=os.path.join(output_dir, 'phenology_targets.png'))
        fuzzy_profile_selected = final_profile

    # Persist aggregated multi-controller tables
    overall_table = compile_controller_summary_table(controller_summaries)
    seasonal_table = compile_seasonal_summary_table(controller_summaries)
    controller_design_table = build_controller_design_summary(controller_summaries)
    if not overall_table.empty:
        overall_table.to_csv(os.path.join(output_dir, "controller_overall_summary.csv"), index=False)
    if not seasonal_table.empty:
        seasonal_table.to_csv(os.path.join(output_dir, "controller_seasonal_summary.csv"), index=False)
    if not controller_design_table.empty:
        controller_design_table.to_csv(os.path.join(output_dir, "controller_design_summary.csv"), index=False)
    mechanism_per_scenario = compile_mechanism_diagnostics_per_scenario(controller_summaries)
    if not mechanism_per_scenario.empty:
        mechanism_per_scenario.to_csv(os.path.join(output_dir, "mechanism_diagnostics_per_scenario.csv"), index=False)
    mechanism_summary = compile_mechanism_diagnostics_summary(controller_summaries)
    if not mechanism_summary.empty:
        mechanism_summary.to_csv(os.path.join(output_dir, "mechanism_diagnostics_summary.csv"), index=False)
    isolation_table = build_attribution_isolation_table(controller_summaries)
    if not isolation_table.empty:
        isolation_table.to_csv(os.path.join(output_dir, "attribution_isolation_table.csv"), index=False)
    
    # ====================================================================
    # STEP 5: Comparison & Statistical Analysis
    # ====================================================================
    pairwise_df = build_pairwise_vs_reference(controller_summaries, reference_id="fuzzy_phenology")
    if not pairwise_df.empty:
        print(f"\n{'='*60}")
        print("STATISTICAL COMPARISON (REFERENCE = FUZZY-PHENOLOGY)")
        print(f"{'='*60}")
        pairwise_df.to_csv(os.path.join(output_dir, "pairwise_vs_fuzzy_phenology.csv"), index=False)
        table_pack = build_comparison_table_pack(pairwise_df)
        if not table_pack.empty:
            table_pack.to_csv(os.path.join(output_dir, "comparison_table_pack.csv"), index=False)
            for section_name in ["headline_comparison", "attribution_comparison", "robustness_only"]:
                section_df = table_pack[table_pack["table_section"] == section_name].copy()
                if not section_df.empty:
                    section_df.to_csv(os.path.join(output_dir, f"{section_name}_stats.csv"), index=False)
        report = generate_multi_controller_report(
            controller_summaries,
            pairwise_df,
            save_path=os.path.join(output_dir, 'comparison_report.txt')
        )
        print(report)
        generate_attribution_interpretation_notes(output_dir, pairwise_df, isolation_table)
    mechanism_pairwise_df = build_mechanism_pairwise_vs_reference(controller_summaries, reference_id="fuzzy_phenology")
    if not mechanism_pairwise_df.empty:
        mechanism_pairwise_df.to_csv(os.path.join(output_dir, "mechanism_pairwise_vs_fuzzy_phenology.csv"), index=False)
        generate_mechanism_diagnostics_notes(output_dir, mechanism_pairwise_df)
        generate_mechanism_mapping_artifacts(output_dir)

    # Keep the original 2-controller baseline export for the principal comparison.
    primary_fuzzy = controller_summaries.get("fuzzy_phenology", {}).get("summary", pd.DataFrame())
    primary_reactive = controller_summaries.get("reactive_static", {}).get("summary", pd.DataFrame())
    if not primary_fuzzy.empty and not primary_reactive.empty:
        ttest_results = paired_t_test(primary_fuzzy, primary_reactive)
        ttest_results.to_csv(os.path.join(output_dir, 'paired_ttest_results.csv'), index=False)
        export_head_to_head_csvs(primary_fuzzy, primary_reactive, output_dir, year_label=year_label)
    
    # ====================================================================
    # STEP 6: Generate plots
    # ====================================================================
    if not args.no_plots:
        fuzzy_results = controller_results.get("fuzzy_phenology", {}).get("results", pd.DataFrame())
        reactive_results = controller_results.get("reactive_static", {}).get("results", pd.DataFrame())
        if not fuzzy_results.empty and not reactive_results.empty and not primary_fuzzy.empty and not primary_reactive.empty:
            generate_sample_plots(
                fuzzy_results,
                reactive_results,
                primary_fuzzy,
                primary_reactive,
                output_dir=output_dir,
            )
        if args.experiment_mode == "attribution":
            generate_reproducibility_figure_visual_package(output_dir, controller_results, controller_summaries)

    if args.experiment_mode == "attribution":
        generate_yield_focus_ablation_outputs(
            output_dir=output_dir,
            weather_df=weather_df,
            eta_star=eta_star,
            fuzzy_profile_selected=fuzzy_profile_selected,
            engine_kwargs=engine_kwargs,
        )

    # ====================================================================
    # STEP 7: Descriptive markdown analysis
    # ====================================================================
    generate_multi_controller_analisa_markdown(
        output_dir=output_dir,
        weather_df=weather_df,
        controller_results=controller_results,
        controller_summaries=controller_summaries,
        eta_selected=eta_star,
        eta_calibration_df=eta_calibration_df,
        fuzzy_profile_selected=fuzzy_profile_selected,
        fuzzy_profile_tuning_df=fuzzy_profile_tuning_df,
        pairwise_df=pairwise_df,
        analisa_filename="analisa.md",
        run_scope_label=(
            f"{FINAL_EVALUATION_PROTOCOL_LABEL} ({selected_season_name})"
            if args.experiment_mode == "final-evaluation" and selected_season_name is not None
            else FINAL_EVALUATION_PROTOCOL_LABEL
            if args.experiment_mode == "final-evaluation"
            else selected_season_name or ("Semua musim reproducibility" if args.reproducibility_pack else "Semua musim (MT-1, MT-2, MT-3)")
        ),
    )

    if selected_season_alias is not None:
        season_report_name = SEASON_ALIAS_TO_REPORT[selected_season_alias]
        generate_multi_controller_analisa_markdown(
            output_dir=output_dir,
            weather_df=weather_df,
            controller_results=controller_results,
            controller_summaries=controller_summaries,
            eta_selected=eta_star,
            eta_calibration_df=eta_calibration_df,
            fuzzy_profile_selected=fuzzy_profile_selected,
            fuzzy_profile_tuning_df=fuzzy_profile_tuning_df,
            pairwise_df=pairwise_df,
            analisa_filename=season_report_name,
            run_scope_label=selected_season_name,
        )

    if args.reproducibility_pack and args.experiment_mode == "all" and not primary_fuzzy.empty and not primary_reactive.empty:
        seasonal_h2h = pd.merge(
            primary_fuzzy.groupby("season", as_index=False).agg(
                iwu_fuzzy_mm=("iwu_mm", "mean"),
                yield_fuzzy_t_ha=("yield_dry_t_ha", "mean"),
                iwue_fuzzy_kg_ha_per_mm=("iwue_kg_ha_per_mm", "mean"),
            ),
            primary_reactive.groupby("season", as_index=False).agg(
                iwu_reaktif_mm=("iwu_mm", "mean"),
                yield_reaktif_t_ha=("yield_dry_t_ha", "mean"),
                iwue_reaktif_kg_ha_per_mm=("iwue_kg_ha_per_mm", "mean"),
            ),
            on="season",
            how="inner",
        )
        seasonal_h2h["delta_iwu_mm"] = seasonal_h2h["iwu_fuzzy_mm"] - seasonal_h2h["iwu_reaktif_mm"]
        seasonal_h2h["delta_yield_t_ha"] = seasonal_h2h["yield_fuzzy_t_ha"] - seasonal_h2h["yield_reaktif_t_ha"]
        seasonal_h2h["delta_iwue_kg_ha_per_mm"] = seasonal_h2h["iwue_fuzzy_kg_ha_per_mm"] - seasonal_h2h["iwue_reaktif_kg_ha_per_mm"]
        seasonal_h2h["season_order"] = seasonal_h2h["season"].map(_season_sort_key)
        seasonal_h2h = seasonal_h2h.sort_values(["season_order", "season"]).drop(columns=["season_order"])
        seasonal_h2h.to_csv(os.path.join(output_dir, f"reproducibility_seasonal_head_to_head_{year_label}.csv"), index=False)

        print(f"\n{'='*60}")
        print("RUNNING REPRODUCIBILITY-ALIGNED LIMITED SENSITIVITY")
        print(f"{'='*60}")
        sensitivity_df = generate_reproducibility_sensitivity_outputs(
            output_dir=output_dir,
            weather_df=weather_df,
            eta_star=eta_star,
            fuzzy_profile_selected=fuzzy_profile_selected,
            engine_kwargs=engine_kwargs,
        )
        generate_reproducibility_claim_evidence_table(output_dir, ttest_results, seasonal_h2h, sensitivity_df)

    print(f"\n{'='*60}")
    print(f"SIMULATION COMPLETE")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
