"""
Bridge utilities for running AquaCrop-OSPy on controller-generated irrigation schedules.

Tujuan:
- Mengambil jadwal irigasi harian dari simulasi closed-loop custom
- Menjalankan AquaCrop per skenario (year x season)
- Mengekstrak yield/stress response untuk evaluasi agronomis dan IWUE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from aquacrop import AquaCropModel, Crop, InitialWaterContent, IrrigationManagement, Soil

from src.config import AQUACROP_SOIL_TYPE
from src.data_cleansing import prepare_aquacrop_weather


@dataclass
class AquaCropRunConfig:
    app_efficiency: float = 0.8
    soil_type: str = AQUACROP_SOIL_TYPE
    crop_name: str = "PaddyRice"
    max_irrigation_event_mm: float = 1000.0
    max_irrigation_season_mm: float = 50000.0
    irrigation_event_threshold_mm: float = 1e-6
    min_iwu_for_iwue_mm: float = 10.0  # hindari ledakan rasio pada musim hujan nyaris tanpa irigasi


def _build_irrigation_schedule_from_season(season_df: pd.DataFrame, cfg: AquaCropRunConfig) -> pd.DataFrame:
    sched = season_df.loc[
        season_df["irrigation_mm"] > cfg.irrigation_event_threshold_mm, ["date", "irrigation_mm"]
    ].copy()
    if sched.empty:
        return pd.DataFrame(columns=["Date", "Depth"])
    sched = sched.rename(columns={"date": "Date", "irrigation_mm": "Depth"})
    sched["Date"] = pd.to_datetime(sched["Date"])
    sched["Depth"] = pd.to_numeric(sched["Depth"], errors="coerce").fillna(0.0)
    return sched


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio only when the denominator is meaningfully positive."""
    if np.isfinite(numerator) and np.isfinite(denominator) and denominator > 1e-9:
        return float(numerator / denominator)
    return np.nan


def _stress_days_from_flux(flux_df: pd.DataFrame, tr_col: str = "Tr", trpot_col: str = "TrPot") -> float:
    """Count days with transpiration deficit relative to transpiration potential."""
    if flux_df.empty or {tr_col, trpot_col}.difference(flux_df.columns):
        return np.nan
    mask = flux_df[trpot_col] > 1e-6
    if not mask.any():
        return np.nan
    return int((flux_df.loc[mask, tr_col] < 0.9 * flux_df.loc[mask, trpot_col]).sum())


def _phase_mask(series: pd.Series, keyword: str) -> pd.Series:
    return series.astype(str).str.contains(keyword, case=False, na=False)


def _reproductive_mask(phase_diag: pd.DataFrame) -> pd.Series:
    """Capture repository-specific reproductive labeling plus critical rice windows."""
    phase_mask = phase_diag["phase"].astype(str).str.contains("REPRO|REPRODUCT", case=False, na=False)
    sub_mask = phase_diag["subphase"].astype(str).str.contains(
        "Panicle|Booting|Flowering|Grain_Filling",
        case=False,
        na=False,
    )
    return phase_mask | sub_mask


def run_aquacrop_for_season(
    season_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    cfg: Optional[AquaCropRunConfig] = None,
) -> dict:
    """
    Jalankan AquaCrop untuk satu skenario berdasarkan jadwal irigasi hasil controller.

    Parameters
    ----------
    season_df : pd.DataFrame
        Output harian custom simulation untuk satu (year, season)
    weather_df : pd.DataFrame
        Weather hasil cleansing (kolom Date/Tmin/Tmax/Prcp/Et0 dst.)
    """
    cfg = cfg or AquaCropRunConfig()
    if season_df.empty:
        return {}

    s = season_df.sort_values("date").copy()
    start = pd.to_datetime(s["date"].iloc[0])
    end = pd.to_datetime(s["date"].iloc[-1])

    weather_slice = weather_df[(weather_df["Date"] >= start) & (weather_df["Date"] <= end)].copy()
    weather_ac = prepare_aquacrop_weather(weather_slice)
    if weather_ac.empty:
        return {}

    irr_schedule = _build_irrigation_schedule_from_season(s, cfg)
    irr_mgmt = IrrigationManagement(
        irrigation_method=3,
        Schedule=irr_schedule.copy(),
        AppEff=float(cfg.app_efficiency) * 100.0,
        MaxIrr=float(cfg.max_irrigation_event_mm),
        MaxIrrSeason=float(cfg.max_irrigation_season_mm),
    )

    crop = Crop(
        cfg.crop_name,
        planting_date=start.strftime("%m/%d"),
        harvest_date=end.strftime("%m/%d"),
    )
    soil = Soil(cfg.soil_type)
    iwc = InitialWaterContent(value=["FC"])

    model = AquaCropModel(
        sim_start_time=start.strftime("%Y/%m/%d"),
        sim_end_time=end.strftime("%Y/%m/%d"),
        weather_df=weather_ac,
        soil=soil,
        crop=crop,
        initial_water_content=iwc,
        irrigation_management=irr_mgmt,
    )
    model.run_model(till_termination=True, process_outputs=True)

    final_stats = model.get_simulation_results()
    water_flux = model.get_water_flux()
    crop_growth = model.get_crop_growth()

    # Extract season row (single-row dataframe expected for our single season window)
    fs_row = final_stats.iloc[0] if isinstance(final_stats, pd.DataFrame) and not final_stats.empty else None

    # Active crop period in AquaCrop daily outputs
    active_flux = water_flux[(water_flux["dap"] > 0)].copy() if "dap" in water_flux else pd.DataFrame()
    active_growth = crop_growth[(crop_growth["dap"] > 0)].copy() if "dap" in crop_growth else pd.DataFrame()

    dry_yield_t_ha = float(fs_row.get("Dry yield (tonne/ha)", np.nan)) if fs_row is not None else np.nan
    fresh_yield_t_ha = float(fs_row.get("Fresh yield (tonne/ha)", np.nan)) if fs_row is not None else np.nan
    yield_potential_t_ha = float(fs_row.get("Yield potential (tonne/ha)", np.nan)) if fs_row is not None else np.nan
    seasonal_irrig_aq_mm = float(fs_row.get("Seasonal irrigation (mm)", np.nan)) if fs_row is not None else np.nan

    iwu_custom_mm = float(s["irrigation_mm"].sum())
    dry_yield_kg_ha = dry_yield_t_ha * 1000.0 if np.isfinite(dry_yield_t_ha) else np.nan
    iwue_kg_ha_per_mm = (
        dry_yield_kg_ha / iwu_custom_mm if (np.isfinite(dry_yield_kg_ha) and iwu_custom_mm >= cfg.min_iwu_for_iwue_mm) else np.nan
    )

    # Stress and productivity diagnostics from AquaCrop daily outputs
    tr_sum = float(active_flux["Tr"].sum()) if ("Tr" in active_flux and not active_flux.empty) else np.nan
    trpot_sum = float(active_flux["TrPot"].sum()) if ("TrPot" in active_flux and not active_flux.empty) else np.nan
    transpiration_ratio = _safe_ratio(tr_sum, trpot_sum)

    stress_days_tr = _stress_days_from_flux(active_flux)

    es_sum = float(active_flux["Es"].sum()) if ("Es" in active_flux and not active_flux.empty) else np.nan
    espot_sum = float(active_flux["EsPot"].sum()) if ("EsPot" in active_flux and not active_flux.empty) else np.nan
    evaporation_ratio = _safe_ratio(es_sum, espot_sum)
    deep_perc_sum = float(active_flux["DeepPerc"].sum()) if ("DeepPerc" in active_flux and not active_flux.empty) else np.nan
    runoff_sum = float(active_flux["Runoff"].sum()) if ("Runoff" in active_flux and not active_flux.empty) else np.nan
    infl_sum = float(active_flux["Infl"].sum()) if ("Infl" in active_flux and not active_flux.empty) else np.nan
    seasonal_irrig_flux_mm = float(active_flux["IrrDay"].sum()) if ("IrrDay" in active_flux and not active_flux.empty) else np.nan

    biomass_t_ha = np.nan
    if not active_growth.empty and "biomass" in active_growth.columns:
        # AquaCrop biomass is in ton/ha in outputs for this package version
        biomass_t_ha = float(active_growth["biomass"].max())
    harvest_index_max = float(active_growth["harvest_index_adj"].max()) if ("harvest_index_adj" in active_growth and not active_growth.empty) else np.nan

    # Join AquaCrop daily outputs with the custom controller phase labels by DAP/HST.
    phase_diag = pd.DataFrame()
    if not active_flux.empty:
        phase_diag = s.copy()
        phase_diag["dap"] = pd.to_numeric(phase_diag["hst"], errors="coerce") + 1.0
        merge_cols = [col for col in ["dap", "Tr", "TrPot", "Es", "EsPot", "DeepPerc", "Runoff", "Infl", "IrrDay"] if col in active_flux.columns]
        phase_diag = phase_diag.merge(active_flux.loc[:, merge_cols], on="dap", how="left")

    reproductive_flux = pd.DataFrame()
    flowering_flux = pd.DataFrame()
    grain_fill_flux = pd.DataFrame()
    reproductive_irrigation_mm = np.nan
    flowering_irrigation_mm = np.nan
    grain_fill_irrigation_mm = np.nan
    reproductive_irrigation_share_pct = np.nan
    flowering_irrigation_share_pct = np.nan
    grain_fill_irrigation_share_pct = np.nan

    if not phase_diag.empty:
        reproductive_flux = phase_diag[_reproductive_mask(phase_diag)].copy()
        flowering_flux = phase_diag[_phase_mask(phase_diag["subphase"], "flower")].copy()
        grain_fill_flux = phase_diag[_phase_mask(phase_diag["subphase"], "grain")].copy()

        reproductive_irrigation_mm = float(reproductive_flux["irrigation_mm"].sum()) if "irrigation_mm" in reproductive_flux else np.nan
        flowering_irrigation_mm = float(flowering_flux["irrigation_mm"].sum()) if "irrigation_mm" in flowering_flux else np.nan
        grain_fill_irrigation_mm = float(grain_fill_flux["irrigation_mm"].sum()) if "irrigation_mm" in grain_fill_flux else np.nan
        reproductive_irrigation_share_pct = _safe_ratio(reproductive_irrigation_mm, iwu_custom_mm) * 100.0 if np.isfinite(iwu_custom_mm) else np.nan
        flowering_irrigation_share_pct = _safe_ratio(flowering_irrigation_mm, iwu_custom_mm) * 100.0 if np.isfinite(iwu_custom_mm) else np.nan
        grain_fill_irrigation_share_pct = _safe_ratio(grain_fill_irrigation_mm, iwu_custom_mm) * 100.0 if np.isfinite(iwu_custom_mm) else np.nan

    reproductive_tr_sum = float(reproductive_flux["Tr"].sum()) if ("Tr" in reproductive_flux and not reproductive_flux.empty) else np.nan
    reproductive_trpot_sum = float(reproductive_flux["TrPot"].sum()) if ("TrPot" in reproductive_flux and not reproductive_flux.empty) else np.nan
    reproductive_tr_ratio = _safe_ratio(reproductive_tr_sum, reproductive_trpot_sum)
    reproductive_stress_days_tr = _stress_days_from_flux(reproductive_flux)

    flowering_tr_sum = float(flowering_flux["Tr"].sum()) if ("Tr" in flowering_flux and not flowering_flux.empty) else np.nan
    flowering_trpot_sum = float(flowering_flux["TrPot"].sum()) if ("TrPot" in flowering_flux and not flowering_flux.empty) else np.nan
    flowering_tr_ratio = _safe_ratio(flowering_tr_sum, flowering_trpot_sum)
    flowering_stress_days_tr = _stress_days_from_flux(flowering_flux)

    grain_fill_tr_sum = float(grain_fill_flux["Tr"].sum()) if ("Tr" in grain_fill_flux and not grain_fill_flux.empty) else np.nan
    grain_fill_trpot_sum = float(grain_fill_flux["TrPot"].sum()) if ("TrPot" in grain_fill_flux and not grain_fill_flux.empty) else np.nan
    grain_fill_tr_ratio = _safe_ratio(grain_fill_tr_sum, grain_fill_trpot_sum)
    grain_fill_stress_days_tr = _stress_days_from_flux(grain_fill_flux)

    return {
        "year": int(s["year"].iloc[0]),
        "season": str(s["season"].iloc[0]),
        "aquacrop_harvest_date": str(fs_row.get("Harvest Date (YYYY/MM/DD)")) if fs_row is not None else None,
        "yield_dry_t_ha": dry_yield_t_ha,
        "yield_fresh_t_ha": fresh_yield_t_ha,
        "yield_potential_t_ha": yield_potential_t_ha,
        "yield_gap_t_ha": (yield_potential_t_ha - dry_yield_t_ha)
        if np.isfinite(yield_potential_t_ha) and np.isfinite(dry_yield_t_ha)
        else np.nan,
        "biomass_t_ha": biomass_t_ha,
        "aquacrop_irrigation_mm": seasonal_irrig_aq_mm,
        "iwu_custom_mm_for_iwue": iwu_custom_mm,
        "iwue_kg_ha_per_mm": iwue_kg_ha_per_mm,
        "aq_tr_sum_mm": tr_sum,
        "aq_trpot_sum_mm": trpot_sum,
        "aq_transpiration_ratio": transpiration_ratio,
        "aq_stress_days_tr": stress_days_tr,
        "aq_es_sum_mm": es_sum,
        "aq_espot_sum_mm": espot_sum,
        "aq_evaporation_ratio": evaporation_ratio,
        "aq_deep_perc_mm": deep_perc_sum,
        "aq_runoff_mm": runoff_sum,
        "aq_infl_mm": infl_sum,
        "aq_irrigation_flux_mm": seasonal_irrig_flux_mm,
        "aq_harvest_index_adj": harvest_index_max,
        "aq_reproductive_tr_sum_mm": reproductive_tr_sum,
        "aq_reproductive_trpot_sum_mm": reproductive_trpot_sum,
        "aq_reproductive_transpiration_ratio": reproductive_tr_ratio,
        "aq_reproductive_stress_days_tr": reproductive_stress_days_tr,
        "aq_reproductive_irrigation_mm": reproductive_irrigation_mm,
        "aq_reproductive_irrigation_share_pct": reproductive_irrigation_share_pct,
        "aq_flowering_tr_sum_mm": flowering_tr_sum,
        "aq_flowering_trpot_sum_mm": flowering_trpot_sum,
        "aq_flowering_transpiration_ratio": flowering_tr_ratio,
        "aq_flowering_stress_days_tr": flowering_stress_days_tr,
        "aq_flowering_irrigation_mm": flowering_irrigation_mm,
        "aq_flowering_irrigation_share_pct": flowering_irrigation_share_pct,
        "aq_grain_fill_tr_sum_mm": grain_fill_tr_sum,
        "aq_grain_fill_trpot_sum_mm": grain_fill_trpot_sum,
        "aq_grain_fill_transpiration_ratio": grain_fill_tr_ratio,
        "aq_grain_fill_stress_days_tr": grain_fill_stress_days_tr,
        "aq_grain_fill_irrigation_mm": grain_fill_irrigation_mm,
        "aq_grain_fill_irrigation_share_pct": grain_fill_irrigation_share_pct,
    }


def run_aquacrop_for_all_scenarios(
    all_results_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    cfg: Optional[AquaCropRunConfig] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run AquaCrop per (year, season) and return aggregated results."""
    cfg = cfg or AquaCropRunConfig()
    rows = []
    groups = list(all_results_df.groupby(["year", "season"]))
    for idx, ((year, season), group) in enumerate(groups, start=1):
        if verbose:
            print(f"    AquaCrop [{idx:02d}/{len(groups)}] {year} {season} ... ", end="")
        out = run_aquacrop_for_season(group, weather_df, cfg=cfg)
        if out:
            rows.append(out)
            if verbose:
                y = out.get("yield_dry_t_ha", np.nan)
                iwue = out.get("iwue_kg_ha_per_mm", np.nan)
                print(f"yield={y:.3f} t/ha, IWUE={iwue:.2f} kg/ha/mm" if np.isfinite(y) else "done")
        elif verbose:
            print("failed/empty")
    return pd.DataFrame(rows)
