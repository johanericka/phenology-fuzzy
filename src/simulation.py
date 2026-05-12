"""
Simulation Engine — Closed-Loop (Persamaan 9, FASE 7)
======================================================
Runs the daily closed-loop simulation: irrigation decision → water balance
update → new soil moisture → next day's decision.

Equation 9 (closed-loop):
  rᵢ = rᵢ₋₁ + (Pᵢ−ROᵢ+CRᵢ−ETᵢ−DPᵢ)/(1000·Zr·θFC) + (1/η)·uᵢ·max(0, rU(i)−rᵢ₋₁)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.config import (
    TOTAL_DAYS, SIMULATION_YEARS, PLANTING_SEASONS, get_output_dir,
    INITIAL_MOISTURE_FC, YIELD_FOCUS_MODE, KEMARAU_TARGET_BOOSTS,
    WEATHER_COVERAGE_MIN_FRACTION,
)
from src.water_balance import WaterBalanceModel
from src.phenology import (
    compute_dynamic_targets,
    get_kc_for_hst,
    get_dominant_phase,
    get_dominant_subphase,
)
from src.fuzzy_controller import FuzzyIrrigationController
from src.reactive_controller import ReactiveController


class SimulationEngine:
    """
    Closed-loop irrigation simulation engine.
    
    Runs daily simulation for one cropping season, recording all variables.
    """
    
    def __init__(self, controller, weather_df: pd.DataFrame,
                 total_days: int = None,
                 initial_r: float = None,
                 simulation_years: list[int] | None = None,
                 planting_seasons: dict | None = None,
                 overlap_days: int | None = None,
                 yield_focus_mode: bool | None = None,
                 target_boosts: dict | None = None):
        """
        Args:
            controller: Irrigation controller (FuzzyIrrigationController or ReactiveController)
            weather_df: Weather DataFrame with Date, Tmin, Tmax, Prcp, RH, SS, Wind, Et0
            total_days: Growing season length (default: TOTAL_DAYS)
            initial_r: Initial soil moisture as fraction of FC
        """
        self.controller = controller
        self.weather_df = weather_df.copy()
        self.total_days = total_days or TOTAL_DAYS
        self.initial_r = initial_r or INITIAL_MOISTURE_FC
        self.simulation_years = list(simulation_years) if simulation_years is not None else list(SIMULATION_YEARS)
        self.planting_seasons = dict(planting_seasons) if planting_seasons is not None else dict(PLANTING_SEASONS)
        self.overlap_days = overlap_days
        self.yield_focus_mode = YIELD_FOCUS_MODE if yield_focus_mode is None else bool(yield_focus_mode)
        self.target_boosts = dict(KEMARAU_TARGET_BOOSTS if target_boosts is None else target_boosts)
        
        # Ensure Date is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.weather_df['Date']):
            self.weather_df['Date'] = pd.to_datetime(self.weather_df['Date'])
    
    def run_season(self, planting_date: datetime, season_name: str = "",
                   year: int = None) -> pd.DataFrame:
        """
        Run one complete cropping season simulation.
        
        Args:
            planting_date: Date of planting
            season_name: Name of the season (e.g., 'Penghujan')
            year: Year of the simulation
            
        Returns:
            DataFrame with daily simulation results
        """
        # Initialize water balance model
        wb = WaterBalanceModel(initial_r=self.initial_r)
        
        # Storage for daily results
        results = []
        
        for hst in range(self.total_days):
            current_date = planting_date + timedelta(days=hst)
            
            # Get weather data for this day
            weather_row = self.weather_df[
                self.weather_df['Date'] == pd.Timestamp(current_date)
            ]
            
            if weather_row.empty:
                # Skip if no weather data available
                continue
            
            weather = weather_row.iloc[0]
            et0 = float(weather['Et0'])
            precipitation = float(weather['Prcp'])
            
            # Get dynamic targets and Kc
            phase = get_dominant_phase(hst, overlap_days=self.overlap_days)
            subphase = get_dominant_subphase(hst, overlap_days=self.overlap_days)
            targets = self._adjust_targets_for_season(
                compute_dynamic_targets(hst, overlap_days=self.overlap_days),
                season_name=season_name,
                phase=phase,
            )
            kc = get_kc_for_hst(hst, overlap_days=self.overlap_days)
            
            # Irrigation decision
            irrig_result = self.controller.compute_irrigation(
                r_prev=wb.r,
                hst=hst,
                rU=targets['rU'],
                rL=targets['rL'],
                rWP=targets['rWP']
            )
            irrigation_mm = irrig_result['irrigation_mm']
            
            # Water balance update
            wb_result = wb.update(
                precipitation=precipitation,
                irrigation=irrigation_mm,
                et0=et0,
                kc=kc
            )
            
            # Record results
            results.append({
                'date': current_date,
                'year': year or current_date.year,
                'season': season_name,
                'hst': hst,
                'phase': phase,
                'subphase': subphase,
                'kc': kc,
                # Weather
                'precipitation': precipitation,
                'et0': et0,
                'tmin': float(weather.get('Tmin', 0)),
                'tmax': float(weather.get('Tmax', 0)),
                # Water balance
                'theta_prev': wb_result['theta_prev'],
                'theta': wb_result['theta'],
                'r_prev': wb_result['r_prev'],
                'r': wb_result['r'],
                'r_pct': wb_result['r'] * 100,
                'runoff': wb_result['runoff'],
                'etc': wb_result['etc'],
                'deep_percolation': wb_result['deep_percolation'],
                # Irrigation decision
                'drought_indicator': irrig_result['drought_indicator'],
                'phase_sensitivity': irrig_result['phase_sensitivity'],
                'phase_critical': irrig_result.get('phase_critical', int(irrig_result.get('phase_sensitivity', 0) >= 0.5)),
                'urgency': irrig_result['urgency'],
                'irrigation_mm': irrigation_mm,
                'sm_ref': irrig_result.get('sm_ref', np.nan),
                'sm_cap': irrig_result.get('sm_cap', np.nan),
                'refill_deficit': irrig_result.get('refill_deficit', np.nan),
                # Dynamic targets
                'rL': targets['rL'],
                'rU': targets['rU'],
                'rWP': targets['rWP'],
                'rL_pct': targets['rL'] * 100,
                'rU_pct': targets['rU'] * 100,
                'rWP_pct': targets['rWP'] * 100,
            })
        
        return pd.DataFrame(results)

    def _adjust_targets_for_season(self, targets: dict, season_name: str, phase: str) -> dict:
        """
        Yield-first adjustment:
        Tingkatkan target kelembapan fuzzy pada musim kemarau, terutama fase reproduktif,
        agar kontrol lebih protektif terhadap penalti hasil panen.
        """
        t = {k: float(v) for k, v in targets.items()}
        if not self.yield_focus_mode:
            return t
        if "kemarau" not in str(season_name).strip().lower():
            return t

        boost = self.target_boosts.get(phase, {"rL": 0.0, "rU": 0.0})
        t["rL"] = min(1.0, t["rL"] + float(boost.get("rL", 0.0)))
        t["rU"] = min(1.0, t["rU"] + float(boost.get("rU", 0.0)))

        # Maintain a valid target band and avoid lower threshold below wilting margin.
        t["rL"] = max(t["rL"], t["rWP"] + 0.10)
        t["rU"] = max(t["rU"], t["rL"] + 0.05)
        t["rU"] = min(1.0, t["rU"])
        t["rL"] = min(t["rL"], t["rU"] - 0.05)
        return t
    
    def run_all_scenarios(self, verbose: bool = True) -> pd.DataFrame:
        """
        Run all 30 scenarios (10 years × 3 seasons) for one controller.
        
        Returns:
            DataFrame with all simulation results concatenated
        """
        all_results = []
        scenario_count = 0
        
        for year in self.simulation_years:
            for season_name, season_info in self.planting_seasons.items():
                # Determine planting date
                plant_month = season_info['start_month']
                plant_day = season_info['start_day']
                
                try:
                    planting_date = datetime(year, plant_month, plant_day)
                except ValueError:
                    continue
                
                # Check if weather data covers this season
                end_date = planting_date + timedelta(days=self.total_days - 1)
                weather_range = self.weather_df[
                    (self.weather_df['Date'] >= pd.Timestamp(planting_date)) &
                    (self.weather_df['Date'] <= pd.Timestamp(end_date))
                ]
                
                if len(weather_range) < self.total_days * WEATHER_COVERAGE_MIN_FRACTION:
                    if verbose:
                        print(f"  ⚠ Skipping {year} {season_name}: insufficient weather data "
                              f"({len(weather_range)}/{self.total_days} days)")
                    continue
                
                scenario_count += 1
                if verbose:
                    print(f"  [{scenario_count:2d}] {year} {season_name:<12s} "
                          f"({planting_date.strftime('%Y-%m-%d')}) ... ", end="")
                
                # Run simulation
                season_result = self.run_season(
                    planting_date=planting_date,
                    season_name=season_name,
                    year=year
                )
                
                if verbose:
                    if not season_result.empty:
                        total_irrig = season_result['irrigation_mm'].sum()
                        mean_r = season_result['r'].mean() * 100
                        print(f"done. IWU={total_irrig:.1f}mm, mean_SM={mean_r:.1f}%FC")
                    else:
                        print(f"no data")
                
                all_results.append(season_result)
        
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    @staticmethod
    def compute_season_metrics(season_df: pd.DataFrame) -> dict:
        """
        Compute performance metrics for one season.
        
        Returns:
            Dict with IWU, IWUE, mean_r, R², MSE, etc.
        """
        if season_df.empty:
            return {}
        
        # Irrigation Water Use (IWU) — total mm
        iwu = season_df['irrigation_mm'].sum()
        
        # Number of irrigation events
        irrigation_mask = season_df['irrigation_mm'] > 0
        n_irrigation_events = irrigation_mask.sum()
        
        # Mean soil moisture (% FC)
        mean_r = season_df['r'].mean()
        
        # Soil moisture tracking accuracy against dynamic target midpoint
        target_mid = (season_df['rL'] + season_df['rU']) / 2
        residuals = season_df['r'] - target_mid
        mse = (residuals ** 2).mean()
        
        # R² — how much variance in SM is explained by target tracking
        ss_res = (residuals ** 2).sum()
        ss_tot = ((season_df['r'] - season_df['r'].mean()) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Stress days — days where r < rWP
        stress_days = (season_df['r'] < season_df['rWP']).sum()
        
        # Excess days — days where r > 1.0 (above FC)
        excess_days = (season_df['r'] > 1.0).sum()
        
        # Days within target range
        in_target = (
            (season_df['r'] >= season_df['rL']) & 
            (season_df['r'] <= season_df['rU'])
        ).sum()
        target_pct = in_target / len(season_df) * 100

        # Irrigation pattern diagnostics
        irrigation_depths = season_df.loc[irrigation_mask, 'irrigation_mm'].astype(float)
        irrigation_frequency_pct = (n_irrigation_events / len(season_df) * 100) if len(season_df) > 0 else 0.0
        mean_depth_per_event_mm = float(irrigation_depths.mean()) if not irrigation_depths.empty else 0.0
        median_depth_per_event_mm = float(irrigation_depths.median()) if not irrigation_depths.empty else 0.0
        cv_event_depth = (
            float(irrigation_depths.std() / irrigation_depths.mean())
            if (not irrigation_depths.empty and irrigation_depths.mean() > 1e-9)
            else 0.0
        )

        event_hst = season_df.loc[irrigation_mask, 'hst'].astype(int).tolist()
        if len(event_hst) >= 2:
            intervals = np.diff(event_hst)
            mean_interval_between_events_days = float(np.mean(intervals))
            median_interval_between_events_days = float(np.median(intervals))
            max_interval_between_events_days = float(np.max(intervals))
        else:
            mean_interval_between_events_days = float(len(season_df)) if n_irrigation_events == 1 else float("nan")
            median_interval_between_events_days = float(len(season_df)) if n_irrigation_events == 1 else float("nan")
            max_interval_between_events_days = float(len(season_df)) if n_irrigation_events == 1 else float("nan")

        if event_hst:
            first_event_hst = int(event_hst[0])
            last_event_hst = int(event_hst[-1])
            max_interval_without_irrigation_days = max(
                first_event_hst,
                int(max_interval_between_events_days) if np.isfinite(max_interval_between_events_days) else 0,
                int((len(season_df) - 1) - last_event_hst),
            )
        else:
            max_interval_without_irrigation_days = int(len(season_df))

        total_etc = season_df['etc'].sum()
        total_precip = season_df['precipitation'].sum()
        # Bukan IWUE agronomis (kg/mm) karena yield belum dimodelkan.
        # Ini proxy efisiensi hidrologi-kontrol untuk analisis teknis.
        iwue_proxy_etc_per_iwu = (total_etc / iwu) if iwu > 1e-9 else np.nan
        target_per_iwu = (target_pct / iwu) if iwu > 1e-9 else np.nan

        return {
            'iwu_mm': iwu,
            'n_irrigation_events': int(n_irrigation_events),
            'irrigation_frequency_pct': irrigation_frequency_pct,
            'mean_depth_per_event_mm': mean_depth_per_event_mm,
            'median_depth_per_event_mm': median_depth_per_event_mm,
            'cv_event_depth': cv_event_depth,
            'mean_interval_between_events_days': mean_interval_between_events_days,
            'median_interval_between_events_days': median_interval_between_events_days,
            'max_interval_between_events_days': max_interval_between_events_days,
            'max_interval_without_irrigation_days': int(max_interval_without_irrigation_days),
            'mean_r': mean_r,
            'mean_r_pct': mean_r * 100,
            'mse': mse,
            'r_squared': r_squared,
            'stress_days': int(stress_days),
            'excess_days': int(excess_days),
            'days_in_target': int(in_target),
            'target_pct': target_pct,
            'total_days': len(season_df),
            'total_precip': total_precip,
            'total_etc': total_etc,
            'iwue_proxy_etc_per_iwu': iwue_proxy_etc_per_iwu,
            'target_per_iwu': target_per_iwu,
        }
