"""
Water Balance Model (Persamaan 1–3)
====================================
Implements the daily water balance equation for the root zone,
both in absolute (θ) and relative (r = θ/θ_FC) forms.

Equations from reproducibility:
  (1) θᵢ = θᵢ₋₁ + (Pᵢ − ROᵢ + Iᵢ + CRᵢ − ETᵢ − DPᵢ) / (1000 · Zr,i)
  (2) rᵢ = θᵢ / θ_FC
  (3) rᵢ = rᵢ₋₁ + (Pᵢ − ROᵢ + Iᵢ + CRᵢ − ETᵢ − DPᵢ) / (1000 · Zr,i · θ_FC)
"""

import numpy as np
from src.config import (
    THETA_SAT, THETA_FC, THETA_WP, ROOT_DEPTH,
    KSAT, CN, PERCOLATION_RATE, INITIAL_MOISTURE_FC, RAW
)


class WaterBalanceModel:
    """
    Physics-based soil moisture model using water balance equation.
    
    Tracks soil moisture in the root zone as both:
    - θ (m³/m³): absolute volumetric water content
    - r (fraction of FC): relative moisture = θ / θ_FC
    """
    
    def __init__(self, initial_r: float = None):
        """
        Initialize water balance model.
        
        Args:
            initial_r: Initial soil moisture as fraction of FC (default: INITIAL_MOISTURE_FC)
        """
        if initial_r is None:
            initial_r = INITIAL_MOISTURE_FC
        
        self.theta = initial_r * THETA_FC  # Convert to absolute
        self.r = initial_r                  # Relative to FC
        
        # Store parameters
        self.theta_sat = THETA_SAT
        self.theta_fc = THETA_FC
        self.theta_wp = THETA_WP
        self.z_root = ROOT_DEPTH  # meters
        self.cn = CN
        self.ksat = KSAT
        self.percolation_rate = PERCOLATION_RATE
    
    def reset(self, initial_r: float = None):
        """Reset soil moisture to initial state."""
        if initial_r is None:
            initial_r = INITIAL_MOISTURE_FC
        self.theta = initial_r * THETA_FC
        self.r = initial_r
    
    def calculate_runoff(self, precipitation: float) -> float:
        """
        Calculate surface runoff using SCS Curve Number method.
        
        RO = (P − 0.2·S)² / (P + 0.8·S)  if P > 0.2·S, else 0
        where S = 25400/CN − 254
        
        Args:
            precipitation: Daily rainfall in mm
            
        Returns:
            Runoff in mm/day
        """
        if precipitation <= 0:
            return 0.0
        
        S = (25400.0 / self.cn) - 254.0
        Ia = 0.2 * S  # Initial abstraction
        
        if precipitation > Ia:
            runoff = (precipitation - Ia) ** 2 / (precipitation + 0.8 * S)
        else:
            runoff = 0.0
        
        return runoff
    
    def calculate_percolation(self, theta_candidate: float = None) -> float:
        """
        Calculate deep percolation when soil moisture exceeds field capacity.
        
        Returns:
            Deep percolation in mm/day
        """
        theta_ref = self.theta if theta_candidate is None else theta_candidate
        if theta_ref > self.theta_fc:
            # Percolation proportional to excess above FC, capped at PERCOLATION_RATE
            excess_mm = (theta_ref - self.theta_fc) * 1000 * self.z_root
            dp = min(excess_mm, self.percolation_rate, self.ksat)
        else:
            dp = 0.0
        
        return dp
    
    def calculate_etc(self, et0: float, kc: float) -> float:
        """
        Calculate crop evapotranspiration with water stress coefficient.
        
        ETc = Ks · Kc · ET0
        
        Where Ks is the water stress coefficient:
        - Ks = 1 when θ > θ_FC × RAW_threshold
        - Ks decreases linearly to 0 as θ approaches θ_WP
        
        Args:
            et0: Reference evapotranspiration (mm/day)
            kc: Crop coefficient for current growth stage
            
        Returns:
            Crop evapotranspiration in mm/day
        """
        # Water stress coefficient (Ks)
        # Stress starts after readily available water (RAW) is depleted.
        theta_stress_onset = self.theta_fc - RAW
        if self.theta >= theta_stress_onset:
            ks = 1.0
        elif self.theta <= self.theta_wp:
            ks = 0.0
        else:
            # Linear reduction between WP and stress onset threshold
            ks = (self.theta - self.theta_wp) / (theta_stress_onset - self.theta_wp)
            ks = np.clip(ks, 0.0, 1.0)
        
        return ks * kc * et0
    
    def calculate_capillary_rise(self) -> float:
        """
        Calculate capillary rise from shallow water table.
        Assumed negligible (CR = 0) per typical AquaCrop-OSPy simulations.
        
        Returns:
            Capillary rise in mm/day
        """
        return 0.0
    
    def update(self, precipitation: float, irrigation: float,
               et0: float, kc: float) -> dict:
        """
        Perform one-day water balance update (Persamaan 1).
        
        θᵢ = θᵢ₋₁ + (Pᵢ − ROᵢ + Iᵢ + CRᵢ − ETᵢ − DPᵢ) / (1000 · Zr,i)
        
        Args:
            precipitation: Rainfall in mm/day
            irrigation: Irrigation applied in mm/day
            et0: Reference evapotranspiration in mm/day
            kc: Crop coefficient for current growth stage
            
        Returns:
            Dict with all water balance components for this day
        """
        # Calculate components
        RO = self.calculate_runoff(precipitation)
        P_eff = precipitation - RO  # Effective precipitation
        CR = self.calculate_capillary_rise()
        ET = self.calculate_etc(et0, kc)
        # Store previous state
        theta_prev = self.theta
        r_prev = self.r

        # Apply inflow/outflow (except deep percolation) first
        net_flux_before_dp = P_eff + irrigation + CR - ET  # mm/day
        theta_candidate = theta_prev + net_flux_before_dp / (1000.0 * self.z_root)
        theta_candidate = float(np.clip(theta_candidate, 0.0, self.theta_sat))

        # Deep percolation based on candidate moisture (captures same-day excess)
        DP = self.calculate_percolation(theta_candidate=theta_candidate)
        delta_theta_dp = DP / (1000.0 * self.z_root)

        # Final state after percolation
        self.theta = float(np.clip(theta_candidate - delta_theta_dp, 0.0, self.theta_sat))
        net_flux = P_eff + irrigation + CR - ET - DP
        delta_theta = self.theta - theta_prev
        
        # Update relative moisture (Eq. 2)
        self.r = self.theta / self.theta_fc
        
        return {
            'theta_prev': theta_prev,
            'theta': self.theta,
            'r_prev': r_prev,
            'r': self.r,
            'precipitation': precipitation,
            'runoff': RO,
            'p_effective': P_eff,
            'irrigation': irrigation,
            'capillary_rise': CR,
            'et0': et0,
            'kc': kc,
            'etc': ET,
            'deep_percolation': DP,
            'theta_candidate': theta_candidate,
            'net_flux': net_flux,
            'delta_theta': delta_theta,
        }
    
    @property
    def moisture_fc_percent(self) -> float:
        """Current soil moisture as percentage of FC."""
        return self.r * 100.0
    
    @property
    def depletion_fraction(self) -> float:
        """Current depletion as fraction of TAW."""
        return (self.theta_fc - self.theta) / (self.theta_fc - self.theta_wp)
