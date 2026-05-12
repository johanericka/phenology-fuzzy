"""
Reactive Controller (Baseline Comparison — FASE 6)
====================================================
Simple threshold-based irrigation controller WITHOUT phenology awareness.
Uses a single static threshold throughout the entire growing season.
This serves as the baseline comparison for the fuzzy-phenology algorithm.
"""

import numpy as np
from src.config import THETA_FC, ROOT_DEPTH, IRRIGATION_EFFICIENCY


class ReactiveController:
    """
    Static threshold-based irrigation controller.
    
    Logic: If soil moisture drops below a fixed threshold,
    irrigate to bring it back up to a fixed target.
    No consideration of growth phase or phenology.
    """
    
    def __init__(self, threshold_fc: float = 0.80, target_fc: float = 1.00,
                 efficiency: float = None):
        """
        Args:
            threshold_fc: Trigger irrigation when r < threshold (fraction of FC)
            target_fc: Irrigate up to this level (fraction of FC)
            efficiency: Irrigation efficiency η (default from config)
        """
        self.threshold = threshold_fc
        self.target = target_fc
        self.efficiency = efficiency or IRRIGATION_EFFICIENCY
    
    def compute_irrigation(self, r_prev: float, hst: float = None,
                           **kwargs) -> dict:
        """
        Compute irrigation decision using static threshold.
        
        Args:
            r_prev: Current soil moisture as fraction of FC
            hst: Days after planting (not used, kept for interface compatibility)
            
        Returns:
            Dict with irrigation decision details
        """
        if r_prev < self.threshold:
            # Need irrigation — bring moisture to target
            deficit = max(0.0, self.target - r_prev)
            storage_capacity = 1000 * ROOT_DEPTH * THETA_FC  # mm
            irrigation_mm = (storage_capacity / self.efficiency) * deficit
            urgency = 1.0
        else:
            deficit = 0.0
            irrigation_mm = 0.0
            urgency = 0.0
        
        return {
            'drought_indicator': max(0, (self.threshold - r_prev) / self.threshold),
            'phase_sensitivity': 0.5,  # Not used but kept for compatibility
            'urgency': urgency,
            'irrigation_mm': irrigation_mm,
            'rL': self.threshold,
            'rU': self.target,
            'rWP': 0.0,
            'r_prev': r_prev,
            'deficit': deficit,
            'storage_capacity': 1000 * ROOT_DEPTH * THETA_FC,
        }
