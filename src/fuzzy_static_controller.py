from __future__ import annotations

import numpy as np

from src.config import IRRIGATION_EFFICIENCY, ROOT_DEPTH, THETA_FC
from src.fuzzy_controller import FuzzyIrrigationController


class FuzzyStaticIrrigationController:
    def __init__(
        self,
        efficiency: float = IRRIGATION_EFFICIENCY,
        static_rL: float = 0.65,
        static_rU: float = 0.75,
        static_rWP: float = 0.35,
    ):
        self.efficiency = float(efficiency)
        self.static_rL = float(static_rL)
        self.static_rU = float(static_rU)
        self.static_rWP = float(static_rWP)
        self.storage_capacity_mm = 1000.0 * ROOT_DEPTH * THETA_FC
        self.fis = FuzzyIrrigationController(efficiency=efficiency)

    def compute_irrigation(
        self,
        r_prev: float,
        hst: float,
        rU: float = None,
        rL: float = None,
        rWP: float = None,
    ) -> dict:
        r_prev = float(r_prev)
        rL = self.static_rL if rL is None else float(rL)
        rU = self.static_rU if rU is None else float(rU)
        rWP = self.static_rWP if rWP is None else float(rWP)

        if r_prev >= rL:
            return {
                "drought_indicator": 0.0,
                "phase_sensitivity": 0.0,
                "phase_critical": 0,
                "urgency": 0.0,
                "irrigation_mm": 0.0,
                "rL": rL,
                "rU": rU,
                "rWP": rWP,
                "r_prev": r_prev,
                "deficit": 0.0,
                "storage_capacity": self.storage_capacity_mm,
            }

        d_i = self.fis.compute_deficit_indicator(r_prev=r_prev, rL=rL, rU=rU)
        u_i = self.fis.compute_urgency(drought_indicator=d_i, phase_sensitivity=0.0)
        refill_deficit = max(0.0, rU - r_prev)
        irrigation_mm = (self.storage_capacity_mm / self.efficiency) * u_i * refill_deficit
        irrigation_mm = float(np.clip(irrigation_mm, 0.0, self.fis.max_irrigation_daily_mm))

        return {
            "drought_indicator": float(d_i),
            "phase_sensitivity": 0.0,
            "phase_critical": 0,
            "urgency": float(u_i),
            "irrigation_mm": irrigation_mm,
            "rL": rL,
            "rU": rU,
            "rWP": rWP,
            "r_prev": r_prev,
            "deficit": max(0.0, rL - r_prev),
            "storage_capacity": self.storage_capacity_mm,
        }
