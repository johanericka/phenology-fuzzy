from __future__ import annotations

from src.config import IRRIGATION_EFFICIENCY, ROOT_DEPTH, THETA_FC


class ReactivePhenologyController:
    def __init__(self, efficiency: float = IRRIGATION_EFFICIENCY):
        self.efficiency = float(efficiency)
        self.storage_capacity_mm = 1000.0 * ROOT_DEPTH * THETA_FC

    def compute_irrigation(
        self,
        r_prev: float,
        hst: float,
        rU: float = None,
        rL: float = None,
        rWP: float = None,
    ) -> dict:
        r_prev = float(r_prev)
        rL = 0.80 if rL is None else float(rL)
        rU = 1.00 if rU is None else float(rU)
        rWP = 0.30 if rWP is None else float(rWP)

        if r_prev < rL:
            deficit = max(0.0, rU - r_prev)
            irrigation_mm = (self.storage_capacity_mm / self.efficiency) * deficit
            urgency = 1.0
        else:
            deficit = 0.0
            irrigation_mm = 0.0
            urgency = 0.0

        return {
            "drought_indicator": max(0.0, (rL - r_prev) / max(rL, 1e-9)),
            "phase_sensitivity": 1.0,
            "phase_critical": 1,
            "urgency": float(urgency),
            "irrigation_mm": float(max(irrigation_mm, 0.0)),
            "rL": rL,
            "rU": rU,
            "rWP": rWP,
            "r_prev": r_prev,
            "deficit": float(deficit),
            "storage_capacity": self.storage_capacity_mm,
        }
