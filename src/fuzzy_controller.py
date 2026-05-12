"""
Fuzzy Logic Controller — Mamdani FIS (Persamaan 6–8)
====================================================
Implementasi tanpa dependensi `skfuzzy`.
Metode:
- Fuzzification: trap/tri membership
- Inference: Mamdani Max-Min
- Defuzzification: Centroid (numerical integration)
"""

from __future__ import annotations

import numpy as np

from src.config import THETA_FC, ROOT_DEPTH, IRRIGATION_EFFICIENCY
from src.phenology import compute_dynamic_targets, get_phase_memberships


def _trimf(x, a, b, c):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b != a:
        left = (a < x) & (x < b)
        y[left] = (x[left] - a) / (b - a)
    y[x == b] = 1.0
    if c != b:
        right = (b < x) & (x < c)
        y[right] = (c - x[right]) / (c - b)
    return np.clip(y, 0.0, 1.0)


def _trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b != a:
        rise = (a < x) & (x < b)
        y[rise] = (x[rise] - a) / (b - a)
    y[(b <= x) & (x <= c)] = 1.0
    if d != c:
        fall = (c < x) & (x < d)
        y[fall] = (d - x[fall]) / (d - c)
    return np.clip(y, 0.0, 1.0)


class FuzzyIrrigationController:
    """
    Phenology-aware fuzzy deficit-to-refill controller (single irrigation decision/day).

    Inputs:
    - defisit ternormalisasi terhadap batas bawah target (rL)
    - fase_kritis biner (0=non-kritis, 1=kritis)
    Output:
    - fraksi refill (u) untuk menghitung volume irigasi menuju area dekat rU
    """

    def __init__(
        self,
        efficiency: float = IRRIGATION_EFFICIENCY,
        upper_ref_margin_frac: float = 0.05,
        overshoot_tol_frac: float = 0.10,
        max_irrigation_daily_mm: float = 40.0,
        min_irrigation_event_mm: float = 0.1,
        vegetative_weight: float = 0.45,
        reproductive_weight: float = 1.00,
        maturation_weight: float = 0.20,
        target_lower_offset: float = 0.0,
        target_upper_offset: float = 0.0,
        profile_name: str = "default",
    ):
        self.output_universe = np.linspace(0.0, 1.2, 1201)  # fraksi refill
        self.storage_capacity_mm = 1000.0 * ROOT_DEPTH * THETA_FC
        self.efficiency = float(efficiency)
        self.profile_name = str(profile_name)
        self.vegetative_weight = float(vegetative_weight)
        self.reproductive_weight = float(reproductive_weight)
        self.maturation_weight = float(maturation_weight)
        self.target_lower_offset = float(target_lower_offset)
        self.target_upper_offset = float(target_upper_offset)
        # Yield-oriented refill: target mendekati batas atas, tetapi tetap dibatasi anti-overshoot.
        self.upper_ref_margin_frac = float(upper_ref_margin_frac)  # SM_ref = rU - margin*band
        self.overshoot_tol_frac = float(overshoot_tol_frac)        # izinkan overshoot <= tol*band
        self.max_irrigation_daily_mm = float(max_irrigation_daily_mm)
        self.min_irrigation_event_mm = float(min_irrigation_event_mm)

    # ------------------------------------------------------------------
    # Membership functions (input 1: normalized deficit d_norm in [0, 2])
    # ------------------------------------------------------------------
    @staticmethod
    def mu_defisit_kecil(x):
        return _trapmf(np.atleast_1d(x), 0.0, 0.0, 0.20, 0.55)

    @staticmethod
    def mu_defisit_sedang(x):
        return _trimf(np.atleast_1d(x), 0.25, 0.75, 1.25)

    @staticmethod
    def mu_defisit_besar(x):
        return _trapmf(np.atleast_1d(x), 0.90, 1.20, 2.0, 2.0)

    # ------------------------------------------------------------------
    # Membership functions (input 2: binary critical phase flag)
    # ------------------------------------------------------------------
    @staticmethod
    def mu_phase_non_kritis(x):
        return _trapmf(np.atleast_1d(x), 0.0, 0.0, 0.10, 0.40)

    @staticmethod
    def mu_phase_kritis(x):
        return _trapmf(np.atleast_1d(x), 0.60, 0.90, 1.0, 1.0)

    # ------------------------------------------------------------------
    # Membership functions (output: urgency)
    # ------------------------------------------------------------------
    @staticmethod
    def mu_refill_rendah(x):
        return _trapmf(np.atleast_1d(x), 0.0, 0.0, 0.25, 0.50)

    @staticmethod
    def mu_refill_sedang(x):
        return _trimf(np.atleast_1d(x), 0.35, 0.70, 1.0)

    @staticmethod
    def mu_refill_tinggi(x):
        return _trapmf(np.atleast_1d(x), 0.85, 1.0, 1.2, 1.2)

    def compute_deficit_indicator(self, r_prev: float, rL: float, rU: float) -> float:
        """
        Defisit ternormalisasi terhadap lebar band target:
          d = max(0, rL-r_prev) / max(rU-rL, eps)
        Domain fuzzy dibuka hingga 2 untuk menangkap defisit besar.
        """
        band = max(float(rU) - float(rL), 1e-6)
        d = max(0.0, float(rL) - float(r_prev)) / band
        return float(np.clip(d, 0.0, 2.0))

    def compute_drought_indicator(self, r_prev: float, rL: float, rWP: float, rU: float = None) -> float:
        """
        Backward-compatible alias. Untuk desain baru, indikator mengikuti defisit ke rL
        dan menormalkan terhadap lebar band rU-rL.
        """
        if rU is None:
            rU = max(rL, r_prev)
        return self.compute_deficit_indicator(r_prev=r_prev, rL=rL, rU=rU)

    def compute_phase_critical_flag(self, hst: float) -> float:
        """
        Continuous phase sensitivity in [0, 1] using fuzzy memberships.

        Rationale:
        - Vegetatif: moderate sensitivity (establishment still matters)
        - Reproduktif: highest sensitivity
        - Pemasakan: lower sensitivity

        This keeps the same fuzzy engine while making the phenology input
        actually aware of overlap transitions, instead of collapsing phase to
        a hard binary flag.
        """
        mu = get_phase_memberships(hst)
        veg = float(mu.get("VEGETATIF", 0.0))
        rep = float(mu.get("REPRODUKTIF", 0.0))
        mat = float(mu.get("PEMASAKAN", 0.0))
        sensitivity = (
            self.vegetative_weight * veg
            + self.reproductive_weight * rep
            + self.maturation_weight * mat
        )
        return float(np.clip(sensitivity, 0.0, 1.0))

    def compute_phase_sensitivity(self, hst: float) -> float:
        """Backward-compatible key; now represents continuous phase sensitivity."""
        return self.compute_phase_critical_flag(hst)

    def _fuzzify_inputs(self, drought_indicator: float, phase_sensitivity: float) -> dict:
        d = float(np.clip(drought_indicator, 0.0, 2.0))
        p = float(np.clip(phase_sensitivity, 0.0, 1.0))
        return {
            "d_kecil": float(self.mu_defisit_kecil(d)[0]),
            "d_sedang": float(self.mu_defisit_sedang(d)[0]),
            "d_besar": float(self.mu_defisit_besar(d)[0]),
            "p_non_kritis": float(self.mu_phase_non_kritis(p)[0]),
            "p_kritis": float(self.mu_phase_kritis(p)[0]),
        }

    def compute_urgency(self, drought_indicator: float, phase_sensitivity: float) -> float:
        """
        Output fuzzy baru: fraksi refill u in [0, 1.2]
        Input: defisit ternormalisasi terhadap rL + fase_kritis (0/1).
        """
        m = self._fuzzify_inputs(drought_indicator, phase_sensitivity)
        x = self.output_universe
        agg = np.zeros_like(x)

        # Rule base minimum (2 input x 3 defisit)
        # Non-kritis: kecil->rendah, sedang->sedang, besar->tinggi
        # Kritis: kecil->sedang, sedang->tinggi, besar->tinggi
        rules = [
            (min(m["d_kecil"], m["p_non_kritis"]), "rendah"),
            (min(m["d_sedang"], m["p_non_kritis"]), "sedang"),
            (min(m["d_besar"], m["p_non_kritis"]), "tinggi"),
            (min(m["d_kecil"], m["p_kritis"]), "sedang"),
            (min(m["d_sedang"], m["p_kritis"]), "tinggi"),
            (min(m["d_besar"], m["p_kritis"]), "tinggi"),
        ]

        output_sets = {
            "rendah": self.mu_refill_rendah(x),
            "sedang": self.mu_refill_sedang(x),
            "tinggi": self.mu_refill_tinggi(x),
        }

        # Aggregation (max of clipped consequents)
        for alpha, out_name in rules:
            if alpha <= 0:
                continue
            agg = np.maximum(agg, np.minimum(alpha, output_sets[out_name]))

        area = np.trapezoid(agg, x)
        if area <= 1e-12:
            # Fallback: refill fraksi naik seiring defisit; fase kritis memberi penguat.
            base = 0.25 + 0.55 * np.clip(drought_indicator / 2.0, 0.0, 1.0)
            if phase_sensitivity >= 0.5:
                base += 0.20
            return float(np.clip(base, 0.0, 1.2))
        centroid = np.trapezoid(x * agg, x) / area
        return float(np.clip(centroid, 0.0, 1.2))

    def compute_irrigation(self, r_prev: float, hst: float, rU: float = None, rL: float = None, rWP: float = None) -> dict:
        """
        Fuzzy deficit-to-refill (single decision/day):
        - deadband jika r_prev >= rL
        - fuzzy menentukan fraksi refill menuju area dekat rU
        - volume dikunci agar tidak melampaui SM_up terlalu jauh
        """
        if rU is None or rL is None or rWP is None:
            t = compute_dynamic_targets(hst)
            rU = t["rU"] if rU is None else rU
            rL = t["rL"] if rL is None else rL
            rWP = t["rWP"] if rWP is None else rWP

        # Sensitivity-ready target perturbation for reproducibility robustness checks.
        rL = float(np.clip(float(rL) + self.target_lower_offset, 0.0, 1.20))
        rU = float(np.clip(float(rU) + self.target_upper_offset, 0.0, 1.20))
        if rU < rL:
            rU = rL
        rWP = float(np.clip(float(rWP), 0.0, rL))

        r_prev = float(r_prev)
        band = max(rU - rL, 1e-6)
        phase_sens = self.compute_phase_sensitivity(hst)
        phase_critical_flag = 1 if phase_sens >= 0.5 else 0

        # Deadband: no irrigation while still inside target band (or above lower bound).
        if r_prev >= rL:
            d_i = 0.0
            u_i = 0.0
            irrigation_mm = 0.0
            deficit_low = 0.0
            refill_deficit = max(0.0, (rU - self.upper_ref_margin_frac * band) - r_prev)
            sm_ref = rU - self.upper_ref_margin_frac * band
            sm_cap = rU + self.overshoot_tol_frac * band
        else:
            d_i = self.compute_deficit_indicator(r_prev, rL, rU)
            u_i = self.compute_urgency(d_i, phase_sens)
            deficit_low = max(0.0, rL - r_prev)
            # Yield-oriented refill target near upper bound.
            sm_ref = rU - self.upper_ref_margin_frac * band
            sm_ref = float(np.clip(sm_ref, rL, rU))
            refill_deficit = max(0.0, sm_ref - r_prev)

            irrigation_mm_raw = (self.storage_capacity_mm / self.efficiency) * u_i * refill_deficit

            # Anti-overshoot cap based on upper target + tolerance band.
            sm_cap = rU + self.overshoot_tol_frac * band
            max_def_to_cap = max(0.0, sm_cap - r_prev)
            irrigation_cap_mm = (self.storage_capacity_mm / self.efficiency) * max_def_to_cap
            irrigation_mm = min(irrigation_mm_raw, irrigation_cap_mm, self.max_irrigation_daily_mm)
            if irrigation_mm < self.min_irrigation_event_mm:
                irrigation_mm = 0.0

        return {
            "drought_indicator": d_i,
            "phase_sensitivity": float(phase_sens),
            "urgency": u_i,
            "irrigation_mm": float(max(irrigation_mm, 0.0)),
            "rL": float(rL),
            "rU": float(rU),
            "rWP": float(rWP),
            "r_prev": float(r_prev),
            "deficit": float(deficit_low),
            "deficit_to_low": float(deficit_low),
            "refill_deficit": float(refill_deficit),
            "phase_critical": int(phase_critical_flag),
            "sm_ref": float(sm_ref),
            "sm_cap": float(sm_cap),
            "storage_capacity": float(self.storage_capacity_mm),
            "efficiency": float(self.efficiency),
        }


def visualize_fis(controller: FuzzyIrrigationController = None, save_path: str = "output/fuzzy_mf_visualization.png"):
    """Plot fuzzy membership functions for documentation/reporting."""
    import matplotlib.pyplot as plt

    controller = controller or FuzzyIrrigationController()
    x_def = np.linspace(0, 2, 1001)
    x_bin = np.linspace(0, 1, 1001)
    x_out = controller.output_universe

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Defisit normalized to lower target band
    axes[0].plot(x_def, controller.mu_defisit_kecil(x_def), lw=2, label="Kecil")
    axes[0].plot(x_def, controller.mu_defisit_sedang(x_def), lw=2, label="Sedang")
    axes[0].plot(x_def, controller.mu_defisit_besar(x_def), lw=2, label="Besar")
    axes[0].set_title("Input 1: Defisit Ternormalisasi terhadap rL")
    axes[0].set_xlabel("d_norm")
    axes[0].set_ylabel("Membership")
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Critical phase flag
    axes[1].plot(x_bin, controller.mu_phase_non_kritis(x_bin), lw=2, label="Non-Kritis")
    axes[1].plot(x_bin, controller.mu_phase_kritis(x_bin), lw=2, label="Kritis")
    axes[1].set_title("Input 2: Fase Kritis (0/1)")
    axes[1].set_xlabel("fase_kritis")
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Output refill fraction
    axes[2].plot(x_out, controller.mu_refill_rendah(x_out), lw=2, label="Rendah")
    axes[2].plot(x_out, controller.mu_refill_sedang(x_out), lw=2, label="Sedang")
    axes[2].plot(x_out, controller.mu_refill_tinggi(x_out), lw=2, label="Tinggi")
    axes[2].set_title("Output: Fraksi Refill Irigasi (u_i)")
    axes[2].set_xlabel("u_i (fraksi refill)")
    axes[2].set_ylim(-0.02, 1.05)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
