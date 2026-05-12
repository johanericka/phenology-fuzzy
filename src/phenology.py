"""
Phenology Module (Persamaan 4–5)
================================
Sumber parameter fase dan target kelembapan diambil dari:
`data/paddy_growth_phenology.csv` (8 sub-fase padi, HST 0-120).

Modul ini:
1. Membaca parameter fase dari CSV
2. Membentuk derajat keanggotaan fuzzy μ_k(i) untuk 8 sub-fase
3. Menghitung target dinamis rL, rU, rWP (Pers. 4–5)
4. Menghasilkan Kc dinamis berbasis pembobotan fuzzy
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import TOTAL_DAYS, PHENOLOGY_OVERLAP_DAYS, load_phenology_data


PHASE_COLUMNS = {
    "hst": "HST",
    "major": "Fase_Utama",
    "sub": "Sub_Fase",
    "rL_pct": "Target_SM_Bawah_pct",
    "rU_pct": "Target_SM_Atas_pct",
    "rWP_pct": "Wilting_Point_pct",
    "kc": "Kc",
}


def _clean_phase_key(name: str) -> str:
    return str(name).strip().replace(" ", "_")


@lru_cache(maxsize=1)
def get_phenology_df() -> pd.DataFrame:
    """Load and normalize daily phenology CSV."""
    df = load_phenology_data().copy()
    df = df.rename(columns={v: k for k, v in PHASE_COLUMNS.items()})
    required = ["hst", "major", "sub", "rL_pct", "rU_pct", "rWP_pct", "kc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom fenologi tidak lengkap: {missing}")

    df["hst"] = df["hst"].astype(int)
    df["major"] = df["major"].astype(str).str.strip().str.upper()
    df["sub"] = df["sub"].map(_clean_phase_key)
    for col in ["rL_pct", "rU_pct", "rWP_pct", "kc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("hst").reset_index(drop=True)
    return df


@lru_cache(maxsize=1)
def get_subphase_table() -> pd.DataFrame:
    """
    Build one row per sub-phase with HST interval and parameter values.
    Assumes each sub-phase is contiguous in the CSV.
    """
    df = get_phenology_df()
    rows: List[dict] = []

    for sub_name, group in df.groupby("sub", sort=False):
        g = group.sort_values("hst")
        rows.append(
            {
                "sub": sub_name,
                "major": g["major"].iloc[0],
                "start_hst": int(g["hst"].min()),
                "end_hst": int(g["hst"].max()),
                "rL": float(g["rL_pct"].iloc[0]) / 100.0,
                "rU": float(g["rU_pct"].iloc[0]) / 100.0,
                "rWP": float(g["rWP_pct"].iloc[0]) / 100.0,
                "kc": float(g["kc"].iloc[0]),
            }
        )

    table = pd.DataFrame(rows).sort_values("start_hst").reset_index(drop=True)
    table["phase_index"] = np.arange(len(table))
    return table


@lru_cache(maxsize=1)
def get_major_phase_order() -> List[str]:
    df = get_subphase_table()
    ordered = []
    for major in df["major"]:
        if major not in ordered:
            ordered.append(major)
    return ordered


RUNTIME_PHENOLOGY_OVERLAP_DAYS: int | None = None


def set_runtime_overlap_days(overlap_days: int | None) -> None:
    """Allow temporary runtime override for fuzzy overlap width."""
    global RUNTIME_PHENOLOGY_OVERLAP_DAYS
    RUNTIME_PHENOLOGY_OVERLAP_DAYS = None if overlap_days is None else int(overlap_days)


def get_effective_overlap_days(overlap_days: int | None = None) -> int:
    """Resolve overlap precedence: explicit argument, runtime override, then config."""
    if overlap_days is not None:
        return max(1, int(overlap_days))
    if RUNTIME_PHENOLOGY_OVERLAP_DAYS is not None:
        return max(1, int(RUNTIME_PHENOLOGY_OVERLAP_DAYS))
    return max(1, int(PHENOLOGY_OVERLAP_DAYS))


def _phase_membership_from_boundaries(
    hst: float,
    idx: int,
    phases: pd.DataFrame,
    overlap_days: int | None = None,
) -> float:
    """
    Linear blend with overlap at each transition.

    Interpretasi implementasi:
    - `overlap_days = 5` berarti fase sebelumnya dan fase berikutnya
      beririsan selama 5 hari menjelang batas awal fase berikutnya.
    """
    row = phases.iloc[idx]
    s = float(row["start_hst"])
    e = float(row["end_hst"])
    x = float(hst)
    w = get_effective_overlap_days(overlap_days)

    # Outside global support of this phase.
    if x < s - w or x > e:
        # Exception: last phase keeps support until end_hst only (no extension after end)
        return 0.0

    prev_boundary = None if idx == 0 else float(phases.iloc[idx]["start_hst"])
    next_boundary = None if idx == len(phases) - 1 else float(phases.iloc[idx + 1]["start_hst"])

    # Left side (rising from previous phase)
    if idx == 0:
        if x < s:
            left_mu = 0.0
        else:
            left_mu = 1.0
    else:
        a = prev_boundary - w
        b = prev_boundary
        if x <= a:
            left_mu = 0.0
        elif x >= b:
            left_mu = 1.0
        else:
            left_mu = (x - a) / max(b - a, 1e-9)

    # Right side (falling to next phase)
    if idx == len(phases) - 1:
        if x > e:
            right_mu = 0.0
        else:
            right_mu = 1.0
    else:
        c = next_boundary - w
        d = next_boundary
        if x <= c:
            right_mu = 1.0
        elif x >= d:
            right_mu = 0.0
        else:
            right_mu = (d - x) / max(d - c, 1e-9)

    return float(np.clip(min(left_mu, right_mu), 0.0, 1.0))


def get_subphase_memberships(hst: float, overlap_days: int | None = None) -> Dict[str, float]:
    """Return μ_k(i) for all configured sub-phases."""
    phases = get_subphase_table()
    memberships = {}
    for idx, row in phases.iterrows():
        memberships[row["sub"]] = _phase_membership_from_boundaries(
            hst=hst, idx=int(idx), phases=phases, overlap_days=overlap_days
        )
    return memberships


def get_phase_memberships(hst: float, overlap_days: int | None = None) -> Dict[str, float]:
    """
    Return memberships for 3 major groups (VEGETATIF/REPRODUKTIF/PEMASAKAN)
    by taking max membership among sub-fases in the same group.
    """
    phases = get_subphase_table()
    sub_mu = get_subphase_memberships(hst, overlap_days=overlap_days)

    out = {major: 0.0 for major in get_major_phase_order()}
    for _, row in phases.iterrows():
        out[row["major"]] = max(out[row["major"]], sub_mu[row["sub"]])
    return out


def get_dominant_phase(hst: float, overlap_days: int | None = None) -> str:
    """Dominant major phase at given HST."""
    memberships = get_phase_memberships(hst, overlap_days=overlap_days)
    return max(memberships, key=memberships.get)


def get_dominant_subphase(hst: float, overlap_days: int | None = None) -> str:
    """Dominant sub-phase at given HST."""
    memberships = get_subphase_memberships(hst, overlap_days=overlap_days)
    return max(memberships, key=memberships.get)


def compute_dynamic_targets(hst: float, overlap_days: int | None = None) -> dict:
    """
    Persamaan (4) dan (5):
      rL(i) = Σ μ_k(i) rL,k / Σ μ_k(i)
      rU(i) = Σ μ_k(i) rU,k / Σ μ_k(i)
      rWP(i)= Σ μ_k(i) rWP,k / Σ μ_k(i)
    """
    phases = get_subphase_table()
    mu = get_subphase_memberships(hst, overlap_days=overlap_days)
    mu_sum = float(sum(mu.values()))

    if mu_sum <= 1e-12:
        # Fallback ke fase pertama
        row0 = phases.iloc[0]
        return {"rL": float(row0["rL"]), "rU": float(row0["rU"]), "rWP": float(row0["rWP"])}

    rL = 0.0
    rU = 0.0
    rWP = 0.0
    for _, row in phases.iterrows():
        w = mu[row["sub"]]
        rL += w * float(row["rL"])
        rU += w * float(row["rU"])
        rWP += w * float(row["rWP"])

    return {"rL": rL / mu_sum, "rU": rU / mu_sum, "rWP": rWP / mu_sum}


def get_kc_for_hst(hst: float, overlap_days: int | None = None) -> float:
    """Weighted Kc from 7-subphase memberships."""
    phases = get_subphase_table()
    mu = get_subphase_memberships(hst, overlap_days=overlap_days)
    mu_sum = float(sum(mu.values()))
    if mu_sum <= 1e-12:
        return float(phases.iloc[0]["kc"])
    kc = sum(mu[row["sub"]] * float(row["kc"]) for _, row in phases.iterrows()) / mu_sum
    return float(kc)


def get_daily_reference_row(hst: int) -> dict:
    """Return the exact (unsmoothed) daily row from CSV for a given HST."""
    df = get_phenology_df()
    row = df[df["hst"] == int(hst)]
    if row.empty:
        raise KeyError(f"HST {hst} tidak ditemukan pada phenology CSV")
    r = row.iloc[0]
    return {
        "hst": int(r["hst"]),
        "major": str(r["major"]),
        "sub": str(r["sub"]),
        "rL": float(r["rL_pct"]) / 100.0,
        "rU": float(r["rU_pct"]) / 100.0,
        "rWP": float(r["rWP_pct"]) / 100.0,
        "kc": float(r["kc"]),
    }


def compute_all_daily_targets(total_days: int = None, overlap_days: int | None = None) -> list:
    """Precompute daily fuzzy-smoothed targets for the whole season."""
    total_days = TOTAL_DAYS if total_days is None else int(total_days)
    records = []
    for hst in range(total_days):
        targets = compute_dynamic_targets(hst, overlap_days=overlap_days)
        major_mu = get_phase_memberships(hst, overlap_days=overlap_days)
        sub_mu = get_subphase_memberships(hst, overlap_days=overlap_days)
        records.append(
            {
                "hst": hst,
                "rL": targets["rL"],
                "rU": targets["rU"],
                "rWP": targets["rWP"],
                "kc": get_kc_for_hst(hst, overlap_days=overlap_days),
                "dominant_phase": get_dominant_phase(hst),
                "dominant_subphase": get_dominant_subphase(hst),
                **{f"mu_{k.lower()}": v for k, v in major_mu.items()},
                **{f"mu_sub_{k.lower()}": v for k, v in sub_mu.items()},
            }
        )
    return records


def plot_phenology_targets(save_path: str = "output/phenology_targets.png") -> None:
    """Visualize memberships, dynamic targets, and Kc."""
    import matplotlib.pyplot as plt

    data = compute_all_daily_targets()
    df = pd.DataFrame(data)
    h = df["hst"].values

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Major phase memberships
    for major in get_major_phase_order():
        col = f"mu_{major.lower()}"
        if col in df:
            axes[0].plot(h, df[col], linewidth=2, label=major.title())
    axes[0].set_ylabel("Derajat Keanggotaan")
    axes[0].set_title("Keanggotaan Fuzzy Fase Fenologi (Agregat 3 Kelompok)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dynamic targets
    axes[1].fill_between(h, df["rL"] * 100, df["rU"] * 100, alpha=0.25, color="green", label="Rentang Target")
    axes[1].plot(h, df["rL"] * 100, "g--", linewidth=1.5, label="rL")
    axes[1].plot(h, df["rU"] * 100, "g-", linewidth=1.5, label="rU")
    axes[1].plot(h, df["rWP"] * 100, "r-", linewidth=1.2, label="rWP")
    axes[1].set_ylabel("% FC")
    axes[1].set_title("Target Dinamis Kelembapan Tanah (Pers. 4–5)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Kc
    axes[2].plot(h, df["kc"], color="navy", linewidth=2)
    axes[2].set_ylabel("Kc")
    axes[2].set_xlabel("Hari Setelah Tanam (HST)")
    axes[2].set_title("Kc Dinamis Berbasis Fase (CSV + Smoothing)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_phenology_targets()
