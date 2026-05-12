"""
Configuration Module
====================
Contains all parameters for soil, crop, and simulation
as defined in the reproducibility Tables 2 and 3.
"""

import os
import pandas as pd


# ============================================================================
# SOIL PARAMETERS (reproducibility custom water-balance layer)
# ============================================================================

SOIL_TYPE = "Clay"
AQUACROP_SOIL_TYPE = "ClayLoam"
THETA_SAT = 0.50          # m³/m³ — Saturasi
THETA_FC = 0.39           # m³/m³ — Field Capacity
THETA_WP = 0.23           # m³/m³ — Wilting Point permanen
TAW = THETA_FC - THETA_WP # m³/m³ — Total Available Water (0.16)
RAW_FRACTION = 0.60       # Readily Available Water fraction
RAW = RAW_FRACTION * TAW  # m³/m³
ROOT_DEPTH = 0.30         # m (300 mm)
KSAT = 5.0                # mm/hari — Saturated Hydraulic Conductivity
CN = 85                   # Curve Number (SCS)
PERCOLATION_RATE = 3.0    # mm/hari — Deep percolation rate


# ============================================================================
# CROP PARAMETERS (Tabel 3 Reproducibility — Paddy Rice)
# ============================================================================

CROP_NAME = "Paddy Rice"
TOTAL_DAYS = 121           # Total growing period (HST 0-120, inklusif)

# Crop coefficients (Kc) per sub-phase
KC_GERMINATION = 1.05      # Perkecambahan
KC_TILLERING = 1.10        # Pertunasan/Anakan
KC_STEM_ELONGATION = 1.15  # Pemanjangan Batang
KC_REPRODUCTIVE = 1.20     # Reproduktif (Pembungaan + Pengisian Biji)
KC_MATURATION_EARLY = 0.95 # Pematangan Awal
KC_MATURATION_LATE = 0.75  # Pematangan Akhir
KC_HARVEST = 0.50          # Panen


# ============================================================================
# FUZZY CONTROLLER PARAMETERS
# ============================================================================

IRRIGATION_EFFICIENCY = 0.80   # η (efisiensi irigasi, tetap 0.8)
INITIAL_MOISTURE_FC = 1.0      # Mulai dari 100% Field Capacity
PHENOLOGY_OVERLAP_DAYS = 5     # overlap fuzzy antar fase (hari)
REACTIVE_THRESHOLD_FC = 0.80   # baseline reaktif: ambang tunggal 80% FC
REACTIVE_TARGET_FC = 1.00      # baseline reaktif: refill hingga 100% FC
WEATHER_COVERAGE_MIN_FRACTION = 0.90  # minimal cakupan cuaca per musim untuk evaluasi reproducibility-facing

# Yield-first tuning (opsional) untuk kontrol fuzzy
YIELD_FOCUS_MODE = True
KEMARAU_TARGET_BOOSTS = {
    "VEGETATIF": {"rL": 0.08, "rU": 0.10},
    "REPRODUKTIF": {"rL": 0.15, "rU": 0.20},
    "PEMASAKAN": {"rL": 0.05, "rU": 0.10},
}


# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

SIMULATION_YEARS = list(range(2010, 2025))  # 15 tahun: 2010–2024

# 3 musim tanam per tahun — tanggal mulai tanam (mm-dd)
# MT-1 (penghujan) lintas tahun: Oktober -> Januari/Februari tahun berikutnya
PLANTING_SEASONS = {
    'MT-1_Penghujan': {'start_month': 10, 'start_day': 1},
    'MT-2_Peralihan': {'start_month': 2,  'start_day': 1},
    'MT-3_Kemarau':   {'start_month': 6,  'start_day': 1},
}

# Total scenarios: 2 algorithms × 10 years × 3 seasons = 60
TOTAL_SCENARIOS = len(SIMULATION_YEARS) * len(PLANTING_SEASONS) * 2

# Location — Kabupaten Malang, Jawa Timur
LATITUDE = -8.0            # degrees
LONGITUDE = 112.6          # degrees
ELEVATION = 500            # meters above sea level


# ============================================================================
# PHENOLOGY DATA
# ============================================================================

def load_phenology_data(filepath: str = None) -> pd.DataFrame:
    """Load phenology growth stage data from CSV."""
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, 'data', 'paddy_growth_phenology.csv')
    
    return pd.read_csv(filepath)


# Phase boundaries (HST) aligned with the reproducibility literature-informed table
PHASE_BOUNDARIES = {
    'Seedling_Establishment':       (0, 15),
    'Early_Tillering':              (16, 35),
    'Active_Tillering':             (36, 45),
    'Panicle_Initiation':           (46, 60),
    'Booting_Early_Reproductive':   (61, 75),
    'Flowering_Anthesis':           (76, 90),
    'Early_Grain_Filling':          (91, 105),
    'Late_Grain_Filling_Ripening':  (106, 120),
}

# Major phase groups for fuzzy input
MAJOR_PHASES = {
    'VEGETATIF':   (0, 45),
    'REPRODUKTIF': (46, 105),
    'PEMASAKAN':   (106, 120),
}


# ============================================================================
# OUTPUT PATHS
# ============================================================================

def get_output_dir():
    """Get the output directory path."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(base_dir, 'output')
    os.makedirs(out, exist_ok=True)
    return out
