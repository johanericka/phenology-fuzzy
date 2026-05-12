"""
Data Cleansing Module for Weather Data
=======================================
Reads and validates BMKG weather data from cuaca-complete.txt,
performs quality checks, handles missing values, and prepares
data for AquaCrop-OSPy simulation.
"""

import pandas as pd
import numpy as np
import warnings
import os


def load_weather_data(filepath: str) -> pd.DataFrame:
    """
    Load weather data from BMKG text file format.
    
    Expected columns: Day, Month, Year, Tmin(C), Tmax(C), Prcp(mm),
                      RH(%), SS(hrs), Wind(m/s), Et0(mm)
    """
    col_names = [
        'Day', 'Month', 'Year', 'Tmin', 'Tmax', 'Prcp',
        'RH', 'SS', 'Wind', 'Et0'
    ]
    
    df = pd.read_csv(
        filepath,
        sep=r'\s+',
        skiprows=1,  # skip header line
        names=col_names,
        na_values=['-', 'NA', 'NaN', '', '8888', '9999']
    )
    
    # Create date column
    df['Date'] = pd.to_datetime(
        df[['Year', 'Month', 'Day']].astype(int),
        format='%Y%m%d',
        errors='coerce'
    )
    
    return df


def validate_ranges(df: pd.DataFrame) -> dict:
    """
    Validate that weather variables are within physically plausible ranges.
    Returns a dict with counts of out-of-range values per variable.
    """
    range_checks = {
        'Tmin': (-10, 45),    # °C
        'Tmax': (-5, 50),     # °C
        'Prcp': (0, 500),     # mm/day
        'RH': (0, 100),       # %
        'SS': (0, 14),        # hours of sunshine
        'Wind': (0, 30),      # m/s
        'Et0': (0, 15),       # mm/day
    }
    
    issues = {}
    for col, (lo, hi) in range_checks.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            n_bad = mask.sum()
            if n_bad > 0:
                issues[col] = {
                    'count': int(n_bad),
                    'min_found': float(df[col].min()),
                    'max_found': float(df[col].max())
                }
                # Clip to valid range
                df[col] = df[col].clip(lo, hi)
    
    return issues


def check_temperature_consistency(df: pd.DataFrame) -> int:
    """
    Ensure Tmin <= Tmax. Fix by swapping where violated.
    Returns number of rows fixed.
    """
    mask = df['Tmin'] > df['Tmax']
    n_fixed = mask.sum()
    if n_fixed > 0:
        df.loc[mask, ['Tmin', 'Tmax']] = df.loc[mask, ['Tmax', 'Tmin']].values
    return int(n_fixed)


def handle_missing_values(df: pd.DataFrame) -> dict:
    """
    Handle missing values using interpolation for continuous variables.
    Returns dict with counts of filled values per column.
    """
    fill_counts = {}
    numeric_cols = ['Tmin', 'Tmax', 'Prcp', 'RH', 'SS', 'Wind', 'Et0']
    
    for col in numeric_cols:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                if col == 'Prcp':
                    # For precipitation, fill missing with 0 (most conservative)
                    df[col] = df[col].fillna(0.0)
                else:
                    # For other variables, use linear interpolation
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
                fill_counts[col] = int(n_missing)
    
    return fill_counts


def filter_years(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Filter data to specified year range (inclusive)."""
    return df[(df['Year'] >= start_year) & (df['Year'] <= end_year)].copy()


def compute_et0_penman_monteith(row, latitude_deg: float = -8.0):
    """
    Compute reference evapotranspiration (ET0) using FAO-56 Penman-Monteith.
    
    Parameters:
        row: DataFrame row with Tmin, Tmax, RH, SS, Wind
        latitude_deg: latitude in degrees (default -8.0 for Malang, East Java)
    
    Returns:
        ET0 in mm/day
    """
    # Constants
    Tmin = row['Tmin']
    Tmax = row['Tmax']
    Tmean = (Tmin + Tmax) / 2.0
    RH = row['RH']
    n = row['SS']       # actual sunshine hours
    u2 = row['Wind']    # wind speed at 2m (m/s)
    
    # Day of year
    doy = row['Date'].timetuple().tm_yday
    
    # Psychrometric constant (kPa/°C) at ~500m elevation
    P = 101.3 * ((293 - 0.0065 * 500) / 293) ** 5.26
    gamma = 0.000665 * P
    
    # Saturation vapor pressure
    e_Tmin = 0.6108 * np.exp(17.27 * Tmin / (Tmin + 237.3))
    e_Tmax = 0.6108 * np.exp(17.27 * Tmax / (Tmax + 237.3))
    es = (e_Tmin + e_Tmax) / 2.0
    
    # Actual vapor pressure
    ea = es * RH / 100.0
    
    # Slope of saturation vapor pressure curve
    delta = 4098 * (0.6108 * np.exp(17.27 * Tmean / (Tmean + 237.3))) / (Tmean + 237.3) ** 2
    
    # Extraterrestrial radiation
    lat = np.radians(latitude_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    solar_dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(-np.tan(lat) * np.tan(solar_dec))
    
    Gsc = 0.0820  # MJ m⁻² min⁻¹
    Ra = (24 * 60 / np.pi) * Gsc * dr * (
        ws * np.sin(lat) * np.sin(solar_dec) +
        np.cos(lat) * np.cos(solar_dec) * np.sin(ws)
    )
    
    # Daylight hours
    N = 24 * ws / np.pi
    
    # Solar radiation (Angstrom)
    Rs = (0.25 + 0.50 * n / max(N, 0.1)) * Ra
    
    # Clear-sky radiation
    Rso = (0.75 + 2e-5 * 500) * Ra
    
    # Net shortwave
    Rns = (1 - 0.23) * Rs
    
    # Net longwave
    sigma = 4.903e-9  # MJ m⁻² day⁻¹ K⁻⁴
    Rnl = sigma * ((Tmax + 273.16) ** 4 + (Tmin + 273.16) ** 4) / 2 * \
          (0.34 - 0.14 * np.sqrt(ea)) * \
          (1.35 * Rs / max(Rso, 0.1) - 0.35)
    
    # Net radiation
    Rn = Rns - Rnl
    
    # Soil heat flux (daily ~ 0)
    G = 0.0
    
    # FAO-56 Penman-Monteith
    numerator = 0.408 * delta * (Rn - G) + gamma * (900 / (Tmean + 273)) * u2 * (es - ea)
    denominator = delta + gamma * (1 + 0.34 * u2)
    
    ET0 = numerator / denominator
    return max(ET0, 0.0)


def cleanse_weather_data(filepath: str, start_year: int = 2015, end_year: int = 2024,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Main data cleansing pipeline.
    
    Args:
        filepath: Path to cuaca-complete.txt
        start_year: Start year filter
        end_year: End year filter
        verbose: Print cleansing report
    
    Returns:
        Cleaned DataFrame ready for simulation
    """
    if verbose:
        print("=" * 60)
        print("DATA CLEANSING REPORT")
        print("=" * 60)
    
    # 1. Load
    df = load_weather_data(filepath)
    if verbose:
        print(f"\n1. Loaded {len(df)} rows, date range: "
              f"{df['Year'].min()}-{df['Year'].max()}")
    
    # 2. Filter years
    df = filter_years(df, start_year, end_year)
    if verbose:
        print(f"2. Filtered to {start_year}-{end_year}: {len(df)} rows")
    
    # 3. Handle missing values
    fill_report = handle_missing_values(df)
    if verbose:
        print(f"3. Missing values filled: {fill_report if fill_report else 'None'}")
    
    # 4. Validate ranges
    range_report = validate_ranges(df)
    if verbose:
        if range_report:
            print(f"4. Out-of-range values clipped:")
            for col, info in range_report.items():
                print(f"   - {col}: {info['count']} values "
                      f"(range found: {info['min_found']:.1f} to {info['max_found']:.1f})")
        else:
            print(f"4. All values within valid ranges")
    
    # 5. Temperature consistency
    n_temp_fixed = check_temperature_consistency(df)
    if verbose:
        print(f"5. Temperature consistency: {n_temp_fixed} rows fixed (Tmin > Tmax)")
    
    # 6. Verify ET0
    if verbose:
        et0_stats = df['Et0'].describe()
        print(f"6. ET0 statistics: min={et0_stats['min']:.2f}, "
              f"mean={et0_stats['mean']:.2f}, max={et0_stats['max']:.2f} mm/day")
    
    # 7. Check date continuity
    df = df.sort_values('Date').reset_index(drop=True)
    date_diff = df['Date'].diff().dt.days
    gaps = date_diff[date_diff > 1]
    if verbose:
        if len(gaps) > 0:
            print(f"7. Date gaps found: {len(gaps)} (dates with missing days)")
        else:
            print(f"7. Date continuity: OK (no gaps)")
    
    if verbose:
        print(f"\nFinal dataset: {len(df)} rows, "
              f"{df['Date'].min().strftime('%Y-%m-%d')} to "
              f"{df['Date'].max().strftime('%Y-%m-%d')}")
        print("=" * 60)
    
    return df


def prepare_aquacrop_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare weather data in the format expected by AquaCrop-OSPy.
    
    AquaCrop expects: MinTemp, MaxTemp, Precipitation, 
                      ReferenceET, Date
    """
    ac_df = pd.DataFrame({
        'MinTemp': df['Tmin'],
        'MaxTemp': df['Tmax'],
        'Precipitation': df['Prcp'],
        'ReferenceET': df['Et0'],
        'Date': df['Date']
    })
    return ac_df


if __name__ == "__main__":
    # Run cleansing on project data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weather_file = os.path.join(base_dir, 'data', 'cuaca-complete.txt')
    
    if os.path.exists(weather_file):
        df = cleanse_weather_data(weather_file, start_year=2015, end_year=2024)
        
        # Save cleaned data
        output_file = os.path.join(base_dir, 'data', 'cuaca-cleaned.csv')
        df.to_csv(output_file, index=False)
        print(f"\nCleaned data saved to: {output_file}")
    else:
        print(f"Weather file not found: {weather_file}")
