"""
utils/data.py
Data loading, preprocessing, city geospatial metadata, and synthetic fallback generator.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# 26 Cities Coordinates (Latitude, Longitude)
CITY_COORDINATES = {
    'Ahmedabad': (23.0225, 72.5714),
    'Aizawl': (23.7271, 92.7176),
    'Amaravati': (16.5417, 80.5158),
    'Amritsar': (31.6340, 74.8723),
    'Bengaluru': (12.9716, 77.5946),
    'Bhopal': (23.2599, 77.4126),
    'Brajrajnagar': (21.8234, 83.9189),
    'Chandigarh': (30.7333, 76.7794),
    'Chennai': (13.0827, 80.2707),
    'Coimbatore': (11.0168, 76.9558),
    'Delhi': (28.6139, 77.2090),
    'Ernakulam': (9.9816, 76.2999),
    'Gurugram': (28.4595, 77.0266),
    'Guwahati': (26.1445, 91.7362),
    'Hyderabad': (17.3850, 78.4867),
    'Jaipur': (26.9124, 75.7873),
    'Jorapokhar': (23.7058, 86.4131),
    'Kochi': (9.9312, 76.2673),
    'Kolkata': (22.5726, 88.3639),
    'Lucknow': (26.8467, 80.9462),
    'Mumbai': (19.0760, 72.8777),
    'Patna': (25.5941, 85.1376),
    'Shillong': (25.5788, 91.8933),
    'Talcher': (20.9509, 85.2167),
    'Thiruvananthapuram': (8.5241, 76.9366),
    'Visakhapatnam': (17.6868, 83.2185),
}

# City to Region Mapping (4 broad ecological & geographic zones)
CITY_REGIONS = {
    # Indo-Gangetic Plain / North (7 cities)
    'Amritsar': 'Indo-Gangetic Plain / North',
    'Chandigarh': 'Indo-Gangetic Plain / North',
    'Delhi': 'Indo-Gangetic Plain / North',
    'Gurugram': 'Indo-Gangetic Plain / North',
    'Jaipur': 'Indo-Gangetic Plain / North',
    'Lucknow': 'Indo-Gangetic Plain / North',
    'Patna': 'Indo-Gangetic Plain / North',
    # Central & West (3 cities)
    'Ahmedabad': 'Central & West',
    'Bhopal': 'Central & West',
    'Mumbai': 'Central & West',
    # South (9 cities)
    'Amaravati': 'South',
    'Bengaluru': 'South',
    'Chennai': 'South',
    'Coimbatore': 'South',
    'Ernakulam': 'South',
    'Hyderabad': 'South',
    'Kochi': 'South',
    'Thiruvananthapuram': 'South',
    'Visakhapatnam': 'South',
    # East & Northeast (7 cities)
    'Aizawl': 'East & Northeast',
    'Brajrajnagar': 'East & Northeast',
    'Guwahati': 'East & Northeast',
    'Jorapokhar': 'East & Northeast',
    'Kolkata': 'East & Northeast',
    'Shillong': 'East & Northeast',
    'Talcher': 'East & Northeast',
}

# Standard Indian National AQI bands and colors (CPCB Standard)
AQI_CATEGORIES = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']

AQI_COLORS = {
    'Good': '#009666',          # Deep emerald green
    'Satisfactory': '#84CF32',  # Light lime green
    'Moderate': '#FACF39',      # Warm yellow / amber
    'Poor': '#F29C38',          # Orange
    'Very Poor': '#E93F33',     # Bright red
    'Severe': '#822327',        # Deep maroon
    'Unknown': '#9E9E9E',       # Neutral grey
}

# Plotly Continuous Color Scale corresponding to AQI bands
AQI_CONTINUOUS_COLORSCALE = [
    [0.00, '#009666'],  # Good (0-50)
    [0.12, '#84CF32'],  # Satisfactory (51-100)
    [0.28, '#FACF39'],  # Moderate (101-200)
    [0.48, '#F29C38'],  # Poor (201-300)
    [0.72, '#E93F33'],  # Very Poor (301-400)
    [1.00, '#822327'],  # Severe (401+)
]

POLLUTANT_UNITS = {
    'PM2.5': 'µg/m³',
    'PM10': 'µg/m³',
    'NO2': 'µg/m³',
    'AQI': 'Index points',
}

def get_season(month: int) -> str:
    """Map month (1-12) to Indian meteorological seasons."""
    if month in (12, 1, 2):
        return 'Winter'
    elif month in (3, 4, 5):
        return 'Summer'
    elif month in (6, 7, 8, 9):
        return 'Monsoon'
    else:
        return 'Post-monsoon'

def compute_aqi_bucket(aqi_val: float) -> str:
    """Indian National AQI band classifier (CPCB standards)."""
    if pd.isna(aqi_val):
        return np.nan
    if aqi_val <= 50:
        return 'Good'
    elif aqi_val <= 100:
        return 'Satisfactory'
    elif aqi_val <= 200:
        return 'Moderate'
    elif aqi_val <= 300:
        return 'Poor'
    elif aqi_val <= 400:
        return 'Very Poor'
    else:
        return 'Severe'

def generate_synthetic_air_quality_data() -> pd.DataFrame:
    """
    Generate realistic synthetic daily air quality data for all 26 cities
    spanning 2018-01-01 to 2020-07-01 if real data is missing.
    Mimics Indo-Gangetic winter spikes, monsoon wash-out, and 2020 lockdown drop.
    """
    np.random.seed(42)
    dates = pd.date_range(start='2018-01-01', end='2020-07-01', freq='D')
    records = []

    for city, (lat, lon) in CITY_COORDINATES.items():
        region = CITY_REGIONS.get(city, 'Other')
        # Base regional baseline
        if region == 'Indo-Gangetic Plain / North':
            base_pm25 = 85.0
        elif region == 'East & Northeast':
            base_pm25 = 55.0
        elif region == 'Central & West':
            base_pm25 = 50.0
        else:
            base_pm25 = 38.0

        for d in dates:
            m = d.month
            # Seasonal factor: high in winter (Nov-Feb), lowest in monsoon (Jul-Aug)
            if m in (11, 12, 1, 2):
                season_mult = 1.65 if region == 'Indo-Gangetic Plain / North' else 1.35
            elif m in (6, 7, 8, 9):
                season_mult = 0.52
            elif m in (10,):
                season_mult = 1.30
            else:
                season_mult = 0.90

            # 2020 Lockdown effect (March 24 to May 31, 2020)
            lockdown_mult = 1.0
            if d.year == 2020 and d.month in (3, 4, 5, 6):
                lockdown_mult = 0.58

            noise = np.random.normal(0, 10)
            pm25 = max(5.0, base_pm25 * season_mult * lockdown_mult + noise)
            pm10 = pm25 * np.random.uniform(1.6, 2.1)
            no2 = pm25 * np.random.uniform(0.3, 0.6)
            aqi = pm25 * np.random.uniform(1.1, 1.4) + (pm10 * 0.15)

            # Random 2% missingness
            if np.random.rand() < 0.02:
                pm25 = np.nan
            if np.random.rand() < 0.02:
                aqi = np.nan

            records.append({
                'City': city,
                'Date': d,
                'PM2.5': round(pm25, 2) if not np.isnan(pm25) else np.nan,
                'PM10': round(pm10, 2),
                'NO2': round(no2, 2),
                'AQI': round(aqi, 1) if not np.isnan(aqi) else np.nan,
            })

    return pd.DataFrame(records)

@st.cache_data(show_spinner="Loading and preparing air quality data...")
def load_and_preprocess_data(file_path: str = "data/city_day.csv") -> tuple[pd.DataFrame, bool, str]:
    """
    Cached pipeline that loads, cleans, interpolates, and enriches air quality records.
    Returns:
        df: Processed DataFrame
        is_synthetic: Boolean indicating if synthetic fallback was used
        status_message: Informational note on data origin and row counts
    """
    is_synthetic = False
    raw_df = None

    # Check file existence and non-zero size
    if os.path.exists(file_path) and os.path.getsize(file_path) > 100:
        try:
            raw_df = pd.read_csv(file_path)
            status_message = f"Loaded real CPCB dataset from '{file_path}' ({len(raw_df):,} raw records)."
        except Exception as e:
            raw_df = None
            status_message = f"Error reading '{file_path}': {e}. Falling back to synthetic sample."
    else:
        status_message = f"Data file '{file_path}' not found or empty. Using synthetic fallback dataset."

    if raw_df is None or raw_df.empty:
        raw_df = generate_synthetic_air_quality_data()
        is_synthetic = True

    df = raw_df.copy()

    # Dynamic column inspection & normalization
    col_map = {c.strip(): c.strip() for c in df.columns}
    # Fix potential casing issues
    for c in list(col_map.keys()):
        if c.lower() == 'city':
            df.rename(columns={c: 'City'}, inplace=True)
        elif c.lower() == 'date':
            df.rename(columns={c: 'Date'}, inplace=True)
        elif c.lower() == 'pm2.5':
            df.rename(columns={c: 'PM2.5'}, inplace=True)
        elif c.lower() == 'pm10':
            df.rename(columns={c: 'PM10'}, inplace=True)
        elif c.lower() == 'no2':
            df.rename(columns={c: 'NO2'}, inplace=True)
        elif c.lower() == 'aqi':
            df.rename(columns={c: 'AQI'}, inplace=True)

    # Convert Date
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'City']).sort_values(['City', 'Date']).reset_index(drop=True)

    # Drop rows where both PM2.5 and AQI are missing
    pollutants_present = [p for p in ['PM2.5', 'AQI'] if p in df.columns]
    if pollutants_present:
        df = df.dropna(subset=pollutants_present, how='all').copy()

    # Time-based / linear interpolation within each city with maximum gap limit of 7 days
    interpolation_cols = [p for p in ['PM2.5', 'PM10', 'NO2', 'AQI'] if p in df.columns]
    for col in interpolation_cols:
        # Cast to numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Linear interpolation within city up to 7 consecutive days limit
        df[col] = df.groupby('City')[col].transform(lambda s: s.interpolate(method='linear', limit=7))

    # Recompute or validate AQI bucket using Indian National AQI bands
    if 'AQI' in df.columns:
        df['AQI_Bucket'] = df['AQI'].apply(compute_aqi_bucket)
    elif 'AQI_Bucket' not in df.columns:
        df['AQI_Bucket'] = 'Unknown'

    # Temporal feature engineering
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%b')
    df['Month_Year'] = df['Date'].dt.strftime('%Y-%m')
    df['Season'] = df['Month'].apply(get_season)

    # Spatial feature engineering
    df['Region'] = df['City'].map(CITY_REGIONS).fillna('Other')
    df['Latitude'] = df['City'].map(lambda c: CITY_COORDINATES.get(c, (None, None))[0])
    df['Longitude'] = df['City'].map(lambda c: CITY_COORDINATES.get(c, (None, None))[1])

    return df, is_synthetic, status_message


# ---------------------------------------------------------
# Hourly Data Loader (city_hour_transformed.csv)
# ---------------------------------------------------------

# Ordered labels for hour-of-day display
HOUR_LABELS = [f"{h:02d}:00" for h in range(24)]

# Day-period order for display
DAY_PERIOD_ORDER = ['1. Morning', '2. Afternoon', '3. Evening', '4. Night']
DAY_PERIOD_LABELS = {'1. Morning': 'Morning', '2. Afternoon': 'Afternoon', '3. Evening': 'Evening', '4. Night': 'Night'}

# Season order for display
SEASON_ORDER = ['1. Winter', '2. Summer', '3. Monsoon', '4. Post-Monsoon']
SEASON_LABELS = {
    '1. Winter': 'Winter',
    '2. Summer': 'Summer',
    '3. Monsoon': 'Monsoon',
    '4. Post-Monsoon': 'Post-Monsoon',
}

LOCKDOWN_COLOR = '#E93F33'
PRELOCKDOWN_COLOR = '#2563EB'

@st.cache_data(show_spinner="Loading hourly air quality data (4.4M rows)…")
def load_hourly_data(file_path: str = "data/city_hour_transformed.csv") -> tuple[pd.DataFrame | None, str]:
    """
    Load and lightly preprocess the hourly air quality dataset.
    Returns only the columns needed for Tab 6 to keep memory footprint small.
    Uses chunked reading to handle the 1 GB file gracefully.

    Returns:
        (df, status_message) where df is None if the file is missing.
    """
    needed_cols = [
        'City', 'Datetime', 'PM2.5', 'NO2', 'AQI',
        'Day_period', 'Month', 'Year', 'Season',
        'Weekday_or_weekend', 'AQ_Acceptability',
    ]

    if not os.path.exists(file_path) or os.path.getsize(file_path) < 1000:
        return None, f"Hourly dataset not found at '{file_path}'."

    try:
        chunks = []
        for chunk in pd.read_csv(
            file_path,
            usecols=needed_cols,
            parse_dates=['Datetime'],
            chunksize=500_000,
            low_memory=False,
        ):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    except Exception as exc:
        return None, f"Error reading hourly dataset: {exc}"

    # --- basic cleaning ---
    df = df.dropna(subset=['City', 'Datetime'])
    df['Hour'] = df['Datetime'].dt.hour

    # Coerce pollutants
    for col in ['PM2.5', 'NO2', 'AQI']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where PM2.5 AND AQI are both missing
    df = df.dropna(subset=['PM2.5', 'AQI'], how='all')

    # Enrich with region
    df['Region'] = df['City'].map(CITY_REGIONS).fillna('Other')

    status = (
        f"Loaded hourly dataset: {len(df):,} records across "
        f"{df['City'].nunique()} cities, {df['Year'].min()}–{df['Year'].max()}."
    )
    return df, status
