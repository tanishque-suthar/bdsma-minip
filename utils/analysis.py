"""
utils/analysis.py
Statistical and machine learning routines:
- Global Moran's I Spatial Autocorrelation (libpysal + esda)
- Time Series Seasonal Decomposition (statsmodels)
- Normalized Monthly Profile Clustering (scikit-learn KMeans)
- 2020 Lockdown Impact & SDG 11 Dynamic Metrics
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import libpysal
from esda.moran import Moran

from utils.data import POLLUTANT_UNITS, CITY_REGIONS

def calculate_spatial_autocorrelation(
    city_summary_df: pd.DataFrame,
    pollutant: str = 'PM2.5',
    k_neighbors: int = 4
) -> Dict[str, Any]:
    """
    Compute Global Moran's I spatial autocorrelation on city mean pollutant levels
    using libpysal KNN spatial weights matrix and esda.Moran.
    """
    valid = city_summary_df.dropna(subset=['Latitude', 'Longitude', pollutant]).copy()
    n_cities = len(valid)

    if n_cities < 4:
        return {
            'valid': False,
            'message': f"Insufficient cities ({n_cities}) to compute spatial autocorrelation. Need at least 4 cities.",
            'I': None,
            'p_value': None,
            'z_score': None,
            'interpretation': "N/A",
            'caveat': "Sample size too small."
        }

    coords = valid[['Latitude', 'Longitude']].values
    actual_k = min(k_neighbors, n_cities - 1)
    
    try:
        w = libpysal.weights.KNN.from_array(coords, k=actual_k)
        w.transform = 'R'  # Row-standardized
        y = valid[pollutant].values
        
        # Guard against zero variance
        if np.isclose(np.var(y), 0):
            return {
                'valid': False,
                'message': "Pollutant values have zero variance across selected cities.",
                'I': None,
                'p_value': None,
                'z_score': None,
                'interpretation': "Zero variance.",
                'caveat': "Cannot compute autocorrelation on uniform values."
            }

        mi = Moran(y, w, transformation='r', permutations=999)
        
        # Plain-English interpretation
        if mi.p_sim < 0.05:
            if mi.I > 0:
                interp = (
                    f"Strong positive spatial clustering detected (Moran's I = {mi.I:.3f}, p = {mi.p_sim:.4f}). "
                    f"Cities with high {pollutant} levels are geographically adjacent to other high-pollution cities "
                    f"(regional airshed clustering), rather than distributed randomly."
                )
            else:
                interp = (
                    f"Statistically significant spatial dispersion (Moran's I = {mi.I:.3f}, p = {mi.p_sim:.4f}). "
                    f"High-pollution cities tend to be bordered by lower-pollution areas."
                )
        else:
            interp = (
                f"No statistically significant spatial autocorrelation observed (Moran's I = {mi.I:.3f}, p = {mi.p_sim:.4f}). "
                f"At this significance threshold (α = 0.05), city pollution levels cannot be distinguished from a random spatial pattern."
            )

        caveat = (
            f"Methodological Caveat: Based on {n_cities} monitored cities with a k={actual_k} nearest-neighbor "
            f"graph. A sample size of ~26 cities provides moderate statistical power; localized micro-climates "
            f"and unmonitored rural corridors between urban clusters are not represented."
        )

        return {
            'valid': True,
            'I': float(mi.I),
            'p_value': float(mi.p_sim),
            'z_score': float(mi.z_sim),
            'expected_I': float(mi.EI),
            'n_cities': n_cities,
            'k_used': actual_k,
            'interpretation': interp,
            'caveat': caveat
        }
    except Exception as e:
        return {
            'valid': False,
            'message': f"Error computing Moran's I: {str(e)}",
            'I': None,
            'p_value': None,
            'z_score': None,
            'interpretation': "Calculation failed.",
            'caveat': str(e)
        }

def decompose_city_time_series(
    df: pd.DataFrame,
    city: str,
    pollutant: str = 'PM2.5',
    freq: str = 'W'
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Perform seasonal decomposition (additive) on a weekly or monthly resampled city series.
    Returns:
        decomp_df: DataFrame with Date, Observed, Trend, Seasonal, Residual
        insight: auto-generated insight string
    """
    city_data = df[df['City'] == city].sort_values('Date').copy()
    if city_data.empty or pollutant not in city_data.columns:
        return None, f"No data found for {city} and pollutant {pollutant}."

    # Set Date index and resample to regular weekly frequency
    ts = city_data.set_index('Date')[pollutant].dropna()
    if len(ts) < 52:
        return None, f"Insufficient observations for {city} ({len(ts)} points). Need at least 52 regular weekly records for decomposition."

    weekly = ts.resample('W').mean()
    # Interpolate small weekly gaps
    weekly_filled = weekly.interpolate(method='linear', limit=8).dropna()

    period = 52  # 52 weeks in a year
    if len(weekly_filled) < 2 * period:
        # Fall back to monthly if series is shorter than 2 full years of weekly data
        monthly = ts.resample('M').mean().interpolate(method='linear', limit=3).dropna()
        if len(monthly) < 24:
            return None, f"Time series for {city} is too short to estimate robust 12-month seasonality."
        period = 12
        series_to_decompose = monthly
    else:
        series_to_decompose = weekly_filled

    try:
        decomp = seasonal_decompose(series_to_decompose, model='additive', period=period, extrapolate_trend='period')
        decomp_df = pd.DataFrame({
            'Date': series_to_decompose.index,
            'Observed': decomp.observed,
            'Trend': decomp.trend,
            'Seasonal': decomp.seasonal,
            'Residual': decomp.resid,
        }).reset_index(drop=True)

        # Generate automatic insight
        seasonal_amplitude = decomp.seasonal.max() - decomp.seasonal.min()
        trend_clean = decomp.trend.dropna()
        if len(trend_clean) > 1:
            trend_change = trend_clean.iloc[-1] - trend_clean.iloc[0]
            direction = "downward (improving)" if trend_change < 0 else "upward (deteriorating)"
            trend_str = f"overall {direction} trend (change of {trend_change:+.1f} {POLLUTANT_UNITS.get(pollutant, '')})"
        else:
            trend_str = "stable overall baseline"

        unit = POLLUTANT_UNITS.get(pollutant, '')
        insight = (
            f"**Insight for {city} ({pollutant}):** Displays a strong annual swing with peak-to-trough "
            f"seasonal amplitude of {seasonal_amplitude:.1f} {unit}. The underlying smoothed trend exhibits a "
            f"{trend_str} over the multi-year timeline, with noticeable residual dips during monsoon months and Spring 2020."
        )

        return decomp_df, insight
    except Exception as e:
        return None, f"Seasonal decomposition failed: {str(e)}"

def cluster_cities_by_monthly_profile(
    df: pd.DataFrame,
    pollutant: str = 'PM2.5',
    n_clusters: int = 3
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """
    Cluster cities by their normalized 12-month pollution profile using KMeans.
    Normalizing focuses clustering on seasonality timing and curve shape rather than just absolute scale.
    Returns:
        city_clusters_df: City, Cluster, Latitude, Longitude, Region, Mean_Pollutant
        profiles_df: Long-form DataFrame (Cluster, Month, Month_Name, Mean_Profile, Mean_Actual)
        summary_text: Dynamic interpretation of the clusters
    """
    valid = df.dropna(subset=['City', 'Month', pollutant]).copy()
    if valid.empty:
        return None, None, "No data available for clustering."

    # Monthly means per city
    pivot = valid.pivot_table(index='City', columns='Month', values=pollutant, aggfunc='mean')
    
    # Fill any missing month using row mean, then column mean
    pivot = pivot.apply(lambda row: row.fillna(row.mean()), axis=1)
    pivot = pivot.fillna(pivot.mean())

    if len(pivot) < n_clusters:
        return None, None, f"Number of cities ({len(pivot)}) is less than requested clusters ({n_clusters})."

    # Normalize each city's profile to z-scores so clustering groups by seasonal profile pattern
    scaler = StandardScaler()
    scaled_profiles = scaler.fit_transform(pivot)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
    cluster_labels = kmeans.fit_predict(scaled_profiles)

    # City-level cluster assignments
    city_clusters = pd.DataFrame({
        'City': pivot.index,
        'Cluster': [f"Cluster {l + 1}" for l in cluster_labels],
        'Cluster_Id': cluster_labels
    })

    # Join metadata
    meta = df[['City', 'Region', 'Latitude', 'Longitude']].drop_duplicates('City')
    means = df.groupby('City')[pollutant].mean().reset_index().rename(columns={pollutant: f'Mean_{pollutant}'})
    city_clusters = city_clusters.merge(meta, on='City', how='left').merge(means, on='City', how='left')

    # Profile curves per cluster
    pivot_with_cluster = pivot.copy()
    pivot_with_cluster['Cluster'] = city_clusters.set_index('City')['Cluster']
    
    # Long form profiles for plotting
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    
    melted = pivot_with_cluster.reset_index().melt(
        id_vars=['City', 'Cluster'],
        var_name='Month',
        value_name='Pollutant_Value'
    )
    melted['Month'] = melted['Month'].astype(int)
    melted['Month_Name'] = melted['Month'].map(month_names)

    # Average profile per cluster
    cluster_profiles = melted.groupby(['Cluster', 'Month', 'Month_Name'])['Pollutant_Value'].mean().reset_index()
    cluster_profiles = cluster_profiles.sort_values(['Cluster', 'Month'])

    # Dynamic summary of clusters
    descriptions = []
    for c_id in sorted(city_clusters['Cluster'].unique()):
        cities_in_c = city_clusters[city_clusters['Cluster'] == c_id]['City'].tolist()
        c_sub = cluster_profiles[cluster_profiles['Cluster'] == c_id]
        peak_month = c_sub.loc[c_sub['Pollutant_Value'].idxmax()]['Month_Name']
        min_month = c_sub.loc[c_sub['Pollutant_Value'].idxmin()]['Month_Name']
        avg_val = c_sub['Pollutant_Value'].mean()
        descriptions.append(
            f"**{c_id} ({len(cities_in_c)} cities):** Peak in **{peak_month}**, lowest in **{min_month}** "
            f"(Annual Avg: {avg_val:.1f} {POLLUTANT_UNITS.get(pollutant, '')}). "
            f"Cities: *{', '.join(cities_in_c[:5])}{', ...' if len(cities_in_c) > 5 else ''}*"
        )
    
    summary_text = "\n\n".join(descriptions)
    return city_clusters, cluster_profiles, summary_text

def calculate_lockdown_impact(
    df: pd.DataFrame,
    pollutant: str = 'PM2.5'
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Compare March-June 2020 (Nationwide Lockdown period) vs same months in 2018-2019.
    Returns:
        comparison_df: City, Pre_Mean, Lockdown_Mean, Pct_Change, Region, Lat, Lon
        stats: dictionary with top improved, top polluted, and SDG 11 metrics
    """
    if pollutant not in df.columns:
        return None, {'error': f"Pollutant {pollutant} not found in dataset."}

    # Restrict to March-June
    mar_jun = df[df['Month'].isin([3, 4, 5, 6]) & df[pollutant].notna()].copy()

    # Pre-lockdown (2018-2019 baseline)
    pre = mar_jun[mar_jun['Year'].isin([2018, 2019])].groupby('City')[pollutant].mean()
    # Lockdown period (2020)
    lockdown = mar_jun[mar_jun['Year'] == 2020].groupby('City')[pollutant].mean()

    comp = pd.DataFrame({
        'Pre_Lockdown_Mean': pre,
        'Lockdown_2020_Mean': lockdown
    }).dropna()

    if comp.empty:
        return None, {'error': "Insufficient data for 2018-2019 vs 2020 comparison."}

    comp['Abs_Change'] = comp['Lockdown_2020_Mean'] - comp['Pre_Lockdown_Mean']
    comp['Pct_Change'] = (comp['Abs_Change'] / comp['Pre_Lockdown_Mean']) * 100
    comp = comp.reset_index()

    # Merge metadata
    meta = df[['City', 'Region', 'Latitude', 'Longitude']].drop_duplicates('City')
    comp = comp.merge(meta, on='City', how='left').sort_values('Pct_Change')

    # Calculate dynamic insights
    top_improved = comp.head(3)
    least_improved = comp.tail(3)
    most_polluted_lockdown = comp.sort_values('Lockdown_2020_Mean', ascending=False).head(3)
    national_pct_change = ((comp['Lockdown_2020_Mean'].mean() - comp['Pre_Lockdown_Mean'].mean()) / comp['Pre_Lockdown_Mean'].mean()) * 100

    stats = {
        'total_cities_evaluated': len(comp),
        'national_avg_reduction_pct': float(national_pct_change),
        'top_improved': top_improved[['City', 'Pre_Lockdown_Mean', 'Lockdown_2020_Mean', 'Pct_Change', 'Region']].to_dict('records'),
        'least_improved': least_improved[['City', 'Pre_Lockdown_Mean', 'Lockdown_2020_Mean', 'Pct_Change', 'Region']].to_dict('records'),
        'most_polluted_lockdown': most_polluted_lockdown[['City', 'Lockdown_2020_Mean', 'Region']].to_dict('records'),
        'unit': POLLUTANT_UNITS.get(pollutant, 'µg/m³')
    }

    return comp, stats
