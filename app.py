"""
app.py
India Air Quality Spatio-Temporal Analysis
Micro Project | SDG 11: Sustainable Cities & Communities (Target 11.6)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data import (
    load_and_preprocess_data,
    load_hourly_data,
    CITY_COORDINATES,
    CITY_REGIONS,
    AQI_CATEGORIES,
    AQI_COLORS,
    AQI_CONTINUOUS_COLORSCALE,
    POLLUTANT_UNITS,
    HOUR_LABELS,
    DAY_PERIOD_ORDER,
    DAY_PERIOD_LABELS,
    SEASON_ORDER,
    SEASON_LABELS,
    LOCKDOWN_COLOR,
    PRELOCKDOWN_COLOR,
)
from utils.analysis import (
    calculate_spatial_autocorrelation,
    decompose_city_time_series,
    cluster_cities_by_monthly_profile,
    calculate_lockdown_impact
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="India Air Quality Spatio-Temporal Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling for Clean, Modern Dashboard Look
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .insight-box {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 10px 14px;
        border-radius: 4px;
        margin-top: 8px;
        margin-bottom: 16px;
        font-size: 0.93rem;
        color: #14532D;
    }
    .stat-badge {
        display: inline-block;
        padding: 3px 8px;
        font-weight: 600;
        border-radius: 4px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------
df_all, is_synthetic, status_msg = load_and_preprocess_data("data/city_day.csv")

# Synthetic data banner fallback notice
if is_synthetic:
    st.warning("**Running with Synthetic Sample Data:** The dataset file `data/city_day.csv` was missing or empty. A realistic synthetic dataset (26 cities, 2018-2020) has been generated so the application is fully interactive and demonstrable.")

# ---------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------
st.sidebar.title("Controls & Filters")
st.sidebar.caption("Spatio-Temporal Air Quality Explorer")

# Available pollutants
candidate_pollutants = ['PM2.5', 'AQI', 'PM10', 'NO2']
available_pollutants = [p for p in candidate_pollutants if p in df_all.columns]
if not available_pollutants:
    available_pollutants = ['PM2.5']

selected_pollutant = st.sidebar.selectbox(
    "Select Pollutant / Index",
    options=available_pollutants,
    index=available_pollutants.index('PM2.5') if 'PM2.5' in available_pollutants else 0,
    help="Default is PM2.5 (fine inhalable particulate matter). Units are µg/m³ (AQI is unitless index)."
)
unit = POLLUTANT_UNITS.get(selected_pollutant, 'µg/m³')

# Region Filter
all_regions = sorted(list(set(CITY_REGIONS.values())))
selected_regions = st.sidebar.multiselect(
    "Filter by Geographic Region",
    options=all_regions,
    default=all_regions,
    help="Filter cities by ecological & geographic airshed zones."
)

# City Filter
available_cities_in_regions = sorted(
    [c for c in df_all['City'].unique() if CITY_REGIONS.get(c, 'Other') in selected_regions]
)

col_btn1, col_btn2 = st.sidebar.columns(2)
select_all = col_btn1.button("Select All", use_container_width=True)
clear_all = col_btn2.button("Clear All", use_container_width=True)

if "selected_cities" not in st.session_state:
    st.session_state.selected_cities = available_cities_in_regions

if select_all:
    st.session_state.selected_cities = available_cities_in_regions
if clear_all:
    st.session_state.selected_cities = []

# Guard against stale session state when region changes
valid_defaults = [c for c in st.session_state.selected_cities if c in available_cities_in_regions]
if not valid_defaults and available_cities_in_regions:
    valid_defaults = available_cities_in_regions

selected_cities = st.sidebar.multiselect(
    "Select Cities",
    options=available_cities_in_regions,
    default=valid_defaults,
    key="city_multiselect"
)

# Year Range Filter
min_year = int(df_all['Year'].min())
max_year = int(df_all['Year'].max())

if min_year < max_year:
    selected_year_range = st.sidebar.slider(
        "Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )
else:
    selected_year_range = (min_year, max_year)
    st.sidebar.text(f"Year: {min_year}")

# Sidebar Documentation & Metadata Expander
with st.sidebar.expander("Data Handling & Notes", expanded=False):
    st.markdown(f"""
    - **Dataset Source:** CPCB continuous monitoring, curated by Rohan Rao (Kaggle).
    - **Status:** {status_msg}
    - **Missing Data Policy:** Forward & backward linear interpolation per city capped at a **7-day window**. Longer gaps remain un-interpolated to avoid synthetic biases.
    - **Exclusion:** Days where both PM2.5 and AQI were absent were dropped.
    - **National AQI Bands:** Classified per Indian CPCB standard breaks: Good (0-50), Satisfactory (51-100), Moderate (101-200), Poor (201-300), Very Poor (301-400), Severe (401+).
    """)

# ---------------------------------------------------------
# Filter Dataframe
# ---------------------------------------------------------
df_filtered = df_all[
    (df_all['City'].isin(selected_cities)) &
    (df_all['Year'] >= selected_year_range[0]) &
    (df_all['Year'] <= selected_year_range[1])
].copy()

# Header
st.markdown('<div class="main-header">India Air Quality Spatio-Temporal Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Exploring CPCB Air Quality (2015–2020) across 26 Indian Cities | '
    '<strong>Theme:</strong> Spatio-Temporal Analytics &bull; <strong>SDG 11:</strong> Sustainable Cities and Communities (Target 11.6)</div>',
    unsafe_allow_html=True
)

# Empty selection warning guard
if df_filtered.empty or not selected_cities:
    st.warning("No data available for current filter selection. Please select at least one region and one city in the sidebar.")
    st.stop()

# ---------------------------------------------------------
# Tab Layout
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Tab 1: Overview & Coverage",
    "Tab 2: Temporal & Seasonality",
    "Tab 3: Spatial & Autocorrelation",
    "Tab 4: Spatio-Temporal & Clusters",
    "Tab 5: Lockdown & SDG 11",
    "Tab 6: Diurnal & Hourly Patterns",
])

# =========================================================
# TAB 1: OVERVIEW & COVERAGE
# =========================================================
with tab1:
    st.subheader("Key Macro Indicators & Data Coverage")
    
    # 4 Key Metrics
    total_cities_covered = df_filtered['City'].nunique()
    start_date_str = df_filtered['Date'].min().strftime('%b %Y')
    end_date_str = df_filtered['Date'].max().strftime('%b %Y')
    national_mean = df_filtered[selected_pollutant].mean()
    
    # % of days in Poor or worse (Poor, Very Poor, Severe)
    poor_or_worse_mask = df_filtered['AQI_Bucket'].isin(['Poor', 'Very Poor', 'Severe'])
    pct_poor_or_worse = (poor_or_worse_mask.sum() / len(df_filtered.dropna(subset=['AQI_Bucket']))) * 100 if len(df_filtered.dropna(subset=['AQI_Bucket'])) > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cities Covered", f"{total_cities_covered} of 26")
    with col2:
        st.metric("Date Span", f"{start_date_str} – {end_date_str}")
    with col3:
        st.metric(f"Mean {selected_pollutant}", f"{national_mean:.1f} {unit}")
    with col4:
        st.metric("Days Poor or Worse", f"{pct_poor_or_worse:.1f}%", help="Share of monitored city-days with AQI in Poor, Very Poor, or Severe bands.")

    st.markdown("---")
    st.subheader(f"Data Coverage Heatmap: Non-Missing {selected_pollutant} Days (% of Year)")

    # Data Coverage Heatmap: City x Year
    # Calculate share of days in each year with non-missing pollutant
    coverage = df_filtered.groupby(['City', 'Year'])[selected_pollutant].apply(lambda s: (s.notna().sum() / 365.25) * 100).unstack()
    coverage = coverage.fillna(0).round(1)

    fig_cov = px.imshow(
        coverage,
        labels=dict(x="Year", y="City", color="Valid Days (%)"),
        x=coverage.columns.astype(str),
        y=coverage.index,
        color_continuous_scale="Blues",
        aspect="auto",
        title=f"Monitoring Station Operational Completeness by City & Year ({selected_pollutant})"
    )
    fig_cov.update_layout(
        height=max(400, len(coverage) * 22),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Year",
        yaxis_title="City"
    )
    st.plotly_chart(fig_cov, use_container_width=True)

    st.markdown(
        f'<div class="insight-box"><strong>Coverage Takeaway:</strong> Data availability increased markedly from 2018 onward as the National Clean Air Programme (NCAP) expanded continuous ambient air quality monitoring stations (CAAQMS). Early years (2015–2017) had sporadic station presence in several Tier-2 cities.</div>',
        unsafe_allow_html=True
    )

    with st.expander("About Dataset & Analytical Limitations", expanded=False):
        st.markdown("""
        - **Monitoring Density Limitation:** The dataset tracks 26 major cities representing ~30 CAAQMS networks, which covers only a fraction of India's 4,000+ urban local bodies. Rural, inter-state transport, and peri-urban corridors are under-represented.
        - **Varying Baseline Windows:** Certain cities (e.g., Ernakulam, Jorapokhar, Brajrajnagar) have continuous records only beginning in late 2018 or 2019. Comparisons over the 2015-2020 window should account for these differing station commissioning dates.
        - **Data Cleaning Choices:** Missing values were linearly interpolated within each city up to a 7-day limit. Extended outages were retained as missing to preserve genuine station down-time rather than imputing artificial flat lines.
        """)

# =========================================================
# TAB 2: TEMPORAL & SEASONALITY
# =========================================================
with tab2:
    st.subheader("Temporal Trajectories & Seasonality Patterns")

    # Yearly Trend Lines
    st.markdown(f"#### 1. Multi-Year Trend by City ({selected_pollutant})")
    yearly_city = df_filtered.groupby(['Year', 'City'])[selected_pollutant].mean().reset_index()

    fig_trend = px.line(
        yearly_city,
        x="Year",
        y=selected_pollutant,
        color="City",
        markers=True,
        title=f"Annual Average {selected_pollutant} Concentration across Selected Cities ({unit})",
        labels={selected_pollutant: f"{selected_pollutant} ({unit})", "Year": "Year"}
    )
    fig_trend.update_layout(
        height=450,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Dynamic Insight for yearly trend
    if not yearly_city.empty:
        highest_city = yearly_city.loc[yearly_city[selected_pollutant].idxmax()]
        lowest_city = yearly_city.loc[yearly_city[selected_pollutant].idxmin()]
        st.markdown(
            f'<div class="insight-box"><strong>Trend Insight:</strong> Peak annual average occurred in <strong>{highest_city["City"]}</strong> ({highest_city[selected_pollutant]:.1f} {unit} in {highest_city["Year"]}), whereas <strong>{lowest_city["City"]}</strong> recorded the lowest ({lowest_city[selected_pollutant]:.1f} {unit} in {lowest_city["Year"]}). A widespread downward dip is observable in 2020 corresponding to COVID-19 mobility restrictions.</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Monthly Seasonality Heatmap
    st.markdown(f"#### 2. Monthly Seasonality Heatmap ({selected_pollutant})")
    month_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_matrix = df_filtered.pivot_table(index='City', columns='Month', values=selected_pollutant, aggfunc='mean')
    monthly_matrix = monthly_matrix.reindex(columns=month_order)
    monthly_matrix.columns = month_labels

    fig_month = px.imshow(
        monthly_matrix,
        labels=dict(x="Month", y="City", color=f"{selected_pollutant} ({unit})"),
        color_continuous_scale=AQI_CONTINUOUS_COLORSCALE,
        aspect="auto",
        title=f"Climatological Monthly Mean {selected_pollutant} Heatmap ({unit})"
    )
    fig_month.update_layout(
        height=max(400, len(monthly_matrix) * 22),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Month",
        yaxis_title="City"
    )
    st.plotly_chart(fig_month, use_container_width=True)

    st.markdown(
        f'<div class="insight-box"><strong>Seasonality Insight:</strong> Northern and Indo-Gangetic cities (Delhi, Patna, Lucknow) demonstrate extreme winter spikes (Nov–Jan) driven by low boundary layer height, temperature inversions, and biomass burning. Conversely, July–August exhibits nationwide minimums due to monsoon precipitation wash-out.</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Seasonal Decomposition
    st.markdown(f"#### 3. Time Series Seasonal Decomposition (Statsmodels)")
    decomp_city = st.selectbox(
        "Choose City for Seasonal Decomposition",
        options=selected_cities,
        index=0 if selected_cities else None,
        key="decomp_city_select"
    )

    if decomp_city:
        decomp_df, decomp_insight = decompose_city_time_series(df_all, city=decomp_city, pollutant=selected_pollutant, freq='W')
        if decomp_df is not None:
            fig_decomp = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                subplot_titles=(
                    f"Observed Weekly Mean ({unit})",
                    f"Underlying Trend ({unit})",
                    f"Annual Seasonal Cycle ({unit})",
                    f"Residual / Anomaly ({unit})"
                )
            )
            fig_decomp.add_trace(go.Scatter(x=decomp_df['Date'], y=decomp_df['Observed'], name="Observed", line=dict(color="#2563EB", width=1.5)), row=1, col=1)
            fig_decomp.add_trace(go.Scatter(x=decomp_df['Date'], y=decomp_df['Trend'], name="Trend", line=dict(color="#DC2626", width=2)), row=2, col=1)
            fig_decomp.add_trace(go.Scatter(x=decomp_df['Date'], y=decomp_df['Seasonal'], name="Seasonal", line=dict(color="#16A34A", width=1.5)), row=3, col=1)
            fig_decomp.add_trace(go.Scatter(x=decomp_df['Date'], y=decomp_df['Residual'], name="Residual", line=dict(color="#9333EA", width=1)), row=4, col=1)
            
            fig_decomp.update_layout(
                height=650,
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10),
                title_text=f"Additive Seasonal Decomposition for {decomp_city} ({selected_pollutant})"
            )
            st.plotly_chart(fig_decomp, use_container_width=True)
            st.markdown(f'<div class="insight-box">{decomp_insight}</div>', unsafe_allow_html=True)
        else:
            st.warning(f"Could not compute decomposition: {decomp_insight}")

    with st.expander("About / Method: Temporal Analysis", expanded=False):
        st.markdown("""
        - **Resampling:** Raw daily observations were aggregated to weekly means (`'W'`) to eliminate small high-frequency fluctuations while preserving weekly and seasonal rhythm.
        - **Additive Decomposition:** We model $Y(t) = Trend(t) + Seasonal(t) + Residual(t)$ using `statsmodels.tsa.seasonal.seasonal_decompose` with a 52-week annual period.
        - **Extrapolation:** Period extrapolation was utilized to avoid losing boundary weeks.
        """)

# =========================================================
# TAB 3: SPATIAL & AUTOCORRELATION
# =========================================================
with tab3:
    st.subheader("Geospatial Distribution & Spatial Clustering")

    # City Mean Aggregations
    city_means = df_filtered.dropna(subset=[selected_pollutant]).groupby(['City', 'Region', 'Latitude', 'Longitude'])[selected_pollutant].mean().reset_index()
    city_means['Marker_Size'] = city_means[selected_pollutant].clip(lower=5.0)
    city_means['AQI_Equivalent_Band'] = city_means[selected_pollutant].apply(
        lambda v: 'Good' if v <= 50 else ('Satisfactory' if v <= 100 else ('Moderate' if v <= 200 else ('Poor' if v <= 300 else ('Very Poor' if v <= 400 else 'Severe'))))
    )

    col_map, col_stat = st.columns([3, 2])

    with col_map:
        st.markdown(f"#### Geographic Air Quality Gradient ({selected_pollutant})")
        # Plotly Scatter Map
        fig_map = px.scatter_map(
            city_means,
            lat="Latitude",
            lon="Longitude",
            size="Marker_Size",
            color=selected_pollutant,
            hover_name="City",
            hover_data={
                "Region": True,
                selected_pollutant: ":.1f",
                "AQI_Equivalent_Band": True,
                "Marker_Size": False,
                "Latitude": False,
                "Longitude": False
            },
            color_continuous_scale=AQI_CONTINUOUS_COLORSCALE,
            size_max=22,
            zoom=3.8,
            center={"lat": 22.0, "lon": 80.0},
            map_style="open-street-map",
            title=f"Mean {selected_pollutant} across Monitored Cities ({unit})"
        )
        fig_map.update_layout(margin=dict(l=5, r=5, t=40, b=5), height=520)
        st.plotly_chart(fig_map, use_container_width=True)

    with col_stat:
        st.markdown("#### Regional Distribution Comparison")
        # Regional Box Plot
        fig_box = px.box(
            df_filtered,
            x="Region",
            y=selected_pollutant,
            color="Region",
            points="outliers",
            title=f"Distribution by Region ({unit})",
            labels={selected_pollutant: f"{selected_pollutant} ({unit})"}
        )
        fig_box.update_layout(
            height=320,
            showlegend=False,
            margin=dict(l=10, r=10, t=35, b=10),
            xaxis_tickangle=-25
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # Moran's I Computation
        st.markdown("#### Spatial Autocorrelation: Moran's I")
        moran_res = calculate_spatial_autocorrelation(city_means, pollutant=selected_pollutant, k_neighbors=4)
        
        if moran_res['valid']:
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Moran's I", f"{moran_res['I']:.3f}")
            m_col2.metric("p-value", f"{moran_res['p_value']:.4f}")
            m_col3.metric("z-score", f"{moran_res['z_score']:.2f}")

            st.info(f"**Interpretation:** {moran_res['interpretation']}")
            st.caption(f"Note: {moran_res['caveat']}")
        else:
            st.warning(f"Spatial Autocorrelation Notice: {moran_res.get('message', 'Unable to compute')}")

    with st.expander("About / Method: Spatial Weights & Moran's I", expanded=False):
        st.markdown("""
        - **Spatial Weights Matrix ($W$):** Constructed via $k$-Nearest Neighbors ($k=4$) using geodesic coordinate pairs $(\\text{Latitude}, \\text{Longitude})$ via `libpysal.weights.KNN`. Row-standardization ($W_{ij} / \\sum_j W_{ij}$) was applied so spatial lags represent average neighbor concentrations.
        - **Global Moran's I:** Evaluated using `esda.moran.Moran` with 999 Monte Carlo permutations under the randomization assumption.
        - **Methodological Power Limitation:** With only 26 metropolitan centroids, local sub-regional variations (e.g. agricultural burning along Punjab-Haryana belts vs urban NCR vehicular emissions) are aggregated into city points.
        """)

# =========================================================
# TAB 4: SPATIO-TEMPORAL & CLUSTERS
# =========================================================
with tab4:
    st.subheader("Dynamic Spatio-Temporal Shifting & City Clustering")

    # Animated Map across Years or Months
    st.markdown(f"#### 1. Animated Pollution Shift over Time ({selected_pollutant})")
    anim_freq = st.radio("Animation Frequency", ["By Year", "By Season & Year"], horizontal=True)

    df_anim_base = df_filtered.dropna(subset=[selected_pollutant]).copy()
    if anim_freq == "By Year":
        anim_df = df_anim_base.groupby(['Year', 'City', 'Latitude', 'Longitude'])[selected_pollutant].mean().reset_index()
        anim_df['Time_Frame'] = anim_df['Year'].astype(str)
        anim_df = anim_df.sort_values('Year')
    else:
        anim_df = df_anim_base.groupby(['Year', 'Season', 'City', 'Latitude', 'Longitude'])[selected_pollutant].mean().reset_index()
        anim_df['Time_Frame'] = anim_df['Year'].astype(str) + " " + anim_df['Season']
        anim_df = anim_df.sort_values(['Year', 'Season'])

    anim_df['Marker_Size'] = anim_df[selected_pollutant].clip(lower=5.0)

    fig_anim = px.scatter_map(
        anim_df,
        lat="Latitude",
        lon="Longitude",
        size="Marker_Size",
        color=selected_pollutant,
        hover_name="City",
        hover_data={"Marker_Size": False, selected_pollutant: ":.1f", "Time_Frame": True},
        animation_frame="Time_Frame",
        color_continuous_scale=AQI_CONTINUOUS_COLORSCALE,
        size_max=22,
        zoom=3.8,
        center={"lat": 22.0, "lon": 80.0},
        map_style="open-street-map",
        title=f"Evolution of City-Level {selected_pollutant} Concentrations ({unit})"
    )
    fig_anim.update_layout(margin=dict(l=5, r=5, t=40, b=5), height=520)
    st.plotly_chart(fig_anim, use_container_width=True)

    st.markdown("---")

    # Small Multiples Seasonal Maps
    st.markdown(f"#### 2. Small-Multiples: Climatological Seasonal Comparison")
    seasons_list = ['Winter', 'Summer', 'Monsoon', 'Post-monsoon']
    seasonal_summary = df_filtered.dropna(subset=[selected_pollutant]).groupby(['Season', 'City', 'Latitude', 'Longitude'])[selected_pollutant].mean().reset_index()
    seasonal_summary['Marker_Size'] = seasonal_summary[selected_pollutant].clip(lower=5.0)

    s_cols = st.columns(4)
    for idx, s_name in enumerate(seasons_list):
        s_data = seasonal_summary[seasonal_summary['Season'] == s_name]
        with s_cols[idx]:
            st.markdown(f"**{s_name}**")
            fig_s = px.scatter_map(
                s_data,
                lat="Latitude",
                lon="Longitude",
                size="Marker_Size",
                color=selected_pollutant,
                hover_name="City",
                color_continuous_scale=AQI_CONTINUOUS_COLORSCALE,
                range_color=[df_filtered[selected_pollutant].quantile(0.05), df_filtered[selected_pollutant].quantile(0.95)],
                size_max=16,
                zoom=3.0,
                center={"lat": 22.0, "lon": 82.0},
                map_style="open-street-map"
            )
            fig_s.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                height=320,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("---")

    # KMeans Clustering on Normalized Monthly Profile
    st.markdown(f"#### 3. City Climatological Profile Clustering (K-Means)")
    st.caption("Clustering cities by their standardized 12-month pollution curve shape reveals distinct regional atmospheric profiles.")

    k_clusters = st.slider("Number of Clusters (k)", min_value=2, max_value=5, value=3, step=1)
    city_clusters, cluster_profiles, cluster_summary = cluster_cities_by_monthly_profile(
        df_all[df_all['City'].isin(selected_cities)],
        pollutant=selected_pollutant,
        n_clusters=k_clusters
    )

    if city_clusters is not None and cluster_profiles is not None:
        c_col1, c_col2 = st.columns([1, 1])

        with c_col1:
            # Map of Clusters
            fig_c_map = px.scatter_map(
                city_clusters,
                lat="Latitude",
                lon="Longitude",
                color="Cluster",
                hover_name="City",
                hover_data={"Region": True, f"Mean_{selected_pollutant}": ":.1f"},
                size_max=18,
                zoom=3.6,
                center={"lat": 22.0, "lon": 80.0},
                map_style="open-street-map",
                title=f"Spatial Distribution of Profile Clusters (k={k_clusters})"
            )
            fig_c_map.update_layout(margin=dict(l=5, r=5, t=35, b=5), height=420)
            st.plotly_chart(fig_c_map, use_container_width=True)

        with c_col2:
            # Profile Curves
            fig_curve = px.line(
                cluster_profiles,
                x="Month_Name",
                y="Pollutant_Value",
                color="Cluster",
                markers=True,
                title=f"Average Annual Seasonal Profile Curves ({unit})",
                labels={"Month_Name": "Month", "Pollutant_Value": f"Mean {selected_pollutant} ({unit})"}
            )
            fig_curve.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=35, b=10),
                hovermode="x unified"
            )
            st.plotly_chart(fig_curve, use_container_width=True)

        st.markdown(f'<div class="insight-box">{cluster_summary}</div>', unsafe_allow_html=True)
    else:
        st.warning(f"Could not perform clustering: {cluster_summary}")

    with st.expander("About / Method: Spatio-Temporal Clustering", expanded=False):
        st.markdown(r"""
        - **Monthly Feature Matrix:** Cities are represented as 12-dimensional feature vectors corresponding to mean concentrations across each calendar month (Jan–Dec).
        - **Standardization:** We apply z-score normalization ($z = (x - \mu)/\sigma$) across each city's profile before K-Means. This ensures grouping reflects the *timing and shape of seasonality* (e.g. sharp winter peak vs monsoon drop vs equable year-round climate) rather than merely absolute baseline magnitude.
        - **Algorithm:** Scikit-Learn's `KMeans` with $k$-means++ initialization and 15 restarts for global stability.
        """)

    st.markdown("---")

    # ==================================================
    # SECTION 4: HOURLY DIURNAL ANIMATION
    # ==================================================
    st.markdown("#### 4. Animated Diurnal Map — City-Level Hourly Cycle")
    st.caption(
        "Watch how PM₂.5 (or any pollutant) spreads and recedes across Indian cities "
        "as the clock moves from midnight through rush hours and back. "
        "Each frame is the climatological mean for that hour across all selected years. "
        "Filter by season to isolate winter smog waves vs monsoon suppression."
    )

    df_hour_t4, _ = load_hourly_data("data/city_hour_transformed.csv")

    if df_hour_t4 is None:
        st.info("Hourly dataset not found. Place `city_hour_transformed.csv` in `data/` to enable this animation.")
    else:
        t4h_col1, t4h_col2 = st.columns(2)
        with t4h_col1:
            t4_hourly_pollutants = [p for p in ['PM2.5', 'NO2', 'AQI'] if p in df_hour_t4.columns]
            t4_h_pollutant = st.selectbox(
                "Pollutant (Hourly Animation)",
                options=t4_hourly_pollutants,
                index=0,
                key="t4_h_pollutant",
            )
            t4_h_unit = POLLUTANT_UNITS.get(t4_h_pollutant, "µg/m³")

        with t4h_col2:
            t4_season_options = [s for s in SEASON_ORDER if s in df_hour_t4['Season'].unique()]
            t4_season_options_all = ["All Seasons"] + t4_season_options
            t4_selected_season = st.selectbox(
                "Filter Season",
                options=t4_season_options_all,
                format_func=lambda s: s if s == "All Seasons" else SEASON_LABELS.get(s, s),
                index=0,
                key="t4_h_season",
            )

        # Filter hourly data
        t4_dh = df_hour_t4[df_hour_t4['City'].isin(selected_cities)].copy()
        if t4_selected_season != "All Seasons":
            t4_dh = t4_dh[t4_dh['Season'] == t4_selected_season]

        # Aggregate: mean pollutant per city per hour (climatological average)
        t4_anim_df = (
            t4_dh.dropna(subset=[t4_h_pollutant])
            .groupby(['Hour', 'City'])[t4_h_pollutant]
            .mean()
            .reset_index()
        )

        # Attach coordinates and region
        t4_anim_df['Latitude'] = t4_anim_df['City'].map(lambda c: CITY_COORDINATES.get(c, (None, None))[0])
        t4_anim_df['Longitude'] = t4_anim_df['City'].map(lambda c: CITY_COORDINATES.get(c, (None, None))[1])
        t4_anim_df['Region'] = t4_anim_df['City'].map(CITY_REGIONS).fillna('Other')
        t4_anim_df = t4_anim_df.dropna(subset=['Latitude', 'Longitude'])
        t4_anim_df['Hour_Label'] = t4_anim_df['Hour'].apply(lambda h: f"{h:02d}:00")
        t4_anim_df['Marker_Size'] = t4_anim_df[t4_h_pollutant].clip(lower=5.0)

        # Shared color axis range across all 24 frames for fair comparison
        c_min = t4_anim_df[t4_h_pollutant].quantile(0.05)
        c_max = t4_anim_df[t4_h_pollutant].quantile(0.95)

        if not t4_anim_df.empty:
            season_label_str = (
                SEASON_LABELS.get(t4_selected_season, t4_selected_season)
                if t4_selected_season != "All Seasons"
                else "All Seasons"
            )

            fig_hour_anim = px.scatter_map(
                t4_anim_df.sort_values('Hour'),
                lat="Latitude",
                lon="Longitude",
                size="Marker_Size",
                color=t4_h_pollutant,
                hover_name="City",
                hover_data={
                    "Region": True,
                    t4_h_pollutant: ":.1f",
                    "Hour_Label": True,
                    "Marker_Size": False,
                    "Latitude": False,
                    "Longitude": False,
                },
                animation_frame="Hour_Label",
                color_continuous_scale=AQI_CONTINUOUS_COLORSCALE,
                range_color=[c_min, c_max],
                size_max=22,
                zoom=3.8,
                center={"lat": 22.0, "lon": 80.0},
                map_style="open-street-map",
                title=(
                    f"Climatological Diurnal Animation: Mean {t4_h_pollutant} by Hour "
                    f"({t4_h_unit}) — {season_label_str}"
                ),
            )

            # Slow the animation slightly so changes are visible
            fig_hour_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 600
            fig_hour_anim.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 300

            fig_hour_anim.update_layout(
                height=550,
                margin=dict(l=5, r=5, t=45, b=5),
            )
            st.plotly_chart(fig_hour_anim, use_container_width=True)

            # Insight: which city and hour shows the peak
            peak_row = t4_anim_df.loc[t4_anim_df[t4_h_pollutant].idxmax()]
            trough_row = t4_anim_df.loc[t4_anim_df[t4_h_pollutant].idxmin()]
            st.markdown(
                f'<div class="insight-box"><strong>Diurnal Spatial Insight:</strong> '
                f'The highest mean {t4_h_pollutant} across all frames occurs in '
                f'<strong>{peak_row["City"]}</strong> at <strong>{peak_row["Hour_Label"]}</strong> '
                f'({peak_row[t4_h_pollutant]:.1f} {t4_h_unit}). '
                f'The lowest is in <strong>{trough_row["City"]}</strong> at '
                f'<strong>{trough_row["Hour_Label"]}</strong> '
                f'({trough_row[t4_h_pollutant]:.1f} {t4_h_unit}). '
                f'Press ▶ to watch the pollution wave build through rush hours and '
                f'dissipate in the early morning.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No hourly data available for the selected cities and season.")

# =========================================================
# TAB 5: LOCKDOWN & SDG 11
# =========================================================
with tab5:
    st.subheader("COVID-19 Lockdown Impact & Implications for SDG 11")
    st.markdown(
        "The nationwide lockdown imposed between late March and June 2020 served as an unprecedented natural experiment "
        "in anthropogenic emissions abatement across industrial, transport, and construction sectors."
    )

    lockdown_df, ld_stats = calculate_lockdown_impact(df_all[df_all['City'].isin(selected_cities)], pollutant=selected_pollutant)

    if lockdown_df is not None:
        # Top headline metrics
        lm_col1, lm_col2, lm_col3 = st.columns(3)
        lm_col1.metric("Cities Evaluated", f"{ld_stats['total_cities_evaluated']} cities")
        lm_col2.metric("Mean Lockdown Change", f"{ld_stats['national_avg_reduction_pct']:.1f}%", help="Average percentage change in March-June 2020 compared to March-June 2018-2019 baseline.")
        top_city_name = ld_stats['top_improved'][0]['City'] if ld_stats['top_improved'] else "N/A"
        top_city_val = ld_stats['top_improved'][0]['Pct_Change'] if ld_stats['top_improved'] else 0
        lm_col3.metric("Greatest Drop", f"{top_city_name} ({top_city_val:.1f}%)")

        col_barchart, col_ldmap = st.columns([1, 1])

        with col_barchart:
            # Sorted percentage change bar chart
            lockdown_df_sorted = lockdown_df.sort_values('Pct_Change', ascending=True)
            fig_bar = px.bar(
                lockdown_df_sorted,
                x="Pct_Change",
                y="City",
                orientation='h',
                color="Pct_Change",
                color_continuous_scale="RdYlGn_r",
                title=f"% Change in {selected_pollutant} (Mar–Jun 2020 vs 2018–2019)",
                labels={"Pct_Change": "% Change in Concentration", "City": "City"}
            )
            fig_bar.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.6)
            fig_bar.update_layout(
                height=max(450, len(lockdown_df_sorted) * 20),
                margin=dict(l=10, r=10, t=35, b=10)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_ldmap:
            # Map of % Change
            lockdown_df['Marker_Size'] = lockdown_df['Pre_Lockdown_Mean'].fillna(10.0).clip(lower=8.0)
            fig_ld_map = px.scatter_map(
                lockdown_df,
                lat="Latitude",
                lon="Longitude",
                color="Pct_Change",
                size="Marker_Size",
                hover_name="City",
                hover_data={
                    "Pre_Lockdown_Mean": ":.1f",
                    "Lockdown_2020_Mean": ":.1f",
                    "Pct_Change": ":.1f",
                    "Region": True,
                    "Marker_Size": False,
                    "Latitude": False,
                    "Longitude": False
                },
                color_continuous_scale="RdYlGn_r",
                size_max=20,
                zoom=3.6,
                center={"lat": 22.0, "lon": 80.0},
                map_style="open-street-map",
                title=f"Spatial Abatement: % Change in {selected_pollutant}"
            )
            fig_ld_map.update_layout(margin=dict(l=5, r=5, t=35, b=5), height=max(450, len(lockdown_df_sorted) * 20))
            st.plotly_chart(fig_ld_map, use_container_width=True)

        st.markdown("---")

        # -----------------------------------------------------
        # Dynamic SDG 11 Implications Section
        # -----------------------------------------------------
        st.subheader("Implications for SDG 11 (Sustainable Cities, Target 11.6)")

        top_imp_list = ", ".join([f"**{r['City']}** ({r['Pct_Change']:.1f}%)" for r in ld_stats['top_improved']])
        top_pol_list = ", ".join([f"**{r['City']}** ({r['Lockdown_2020_Mean']:.1f} {unit})" for r in ld_stats['most_polluted_lockdown']])
        least_imp_list = ", ".join([f"**{r['City']}** ({r['Pct_Change']:+.1f}%)" for r in ld_stats['least_improved']])

        st.markdown(rf"""
        ### 1. Priority Cities & The "Clean Air Ceiling"
        - **Most Improved Cities:** The largest relative drops were recorded in {top_imp_list}. Heavy vehicular and industrial abatement resulted in unprecedented blue skies in these metropolitan centers.
        - **Persistent Hotspots Even Under Lockdown:** Strikingly, cities like {top_pol_list} remained substantially elevated above WHO annual guidance ($5\,\mu\text{{g/m}}^3$) even when non-essential activities were halted. This proves that urban mobility restrictions alone cannot solve pollution in the Indo-Gangetic Plain; background sources like thermal power plants, regional biomass burning, residential cooking fuels, and trans-boundary meteorological transport establish an intractable baseline.
        - **Least Responsive Cities:** In contrast, {least_imp_list} experienced marginal reductions, reflecting coastal buffering or local industrial clusters that were exempted as essential infrastructure.

        ### 2. Seasonal Action Windows for Municipal Policy
        Under **SDG 11 Target 11.6**, municipal governments must align intervention timing with meteorological vulnerability:
        - **Post-Monsoon Emergency Window (October – November):** Stubble burning in northwestern agricultural belts combined with post-harvest biomass burning and Diwali festivities triggers acute regional pollution emergencies. Dynamic airshed-level interventions (e.g. coordinated stubble bio-decomposer subsidies and mechanized harvesters) are essential.
        - **Winter Atmospheric Trap (December – February):** Low planetary boundary layer heights and thermal inversions mechanically trap particulates near ground level. Blanket bans on diesel generator sets, strict construction dust controls, and night-time anti-smog water spraying must be front-loaded in these months.
        - **Pre-Monsoon & Monsoon Wash-Out (June – September):** Wet deposition naturally reduces ambient particulates across the peninsula. This represents the optimal operational window for maintenance of CAAQMS sensor networks and major civil works with minimal public health exposure risk.

        ### 3. Methodological Caveat: Weather Confounding
        While the 2020 lockdown enforced dramatic anthropogenic restrictions, air quality is simultaneously governed by meteorology (wind speed, direction, temperature, and planetary boundary layer dynamics). Unusually early rainfall or altered wind vectors can amplify or mask true emission reductions. Causal policy attributions must combine CPCB ground data with chemical transport modeling (e.g. WRF-Chem) to isolate weather from human activity.
        """)

    else:
        st.warning(f"Lockdown analysis could not be calculated: {ld_stats.get('error', 'Unknown issue')}")

    with st.expander("About / Method: Lockdown & SDG 11 Evaluation", expanded=False):
        st.markdown("""
        - **Baseline Window:** Average of March 1 to June 30 across 2018 and 2019 to control for single-year weather anomalies.
        - **Treatment Window:** March 1 to June 30, 2020 (encompassing Lockdown Phases 1–4 and Unlock 1.0).
        - **Formula:** 
          $$\\%\\,\\Delta = \\frac{\\bar{X}_{2020} - \\bar{X}_{2018-2019}}{\\bar{X}_{2018-2019}} \\times 100$$
        - **SDG 11 Target 11.6:** *"By 2030, reduce the adverse per capita environmental impact of cities, including by paying special attention to air quality and municipal and other waste management."*
        """)

# =========================================================
# TAB 6: DIURNAL & HOURLY PATTERNS
# =========================================================
with tab6:
    st.subheader("Diurnal Air Quality Patterns — Hourly Granularity Analysis")
    st.markdown(
        "Using the extended **city-hour** dataset (4.4 M hourly records, 2015–2020), "
        "this tab reveals *within-day* pollution rhythms that daily averages mask: "
        "morning rush-hour spikes, overnight recovery, weekday vs weekend divergence, "
        "and how the COVID-19 lockdown reshaped the diurnal curve."
    )

    # ---- lazy load hourly data ----
    df_hour, hour_status = load_hourly_data("data/city_hour_transformed.csv")

    if df_hour is None:
        st.error(
            f"Hourly dataset unavailable: {hour_status}  \n"
            "Place `city_hour_transformed.csv` inside the `data/` folder and restart."
        )
    else:
        st.caption(hour_status)

        # ---- Tab-6 specific filters ----
        h6c1, h6c2, h6c3 = st.columns(3)

        with h6c1:
            hourly_pollutants = [p for p in ['PM2.5', 'NO2', 'AQI'] if p in df_hour.columns]
            h_pollutant = st.selectbox(
                "Pollutant",
                options=hourly_pollutants,
                index=0,
                key="h6_pollutant"
            )
            h_unit = POLLUTANT_UNITS.get(h_pollutant, "µg/m³")

        with h6c2:
            all_h_cities = sorted(df_hour['City'].unique().tolist())
            h_default_cities = [c for c in selected_cities if c in all_h_cities] or all_h_cities[:5]
            h_cities = st.multiselect(
                "Cities",
                options=all_h_cities,
                default=h_default_cities,
                key="h6_cities"
            )

        with h6c3:
            h_seasons_raw = [s for s in SEASON_ORDER if s in df_hour['Season'].unique()]
            h_seasons = st.multiselect(
                "Seasons",
                options=h_seasons_raw,
                default=h_seasons_raw,
                format_func=lambda s: SEASON_LABELS.get(s, s),
                key="h6_seasons"
            )

        # Apply filters
        mask = (
            df_hour['City'].isin(h_cities) &
            df_hour['Season'].isin(h_seasons if h_seasons else SEASON_ORDER)
        )
        dh = df_hour[mask].copy()

        if dh.empty or not h_cities:
            st.warning("No hourly data for the current selection. Adjust the city or season filters above.")
        else:
            st.markdown("---")

            # ==================================================
            # CHART 1: Mean Diurnal Curve by City
            # ==================================================
            st.markdown(f"#### 1. Average Diurnal Curve by City ({h_pollutant})")
            st.caption(
                "Each line shows how the city's mean concentration evolves across the 24 hours of the day, "
                "averaged over all selected seasons and years."
            )

            diurnal_city = (
                dh.groupby(['City', 'Hour'])[h_pollutant]
                .mean()
                .reset_index()
            )

            fig_diurnal = px.line(
                diurnal_city,
                x='Hour',
                y=h_pollutant,
                color='City',
                markers=True,
                title=f"Mean {h_pollutant} by Hour of Day ({h_unit}) — Selected Cities",
                labels={h_pollutant: f"{h_pollutant} ({h_unit})", 'Hour': 'Hour of Day'},
            )
            fig_diurnal.update_xaxes(
                tickvals=list(range(0, 24, 2)),
                ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
            )
            fig_diurnal.update_layout(
                height=430,
                hovermode="x unified",
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_diurnal, use_container_width=True)

            if not diurnal_city.empty:
                overall_diurnal = dh.groupby('Hour')[h_pollutant].mean()
                peak_h = int(overall_diurnal.idxmax())
                trough_h = int(overall_diurnal.idxmin())
                st.markdown(
                    f'<div class="insight-box"><strong>Diurnal Insight:</strong> '
                    f'Across the selected cities, {h_pollutant} peaks around '
                    f'<strong>{peak_h:02d}:00</strong> and reaches its daily minimum around '
                    f'<strong>{trough_h:02d}:00</strong>. '
                    f'Morning rush hours (06:00–09:00) and late-evening traffic (18:00–21:00) '
                    f'typically generate local emission spikes, while early morning hours see '
                    f'overnight atmospheric recovery.</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # ==================================================
            # CHART 2: Weekday vs Weekend Diurnal Overlay
            # ==================================================
            st.markdown(f"#### 2. Weekday vs Weekend Diurnal Comparison ({h_pollutant})")
            st.caption(
                "Weekend traffic patterns differ significantly from weekdays. "
                "A flatter weekend curve indicates traffic as the dominant emission driver."
            )

            if 'Weekday_or_weekend' in dh.columns:
                diurnal_ww = (
                    dh.groupby(['Hour', 'Weekday_or_weekend'])[h_pollutant]
                    .mean()
                    .reset_index()
                )
                fig_ww = px.line(
                    diurnal_ww,
                    x='Hour',
                    y=h_pollutant,
                    color='Weekday_or_weekend',
                    markers=True,
                    color_discrete_map={'Weekday': '#2563EB', 'Weekend': '#F59E0B'},
                    title=f"Weekday vs Weekend Diurnal Profile ({h_pollutant}, {h_unit})",
                    labels={
                        h_pollutant: f"{h_pollutant} ({h_unit})",
                        'Hour': 'Hour of Day',
                        'Weekday_or_weekend': 'Day Type',
                    },
                )
                fig_ww.update_xaxes(
                    tickvals=list(range(0, 24, 2)),
                    ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
                )
                fig_ww.update_layout(
                    height=380,
                    hovermode="x unified",
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_ww, use_container_width=True)

                ww_pivot = diurnal_ww.pivot(index='Hour', columns='Weekday_or_weekend', values=h_pollutant)
                if 'Weekday' in ww_pivot.columns and 'Weekend' in ww_pivot.columns:
                    max_gap_hour = int((ww_pivot['Weekday'] - ww_pivot['Weekend']).abs().idxmax())
                    weekday_mean = round(ww_pivot['Weekday'].mean(), 1)
                    weekend_mean = round(ww_pivot['Weekend'].mean(), 1)
                    pct_diff = round(abs(weekday_mean - weekend_mean) / max(weekend_mean, 0.1) * 100, 1)
                    st.markdown(
                        f'<div class="insight-box"><strong>Weekday / Weekend Gap:</strong> '
                        f'Daily mean {h_pollutant} is <strong>{weekday_mean} {h_unit}</strong> on weekdays vs '
                        f'<strong>{weekend_mean} {h_unit}</strong> on weekends '
                        f'(~{pct_diff}% difference). '
                        f'The largest hourly divergence occurs at <strong>{max_gap_hour:02d}:00</strong>, '
                        f'consistent with commuter traffic patterns.</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

            # ==================================================
            # CHART 3: Seasonal Diurnal Heatmap
            # ==================================================
            st.markdown(f"#### 3. Seasonal Diurnal Heatmap — Hour × Season ({h_pollutant})")
            st.caption(
                "A 24-hour × 4-season grid showing mean concentration at each hour. "
                "Brighter cells = higher pollution."
            )

            if h_seasons:
                diurnal_season = (
                    dh.groupby(['Season', 'Hour'])[h_pollutant]
                    .mean()
                    .reset_index()
                )
                season_pivot = diurnal_season.pivot(index='Season', columns='Hour', values=h_pollutant)
                season_pivot = season_pivot.reindex([s for s in SEASON_ORDER if s in season_pivot.index])
                season_pivot.index = [SEASON_LABELS.get(s, s) for s in season_pivot.index]

                fig_heat = px.imshow(
                    season_pivot,
                    labels=dict(x="Hour of Day", y="Season", color=f"{h_pollutant} ({h_unit})"),
                    x=[f"{h:02d}:00" for h in season_pivot.columns],
                    color_continuous_scale=AQI_CONTINUOUS_COLORSCALE,
                    aspect="auto",
                    title=f"Seasonal Diurnal Heatmap: Mean {h_pollutant} by Season & Hour ({h_unit})",
                )
                fig_heat.update_xaxes(tickangle=-45)
                fig_heat.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_heat, use_container_width=True)

                st.markdown(
                    '<div class="insight-box"><strong>Seasonal × Diurnal Insight:</strong> '
                    'Winter cells are uniformly brighter, reflecting the compression of the planetary '
                    'boundary layer that traps pollutants regardless of the time of day. '
                    'During Monsoon, all hours show near-uniform low values due to wet '
                    'deposition—even peak traffic hours cannot overcome rain-driven '
                    'atmospheric flushing.</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # ==================================================
            # CHART 4: Lockdown Diurnal Shift
            # ==================================================
            st.markdown(f"#### 4. COVID-19 Lockdown: Before vs During Diurnal Shift ({h_pollutant})")
            st.caption(
                "Comparing the March–June hourly profile in 2019 (pre-lockdown baseline) "
                "against March–June 2020 (lockdown). A flattened or lowered curve "
                "shows the specific hours most impacted by mobility restrictions."
            )

            dh_lockdown_window = df_hour[
                df_hour['City'].isin(h_cities) &
                df_hour['Datetime'].dt.month.isin([3, 4, 5, 6])
            ].copy()

            pre_ld = dh_lockdown_window[dh_lockdown_window['Year'] == 2019]
            during_ld = dh_lockdown_window[dh_lockdown_window['Year'] == 2020]

            if not pre_ld.empty and not during_ld.empty:
                pre_diurnal = pre_ld.groupby('Hour')[h_pollutant].mean().reset_index()
                pre_diurnal['Period'] = '2019 (Pre-Lockdown Baseline)'

                during_diurnal = during_ld.groupby('Hour')[h_pollutant].mean().reset_index()
                during_diurnal['Period'] = '2020 (Lockdown)'

                ld_compare = pd.concat([pre_diurnal, during_diurnal], ignore_index=True)

                fig_ld_diurnal = px.line(
                    ld_compare,
                    x='Hour',
                    y=h_pollutant,
                    color='Period',
                    markers=True,
                    color_discrete_map={
                        '2019 (Pre-Lockdown Baseline)': PRELOCKDOWN_COLOR,
                        '2020 (Lockdown)': LOCKDOWN_COLOR,
                    },
                    title=f"Diurnal Profile: Mar–Jun 2019 vs Mar–Jun 2020 Lockdown ({h_pollutant}, {h_unit})",
                    labels={
                        h_pollutant: f"{h_pollutant} ({h_unit})",
                        'Hour': 'Hour of Day',
                        'Period': 'Period',
                    },
                )
                fig_ld_diurnal.update_xaxes(
                    tickvals=list(range(0, 24, 2)),
                    ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
                )
                fig_ld_diurnal.update_layout(
                    height=400,
                    hovermode="x unified",
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_ld_diurnal, use_container_width=True)

                merged_ld = pre_diurnal.set_index('Hour')[[h_pollutant]].rename(columns={h_pollutant: 'pre'})
                merged_ld['during'] = during_diurnal.set_index('Hour')[h_pollutant]
                merged_ld['pct_drop'] = (merged_ld['during'] - merged_ld['pre']) / merged_ld['pre'] * 100
                peak_drop_h = int(merged_ld['pct_drop'].idxmin())
                peak_drop_pct = round(merged_ld['pct_drop'].min(), 1)
                overall_drop = round(merged_ld['pct_drop'].mean(), 1)

                st.markdown(
                    f'<div class="insight-box"><strong>Lockdown Diurnal Shift:</strong> '
                    f'Across the selected cities, the overall hourly mean dropped by '
                    f'<strong>{overall_drop}%</strong> during lockdown. '
                    f'The steepest reduction was at <strong>{peak_drop_h:02d}:00</strong> '
                    f'({peak_drop_pct:.1f}%), typically the morning rush-hour slot, '
                    f'directly attributable to near-zero vehicular traffic on roads. '
                    f'Night-time hours show smaller reductions, consistent with '
                    f'industrial and thermal baseline sources continuing to operate.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("Insufficient 2019 or 2020 data in the selected cities for lockdown diurnal comparison.")

            st.markdown("---")

            # ==================================================
            # CHART 5: Day-Period AQ Acceptability Breakdown
            # ==================================================
            st.markdown("#### 5. Air Quality Acceptability by Day Period & Region")
            st.caption(
                "The dataset pre-classifies each hourly record as 'Acceptable' or 'Unacceptable' "
                "based on CPCB thresholds. This chart shows the share of unacceptable hours "
                "broken down by time-of-day period and geographic region."
            )

            if 'AQ_Acceptability' in dh.columns and 'Day_period' in dh.columns:
                dh_acc = dh[dh['Region'] != 'Other'].copy()
                dh_acc['Day_period_Label'] = dh_acc['Day_period'].map(DAY_PERIOD_LABELS).fillna(dh_acc['Day_period'])

                acc_summary = (
                    dh_acc.groupby(['Region', 'Day_period', 'Day_period_Label'])
                    .apply(lambda g: (g['AQ_Acceptability'] == 'Unacceptable').mean() * 100, include_groups=False)
                    .reset_index(name='Pct_Unacceptable')
                )
                acc_summary['Day_period_rank'] = acc_summary['Day_period'].map(
                    {v: i for i, v in enumerate(DAY_PERIOD_ORDER)}
                )
                acc_summary = acc_summary.sort_values(['Region', 'Day_period_rank'])

                fig_acc = px.bar(
                    acc_summary,
                    x='Day_period_Label',
                    y='Pct_Unacceptable',
                    color='Region',
                    barmode='group',
                    title="% of Hours with Unacceptable AQ by Day Period & Region",
                    labels={
                        'Day_period_Label': 'Time of Day',
                        'Pct_Unacceptable': '% Unacceptable Hours',
                        'Region': 'Region',
                    },
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_acc.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=40, b=10),
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
                )
                st.plotly_chart(fig_acc, use_container_width=True)

                igp_acc = acc_summary[acc_summary['Region'] == 'Indo-Gangetic Plain / North']['Pct_Unacceptable'].mean()
                south_acc = acc_summary[acc_summary['Region'] == 'South']['Pct_Unacceptable'].mean()
                if not (pd.isna(igp_acc) or pd.isna(south_acc)):
                    st.markdown(
                        f'<div class="insight-box"><strong>Regional Disparity:</strong> '
                        f'On average, <strong>{igp_acc:.1f}%</strong> of hourly readings in the '
                        f'Indo-Gangetic Plain are classified Unacceptable vs only '
                        f'<strong>{south_acc:.1f}%</strong> in Southern cities — a '
                        f'{round(igp_acc - south_acc, 1)} percentage-point gap that persists '
                        f'across all times of day, reflecting structural rather than '
                        f'behavioural pollution sources in the North.</div>',
                        unsafe_allow_html=True,
                    )

        with st.expander("About / Method: Diurnal Analysis", expanded=False):
            st.markdown("""
            - **Data Source:** `city_hour_transformed.csv` — an extended CPCB hourly dataset
              with pre-engineered columns for `Day_period`, `Season`, `Weekday_or_weekend`,
              and `AQ_Acceptability` (4.4 M rows, ~1 GB).
            - **Diurnal Curve:** Mean pollutant grouped by `Hour` (0–23) across all selected cities and seasons.
            - **Lockdown Comparison:** Restricts to March–June window in 2019 (baseline) vs 2020
              (lockdown), consistent with the Tab 5 quasi-experimental design.
            - **AQ Acceptability:** Pre-classified by the dataset provider using CPCB threshold bands;
              'Unacceptable' corresponds broadly to AQI > 100 (Moderate or worse).
            - **Memory Efficiency:** Only 11 columns are read from the 1 GB file using chunked
              reading (500 k rows/chunk) and Streamlit `@st.cache_data` to avoid repeated I/O.
            """)

# Footer
st.markdown("---")
st.caption("CPCB India Air Quality Spatio-Temporal Demo &bull; Built with Streamlit, Plotly, Statsmodels, Scikit-Learn & Libpysal &bull; SDG 11.6 Micro Project")
