# India Air Quality Spatio-Temporal Analysis

A clean, interactive Streamlit application exploring spatio-temporal air quality dynamics across 26 Indian metropolitan centers (2015–2020), contextualized within **United Nations Sustainable Development Goal 11 (Sustainable Cities and Communities)**, specifically **Target 11.6**:
> *"By 2030, reduce the adverse per capita environmental impact of cities, including by paying special attention to air quality and municipal and other waste management."*

---

## Quickstart & Run Instructions

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Recommended: A virtual environment

### 2. Environment Setup
```bash
# Clone or navigate to the repository directory
cd bdsma-minip

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Data Setup & Synthetic Fallback

The project is designed to use the Kaggle dataset **[Air Quality Data in India (2015-2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)** by Rohan Rao (CPCB-sourced).

### Using the Real Dataset:
1. Place `city_day.csv` (and optionally `stations.csv`) inside a `data/` folder in the project root:
   ```
   bdsma-minip/
   ├── data/
   │   ├── city_day.csv
   │   └── stations.csv
   ├── app.py
   ...
   ```
*(Note: If you have `archive.zip` in the root folder, extracting `city_day.csv` into `data/` will populate it immediately).*

### Synthetic Fallback:
If the `data/` folder is empty or `city_day.csv` is missing, the application **automatically detects this and falls back to a realistic synthetic dataset** (26 cities, 2018–2020) reflecting realistic seasonal winter spikes, monsoon wash-outs, and the 2020 lockdown drop. A prominent warning banner indicates sample data mode so the application can be previewed seamlessly in any environment.

---

## Architecture & Project Structure

```
bdsma-minip/
├── app.py                 # Streamlit UI, multi-tab layout, interactive Plotly visualizations
├── utils/
│   ├── __init__.py
│   ├── data.py            # Caching, file inspection, missing data interpolation, AQI bucket classification, city coordinates & regions
│   └── analysis.py        # Spatial autocorrelation (Moran's I), seasonal decomposition, KMeans profile clustering, lockdown analysis
├── requirements.txt       # Python dependencies
└── README.md              # Documentation and run instructions
```

---

## Features & Tab Breakdown

1. **Sidebar Filters**:
   - Dynamic pollutant selection (`PM2.5` [default], `AQI`, `PM10`, `NO2`).
   - Regional filtering (`Indo-Gangetic Plain / North`, `Central & West`, `South`, `East & Northeast`).
   - City multi-select with convenient "Select All" / "Clear All" toggles.
   - Year range slider (2015 to 2020).
   - In-app documentation on preprocessing decisions.

2. **Tab 1: Overview & Coverage**:
   - 4 Macro-KPIs: Monitored cities count, date span, national mean pollutant concentration, and % of monitored city-days in *Poor* or worse AQI categories.
   - City $\times$ Year operational completeness heatmap showing data coverage and historical CAAQMS grid expansion.
   - Clear statement of monitoring limitations.

3. **Tab 2: Temporal & Seasonality**:
   - Annual trend trajectories per city with dynamic takeaway notes.
   - Climatological monthly heatmap (City $\times$ Month) contrasting Indo-Gangetic winter peaks with southern coastal baselines.
   - Classical Additive Seasonal Decomposition ($Y_t = Trend_t + Seasonal_t + Residual_t$) using `statsmodels` on weekly resampled series for any chosen city.

4. **Tab 3: Spatial & Autocorrelation**:
   - Geospatial Scatter Map of India with points colored and sized by mean pollutant level using Indian CPCB AQI color scales.
   - Regional box plots highlighting cross-regional inequality.
   - Global Moran's I spatial autocorrelation via `libpysal` ($k$-NN spatial weights graph, $k=4$) and `esda.moran.Moran`, with p-value, z-score, and plain-English airshed interpretation.

5. **Tab 4: Spatio-Temporal & Clustering**:
   - Time-animated map showing annual and seasonal shifts across the sub-continent.
   - Small-multiples seasonal maps (Winter, Summer, Monsoon, Post-monsoon).
   - Unsupervised K-Means clustering on standardized 12-month annual profiles ($k=2\dots 5$) grouping cities with shared seasonal dynamics.

6. **Tab 5: Lockdown & SDG 11 Policy Implications**:
   - Quantitative evaluation of the March–June 2020 national lockdown against 2018–2019 baseline (sorted % change bar chart and spatial change map).
   - Dynamic identification of top improved cities vs persistent high-pollution airsheds.
   - Targeted municipal policy insights for SDG 11.6 covering seasonal intervention windows (post-monsoon stubble emergency vs winter atmospheric inversion vs monsoon wash-out) and meteorological confounding caveats.

---

## Methodology & Assumptions

- **Missing Data Policy:** Forward & backward linear interpolation per city capped at a **7-day window**. Prolonged outages remain NaN to preserve true station offline status.
- **National AQI Bands:** Classified per CPCB standard breaks: Good (0–50), Satisfactory (51–100), Moderate (101–200), Poor (201–300), Very Poor (301–400), Severe (401+).
- **Spatial Weights:** $k$-nearest neighbors ($k=4$, row-standardized) derived from hardcoded coordinates of the 26 monitored municipal centroids.
