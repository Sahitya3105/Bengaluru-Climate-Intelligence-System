# 🌆 Bengaluru Climate Intelligence Platform
**Comprehensive Design and Analysis Report**

This document serves as the detailed architectural and algorithmic breakdown for the **Spatio-Temporal Climate Prediction and Human Risk Assessment** project. It outlines every file, data source, algorithm, and machine learning model used in the platform.

---

## 📂 1. Directory Structure & File Architecture

The project follows a modular, production-grade structure separating data ingestion, machine learning, and UI rendering.

```text
daaProject/
│
├── config/
│   └── settings.py               # Global constants (Geographic bounds, API endpoints, random seeds)
│
├── data/
│   ├── raw/                      # Empty drop-zone for manual CSV datasets
│   ├── synthetic/                # Raw data fetched directly from APIs
│   └── processed/                # Cleaned, merged, and featurized data ready for ML
│
├── pipeline/
│   ├── data_collector.py         # Connects to NASA & Open-Meteo APIs to fetch 20 years of data
│   ├── preprocessor.py           # Handles missing values, outliers, and dataset merging
│   └── feature_engineer.py       # Calculates advanced metrics (Heat Index, rolling averages, lags)
│
├── models/
│   ├── model_trainer.py          # The core Machine Learning script that trains all 7 algorithms
│   └── saved/                    # Stores the evaluation metrics (RMSE, Accuracy) in JSON format
│
├── analysis/
│   ├── health_risk.py            # Calculates human impact scores (Dengue, Respiratory, Heat)
│   ├── seasonal_analysis.py      # Aggregates daily data into Bengaluru's 4 distinct seasons
│   └── spatial_analysis.py       # (Optional) K-Means clustering for geographic hotspots
│
├── visualization/
│   ├── climate_charts.py         # Plotly rendering logic (Trend lines, Bar charts, Radar charts)
│   └── geo_maps.py               # Folium rendering logic (Interactive Ward Heatmaps)
│
├── app/
│   └── dashboard.py              # The main Streamlit application (The UI you interact with)
│
└── requirements.txt              # Python dependencies (pandas, scikit-learn, xgboost, streamlit, etc.)
```

---

## 📡 2. Data Sources

The platform relies on real-world, open-access meteorological APIs to gather historical data spanning **2005 to 2026**:

1. **Open-Meteo ERA5 Reanalysis Archive:**
   - **What it provides:** Daily Temperature (Min/Max/Mean), Rainfall, Wind Speed.
   - **Why we use it:** ERA5 is a highly respected global climate reanalysis dataset produced by the European Centre for Medium-Range Weather Forecasts (ECMWF).
2. **NASA POWER API (Prediction of Worldwide Energy Resources):**
   - **What it provides:** Relative Humidity, Atmospheric Pressure, Solar Radiation.
   - **Why we use it:** NASA provides hyper-accurate localized atmospheric pressure and humidity data critical for calculating human discomfort (Heat Index).
3. **Open-Meteo CAMS (Copernicus Atmosphere Monitoring Service):**
   - **What it provides:** Daily Air Quality Index (AQI), PM2.5, PM10, NO2, SO2, and Ozone levels.
4. **Simulated Geospatial Data:**
   - Because real Ward Shapefiles and GeoTIFF satellite imagery (Landsat/Sentinel) require manual, paid, or authenticated downloads, the `data_collector.py` generates highly realistic statistical simulations for Land-Use (NDVI, Built-up area) and the 150 Bengaluru geographic wards based on known urban expansion rates.

---

## ⚙️ 3. The Data Engineering Pipeline

Before algorithms can be trained, the raw data must pass through an automated ETL (Extract, Transform, Load) pipeline.

### Step A: Preprocessing (`preprocessor.py`)
*   **Missing Value Imputation:** Uses Forward-Fill (`ffill`) and Backward-Fill (`bfill`) to interpolate missing API data.
*   **Outlier Detection:** Removes mathematically impossible anomalies (e.g., temperatures below 0°C in Bengaluru).
*   **Categorical Encoding:** Converts continuous numbers into distinct classes (e.g., mapping AQI `> 100` to `"Poor"`).

### Step B: Feature Engineering (`feature_engineer.py`)
This is the most critical step for the Time-Series forecasting algorithms.
*   **Steadman Heat Index:** Calculates what the temperature "feels like" by combining Temp and Humidity.
*   **Urban Heat Island (UHI) Intensity:** Estimates localized temperature spikes caused by concrete density.
*   **Lag Features:** Creates t-1 and t-7 columns. It tells the AI what the weather was exactly 1 day ago and 1 week ago, allowing the algorithms to detect momentum.
*   **Rolling Averages:** 7-day and 30-day moving averages to smooth out daily noise.

---

## 🧠 4. Algorithms & Machine Learning Models

The platform trains **7 distinct predictive models**, spanning both Classical Machine Learning and Deep Learning architectures. 

*(Note: If TensorFlow/Keras is not installed on the host machine, the `model_trainer.py` script automatically falls back to powerful Random Forest Proxies to ensure the pipeline doesn't crash).*

### 1. LSTM (Long Short-Term Memory)
*   **Task:** Time-Series Temperature Forecasting.
*   **Why this algorithm?** LSTMs are Recurrent Neural Networks (RNNs) specifically designed to remember past data (using "memory cells"). This makes them the industry standard for predicting weather, as tomorrow's temperature is highly dependent on the last 7 days.

### 2. GRU (Gated Recurrent Unit)
*   **Task:** Time-Series AQI (Pollution) Forecasting.
*   **Why this algorithm?** Similar to LSTM but with a simplified architecture. It trains faster and is highly effective at predicting volatile, spiky data like winter pollution levels.

### 3. CNN-LSTM (Convolutional LSTM)
*   **Task:** Rainfall Prediction.
*   **Why this algorithm?** Combines the spatial pattern recognition of a CNN with the temporal memory of an LSTM. It is excellent at detecting sudden anomalies (like monsoon cloud bursts).

### 4. Transformer Architecture
*   **Task:** Long-Range Climate Trend Forecasting.
*   **Why this algorithm?** Transformers use "Self-Attention" mechanisms to look at the entire 20-year dataset simultaneously, making them superior for detecting long-term macro-trends (like global warming over decades).

### 5. Random Forest Classifier
*   **Task:** Heat-Zone Classification (Normal vs. Heatwave).
*   **Why this algorithm?** An ensemble learning method that builds hundreds of decision trees and merges them. It is highly resistant to overfitting and excellent for distinct categorical classification.

### 6. XGBoost (Extreme Gradient Boosting)
*   **Task:** Flood Risk Classification (Low, Medium, High).
*   **Why this algorithm?** XGBoost builds decision trees sequentially, where each new tree specifically tries to correct the errors of the previous one. It is currently the most dominant algorithm in tabular data competitions (like Kaggle).

### 7. RF Health-Risk Multi-Classifier
*   **Task:** Human Health Impact Assessment.
*   **Why this algorithm?** Takes complex multi-dimensional data (Pollution + Heat + Humidity) and classifies the exact human danger level (e.g., "Severe Respiratory Risk").

---

## 📊 5. Evaluation Metrics

The system automatically grades the algorithms during training. The metrics used are:

*   **RMSE (Root Mean Square Error):** Used for forecasting (LSTM, GRU). Measures the average distance between the predicted temperature and the actual temperature. Lower is better.
*   **MAE (Mean Absolute Error):** Similar to RMSE but less sensitive to massive outliers.
*   **R² (R-Squared):** Measures how well the model explains the variance in the data. A score of `1.0` is perfect; `0.90+` is excellent.
*   **Accuracy & F1-Score:** Used for Classifiers (XGBoost, Random Forest). Measures the percentage of times the AI correctly guessed the exact category (e.g., accurately guessing a "High Flood Risk" day).

---

## 🌍 6. Visualization & Dashboard (`app/dashboard.py`)

The final output is rendered via a **Streamlit** Web Dashboard, utilizing:
*   **Plotly Express & Graph Objects:** For interactive, zoomable trend lines, radar charts, and stacked area graphs.
*   **Folium:** For rendering geographic JSON data over real-world interactive map tiles (Heatmaps).
*   **Pandas:** For real-time, in-memory data slicing when the user adjusts the "Year Range" slider.
