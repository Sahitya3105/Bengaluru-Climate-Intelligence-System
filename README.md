# 🌆 Spatio-Temporal Climate Prediction & Human Risk Assessment
## Bengaluru Climate Intelligence Platform (2005–2026)

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> An AI-powered climate intelligence and human risk assessment platform for Bengaluru/Karnataka, integrating multi-source environmental data from 2005–2026.

---
## 🎯 Objective

Develop a research-grade climate analytics and prediction system that analyzes environmental change, predicts future climate conditions, detects urban heat island effects, estimates pollution and flood risks, and evaluates potential human health impacts using AI/ML/DL techniques.

---

## 🏗️ Project Structure

```
daaProject/
├── app/                        # Main Streamlit dashboard
│   └── dashboard.py
├── data/
│   ├── raw/                    # Raw datasets (CSV, NetCDF, GeoTIFF)
│   ├── processed/              # Preprocessed data
│   └── synthetic/              # Synthetically generated demo data
├── pipeline/
│   ├── data_collector.py       # Multi-source data ingestion
│   ├── preprocessor.py         # Cleaning, normalization, alignment
│   └── feature_engineer.py     # Climate indicator engineering
├── models/
│   ├── lstm_model.py           # LSTM temperature/rainfall forecasting
│   ├── gru_model.py            # GRU-based forecasting
│   ├── cnn_lstm_model.py       # CNN-LSTM hybrid
│   ├── transformer_model.py    # Transformer forecaster
│   ├── rf_xgb_model.py         # Random Forest + XGBoost
│   ├── convlstm_model.py       # ConvLSTM for spatial forecasting
│   └── model_trainer.py        # Unified training & evaluation pipeline
├── analysis/
│   ├── seasonal_analysis.py    # Seasonal pattern analysis
│   ├── spatial_analysis.py     # Ward-wise spatial clustering
│   ├── uhi_analysis.py         # Urban Heat Island analysis
│   └── health_risk.py          # Human health risk assessment
├── visualization/
│   ├── climate_charts.py       # Plotly climate visualizations
│   ├── risk_dashboard.py       # Risk indicator dashboards
│   └── geo_maps.py             # Folium/Leaflet GIS maps
├── config/
│   └── settings.py             # Global configuration
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data (for demo without real datasets)
```bash
python pipeline/data_collector.py --synthetic
```

### 3. Run preprocessing
```bash
python pipeline/preprocessor.py
```

### 4. Train models
```bash
python models/model_trainer.py
```

### 5. Launch dashboard
```bash
streamlit run app/dashboard.py
```

---

## 📊 Datasets Integrated

| Source | Type | Variables |
|--------|------|-----------|
| IMD Gridded Data | CSV/NetCDF | Temperature, Rainfall |
| NASA POWER API | API/JSON | Solar Radiation, Wind, Humidity |
| ERA5/ERA5-Land | NetCDF | Reanalysis climate variables |
| CPCB/OpenCity | CSV | AQI, PM2.5, PM10, NO₂, SO₂ |
| Bhuvan/NRSC | GeoTIFF | LULC, NDVI, LST |
| Landsat/Sentinel | GeoTIFF | Satellite imagery bands |

---

## 🤖 AI/ML Models

| Model | Task |
|-------|------|
| LSTM | Temperature & Rainfall time-series forecasting |
| GRU | AQI prediction |
| CNN-LSTM | Multi-variable climate forecasting |
| Transformer | Long-range climate forecasting |
| Random Forest | Heat-zone classification |
| XGBoost | Flood-risk classification |
| ConvLSTM | Spatial climate pattern prediction |

---

## 🏥 Health Risk Outputs

- Heat exhaustion & dehydration risk
- Respiratory disease risk index
- Dengue/mosquito susceptibility
- Flood-related health hazards
- Human productivity reduction index

---

## 📈 Prediction Classes

| Variable | Classes |
|----------|---------|
| Temperature | Normal / Hot / Very Hot / Heat-Wave-Like |
| Rainfall | No Rain / Light / Moderate / Heavy / Extreme |
| AQI | Good / Moderate / Poor / Very Poor |
| Human Risk | Low / Moderate / High / Severe |
| Flood Risk | Low / Medium / High |

---

## 👥 Authors

DAA Project — Bengaluru Climate Intelligence Platform  
Academic Year 2025–2026
