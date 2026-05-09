"""
config/settings.py
Global configuration for the Bengaluru Climate Intelligence Platform.
"""

# ─────────────────────────────────────────────
# Geographic Bounds — Bengaluru Urban Region
# ─────────────────────────────────────────────
BENGALURU_BOUNDS = {
    "lat_min": 12.7342,
    "lat_max": 13.1736,
    "lon_min": 77.3791,
    "lon_max": 77.8235,
    "center_lat": 12.9716,
    "center_lon": 77.5946,
}

# ─────────────────────────────────────────────
# Time Range
# ─────────────────────────────────────────────
DATA_START_YEAR = 2005
DATA_END_YEAR   = 2026
FORECAST_YEARS  = 3  # years ahead to predict

# ─────────────────────────────────────────────
# Seasons (month ranges)
# ─────────────────────────────────────────────
SEASONS = {
    "Winter":           [12, 1, 2],
    "Summer":           [3, 4, 5],
    "Southwest Monsoon":[6, 7, 8, 9],
    "Post-Monsoon":     [10, 11],
}

# ─────────────────────────────────────────────
# Classification Thresholds
# ─────────────────────────────────────────────
TEMP_CLASSES = {
    "Normal":       (0,   30),
    "Hot":          (30,  35),
    "Very Hot":     (35,  40),
    "Heat-Wave":    (40, 999),
}

RAINFALL_CLASSES = {
    "No Rain":  (0,    0.1),
    "Light":    (0.1,  7.5),
    "Moderate": (7.5,  35.5),
    "Heavy":    (35.5, 64.4),
    "Extreme":  (64.4, 9999),
}

AQI_CLASSES = {
    "Good":      (0,   50),
    "Moderate":  (51,  100),
    "Poor":      (101, 200),
    "Very Poor": (201, 9999),
}

FLOOD_RISK_CLASSES  = ["Low", "Medium", "High"]
HEALTH_RISK_CLASSES = ["Low", "Moderate", "High", "Severe"]
DENGUE_RISK_CLASSES = ["Low", "Medium", "High"]

# ─────────────────────────────────────────────
# ML / DL Settings
# ─────────────────────────────────────────────
SEQUENCE_LENGTH    = 30   # days lookback for sequential models
FORECAST_HORIZON   = 7    # days to predict ahead
BATCH_SIZE         = 32
EPOCHS             = 50
LEARNING_RATE      = 0.001
TRAIN_SPLIT        = 0.7
VALIDATION_SPLIT   = 0.15
TEST_SPLIT         = 0.15

RANDOM_SEED = 42

# ─────────────────────────────────────────────
# Feature Sets
# ─────────────────────────────────────────────
METEOROLOGICAL_FEATURES = [
    "temperature_max", "temperature_min", "temperature_mean",
    "rainfall", "humidity", "wind_speed", "pressure",
]

POLLUTION_FEATURES = [
    "pm25", "pm10", "no2", "so2", "co", "o3", "aqi",
]

SATELLITE_FEATURES = [
    "ndvi", "ndbi", "ndwi", "lst",
]

URBAN_FEATURES = [
    "road_density", "buildup_area", "population_density",
]

ENGINEERED_FEATURES = [
    "heat_index", "rainfall_anomaly", "uhi_score",
    "vegetation_loss_score", "water_body_shrinkage",
    "pollution_severity", "discomfort_index",
]

ALL_FEATURES = (
    METEOROLOGICAL_FEATURES
    + POLLUTION_FEATURES
    + SATELLITE_FEATURES
    + URBAN_FEATURES
    + ENGINEERED_FEATURES
)

# ─────────────────────────────────────────────
# Target Variables
# ─────────────────────────────────────────────
REGRESSION_TARGETS     = ["temperature_mean", "rainfall", "aqi"]
CLASSIFICATION_TARGETS = [
    "temp_class", "rain_class", "aqi_class",
    "flood_risk", "health_risk", "dengue_risk",
]

# ─────────────────────────────────────────────
# Bengaluru Wards (representative)
# ─────────────────────────────────────────────
WARD_NAMES = [
    "Yelahanka", "Hebbal", "Sadashivanagar", "Rajajinagar",
    "Malleswaram", "Shivajinagar", "Cubbon Park", "MG Road",
    "Koramangala", "BTM Layout", "Jayanagar", "JP Nagar",
    "Banashankari", "Kengeri", "RR Nagar", "Vijayanagar",
    "Peenya", "Yeshwanthpur", "Mahalakshmi Layout", "Basavanagudi",
    "Lalbagh", "Richmond Town", "Indiranagar", "Whitefield",
    "Marathahalli", "Electronic City", "HSR Layout", "Bommanahalli",
    "Bellandur", "KR Puram",
]

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
import os
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
DATA_SYN_DIR   = os.path.join(BASE_DIR, "data", "synthetic")
MODELS_DIR     = os.path.join(BASE_DIR, "models", "saved")

for _d in [DATA_RAW_DIR, DATA_PROC_DIR, DATA_SYN_DIR, MODELS_DIR]:
    os.makedirs(_d, exist_ok=True)
