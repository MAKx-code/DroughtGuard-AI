"""Synthetic Karnataka drought dataset — clearly labelled DEMO data."""
import pandas as pd
import numpy as np
from config import DISTRICTS, DISTRICT_TALUKS

# Seed for reproducibility
rng = np.random.default_rng(42)

# ── Realistic synthetic parameters per district (baseline stress level) ──────
DISTRICT_STRESS = {
    "Kalaburagi":  0.78,
    "Raichur":     0.72,
    "Koppal":      0.68,
    "Vijayapura":  0.70,
    "Ballari":     0.65,
    "Chitradurga": 0.60,
    "Tumakuru":    0.52,
    "Kolar":       0.55,
}

DISTRICT_COORDS = {
    "Kalaburagi":  (17.329, 76.820),
    "Raichur":     (16.212, 77.356),
    "Koppal":      (15.351, 76.154),
    "Vijayapura":  (16.830, 75.715),
    "Ballari":     (15.140, 76.924),
    "Chitradurga": (14.230, 76.401),
    "Tumakuru":    (13.341, 77.101),
    "Kolar":       (13.136, 78.129),
}

DISTRICT_MAIN_CROP = {
    "Kalaburagi":  "Tur Dal (Pigeon Pea)",
    "Raichur":     "Paddy (Kharif)",
    "Koppal":      "Sunflower",
    "Vijayapura":  "Sugarcane",
    "Ballari":     "Cotton",
    "Chitradurga": "Groundnut",
    "Tumakuru":    "Coconut / Ragi",
    "Kolar":       "Tomato / Mulberry",
}


def _taluk_row(district: str, taluk: str, stress: float) -> dict:
    """Generate one synthetic taluk data row."""
    jitter = rng.uniform(-0.12, 0.12)
    s = max(0.05, min(0.95, stress + jitter))

    normal_rainfall = rng.uniform(480, 720)   # mm per season
    deficit_pct = s * 55 + rng.uniform(-5, 5)  # 0–60 % deficit
    received = normal_rainfall * (1 - deficit_pct / 100)

    temperature = 28 + s * 10 + rng.uniform(-1, 1)
    humidity = max(20, 75 - s * 40 + rng.uniform(-5, 5))
    soil_moisture = max(5, 60 - s * 50 + rng.uniform(-3, 3))
    gw_depth = 5 + s * 25 + rng.uniform(-2, 2)   # meters below ground level
    reservoir_pct = max(5, 85 - s * 75 + rng.uniform(-5, 5))
    dry_spell_days = int(s * 60 + rng.uniform(-5, 5))
    et_anomaly = s * 2.5 + rng.uniform(-0.3, 0.3)   # mm/day above normal
    ndvi = max(0.05, 0.75 - s * 0.60 + rng.uniform(-0.03, 0.03))
    cultivated_area = rng.uniform(15000, 80000)  # hectares
    sm_farmers_pct = rng.uniform(55, 85)
    livestock_risk = "High" if s > 0.65 else "Moderate" if s > 0.40 else "Low"
    water_avail = "Critical" if s > 0.70 else "Stressed" if s > 0.45 else "Adequate"

    lat, lon = DISTRICT_COORDS[district]
    t_lat = lat + rng.uniform(-0.3, 0.3)
    t_lon = lon + rng.uniform(-0.3, 0.3)

    return {
        "district":            district,
        "taluk":               taluk,
        "latitude":            round(t_lat, 4),
        "longitude":           round(t_lon, 4),
        "rainfall_received_mm":round(received, 1),
        "normal_rainfall_mm":  round(normal_rainfall, 1),
        "rainfall_deviation_pct": round(-deficit_pct, 1),
        "dry_spell_days":      dry_spell_days,
        "temperature_c":       round(temperature, 1),
        "humidity_pct":        round(humidity, 1),
        "soil_moisture_pct":   round(soil_moisture, 1),
        "groundwater_depth_m": round(gw_depth, 1),
        "reservoir_storage_pct": round(reservoir_pct, 1),
        "et_anomaly_mm_day":   round(et_anomaly, 2),
        "ndvi":                round(ndvi, 3),
        "main_crop":           DISTRICT_MAIN_CROP[district],
        "cultivated_area_ha":  round(cultivated_area),
        "sm_farmer_pct":       round(sm_farmers_pct, 1),
        "livestock_fodder_risk": livestock_risk,
        "drinking_water_status": water_avail,
        "data_source":         "SYNTHETIC DEMO DATA — Not official government data",
        "season":              "Kharif 2024",
    }


def generate_dataset() -> pd.DataFrame:
    rows = []
    for district in DISTRICTS:
        stress = DISTRICT_STRESS[district]
        for taluk in DISTRICT_TALUKS[district]:
            rows.append(_taluk_row(district, taluk, stress))
    return pd.DataFrame(rows)


# Pre-built dataset (singleton)
_df: pd.DataFrame | None = None


def get_dataset() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = generate_dataset()
    return _df


def get_district_summary() -> pd.DataFrame:
    df = get_dataset()
    grp = df.groupby("district").agg(
        lat=("latitude", "mean"),
        lon=("longitude", "mean"),
        rainfall_received_mm=("rainfall_received_mm", "mean"),
        normal_rainfall_mm=("normal_rainfall_mm", "mean"),
        rainfall_deviation_pct=("rainfall_deviation_pct", "mean"),
        dry_spell_days=("dry_spell_days", "mean"),
        temperature_c=("temperature_c", "mean"),
        soil_moisture_pct=("soil_moisture_pct", "mean"),
        groundwater_depth_m=("groundwater_depth_m", "mean"),
        reservoir_storage_pct=("reservoir_storage_pct", "mean"),
        et_anomaly_mm_day=("et_anomaly_mm_day", "mean"),
        ndvi=("ndvi", "mean"),
        sm_farmer_pct=("sm_farmer_pct", "mean"),
        main_crop=("main_crop", "first"),
    ).reset_index()
    return grp


def get_taluk_data(district: str, taluk: str) -> dict | None:
    df = get_dataset()
    row = df[(df["district"] == district) & (df["taluk"] == taluk)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()
