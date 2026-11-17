"""Deterministic drought risk scoring engine."""
from config import DROUGHT_WEIGHTS, SEVERITY_THRESHOLDS, SEVERITY_COLORS


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _rainfall_score(deviation_pct: float) -> float:
    """Deviation is negative (deficit). Score 0–100."""
    if deviation_pct >= 0:
        return 0.0
    deficit = abs(deviation_pct)
    if deficit < 10:  return deficit * 1.5
    if deficit < 25:  return 15 + (deficit - 10) * 2.0
    if deficit < 50:  return 45 + (deficit - 25) * 1.4
    return min(100, 80 + (deficit - 50) * 0.4)


def _soil_moisture_score(sm_pct: float) -> float:
    """Lower soil moisture → higher score."""
    if sm_pct >= 50: return 0.0
    if sm_pct >= 30: return (50 - sm_pct) * 2.5
    if sm_pct >= 15: return 50 + (30 - sm_pct) * 3.0
    return min(100, 95 + (15 - sm_pct) * 0.5)


def _dry_spell_score(days: int) -> float:
    if days < 7:   return 0.0
    if days < 15:  return days * 2.0
    if days < 30:  return 30 + (days - 15) * 2.5
    if days < 45:  return 67 + (days - 30) * 1.5
    return min(100, 90 + (days - 45) * 0.5)


def _groundwater_score(depth_m: float) -> float:
    """Deeper depth → higher stress."""
    if depth_m < 5:   return 0.0
    if depth_m < 10:  return (depth_m - 5) * 6.0
    if depth_m < 20:  return 30 + (depth_m - 10) * 4.0
    if depth_m < 30:  return 70 + (depth_m - 20) * 2.0
    return min(100, 90 + (depth_m - 30) * 0.5)


def _reservoir_score(storage_pct: float) -> float:
    """Lower storage → higher score."""
    if storage_pct >= 70: return 0.0
    if storage_pct >= 40: return (70 - storage_pct) * 2.0
    if storage_pct >= 20: return 60 + (40 - storage_pct) * 1.5
    return min(100, 90 + (20 - storage_pct) * 0.5)


def _temperature_score(et_anomaly: float) -> float:
    """Higher ET anomaly → higher stress."""
    if et_anomaly <= 0: return 0.0
    if et_anomaly < 1:  return et_anomaly * 25
    if et_anomaly < 2:  return 25 + (et_anomaly - 1) * 35
    return min(100, 60 + (et_anomaly - 2) * 25)


def _vulnerability_score(sm_farmer_pct: float, livestock_risk: str) -> float:
    base = (sm_farmer_pct - 50) / 35 * 60  # 50-85% maps to 0-60
    livestock_add = {"High": 40, "Moderate": 20, "Low": 0}.get(livestock_risk, 0)
    return _clamp((base + livestock_add) / 1.4, 0, 100)


def calculate_risk(row: dict) -> dict:
    """
    Calculate drought risk score (0-100) and severity for a taluk/district.
    Returns a dict with score, severity, factor scores and explanation.
    """
    factors = {
        "rainfall_deficit":       _rainfall_score(row.get("rainfall_deviation_pct", 0)),
        "soil_moisture_deficit":  _soil_moisture_score(row.get("soil_moisture_pct", 40)),
        "dry_spell_duration":     _dry_spell_score(row.get("dry_spell_days", 0)),
        "groundwater_stress":     _groundwater_score(row.get("groundwater_depth_m", 5)),
        "reservoir_storage":      _reservoir_score(row.get("reservoir_storage_pct", 60)),
        "temperature_anomaly":    _temperature_score(row.get("et_anomaly_mm_day", 0)),
        "crop_vulnerability":     _vulnerability_score(
                                      row.get("sm_farmer_pct", 65),
                                      row.get("livestock_fodder_risk", "Moderate")),
    }

    weighted = {k: factors[k] * DROUGHT_WEIGHTS[k] for k in factors}
    score = round(sum(weighted.values()), 1)

    severity = "Low"
    for sev, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= score < hi:
            severity = sev
            break
    if score >= 75:
        severity = "Severe"

    # Identify top 3 contributing factors
    top_factors = sorted(weighted.items(), key=lambda x: x[1], reverse=True)[:3]
    top_labels = [k.replace("_", " ").title() for k, _ in top_factors]

    crop_stress = "Critical" if score >= 75 else "High" if score >= 55 else "Moderate" if score >= 30 else "Low"
    water_shortage = row.get("drinking_water_status", "Adequate")

    # Recommended interventions
    interventions = []
    if severity in ("High", "Severe"):
        interventions.extend([
            "Activate drinking water tanker deployment",
            "Issue crop advisory for contingency measures",
            "Assess MGNREGA employment demand",
        ])
    if severity == "Severe":
        interventions.extend([
            "Declare drought and trigger relief package",
            "Deploy fodder banks in critical taluks",
            "Initiate borewell deepening/rejuvenation",
        ])
    if factors["reservoir_storage"] > 60:
        interventions.append("Prioritize tank and lake storage conservation")
    if not interventions:
        interventions.append("Continue monitoring; no immediate action required")

    return {
        "score":          score,
        "severity":       severity,
        "color":          SEVERITY_COLORS[severity],
        "factor_scores":  {k: round(v, 1) for k, v in factors.items()},
        "weighted_scores": {k: round(v, 2) for k, v in weighted.items()},
        "top_factors":    top_labels,
        "crop_stress":    crop_stress,
        "water_shortage": water_shortage,
        "interventions":  interventions,
        "note":           "Prototype thresholds — not official drought declaration criteria.",
    }


def enrich_dataframe(df):
    """Add risk columns to a taluk dataframe."""
    import pandas as pd
    results = [calculate_risk(row) for _, row in df.iterrows()]
    df = df.copy()
    df["risk_score"]    = [r["score"]    for r in results]
    df["risk_severity"] = [r["severity"] for r in results]
    df["risk_color"]    = [r["color"]    for r in results]
    df["crop_stress"]   = [r["crop_stress"] for r in results]
    return df, results
