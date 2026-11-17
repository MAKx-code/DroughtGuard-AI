"""
MCP Tool Server for DroughtGuard AI.
Exposes 6 tools via FastMCP. Run as a standalone process or imported directly.
"""
import json
from data_service import get_dataset, get_taluk_data, get_district_summary
from drought_engine import calculate_risk
from memory import query_knowledge

# ── Try FastMCP / MCP SDK ─────────────────────────────────────────────────────
try:
    from mcp.server.fastmcp import FastMCP
    _MCP_AVAILABLE = True
except ImportError:
    try:
        from fastmcp import FastMCP
        _MCP_AVAILABLE = True
    except ImportError:
        _MCP_AVAILABLE = False
        FastMCP = None

if _MCP_AVAILABLE:
    mcp = FastMCP("DroughtGuard-Tools")
else:
    mcp = None


# ── Tool definitions (also usable as plain Python functions) ──────────────────

def _get_district_weather_impl(district: str, season: str = "Kharif 2024") -> dict:
    """Return rainfall, temperature and humidity data for a district."""
    df = get_dataset()
    sub = df[df["district"] == district]
    if sub.empty:
        return {"error": f"District '{district}' not found", "data_source": "SYNTHETIC DEMO"}
    row = sub.iloc[0]
    return {
        "district": district,
        "season": season,
        "rainfall_received_mm": float(row["rainfall_received_mm"]),
        "normal_rainfall_mm":   float(row["normal_rainfall_mm"]),
        "rainfall_deviation_pct": float(row["rainfall_deviation_pct"]),
        "dry_spell_days":       int(row["dry_spell_days"]),
        "temperature_c":        float(row["temperature_c"]),
        "humidity_pct":         float(row["humidity_pct"]),
        "et_anomaly_mm_day":    float(row["et_anomaly_mm_day"]),
        "data_source": "SYNTHETIC DEMO DATA",
    }


def _get_taluk_drought_data_impl(district: str, taluk: str) -> dict:
    """Return full drought indicators and risk score for a taluk."""
    row = get_taluk_data(district, taluk)
    if not row:
        return {"error": f"Taluk '{taluk}' in '{district}' not found",
                "data_source": "SYNTHETIC DEMO"}
    risk = calculate_risk(row)
    return {
        "district": district, "taluk": taluk,
        "risk_score": risk["score"], "risk_severity": risk["severity"],
        "top_factors": risk["top_factors"],
        "crop_stress": risk["crop_stress"],
        "soil_moisture_pct": row["soil_moisture_pct"],
        "groundwater_depth_m": row["groundwater_depth_m"],
        "ndvi": row["ndvi"],
        "data_source": "SYNTHETIC DEMO DATA",
    }


def _get_reservoir_status_impl(district: str) -> dict:
    """Return reservoir and groundwater status for a district."""
    df = get_dataset()
    sub = df[df["district"] == district]
    if sub.empty:
        return {"error": f"District '{district}' not found", "data_source": "SYNTHETIC DEMO"}
    avg_res  = float(sub["reservoir_storage_pct"].mean())
    avg_gw   = float(sub["groundwater_depth_m"].mean())
    avg_sm   = float(sub["soil_moisture_pct"].mean())
    critical = avg_res < 20
    return {
        "district": district,
        "avg_reservoir_storage_pct": round(avg_res, 1),
        "avg_groundwater_depth_m":   round(avg_gw, 1),
        "avg_soil_moisture_pct":     round(avg_sm, 1),
        "water_critical": critical,
        "drinking_water_status": sub.iloc[0]["drinking_water_status"],
        "data_source": "SYNTHETIC DEMO DATA",
    }


def _get_crop_information_impl(district: str, season: str = "Kharif") -> dict:
    """Return crop and agricultural data for a district."""
    df = get_dataset()
    sub = df[df["district"] == district]
    if sub.empty:
        return {"error": f"District '{district}' not found", "data_source": "SYNTHETIC DEMO"}
    row = sub.iloc[0]
    return {
        "district": district,
        "season": season,
        "main_crop":          row["main_crop"],
        "cultivated_area_ha": int(row["cultivated_area_ha"]),
        "sm_farmer_pct":      float(row["sm_farmer_pct"]),
        "livestock_risk":     row["livestock_fodder_risk"],
        "ndvi":               float(row["ndvi"]),
        "data_source": "SYNTHETIC DEMO DATA",
    }


def _search_agricultural_knowledge_impl(query: str, n_results: int = 3) -> dict:
    """Search the vector knowledge base for agricultural advisories."""
    results = query_knowledge(query, n_results)
    return {
        "query": query,
        "results": results,
        "source": "DroughtGuard Knowledge Base (ChromaDB)",
    }


def _get_district_vulnerability_impl(district: str) -> dict:
    """Return vulnerability assessment for a district."""
    df = get_dataset()
    sub = df[df["district"] == district]
    if sub.empty:
        return {"error": f"District '{district}' not found", "data_source": "SYNTHETIC DEMO"}
    avg_sm_farmer = float(sub["sm_farmer_pct"].mean())
    high_risk_taluks = sub[sub["drinking_water_status"] == "Critical"]["taluk"].tolist()
    livestock_high   = sub[sub["livestock_fodder_risk"] == "High"]["taluk"].tolist()
    return {
        "district": district,
        "avg_small_marginal_farmer_pct": round(avg_sm_farmer, 1),
        "critical_water_taluks":         high_risk_taluks,
        "high_livestock_risk_taluks":    livestock_high,
        "total_taluks":                  len(sub),
        "data_source": "SYNTHETIC DEMO DATA",
    }


# ── Register tools with MCP if available ─────────────────────────────────────

if _MCP_AVAILABLE and mcp:
    @mcp.tool()
    def get_district_weather(district: str, season: str = "Kharif 2024") -> str:
        return json.dumps(_get_district_weather_impl(district, season))

    @mcp.tool()
    def get_taluk_drought_data(district: str, taluk: str) -> str:
        return json.dumps(_get_taluk_drought_data_impl(district, taluk))

    @mcp.tool()
    def get_reservoir_status(district: str) -> str:
        return json.dumps(_get_reservoir_status_impl(district))

    @mcp.tool()
    def get_crop_information(district: str, season: str = "Kharif") -> str:
        return json.dumps(_get_crop_information_impl(district, season))

    @mcp.tool()
    def search_agricultural_knowledge(query: str, n_results: int = 3) -> str:
        return json.dumps(_search_agricultural_knowledge_impl(query, n_results))

    @mcp.tool()
    def get_district_vulnerability(district: str) -> str:
        return json.dumps(_get_district_vulnerability_impl(district))


# ── Direct call interface (used by agents when MCP protocol not needed) ────────

TOOL_REGISTRY = {
    "get_district_weather":          _get_district_weather_impl,
    "get_taluk_drought_data":        _get_taluk_drought_data_impl,
    "get_reservoir_status":          _get_reservoir_status_impl,
    "get_crop_information":          _get_crop_information_impl,
    "search_agricultural_knowledge": _search_agricultural_knowledge_impl,
    "get_district_vulnerability":    _get_district_vulnerability_impl,
}

ALLOWED_TOOLS = set(TOOL_REGISTRY.keys())


def call_tool(tool_name: str, **kwargs) -> dict:
    """Safe tool dispatcher with allowlist enforcement."""
    if tool_name not in ALLOWED_TOOLS:
        return {"error": f"Tool '{tool_name}' not in allowlist", "allowed": list(ALLOWED_TOOLS)}
    try:
        return TOOL_REGISTRY[tool_name](**kwargs)
    except Exception as e:
        return {"error": str(e), "tool": tool_name}


if __name__ == "__main__":
    if _MCP_AVAILABLE and mcp:
        print("Starting DroughtGuard MCP Tool Server...")
        mcp.run()
    else:
        print("MCP SDK not available. Install 'mcp' or 'fastmcp' package.")
