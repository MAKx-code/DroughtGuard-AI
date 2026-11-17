"""Central configuration for DroughtGuard AI."""
import os
from dotenv import load_dotenv

load_dotenv()

# ── IBM watsonx.ai ──────────────────────────────────────────────────────────
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY", "")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "ibm/granite-4-h-small")

# ── Google Gemini ───────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-3.8-flash")

# ── Groq ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_ID = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-120b")

# ── OmniRoute (optional) ────────────────────────────────────────────────────
OMNIROUTE_BASE_URL = os.getenv("OMNIROUTE_BASE_URL", "")
OMNIROUTE_API_KEY = os.getenv("OMNIROUTE_API_KEY", "")

# ── watsonx.governance (optional) ──────────────────────────────────────────
WATSONX_GOV_APIKEY = os.getenv("WATSONX_GOV_APIKEY", "")
WATSONX_GOV_URL = os.getenv("WATSONX_GOV_URL", "")

# ── App settings ────────────────────────────────────────────────────────────
APP_NAME = "DroughtGuard AI"
APP_VERSION = "1.0.0-hackathon"
DATA_MODE = "SYNTHETIC DEMO"  # Always label data clearly

# ── Drought risk weights (must sum to 100) ──────────────────────────────────
DROUGHT_WEIGHTS = {
    "rainfall_deficit":       0.25,
    "soil_moisture_deficit":  0.20,
    "dry_spell_duration":     0.15,
    "groundwater_stress":     0.15,
    "reservoir_storage":      0.10,
    "temperature_anomaly":    0.08,
    "crop_vulnerability":     0.07,
}

# ── Drought severity thresholds ─────────────────────────────────────────────
SEVERITY_THRESHOLDS = {
    "Low":      (0,  30),
    "Moderate": (30, 55),
    "High":     (55, 75),
    "Severe":   (75, 100),
}

SEVERITY_COLORS = {
    "Low":      "#4caf77",
    "Moderate": "#f0a500",
    "High":     "#e06b2d",
    "Severe":   "#c0392b",
}

# ── Karnataka districts covered ─────────────────────────────────────────────
DISTRICTS = [
    "Kalaburagi", "Raichur", "Koppal", "Vijayapura",
    "Ballari", "Chitradurga", "Tumakuru", "Kolar",
]

DISTRICT_TALUKS = {
    "Kalaburagi": ["Kalaburagi", "Chittapur", "Sedam", "Yadgir", "Shahapur"],
    "Raichur":    ["Raichur", "Manvi", "Sindhanur", "Lingasugur", "Devadurga"],
    "Koppal":     ["Koppal", "Gangavathi", "Kushtagi", "Yelburga"],
    "Vijayapura": ["Vijayapura", "Basavana Bagewadi", "Indi", "Muddebihal", "Sindagi"],
    "Ballari":    ["Ballari", "Hospet", "Sandur", "Siruguppa", "Hagaribommanahalli"],
    "Chitradurga":["Chitradurga", "Hiriyur", "Molakalmuru", "Challakere", "Hosadurga"],
    "Tumakuru":   ["Tumakuru", "Tiptur", "Madhugiri", "Pavagada", "Sira"],
    "Kolar":      ["Kolar", "Mulbagal", "Srinivaspur", "Bangarpet", "Malur"],
}

# ── MCP server settings ─────────────────────────────────────────────────────
MCP_HOST = "127.0.0.1"
MCP_PORT = 7777

# ── Provider availability checks ────────────────────────────────────────────
def watsonx_available() -> bool:
    return bool(WATSONX_APIKEY and WATSONX_PROJECT_ID)

def gemini_available() -> bool:
    return bool(GOOGLE_API_KEY)

def groq_available() -> bool:
    return bool(GROQ_API_KEY)

def omniroute_available() -> bool:
    return bool(OMNIROUTE_BASE_URL and OMNIROUTE_API_KEY)

def gov_guardrails_available() -> bool:
    return bool(WATSONX_GOV_APIKEY)
