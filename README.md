# DroughtGuard AI — Karnataka Drought Early Warning & Farmer Advisory System

**Hackathon Prototype** | Agentic AI Hackathon — Bengaluru, Karnataka | Powered by IBM Bob

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys (optional — app runs in DEMO MODE without keys)
copy .env.example .env
# Edit .env and add your credentials

# 3. Run
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Architecture

```
DroughtGuard AI
├── Agent 1 — Drought Intelligence (IBM Granite / watsonx.ai)
│   └── Tools: get_district_weather, get_taluk_drought_data, get_reservoir_status
├── Agent 2 — Farmer Advisory (Google Gemini)
│   └── Tools: get_taluk_drought_data, get_crop_information, search_agricultural_knowledge
└── Agent 3 — Alert & Relief (Groq Llama 3)
    └── Tools: get_district_vulnerability, get_reservoir_status
```

All agents use **MCP tools** (via `mcp_server.py`), **ChromaDB RAG** for knowledge retrieval,
**SQLite** for persistent memory and audit logs, and **local + IBM guardrails** for safety.

---

## Demo Mode

If API keys are not configured, the application runs in **clearly labelled DEMO MODE** using
deterministic fallback responses. All live vs demo indicators are visible in the UI.

---

## API Keys

| Key | Service | Required for |
|---|---|---|
| `WATSONX_APIKEY` + `WATSONX_PROJECT_ID` | IBM watsonx.ai | Agent 1 live inference |
| `GOOGLE_API_KEY` | Google AI Studio | Agent 2 bilingual advisory |
| `GROQ_API_KEY` | Groq | Agent 3 alert generation |
| `OMNIROUTE_API_KEY` + `OMNIROUTE_BASE_URL` | OmniRoute | Optional unified gateway |
| `WATSONX_GOV_APIKEY` | IBM watsonx.governance | IBM-native guardrails |

---

## Data Notice

All data displayed is **SYNTHETIC DEMONSTRATION DATA** generated for the hackathon prototype.
Values are not official observations from KSNDMC, IMD, ISRO, or any government agency.

---

## Coverage

8 Karnataka districts: Kalaburagi, Raichur, Koppal, Vijayapura, Ballari, Chitradurga, Tumakuru, Kolar

---

## Stack

- **Streamlit** — Dashboard UI
- **IBM Granite (watsonx.ai)** — Drought intelligence agent
- **Google Gemini** — Bilingual farmer advisory
- **Groq (Llama 3)** — Fast alert generation
- **ChromaDB** — Agricultural knowledge RAG
- **SQLite** — Persistent memory & audit logs
- **MCP (FastMCP)** — Tool server & agent-tool communication
- **Plotly** — Interactive maps and charts
