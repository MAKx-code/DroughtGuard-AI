# DroughtGuard AI 🌾

### AI-Powered Drought Early Warning & Farmer Advisory Platform for Karnataka

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit)](https://droughtguard-ai-8nqzaauquqzstftfqjvvnj.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-MAKx--code%2FDroughtGuard--AI-181717?logo=github)](https://github.com/MAKx-code/DroughtGuard-AI)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://python.org)
[![IBM Granite](https://img.shields.io/badge/IBM-Granite%20%2F%20watsonx.ai-054ADA?logo=ibm)](https://www.ibm.com/watsonx)

**Agentic AI Hackathon** | Bengaluru, Karnataka

---

## Problem Statement

Karnataka's drought-prone northern districts — including Kalaburagi, Raichur, Vijayapura, Ballari and Koppal — face recurring agricultural drought every year. Early warning systems exist in isolation across KSNDMC, IMD and state agriculture departments, but farmers and district officers receive fragmented, delayed information. By the time action is taken, crop losses have already occurred.

**DroughtGuard AI** bridges this gap with a multi-agent system that continuously assesses drought risk across taluks, generates bilingual farmer advisories, coordinates relief alerts, and routes all high-severity alerts through a human-in-the-loop officer approval workflow — before they are acted upon.

---

## Live Demo

🔗 **[https://droughtguard-ai-8nqzaauquqzstftfqjvvnj.streamlit.app/](https://droughtguard-ai-8nqzaauquqzstftfqjvvnj.streamlit.app/)**

> **Data Notice:** All data displayed in the demo is **synthetic demonstration data** generated for the hackathon prototype. Values are not official observations from KSNDMC, IMD, ISRO, or any government agency. The architecture is designed to connect to real data sources in production.

---

## Key Features

| Feature | Status |
|---|---|
| 7-factor deterministic drought risk scoring (0–100) | ✅ Implemented |
| Interactive district & taluk risk map (Plotly) | ✅ Implemented |
| Three-agent AI assessment pipeline | ✅ Implemented |
| Bilingual farmer advisories (English + Kannada) | ✅ Implemented |
| Human-in-the-loop officer alert approval / rejection | ✅ Implemented |
| Agricultural knowledge RAG (ChromaDB, 10 seed docs) | ✅ Implemented |
| Agent observability dashboard (tokens, latency, guardrails) | ✅ Implemented |
| Local guardrails (prompt injection, PII, off-topic, claims) | ✅ Implemented |
| MCP tool server (6 tools, FastMCP) | ✅ Implemented |
| IBM watsonx.governance guardrail hook | ⚡ Hook ready — requires active watsonx.governance subscription |
| IBM Granite live inference | ⚡ DEMO/FALLBACK — requires active WML service instance |
| OmniRoute unified gateway | ⚡ Optional — not required for core functionality |

---

## Three-Agent Architecture

```
DroughtGuard AI
│
├── Agent 1 — Drought Intelligence
│   Provider : IBM Granite  (ibm/granite-4-h-small via watsonx.ai)
│   Status   : DEMO / FALLBACK  (WML service activation required for live)
│   MCP Tools: get_district_weather, get_taluk_drought_data, get_reservoir_status
│
├── Agent 2 — Bilingual Farmer Advisory
│   Provider : Google Gemini  (gemini-3.8-flash via google-genai SDK)
│   Status   : LIVE  ✅
│   MCP Tools: get_taluk_drought_data, get_crop_information, search_agricultural_knowledge
│
└── Agent 3 — Alert & Relief Coordination
    Provider : Groq  (openai/gpt-oss-120b via Groq API)
    Status   : LIVE  ✅
    MCP Tools: get_district_vulnerability, get_reservoir_status
```

All agents share:
- **MCP tool server** (`mcp_server.py`, FastMCP, 6 tools)
- **ChromaDB RAG** — agricultural knowledge retrieval
- **SQLite** — persistent memory, agent run logs, alert audit trail
- **Local guardrails** — input/output safety checks before every LLM call

The orchestrator in `agents.py` runs all three agents sequentially, merges outputs, and routes any High or Severe alerts to the HITL officer approval queue.

---

## Technology Stack

| Layer | Technology |
|---|---|
| **UI** | Streamlit |
| **Agent 1** | IBM Granite `ibm/granite-4-h-small` via `ibm-watsonx-ai` SDK |
| **Agent 2** | Google Gemini `gemini-3.8-flash` via `google-genai` SDK |
| **Agent 3** | Groq `openai/gpt-oss-120b` via `groq` SDK |
| **Tool server** | FastMCP (Model Context Protocol) |
| **Vector store** | ChromaDB (persistent, local) |
| **Relational store** | SQLite (alerts, runs, advisories, guardrail events) |
| **Risk engine** | Custom deterministic 7-factor scorer (`drought_engine.py`) |
| **Maps & charts** | Plotly (scatter_map, bar, line) |
| **Guardrails** | Local (`guardrails.py`) + IBM watsonx.governance hook |
| **Deployment** | Streamlit Community Cloud |

---

## Coverage

8 drought-prone Karnataka districts · 39 taluks:

| District | Taluks |
|---|---|
| Kalaburagi | Kalaburagi, Chittapur, Sedam, Yadgir, Shahapur |
| Raichur | Raichur, Manvi, Sindhanur, Lingasugur, Devadurga |
| Koppal | Koppal, Gangavathi, Kushtagi, Yelburga |
| Vijayapura | Vijayapura, Basavana Bagewadi, Indi, Muddebihal, Sindagi |
| Ballari | Ballari, Hospet, Sandur, Siruguppa, Hagaribommanahalli |
| Chitradurga | Chitradurga, Hiriyur, Molakalmuru, Challakere, Hosadurga |
| Tumakuru | Tumakuru, Tiptur, Madhugiri, Pavagada, Sira |
| Kolar | Kolar, Mulbagal, Srinivaspur, Bangarpet, Malur |

---

## Local Setup

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/MAKx-code/DroughtGuard-AI.git
cd DroughtGuard-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials (app runs in DEMO MODE without keys)
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
# Edit .env and add your API keys

# 4. Run
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your credentials. All keys are optional — the app runs in labelled DEMO MODE when keys are absent.

| Variable | Service | Required for |
|---|---|---|
| `WATSONX_APIKEY` | IBM watsonx.ai | Agent 1 live inference |
| `WATSONX_PROJECT_ID` | IBM watsonx.ai | Agent 1 live inference |
| `WATSONX_URL` | IBM watsonx.ai | Defaults to `https://us-south.ml.cloud.ibm.com` |
| `WATSONX_MODEL_ID` | IBM watsonx.ai | Defaults to `ibm/granite-4-h-small` |
| `GOOGLE_API_KEY` | Google AI Studio | Agent 2 live inference |
| `GEMINI_MODEL_ID` | Google AI Studio | Defaults to `gemini-3.8-flash` |
| `GROQ_API_KEY` | Groq | Agent 3 live inference |
| `GROQ_MODEL_ID` | Groq | Defaults to `openai/gpt-oss-120b` |
| `OMNIROUTE_BASE_URL` + `OMNIROUTE_API_KEY` | OmniRoute | Optional unified LLM gateway |
| `WATSONX_GOV_APIKEY` + `WATSONX_GOV_URL` | IBM watsonx.governance | IBM-native output guardrails |

---

## Streamlit Community Cloud Deployment

1. Fork / clone this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo → branch `main` → Main file: `app.py`
4. Click **Advanced settings → Secrets** and add your API keys in TOML format:

```toml
GOOGLE_API_KEY = "your-key"
GROQ_API_KEY = "your-key"
WATSONX_APIKEY = "your-key"
WATSONX_PROJECT_ID = "your-project-id"
```

5. Click **Deploy**

> **Note:** SQLite and ChromaDB use relative paths and are auto-initialised on startup. No external database provisioning is required.

---

## Project Structure

```
DroughtGuard-AI/
├── app.py                  # Streamlit dashboard (6 pages)
├── agents.py               # Three-agent orchestrator
├── llm_router.py           # Provider router with demo fallbacks
├── mcp_server.py           # FastMCP tool server (6 tools)
├── config.py               # Environment config & thresholds
├── data_service.py         # Synthetic Karnataka dataset (39 taluks)
├── drought_engine.py       # 7-factor risk scoring engine
├── memory.py               # SQLite + ChromaDB RAG
├── guardrails.py           # Input/output safety checks
├── monitoring.py           # Agent observability logging
├── requirements.txt
├── .env.example            # Credentials template
├── assets/
│   └── bg_farmland.svg     # UI background asset
└── .streamlit/
    └── config.toml         # Streamlit server config
```

---

## Synthetic Data Disclaimer

> All data displayed in this application — including rainfall figures, soil moisture readings, groundwater levels, reservoir storage percentages, temperature anomalies and crop vulnerability indices — is **entirely synthetic and generated programmatically** for hackathon demonstration purposes.
>
> This data does **not** represent official observations or forecasts from KSNDMC, IMD, ISRO, CWC, the Government of Karnataka, or any other government or scientific body. It must not be used for real agricultural or policy decisions.

---

*DroughtGuard AI — | Agentic AI Hackathon, Bengaluru*
