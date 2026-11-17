# Drought Early Warning & Farmer Advisory System — Implementation Plan

## Overview

Build a Streamlit localhost prototype of an Agentic AI Drought Early Warning System for Karnataka.
The MVP uses three specialized agents (IBM Granite 4, Google Gemini, Groq Llama 3), synthetic data,
MCP-based tool integration, NeMo Guardrails, ChromaDB RAG, a Human-in-the-Loop approval gate,
and an observability sidebar — all wired through an OmniRoute / litellm gateway.

**Non-goals (Phase 2):** Live satellite data ingestion, IBM Cloud deployment, long-term seasonal memory,
reservoir / groundwater live APIs.

---

## Sub-Task 1 — Project Scaffold & Environment

**Intent:** Create the directory structure, `requirements.txt`, `.env.example`, and confirm all
dependencies install cleanly before writing any agent code.

**Expected Outcomes:**
- `mockround/` structure matches the target layout
- `pip install -r requirements.txt` succeeds with no conflicts
- `.env.example` documents every required key

**Todo List:**
- [ ] Create directory tree: `agents/`, `mcp_server/`, `guardrails/`, `data/advisories/`, `memory/`
- [ ] Write `requirements.txt` pinning: `streamlit`, `litellm`, `google-generativeai`, `groq`,
  `ibm-watsonx-ai`, `mcp`, `nemoguardrails`, `chromadb`, `pandas`, `python-dotenv`, `plotly`
- [ ] Write `.env.example` with keys: `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, `GEMINI_API_KEY`,
  `GROQ_API_KEY`, `OMNIROUTE_API_KEY` (optional)
- [ ] Verify install in local venv

**Relevant Context:** Python 3.12 required per problem statement. All packages must be version-compatible.

**Status:** `[ ] pending`

---

## Sub-Task 2 — Synthetic Data & ChromaDB Seed

**Intent:** Create minimal synthetic CSVs (rainfall, crop calendar) and seed ChromaDB with 3–4
advisory text documents so agents have grounded retrieval from day one.

**Expected Outcomes:**
- `data/synthetic_rainfall.csv` with columns: `taluk`, `month`, `rainfall_mm`, `normal_mm`, `deviation_pct`
- `data/crop_calendar.csv` with columns: `crop`, `season`, `sow_month`, `harvest_month`, `districts`
- `data/advisories/` contains 3 plain-text files (drought scheme, insurance deadline, contingency crop guide)
- `memory/vector_store.py` initialises ChromaDB, embeds advisory docs, exposes `query(text, n)` function
- ChromaDB collection persists to `./chroma_db/`

**Todo List:**
- [ ] Write `data/synthetic_rainfall.csv` (10 taluks × 6 months, realistic deficit values)
- [ ] Write `data/crop_calendar.csv` (Kharif / Rabi crops for Karnataka)
- [ ] Write 3 advisory seed documents in `data/advisories/`
- [ ] Write `memory/vector_store.py` using `chromadb.PersistentClient`
- [ ] Test that `query("water shortage")` returns relevant chunks

**Relevant Context:** ChromaDB default embedding (all-MiniLM) works without an extra API key.

**Status:** `[ ] pending`

---

## Sub-Task 3 — MCP Tool Server

**Intent:** Expose three lightweight tools via MCP so agents can call structured data-fetch functions
rather than constructing raw API calls themselves.

**Expected Outcomes:**
- `mcp_server/tools_server.py` defines and registers three tools:
  - `get_weather(taluk, month)` → returns rainfall deviation and temperature from synthetic CSV
  - `get_ndvi_proxy(taluk)` → returns synthetic NDVI score (0–1) and soil moisture category
  - `get_crop_calendar(crop, season)` → returns sow/harvest months and stress flags
- Server starts cleanly with `python mcp_server/tools_server.py`

**Todo List:**
- [ ] Install and import `mcp` SDK
- [ ] Define `get_weather` tool reading from `synthetic_rainfall.csv`
- [ ] Define `get_ndvi_proxy` tool returning hardcoded synthetic values per taluk
- [ ] Define `get_crop_calendar` tool reading from `crop_calendar.csv`
- [ ] Register all three with `mcp.Server` and add `if __name__ == "__main__": server.run()`

**Relevant Context:** Use the `mcp` Python SDK (`pip install mcp`). Tools must have typed input schemas.

**Status:** `[ ] pending`

---

## Sub-Task 4 — NeMo Guardrails Config

**Intent:** Protect all three agent inputs from prompt injection and off-topic queries using a minimal
COLANG guardrail config.

**Expected Outcomes:**
- `guardrails/config.colang` defines:
  - Allowed topics: drought, rainfall, crops, water, Karnataka, farmer advisories
  - Blocked patterns: ignore previous instructions, jailbreak phrases
  - Input rail applied before every LLM call
- `guardrails/guardrails_wrapper.py` exposes `apply_input_rail(text) -> str | None` (None = blocked)

**Todo List:**
- [ ] Write `config.colang` with topic-scoped input rail
- [ ] Write `config.yml` for NeMo (points to `config.colang`)
- [ ] Write `guardrails_wrapper.py` loading NeMo `RailsConfig` and running input check
- [ ] Test that "ignore all previous instructions" returns None (blocked)
- [ ] Test that "What is the drought risk in Raichur?" passes through

**Relevant Context:** `nemoguardrails` >= 0.6; use `RailsConfig.from_path()` and `rails.generate()`.

**Status:** `[ ] pending`

---

## Sub-Task 5 — Agent 1: Weather & Drought Detection (IBM Granite 4)

**Intent:** Implement the first agent that calls the `get_weather` MCP tool, reasons over rainfall
deviation data using ReAct prompting, and outputs a structured drought severity classification.

**Expected Outcomes:**
- `agents/agent1_weather.py` exports `run(taluk: str, month: str) -> dict`
- Output dict: `{ "taluk", "severity": "mild|moderate|severe", "deviation_pct", "reasoning", "tokens_used", "latency_ms" }`
- Uses IBM Granite 4 via `ibm-watsonx-ai` or `litellm` with `watsonx/ibm/granite-4` model ID
- ReAct prompt: Thought → Action (call tool) → Observation → Answer pattern
- Guardrail applied to the constructed prompt before sending

**Todo List:**
- [ ] Set up `litellm.completion()` call for Granite 4 with watsonx credentials from `.env`
- [ ] Write ReAct system prompt scoped to drought severity classification
- [ ] Call `get_weather` tool (simulated function call or direct import from MCP server module)
- [ ] Parse severity from LLM response with fallback regex extraction
- [ ] Return dict with observability fields (tokens, latency)

**Relevant Context:** litellm supports `watsonx/ibm/granite-4-dense-instruct` as model string.
Fallback: use direct `ibm_watsonx_ai.foundation_models.ModelInference`.

**Status:** `[ ] pending`

---

## Sub-Task 6 — Agent 2: Crop & Vulnerability Risk (Google Gemini)

**Intent:** Implement the second agent that combines NDVI proxy data and crop calendar information
to estimate crop stress level and flag the most vulnerable taluks.

**Expected Outcomes:**
- `agents/agent2_crop.py` exports `run(taluk: str, season: str) -> dict`
- Output dict: `{ "taluk", "crop_stress": "low|medium|high", "ndvi_score", "vulnerable_crops": [...], "risk_summary", "tokens_used", "latency_ms" }`
- Uses Gemini 1.5 Flash via `google-generativeai` or `litellm` with `gemini/gemini-1.5-flash`
- Calls `get_ndvi_proxy` and `get_crop_calendar` tools
- Guardrail applied to prompt

**Todo List:**
- [ ] Set up `litellm.completion()` or `genai.GenerativeModel` with Gemini Flash
- [ ] Write prompt that structures NDVI + crop calendar into a vulnerability reasoning chain
- [ ] Parse `crop_stress` and `vulnerable_crops` from response
- [ ] Return observability dict

**Relevant Context:** Gemini 1.5 Flash is free-tier eligible. `GEMINI_API_KEY` from Google AI Studio.

**Status:** `[ ] pending`

---

## Sub-Task 7 — Agent 3: Advisory & Relief (Groq Llama 3)

**Intent:** Implement the third agent that uses ChromaDB RAG to retrieve relevant scheme guidelines
and generates a bilingual (Kannada + English) advisory and relief recommendation.

**Expected Outcomes:**
- `agents/agent3_advisory.py` exports `run(taluk: str, severity: str, crop_stress: str) -> dict`
- Output dict: `{ "taluk", "advisory_en", "advisory_kn", "relief_actions": [...], "tokens_used", "latency_ms" }`
- Uses Groq Llama 3 70B (`groq/llama3-70b-8192`) via litellm
- Retrieves top-3 chunks from ChromaDB before constructing prompt
- Guardrail applied to prompt

**Todo List:**
- [ ] Set up `litellm.completion()` for Groq with `GROQ_API_KEY`
- [ ] Call `memory.vector_store.query()` to get relevant advisory chunks
- [ ] Write bilingual prompt: "Respond first in English, then in Kannada script"
- [ ] Parse advisory sections from response (split on language marker)
- [ ] Return observability dict

**Relevant Context:** Groq free tier supports Llama 3 70B with high token-per-second speed.

**Status:** `[ ] pending`

---

## Sub-Task 8 — Observability Module

**Intent:** Provide a lightweight token-cost and latency tracker that all agents write to, displayed
in the Streamlit sidebar.

**Expected Outcomes:**
- `observability.py` has a simple `ObservabilityLog` class with `record(agent, tokens, latency_ms, tool_calls)` and `summary() -> dict`
- Summary includes: total tokens per agent, average latency, total estimated cost (USD)
- Used by all three agent modules

**Todo List:**
- [ ] Write `ObservabilityLog` with an in-memory list store
- [ ] Add `estimate_cost(model, tokens)` helper using published per-token pricing
- [ ] Export singleton `obs_log` for import across agent files

**Status:** `[ ] pending`

---

## Sub-Task 9 — Streamlit Dashboard (app.py)

**Intent:** Wire all agents, the HITL gate, observability, and synthetic map into a single Streamlit
app that demonstrates the full multi-agent pipeline end-to-end.

**Expected Outcomes:**
- Sidebar: taluk selector, season selector, "Run Analysis" button, observability stats
- Main area – three columns:
  - **Drought Status:** Agent 1 output with severity badge (colour-coded: green/amber/red)
  - **Crop Risk:** Agent 2 output with crop stress and vulnerable crops list
  - **Advisory:** Agent 3 bilingual advisory and relief actions
- HITL gate: if severity == "severe", advisory is hidden behind `st.button("Officer Approve & Publish")`
- District map: `plotly` choropleth using synthetic severity scores for 10 Karnataka taluks
- Guardrail block message shown if input is rejected

**Todo List:**
- [ ] Set up Streamlit page config (title, wide layout, Karnataka theme colours)
- [ ] Add taluk + season selectors in sidebar
- [ ] On "Run Analysis": call Agent 1, Agent 2, Agent 3 sequentially; store results in `st.session_state`
- [ ] Render severity badge, crop stress panel, and advisory panel
- [ ] Implement HITL approval gate for severe alerts
- [ ] Render Plotly choropleth map (GeoJSON or simple scatter map with taluk coordinates)
- [ ] Render observability sidebar panel from `obs_log.summary()`
- [ ] Add `st.spinner` and error handling for API failures

**Relevant Context:** `st.session_state` is used to persist results between reruns.
Map fallback: simple `plotly.express.scatter_mapbox` with lat/lon for 10 taluks if GeoJSON is complex.

**Status:** `[ ] pending`

---

## Sub-Task 10 — Integration Test & Demo Run

**Intent:** Verify the full pipeline works end-to-end on localhost before the demo, confirm guardrails
fire correctly, HITL gate appears for severe cases, and observability stats render.

**Expected Outcomes:**
- `streamlit run app.py` starts with no import errors
- Selecting "Raichur" + "Kharif" and clicking "Run Analysis" produces all three agent outputs
- Setting synthetic data to severe threshold triggers HITL approval button
- Injecting "ignore all instructions" is blocked by guardrails with visible message
- Observability sidebar shows tokens, latency, and estimated cost

**Todo List:**
- [ ] Run `streamlit run app.py` and fix any import or runtime errors
- [ ] Test three different taluks to confirm varying severity outputs
- [ ] Force severe severity (edit synthetic data) and verify HITL gate
- [ ] Test guardrail rejection with an adversarial input
- [ ] Screenshot or note final demo state

**Status:** `[ ] pending`

---

## Dependency / Execution Order

```
Sub-Task 1 → Sub-Task 2 → Sub-Task 3 → Sub-Task 4
                                    ↓
              Sub-Task 5, Sub-Task 6, Sub-Task 7 (parallel)
                                    ↓
                         Sub-Task 8 (observability)
                                    ↓
                         Sub-Task 9 (Streamlit app)
                                    ↓
                         Sub-Task 10 (integration test)
```

## API Keys Required

| Key | Source | Free Tier |
|---|---|---|
| `WATSONX_API_KEY` + `WATSONX_PROJECT_ID` | IBM Cloud / watsonx.ai | Yes (Lite) |
| `GEMINI_API_KEY` | Google AI Studio | Yes |
| `GROQ_API_KEY` | console.groq.com | Yes |
| `OMNIROUTE_API_KEY` | omniroute.online | Optional — litellm fallback |
