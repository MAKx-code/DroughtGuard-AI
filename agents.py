"""
Three specialized AI agents + orchestrator for DroughtGuard AI.

Agent 1 — Drought Intelligence Agent (IBM Granite / watsonx.ai)
Agent 2 — Farmer Advisory Agent (Google Gemini)
Agent 3 — Alert & Relief Response Agent (Groq)
"""
import datetime
import time
from llm_router import call_granite, call_gemini, call_groq, LLMResponse
from mcp_server import call_tool
from guardrails import check_input, check_output
from memory import (
    log_agent_run, save_assessment, save_alert, save_advisory,
    query_knowledge,
)
from monitoring import obs
from drought_engine import calculate_risk


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Drought Intelligence Agent (IBM Granite)
# ═══════════════════════════════════════════════════════════════════════════════

AGENT1_SYSTEM = """You are the Drought Intelligence Agent for Karnataka, India.
You analyze meteorological, satellite, soil, and water resource data to detect
emerging drought conditions and assess their severity.

Your outputs must be grounded in the data provided. Do not invent measurements.
Classify drought as: Low, Moderate, High, or Severe.
Structure your response clearly with: Assessment, Evidence, Affected Areas, Recommendations.
Keep responses factual, concise, and actionable."""


def run_agent1(district: str, taluk: str = None) -> dict:
    """
    Drought Intelligence Agent — ReAct cycle:
    Reason → Tool Call (get weather + drought data) → Observe → Generate Assessment
    """
    started = datetime.datetime.utcnow().isoformat()
    t0 = time.time()
    tool_calls_made = []

    # ── STEP 1: Reason — what data do we need? ────────────────────────────────
    # ── STEP 2: Act — call MCP tools ─────────────────────────────────────────
    weather_data = call_tool("get_district_weather", district=district)
    tool_calls_made.append("get_district_weather")

    target_taluk = taluk or district
    drought_data = call_tool("get_taluk_drought_data",
                             district=district, taluk=target_taluk)
    tool_calls_made.append("get_taluk_drought_data")

    reservoir_data = call_tool("get_reservoir_status", district=district)
    tool_calls_made.append("get_reservoir_status")

    # ── STEP 3: Observe — build context ──────────────────────────────────────
    risk_score    = drought_data.get("risk_score", 50)
    risk_severity = drought_data.get("risk_severity", "Moderate")
    crop_stress   = drought_data.get("crop_stress", "Moderate")

    context = {
        "district": district,
        "taluk": target_taluk,
        "rainfall_deviation_pct": weather_data.get("rainfall_deviation_pct", -30),
        "soil_moisture_pct": drought_data.get("soil_moisture_pct", 25),
        "severity": risk_severity,
        "score": risk_score,
        "crop_stress": crop_stress,
    }

    prompt = f"""Analyze the following drought data for {district} district, {target_taluk} taluk, Karnataka:

WEATHER DATA:
- Rainfall received: {weather_data.get('rainfall_received_mm', 'N/A')} mm
- Normal rainfall: {weather_data.get('normal_rainfall_mm', 'N/A')} mm
- Rainfall deviation: {weather_data.get('rainfall_deviation_pct', 'N/A')}%
- Consecutive dry spell: {weather_data.get('dry_spell_days', 'N/A')} days
- Temperature: {weather_data.get('temperature_c', 'N/A')}°C
- ET anomaly: {weather_data.get('et_anomaly_mm_day', 'N/A')} mm/day above normal

DROUGHT INDICATORS:
- Drought Risk Score: {risk_score}/100
- Severity Classification: {risk_severity}
- Soil moisture: {drought_data.get('soil_moisture_pct', 'N/A')}%
- Groundwater depth: {drought_data.get('groundwater_depth_m', 'N/A')} m
- NDVI (vegetation): {drought_data.get('ndvi', 'N/A')}
- Crop stress: {crop_stress}
- Top risk factors: {', '.join(drought_data.get('top_factors', []))}

WATER RESOURCES:
- Reservoir storage: {reservoir_data.get('avg_reservoir_storage_pct', 'N/A')}%
- Water status: {reservoir_data.get('drinking_water_status', 'N/A')}

Based on this data, provide:
1. DROUGHT ASSESSMENT: Overall risk level and justification
2. KEY EVIDENCE: 3-4 specific data points driving this assessment
3. AFFECTED POPULATIONS: Who is most at risk
4. IMMEDIATE RECOMMENDATIONS: 3-4 specific actions for district authorities
5. MONITORING PRIORITIES: What to watch in the next 2 weeks

Note: This assessment is based on SYNTHETIC DEMONSTRATION DATA for the hackathon prototype."""

    # Guardrail check on prompt
    guard = check_input(prompt[:200], agent="Agent1-Drought")
    if not guard.allowed:
        return {"error": guard.reason, "agent": "Agent1", "demo_mode": True}

    # ── STEP 4: Generate ──────────────────────────────────────────────────────
    resp: LLMResponse = call_granite(prompt, AGENT1_SYSTEM, context=context)

    # Output guardrail
    out_guard = check_output(resp.text, agent="Agent1-Drought")

    lat = (time.time() - t0) * 1000
    completed = datetime.datetime.utcnow().isoformat()

    # Persist
    run_id = log_agent_run(
        agent_name="Drought Intelligence Agent",
        provider=resp.provider,
        model_id=resp.model,
        started_at=started,
        completed_at=completed,
        latency_ms=round(lat, 1),
        tokens_used=resp.tokens,
        tool_calls=tool_calls_made,
        status="success" if not resp.error else "error",
        error=resp.error or "",
        guardrail_outcome="pass" if guard.allowed else "blocked",
        district=district, taluk=target_taluk,
    )
    obs.record("Agent1-Drought", resp.provider, resp.model,
               round(lat, 1), resp.tokens, tool_calls_made,
               "success" if not resp.error else "error",
               resp.error or "")

    save_assessment(district, target_taluk, risk_score, risk_severity,
                    {"assessment": resp.text, "weather": weather_data,
                     "drought": drought_data}, run_id)

    return {
        "agent": "Drought Intelligence Agent",
        "provider": resp.provider,
        "model": resp.model,
        "demo_mode": resp.demo_mode,
        "district": district,
        "taluk": target_taluk,
        "risk_score": risk_score,
        "risk_severity": risk_severity,
        "assessment": resp.text,
        "tool_calls": tool_calls_made,
        "latency_ms": round(lat, 1),
        "tokens": resp.tokens,
        "run_id": run_id,
        "output_flags": out_guard.output_flags,
        "error": resp.error,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Farmer Advisory Agent (Google Gemini)
# ═══════════════════════════════════════════════════════════════════════════════

AGENT2_SYSTEM = """You are the Farmer Advisory Agent for Karnataka, India.
You generate practical, actionable agricultural advisories for farmers affected by drought.

Always respond in TWO sections:
1. ENGLISH ADVISORY — clear, simple language suitable for rural farmers
2. ಕನ್ನಡ ಸಲಹೆ — the same advisory translated to simple Kannada

Base all advice on the provided data. Do not make unsupported claims about yields or guarantees.
Focus on practical, implementable actions. Mention relevant government schemes where appropriate."""


def run_agent2(district: str, taluk: str, crop: str = None,
               season: str = "Kharif", user_query: str = None) -> dict:
    """Farmer Advisory Agent — RAG + Gemini."""
    started = datetime.datetime.utcnow().isoformat()
    t0 = time.time()
    tool_calls_made = []

    # MCP tool calls
    drought_data = call_tool("get_taluk_drought_data", district=district, taluk=taluk)
    tool_calls_made.append("get_taluk_drought_data")

    crop_data = call_tool("get_crop_information", district=district, season=season)
    tool_calls_made.append("get_crop_information")

    main_crop = crop or crop_data.get("main_crop", "Cotton")

    # RAG: get relevant advisory knowledge
    knowledge_query = f"drought advisory {main_crop} {season} Karnataka water conservation"
    if user_query:
        knowledge_query = user_query
    kb_results = call_tool("search_agricultural_knowledge",
                            query=knowledge_query, n_results=3)
    tool_calls_made.append("search_agricultural_knowledge")

    knowledge_ctx = "\n".join([r["text"] for r in kb_results.get("results", [])])

    risk_score    = drought_data.get("risk_score", 50)
    risk_severity = drought_data.get("risk_severity", "Moderate")
    crop_stress   = drought_data.get("crop_stress", "Moderate")

    context = {
        "district": district, "taluk": taluk,
        "rainfall_deviation_pct": -35,
        "soil_moisture_pct": drought_data.get("soil_moisture_pct", 25),
        "severity": risk_severity, "score": risk_score,
        "crop_stress": crop_stress, "crop": main_crop,
    }

    farmer_query_section = f"\nFARMER'S QUESTION: {user_query}" if user_query else ""

    prompt = f"""Generate a farmer advisory for {district} district, {taluk} taluk, Karnataka.

CURRENT CONDITIONS:
- Main crop: {main_crop} ({season} season)
- Drought Risk Score: {risk_score}/100 ({risk_severity})
- Crop stress level: {crop_stress}
- Soil moisture: {drought_data.get('soil_moisture_pct', 'N/A')}%
- Groundwater depth: {drought_data.get('groundwater_depth_m', 'N/A')} m
- Small & marginal farmers: {crop_data.get('sm_farmer_pct', 'N/A')}% of farming community
- Livestock risk: {crop_data.get('livestock_risk', 'Moderate')}
{farmer_query_section}

RELEVANT KNOWLEDGE BASE:
{knowledge_ctx}

Provide:
1. IMMEDIATE ACTIONS (this week): 3-4 practical steps
2. WATER MANAGEMENT: specific advice for available water resources
3. CROP GUIDANCE: sowing/harvesting decisions given drought conditions
4. GOVERNMENT SCHEMES: relevant schemes the farmer should contact
5. URGENCY LEVEL: Low / Moderate / High / Critical

Remember: Respond with BOTH English and Kannada sections.
Data source: SYNTHETIC DEMO DATA for hackathon prototype."""

    guard = check_input(prompt[:200], agent="Agent2-Advisory")
    if not guard.allowed:
        return {"error": guard.reason, "agent": "Agent2", "demo_mode": True}

    resp: LLMResponse = call_gemini(prompt, AGENT2_SYSTEM, context=context)
    out_guard = check_output(resp.text, agent="Agent2-Advisory")

    # Split English/Kannada
    text = resp.text
    advisory_en = text
    advisory_kn = ""
    if "ಕನ್ನಡ" in text:
        parts = text.split("ಕನ್ನಡ", 1)
        advisory_en = parts[0].strip()
        advisory_kn = "ಕನ್ನಡ" + parts[1].strip()

    lat = (time.time() - t0) * 1000
    completed = datetime.datetime.utcnow().isoformat()

    run_id = log_agent_run(
        agent_name="Farmer Advisory Agent",
        provider=resp.provider, model_id=resp.model,
        started_at=started, completed_at=completed,
        latency_ms=round(lat, 1), tokens_used=resp.tokens,
        tool_calls=tool_calls_made,
        status="success" if not resp.error else "error",
        error=resp.error or "", district=district, taluk=taluk,
    )
    obs.record("Agent2-Advisory", resp.provider, resp.model,
               round(lat, 1), resp.tokens, tool_calls_made,
               "success" if not resp.error else "error", resp.error or "")

    save_advisory(district, taluk, main_crop, season,
                  advisory_en, advisory_kn, run_id)

    return {
        "agent": "Farmer Advisory Agent",
        "provider": resp.provider, "model": resp.model,
        "demo_mode": resp.demo_mode,
        "district": district, "taluk": taluk,
        "crop": main_crop, "season": season,
        "advisory_en": advisory_en,
        "advisory_kn": advisory_kn,
        "full_text": text,
        "risk_score": risk_score, "risk_severity": risk_severity,
        "crop_stress": crop_stress,
        "knowledge_used": len(kb_results.get("results", [])),
        "tool_calls": tool_calls_made,
        "latency_ms": round(lat, 1), "tokens": resp.tokens,
        "output_flags": out_guard.output_flags,
        "error": resp.error,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Alert & Relief Response Agent (Groq)
# ═══════════════════════════════════════════════════════════════════════════════

AGENT3_SYSTEM = """You are the Alert and Relief Response Agent for Karnataka Drought Early Warning System.
You generate drought alerts and relief recommendations for district authorities and emergency responders.

Your output must be:
- Clear and actionable
- Prioritized by urgency
- Linked to specific departments (Revenue, Agriculture, RDPR, Animal Husbandry)
- Always marked as REQUIRING HUMAN APPROVAL before public notification

Do not initiate any real-world actions. Generate recommendations only."""


def run_agent3(district: str, taluk: str = None,
               risk_score: float = None, risk_severity: str = None) -> dict:
    """Alert & Relief Agent — fast Groq response with structured alert."""
    started = datetime.datetime.utcnow().isoformat()
    t0 = time.time()
    tool_calls_made = []

    vuln_data = call_tool("get_district_vulnerability", district=district)
    tool_calls_made.append("get_district_vulnerability")

    reservoir_data = call_tool("get_reservoir_status", district=district)
    tool_calls_made.append("get_reservoir_status")

    target_taluk = taluk or district

    # If risk not provided, calculate it
    if risk_score is None:
        from data_service import get_taluk_data
        row = get_taluk_data(district, target_taluk) or {}
        if not row:
            df = get_dataset_summary_row(district)
            row = df
        calc = calculate_risk(row) if row else {"score": 50, "severity": "Moderate"}
        risk_score = calc["score"]
        risk_severity = calc["severity"]

    context = {
        "district": district, "taluk": target_taluk,
        "severity": risk_severity, "score": risk_score,
        "rainfall_deviation_pct": -35,
    }

    critical_taluks = vuln_data.get("critical_water_taluks", [])
    livestock_taluks = vuln_data.get("high_livestock_risk_taluks", [])

    prompt = f"""Generate a drought alert and relief recommendation for {district} district, Karnataka.

ALERT STATUS:
- Risk Score: {risk_score}/100
- Severity: {risk_severity}
- Target Taluk: {target_taluk}

VULNERABILITY DATA:
- Small & marginal farmers: {vuln_data.get('avg_small_marginal_farmer_pct', 'N/A')}%
- Critical water shortage taluks: {critical_taluks or 'None identified'}
- High livestock risk taluks: {livestock_taluks or 'None identified'}

WATER RESOURCES:
- Reservoir storage: {reservoir_data.get('avg_reservoir_storage_pct', 'N/A')}%
- Water critical: {reservoir_data.get('water_critical', False)}

Generate:
1. ALERT HEADER: Severity level and affected area
2. SITUATION SUMMARY: 2-3 sentences on current conditions
3. PRIORITY RELIEF ACTIONS (ranked 1-5):
   - Action, responsible department, timeline
4. VULNERABLE GROUPS: Who needs immediate attention
5. RESOURCE REQUIREMENTS: Tankers, fodder, boreholes needed
6. ESCALATION: Whether state-level support is needed

IMPORTANT: Mark all recommendations as PENDING OFFICER APPROVAL.
Data source: SYNTHETIC DEMO DATA for hackathon prototype."""

    guard = check_input(prompt[:200], agent="Agent3-Alert")
    if not guard.allowed:
        return {"error": guard.reason, "agent": "Agent3", "demo_mode": True}

    resp: LLMResponse = call_groq(prompt, AGENT3_SYSTEM, context=context)
    out_guard = check_output(resp.text, agent="Agent3-Alert")

    lat = (time.time() - t0) * 1000
    completed = datetime.datetime.utcnow().isoformat()

    run_id = log_agent_run(
        agent_name="Alert & Relief Agent",
        provider=resp.provider, model_id=resp.model,
        started_at=started, completed_at=completed,
        latency_ms=round(lat, 1), tokens_used=resp.tokens,
        tool_calls=tool_calls_made,
        status="success" if not resp.error else "error",
        error=resp.error or "", district=district, taluk=target_taluk,
    )
    obs.record("Agent3-Alert", resp.provider, resp.model,
               round(lat, 1), resp.tokens, tool_calls_made,
               "success" if not resp.error else "error", resp.error or "")

    # Auto-save as pending alert if High or Severe
    alert_id = None
    if risk_severity in ("High", "Severe"):
        alert_id = save_alert(district, target_taluk, risk_severity,
                              resp.text[:2000])

    return {
        "agent": "Alert & Relief Agent",
        "provider": resp.provider, "model": resp.model,
        "demo_mode": resp.demo_mode,
        "district": district, "taluk": target_taluk,
        "risk_score": risk_score, "risk_severity": risk_severity,
        "alert_text": resp.text,
        "alert_id": alert_id,
        "requires_approval": risk_severity in ("High", "Severe"),
        "critical_taluks": critical_taluks,
        "tool_calls": tool_calls_made,
        "latency_ms": round(lat, 1), "tokens": resp.tokens,
        "output_flags": out_guard.output_flags,
        "error": resp.error,
    }


def get_dataset_summary_row(district: str) -> dict:
    from data_service import get_district_summary
    df = get_district_summary()
    sub = df[df["district"] == district]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_assessment(district: str, taluk: str = None,
                        crop: str = None, season: str = "Kharif") -> dict:
    """
    Orchestrate all three agents sequentially.
    Agent 1 → Agent 2 → Agent 3
    Pass risk context from Agent 1 to Agent 3.
    """
    results = {}

    # Agent 1: Drought Intelligence
    a1 = run_agent1(district, taluk)
    results["agent1"] = a1

    # Agent 2: Farmer Advisory
    a2 = run_agent2(district, taluk or district, crop, season)
    results["agent2"] = a2

    # Agent 3: Alert & Relief (uses Agent 1 risk score)
    a3 = run_agent3(
        district=district, taluk=taluk,
        risk_score=a1.get("risk_score"),
        risk_severity=a1.get("risk_severity"),
    )
    results["agent3"] = a3

    results["orchestrator"] = {
        "district": district,
        "taluk": taluk or district,
        "overall_severity": a1.get("risk_severity", "Unknown"),
        "risk_score": a1.get("risk_score", 0),
        "requires_officer_approval": a3.get("requires_approval", False),
        "total_tool_calls": (
            len(a1.get("tool_calls", [])) +
            len(a2.get("tool_calls", [])) +
            len(a3.get("tool_calls", []))
        ),
    }

    return results
