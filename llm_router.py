"""
Centralized LLM provider router for DroughtGuard AI.
Supports IBM watsonx.ai (Granite), Google Gemini, Groq, and OmniRoute gateway.
Falls back to DEMO MODE with deterministic responses when credentials are absent.
"""
import time
import datetime
from config import (
    WATSONX_APIKEY, WATSONX_URL, WATSONX_PROJECT_ID, WATSONX_MODEL_ID,
    GOOGLE_API_KEY, GEMINI_MODEL_ID,
    GROQ_API_KEY, GROQ_MODEL_ID,
    OMNIROUTE_BASE_URL, OMNIROUTE_API_KEY,
    watsonx_available, gemini_available, groq_available, omniroute_available,
)


class LLMResponse:
    def __init__(self, text: str, model: str, provider: str,
                 tokens: int = 0, latency_ms: float = 0,
                 demo_mode: bool = False, error: str = ""):
        self.text = text
        self.model = model
        self.provider = provider
        self.tokens = tokens
        self.latency_ms = latency_ms
        self.demo_mode = demo_mode
        self.error = error
        self.timestamp = datetime.datetime.utcnow().isoformat()

    def label(self) -> str:
        tag = "⚠️ DEMO MODE" if self.demo_mode else "✅ LIVE"
        return f"{tag} | {self.provider} / {self.model}"


# ── IBM watsonx.ai / Granite ─────────────────────────────────────────────────

def _call_watsonx(prompt: str, system: str = "", max_tokens: int = 800) -> LLMResponse:
    t0 = time.time()
    try:
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai import Credentials

        creds = Credentials(url=WATSONX_URL, api_key=WATSONX_APIKEY)
        model = ModelInference(
            model_id=WATSONX_MODEL_ID,
            credentials=creds,
            project_id=WATSONX_PROJECT_ID,
            params={"max_new_tokens": max_tokens, "temperature": 0.3},
        )
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        result = model.generate_text(prompt=full_prompt)
        lat = (time.time() - t0) * 1000
        # Try to get token count
        tokens = 0
        if hasattr(result, "usage") and result.usage:
            tokens = getattr(result.usage, "total_tokens", 0)
        return LLMResponse(
            text=str(result),
            model=WATSONX_MODEL_ID,
            provider="IBM watsonx.ai",
            tokens=tokens,
            latency_ms=round(lat, 1),
        )
    except Exception as e:
        lat = (time.time() - t0) * 1000
        return LLMResponse(
            text="", model=WATSONX_MODEL_ID, provider="IBM watsonx.ai",
            latency_ms=round(lat, 1), error=str(e), demo_mode=True,
        )


# ── Google Gemini ─────────────────────────────────────────────────────────────

def _call_gemini(prompt: str, system: str = "", max_tokens: int = 800) -> LLMResponse:
    t0 = time.time()
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GOOGLE_API_KEY)
        config_obj = types.GenerateContentConfig(
            system_instruction=system if system else None,
            max_output_tokens=max_tokens,
            temperature=0.3,
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=prompt,
            config=config_obj,
        )
        lat = (time.time() - t0) * 1000
        text = response.text or ""
        tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens = getattr(response.usage_metadata, "total_token_count", 0)
        return LLMResponse(
            text=text,
            model=GEMINI_MODEL_ID,
            provider="Google Gemini",
            tokens=tokens,
            latency_ms=round(lat, 1),
        )
    except Exception as e:
        lat = (time.time() - t0) * 1000
        return LLMResponse(
            text="", model=GEMINI_MODEL_ID, provider="Google Gemini",
            latency_ms=round(lat, 1), error=str(e), demo_mode=True,
        )


# ── Groq ──────────────────────────────────────────────────────────────────────

def _call_groq(prompt: str, system: str = "", max_tokens: int = 800) -> LLMResponse:
    t0 = time.time()
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=GROQ_MODEL_ID,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        lat = (time.time() - t0) * 1000
        text = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0
        return LLMResponse(
            text=text, model=GROQ_MODEL_ID, provider="Groq",
            tokens=tokens, latency_ms=round(lat, 1),
        )
    except Exception as e:
        lat = (time.time() - t0) * 1000
        return LLMResponse(
            text="", model=GROQ_MODEL_ID, provider="Groq",
            latency_ms=round(lat, 1), error=str(e), demo_mode=True,
        )


# ── OmniRoute gateway (OpenAI-compatible) ─────────────────────────────────────

def _call_omniroute(prompt: str, model: str, system: str = "",
                    max_tokens: int = 800) -> LLMResponse:
    t0 = time.time()
    try:
        import requests
        headers = {"Authorization": f"Bearer {OMNIROUTE_API_KEY}",
                   "Content-Type": "application/json"}
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
        resp = requests.post(
            f"{OMNIROUTE_BASE_URL}/v1/chat/completions",
            json=payload, headers=headers, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        lat = (time.time() - t0) * 1000
        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return LLMResponse(
            text=text, model=model, provider="OmniRoute",
            tokens=tokens, latency_ms=round(lat, 1),
        )
    except Exception as e:
        lat = (time.time() - t0) * 1000
        return LLMResponse(
            text="", model=model, provider="OmniRoute",
            latency_ms=round(lat, 1), error=str(e), demo_mode=True,
        )


# ── Demo-mode fallback responses ──────────────────────────────────────────────

DEMO_RESPONSES = {
    "granite": """**DEMO MODE — IBM Granite (watsonx.ai credentials not configured)**

Drought Intelligence Assessment:
Based on the submitted district data, rainfall deviation of -{rainfall_deficit:.0f}% combined with soil moisture at {soil_moisture:.0f}% indicates {severity} drought stress.

Risk Score: {score}/100 | Severity: {severity}

Key factors driving this assessment:
1. Rainfall deficit: Below-normal precipitation over the past 60 days
2. Soil moisture depletion: Root-zone moisture critically low for Kharif crops
3. Groundwater stress: Water table receding below seasonal normal

Recommended actions: Activate contingency crop measures; issue crop advisory; monitor tank storage levels.

ℹ️ Configure WATSONX_APIKEY and WATSONX_PROJECT_ID in .env to enable live IBM Granite inference.""",

    "gemini": """**DEMO MODE — Google Gemini (GOOGLE_API_KEY not configured)**

Farmer Advisory (English):
Dear Farmer, your {crop} crop is under {stress_level} stress due to below-normal rainfall. We recommend:
• Switch to drip irrigation to conserve available soil moisture
• Apply mulching around crop bases to reduce evaporation
• Consider short-duration contingency crops if main crop fails
• Register for PMFBY crop insurance before the season deadline

ಕನ್ನಡ ಸಲಹೆ (Demo):
ಆತ್ಮೀಯ ರೈತರೇ, ನಿಮ್ಮ ಬೆಳೆಗೆ ಮಳೆ ಕೊರತೆಯಿಂದ ಒತ್ತಡ ಉಂಟಾಗಿದೆ. ತೊಟ್ಟು ನೀರಾವರಿ ಬಳಸಿ ನೀರು ಉಳಿಸಿ. ಮಲ್ಚಿಂಗ್ ಮಾಡಿ. ಬೆಳೆ ವಿಮೆ ನೋಂದಾಯಿಸಿ.

ℹ️ Configure GOOGLE_API_KEY in .env to enable live Gemini advisory generation.""",

    "groq": """**DEMO MODE — Groq (GROQ_API_KEY not configured)**

🚨 Drought Alert — {district}

Alert Level: {severity}
Affected Area: {district} district, {taluk} taluk

Situation Summary:
Current conditions indicate {severity} drought risk. Rainfall is {rainfall_deficit:.0f}% below seasonal normal. Groundwater depth has increased significantly.

Priority Relief Actions:
1. 🚰 Deploy drinking water tankers to critical habitations
2. 🌾 Activate MGNREGA work demand for water conservation activities
3. 🐄 Open fodder banks for livestock in affected taluks
4. 💧 Prioritize borewell deepening in water-stressed villages

Approval Required: Officer must approve before public notification.

ℹ️ Configure GROQ_API_KEY in .env to enable live Groq alert generation.""",
}


def _demo_response(provider: str, context: dict = None) -> LLMResponse:
    ctx = context or {}
    template = DEMO_RESPONSES.get(provider, "DEMO MODE — Provider unavailable")
    try:
        text = template.format(
            rainfall_deficit=abs(ctx.get("rainfall_deviation_pct", -35)),
            soil_moisture=ctx.get("soil_moisture_pct", 25),
            severity=ctx.get("severity", "High"),
            score=ctx.get("score", 68),
            crop=ctx.get("crop", "Cotton"),
            stress_level=ctx.get("crop_stress", "moderate"),
            district=ctx.get("district", "Raichur"),
            taluk=ctx.get("taluk", "Manvi"),
        )
    except Exception:
        text = template
    model_map = {
        "granite": WATSONX_MODEL_ID,
        "gemini":  GEMINI_MODEL_ID,
        "groq":    GROQ_MODEL_ID,
    }
    provider_map = {
        "granite": "IBM watsonx.ai",
        "gemini":  "Google Gemini",
        "groq":    "Groq",
    }
    return LLMResponse(
        text=text,
        model=model_map.get(provider, provider),
        provider=provider_map.get(provider, "Demo"),
        tokens=len(text.split()) * 2,  # rough estimate
        latency_ms=120.0,
        demo_mode=True,
    )


# ── Public router functions ───────────────────────────────────────────────────

def call_granite(prompt: str, system: str = "",
                 context: dict = None, max_tokens: int = 800) -> LLMResponse:
    if omniroute_available():
        resp = _call_omniroute(prompt, WATSONX_MODEL_ID, system, max_tokens)
        if not resp.demo_mode:
            return resp
    if watsonx_available():
        resp = _call_watsonx(prompt, system, max_tokens)
        if not resp.demo_mode:
            return resp
    return _demo_response("granite", context)


def call_gemini(prompt: str, system: str = "",
                context: dict = None, max_tokens: int = 800) -> LLMResponse:
    if omniroute_available():
        resp = _call_omniroute(prompt, GEMINI_MODEL_ID, system, max_tokens)
        if not resp.demo_mode:
            return resp
    if gemini_available():
        resp = _call_gemini(prompt, system, max_tokens)
        if not resp.demo_mode:
            return resp
    return _demo_response("gemini", context)


def call_groq(prompt: str, system: str = "",
              context: dict = None, max_tokens: int = 800) -> LLMResponse:
    if omniroute_available():
        resp = _call_omniroute(prompt, GROQ_MODEL_ID, system, max_tokens)
        if not resp.demo_mode:
            return resp
    if groq_available():
        resp = _call_groq(prompt, system, max_tokens)
        if not resp.demo_mode:
            return resp
    return _demo_response("groq", context)


def provider_status() -> dict:
    return {
        "IBM watsonx.ai (Granite)": {
            "configured": watsonx_available(),
            "model": WATSONX_MODEL_ID,
            "url": WATSONX_URL,
        },
        "Google Gemini": {
            "configured": gemini_available(),
            "model": GEMINI_MODEL_ID,
        },
        "Groq": {
            "configured": groq_available(),
            "model": GROQ_MODEL_ID,
        },
        "OmniRoute Gateway": {
            "configured": omniroute_available(),
            "url": OMNIROUTE_BASE_URL or "not set",
        },
    }
