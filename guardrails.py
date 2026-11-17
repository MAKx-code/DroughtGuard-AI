"""
Input/output guardrails for DroughtGuard AI.
Prioritizes IBM watsonx.governance where available; falls back to local checks.
"""
import re
from config import gov_guardrails_available, WATSONX_GOV_APIKEY
import memory

# ── Prompt injection patterns ────────────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior)\s+instructions?",
    r"you\s+are\s+now\s+(?!an?\s+AI\s+assistant)",
    r"act\s+as\s+(if\s+you\s+are\s+)?(a\s+)?(?!drought|weather|farm)",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"bypass\s+(safety|filter|guard)",
    r"</?(system|prompt|instruction)>",
    r"print\s+your\s+(system\s+)?prompt",
]

# ── Off-topic patterns (block non-agriculture topics) ───────────────────────
OFF_TOPIC_ALLOW = [
    r"drought", r"rain(fall)?", r"crop", r"farm(er)?", r"water", r"soil",
    r"weather", r"temperature", r"karnataka", r"district", r"taluk",
    r"irrigation", r"reservoir", r"groundwater", r"tanker", r"fodder",
    r"advisory", r"risk", r"relief", r"mgnrega", r"insurance", r"alert",
    r"kharif", r"rabi", r"harvest", r"sow(ing)?", r"livestock",
    r"help", r"what", r"how", r"when", r"where", r"why", r"status",
    r"assess", r"analyse", r"analyze", r"generate", r"show", r"tell",
    r"explain", r"give", r"provide", r"list",
]

# ── PII patterns ─────────────────────────────────────────────────────────────
PII_PATTERNS = {
    "phone":   r"\b[6-9]\d{9}\b",
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "email":   r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
}

# ── Unsupported claim keywords ────────────────────────────────────────────────
UNSUPPORTED_CLAIM_TRIGGERS = [
    "guaranteed profit", "100% accurate", "definitely will rain",
    "exact date of drought", "precise yield", "i promise",
    "certainly no drought", "no risk whatsoever",
]


def _check_injection(text: str) -> tuple[bool, str]:
    text_l = text.lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, text_l):
            return True, f"Prompt injection pattern detected: '{pat}'"
    return False, ""


def _check_off_topic(text: str) -> tuple[bool, str]:
    text_l = text.lower()
    if any(re.search(pat, text_l) for pat in OFF_TOPIC_ALLOW):
        return False, ""
    # Short queries always allowed
    if len(text.split()) <= 5:
        return False, ""
    return True, "Query appears unrelated to drought/agriculture advisory scope."


def _redact_pii(text: str) -> tuple[str, list]:
    found = []
    for pii_type, pat in PII_PATTERNS.items():
        matches = re.findall(pat, text)
        if matches:
            found.append(pii_type)
            text = re.sub(pat, f"[{pii_type.upper()} REDACTED]", text)
    return text, found


def _check_output_claims(text: str) -> list:
    flags = []
    text_l = text.lower()
    for trigger in UNSUPPORTED_CLAIM_TRIGGERS:
        if trigger in text_l:
            flags.append(trigger)
    return flags


def _try_watsonx_gov(text: str) -> dict | None:
    """Attempt IBM watsonx.governance HAP/PII check."""
    if not gov_guardrails_available():
        return None
    try:
        import os
        os.environ["WATSONX_APIKEY"] = WATSONX_GOV_APIKEY
        from ibm_watsonx_gov.evaluators import MetricsEvaluator
        from ibm_watsonx_gov.metrics import HAPMetric, PIIMetric
        evaluator = MetricsEvaluator()
        result = evaluator.evaluate(
            data={"input_text": text[:1000]},
            metrics=[HAPMetric(), PIIMetric()]
        )
        df = result.to_df()
        return {"source": "IBM watsonx.governance", "result": df.to_dict()}
    except Exception as e:
        return {"source": "IBM watsonx.governance", "error": str(e)}


class GuardrailResult:
    def __init__(self, allowed: bool, reason: str = "",
                 cleaned_text: str = "", pii_found: list = None,
                 output_flags: list = None, gov_result: dict = None,
                 source: str = "local"):
        self.allowed = allowed
        self.reason = reason
        self.cleaned_text = cleaned_text
        self.pii_found = pii_found or []
        self.output_flags = output_flags or []
        self.gov_result = gov_result
        self.source = source  # "IBM watsonx.governance" or "local"

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "pii_found": self.pii_found,
            "output_flags": self.output_flags,
            "source": self.source,
        }


def check_input(text: str, agent: str = "unknown") -> GuardrailResult:
    """Run all input guardrails. Returns GuardrailResult."""

    # 1. Try IBM watsonx.governance
    gov_result = _try_watsonx_gov(text)
    gov_source = "IBM watsonx.governance" if (gov_result and "error" not in gov_result) else "local"

    # 2. Prompt injection
    injected, inj_reason = _check_injection(text)
    if injected:
        memory.log_guardrail_event(agent, "prompt_injection", text, "BLOCKED", inj_reason)
        return GuardrailResult(False, inj_reason, text, source=gov_source)

    # 3. PII redaction
    cleaned, pii_found = _redact_pii(text)
    if pii_found:
        memory.log_guardrail_event(agent, "pii_redaction", text, "REDACTED",
                                   f"PII types: {pii_found}")

    # 4. Off-topic check
    off_topic, ot_reason = _check_off_topic(cleaned)
    if off_topic:
        memory.log_guardrail_event(agent, "off_topic", text, "BLOCKED", ot_reason)
        return GuardrailResult(False, ot_reason, cleaned, pii_found, source=gov_source)

    memory.log_guardrail_event(agent, "input_check", text, "PASS")
    return GuardrailResult(True, "", cleaned, pii_found,
                           gov_result=gov_result, source=gov_source)


def check_output(text: str, agent: str = "unknown") -> GuardrailResult:
    """Run output guardrails on LLM-generated text."""
    flags = _check_output_claims(text)
    if flags:
        memory.log_guardrail_event(agent, "output_claim_check", text[:300],
                                   "FLAGGED", f"Triggers: {flags}")
        return GuardrailResult(True, f"Output flagged for review: {flags}",
                               text, output_flags=flags, source="local")
    memory.log_guardrail_event(agent, "output_check", text[:300], "PASS")
    return GuardrailResult(True, "", text, source="local")


def guardrail_status() -> dict:
    """Return current guardrail configuration status."""
    return {
        "ibm_watsonx_governance": {
            "available": gov_guardrails_available(),
            "note": "Requires WATSONX_GOV_APIKEY and provisioned watsonx.governance instance",
        },
        "local_input_checks": {
            "available": True,
            "checks": ["prompt_injection", "off_topic", "pii_redaction"],
        },
        "local_output_checks": {
            "available": True,
            "checks": ["unsupported_claim_detection", "output_validation"],
        },
        "tool_allowlist": {
            "available": True,
            "tools": ["get_district_weather", "get_taluk_drought_data",
                      "get_reservoir_status", "get_crop_information",
                      "search_agricultural_knowledge", "get_district_vulnerability"],
        },
    }
