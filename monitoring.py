"""Lightweight agent execution monitoring and observability."""
import datetime
import json
from collections import defaultdict
from memory import get_recent_runs, get_guardrail_events, get_all_alerts


class ObsLog:
    """In-memory accumulator for the current session."""
    def __init__(self):
        self.entries: list[dict] = []

    def record(self, agent: str, provider: str, model: str,
               latency_ms: float, tokens: int = 0,
               tool_calls: list = None, status: str = "success",
               error: str = "", guardrail: str = "pass"):
        self.entries.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agent": agent, "provider": provider, "model": model,
            "latency_ms": latency_ms, "tokens": tokens,
            "tool_calls": tool_calls or [],
            "status": status, "error": error, "guardrail": guardrail,
        })

    def session_summary(self) -> dict:
        if not self.entries:
            return {}
        total_tokens = sum(e["tokens"] for e in self.entries)
        avg_latency  = sum(e["latency_ms"] for e in self.entries) / len(self.entries)
        by_provider  = defaultdict(int)
        by_agent     = defaultdict(lambda: {"calls": 0, "tokens": 0, "latency_ms": 0})
        tool_counts  = defaultdict(int)
        success = sum(1 for e in self.entries if e["status"] == "success")
        for e in self.entries:
            by_provider[e["provider"]] += 1
            by_agent[e["agent"]]["calls"] += 1
            by_agent[e["agent"]]["tokens"] += e["tokens"]
            by_agent[e["agent"]]["latency_ms"] += e["latency_ms"]
            for t in e["tool_calls"]:
                tool_counts[t] += 1
        return {
            "total_calls": len(self.entries),
            "success_calls": success,
            "failed_calls": len(self.entries) - success,
            "total_tokens": total_tokens,
            "avg_latency_ms": round(avg_latency, 1),
            "by_provider": dict(by_provider),
            "by_agent": {k: dict(v) for k, v in by_agent.items()},
            "tool_counts": dict(tool_counts),
        }


obs = ObsLog()  # singleton


def get_monitoring_data() -> dict:
    """Aggregate data for the monitoring dashboard."""
    recent_runs = get_recent_runs(50)
    guardrail_events = get_guardrail_events(30)
    all_alerts = get_all_alerts(50)

    # Compute stats from DB
    total_runs = len(recent_runs)
    success_runs = sum(1 for r in recent_runs if r["status"] == "success")
    fail_runs = total_runs - success_runs
    latencies = [r["latency_ms"] for r in recent_runs if r["latency_ms"]]
    avg_lat = round(sum(latencies) / len(latencies), 1) if latencies else 0
    total_tokens = sum(r["tokens_used"] or 0 for r in recent_runs)

    provider_counts = defaultdict(int)
    for r in recent_runs:
        provider_counts[r["provider"] or "unknown"] += 1

    tool_all = []
    for r in recent_runs:
        try:
            tool_all.extend(json.loads(r.get("tool_calls") or "[]"))
        except Exception:
            pass
    tool_counts = defaultdict(int)
    for t in tool_all:
        tool_counts[t] += 1

    guardrail_blocked = sum(1 for g in guardrail_events if g["outcome"] in ("BLOCKED", "FLAGGED"))
    pending_alerts = sum(1 for a in all_alerts if a["status"] == "PENDING")
    approved_alerts = sum(1 for a in all_alerts if a["status"] == "APPROVED")

    return {
        "total_runs": total_runs,
        "success_runs": success_runs,
        "fail_runs": fail_runs,
        "avg_latency_ms": avg_lat,
        "total_tokens": total_tokens,
        "provider_counts": dict(provider_counts),
        "tool_counts": dict(tool_counts),
        "guardrail_blocked": guardrail_blocked,
        "guardrail_total": len(guardrail_events),
        "pending_alerts": pending_alerts,
        "approved_alerts": approved_alerts,
        "recent_runs": recent_runs[:10],
        "guardrail_events": guardrail_events[:10],
        "all_alerts": all_alerts[:20],
        "session_summary": obs.session_summary(),
    }
