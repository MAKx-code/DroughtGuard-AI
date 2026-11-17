"""SQLite memory + ChromaDB vector store for DroughtGuard AI."""
import sqlite3
import json
import datetime
from pathlib import Path

DB_PATH = Path("droughtguard.db")


# ── SQLite ──────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with _conn() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS agent_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT,
            provider TEXT,
            model_id TEXT,
            started_at TEXT,
            completed_at TEXT,
            latency_ms REAL,
            tokens_used INTEGER,
            tool_calls TEXT,
            status TEXT,
            error TEXT,
            guardrail_outcome TEXT,
            district TEXT,
            taluk TEXT
        );

        CREATE TABLE IF NOT EXISTS drought_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            district TEXT,
            taluk TEXT,
            risk_score REAL,
            risk_severity TEXT,
            assessment_json TEXT,
            agent_run_id INTEGER
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            district TEXT,
            taluk TEXT,
            severity TEXT,
            alert_text TEXT,
            status TEXT DEFAULT 'PENDING',
            approved_by TEXT,
            approved_at TEXT,
            rejection_reason TEXT
        );

        CREATE TABLE IF NOT EXISTS advisories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            district TEXT,
            taluk TEXT,
            crop TEXT,
            season TEXT,
            advisory_en TEXT,
            advisory_kn TEXT,
            agent_run_id INTEGER
        );

        CREATE TABLE IF NOT EXISTS session_context (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS guardrail_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            agent TEXT,
            check_type TEXT,
            input_text TEXT,
            outcome TEXT,
            detail TEXT
        );
        """)


def log_agent_run(
    agent_name: str, provider: str, model_id: str,
    started_at: str, completed_at: str, latency_ms: float,
    tokens_used: int = 0, tool_calls: list = None,
    status: str = "success", error: str = "",
    guardrail_outcome: str = "pass",
    district: str = "", taluk: str = ""
) -> int:
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO agent_runs
               (agent_name,provider,model_id,started_at,completed_at,
                latency_ms,tokens_used,tool_calls,status,error,
                guardrail_outcome,district,taluk)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (agent_name, provider, model_id, started_at, completed_at,
             latency_ms, tokens_used, json.dumps(tool_calls or []),
             status, error, guardrail_outcome, district, taluk)
        )
        return cur.lastrowid


def save_assessment(district: str, taluk: str, risk_score: float,
                    risk_severity: str, assessment: dict, run_id: int = 0) -> int:
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO drought_assessments
               (created_at,district,taluk,risk_score,risk_severity,assessment_json,agent_run_id)
               VALUES (?,?,?,?,?,?,?)""",
            (datetime.datetime.utcnow().isoformat(), district, taluk,
             risk_score, risk_severity, json.dumps(assessment), run_id)
        )
        return cur.lastrowid


def save_alert(district: str, taluk: str, severity: str, alert_text: str) -> int:
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO alerts (created_at,district,taluk,severity,alert_text,status)
               VALUES (?,?,?,?,?,?)""",
            (datetime.datetime.utcnow().isoformat(), district, taluk,
             severity, alert_text, "PENDING")
        )
        return cur.lastrowid


def approve_alert(alert_id: int, officer: str = "Officer"):
    with _conn() as con:
        con.execute(
            """UPDATE alerts SET status='APPROVED', approved_by=?, approved_at=?
               WHERE id=?""",
            (officer, datetime.datetime.utcnow().isoformat(), alert_id)
        )


def reject_alert(alert_id: int, reason: str = "", officer: str = "Officer"):
    with _conn() as con:
        con.execute(
            """UPDATE alerts SET status='REJECTED', approved_by=?, approved_at=?,
               rejection_reason=? WHERE id=?""",
            (officer, datetime.datetime.utcnow().isoformat(), reason, alert_id)
        )


def get_pending_alerts() -> list:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM alerts WHERE status='PENDING' ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_alerts(limit: int = 50) -> list:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_recent_runs(limit: int = 20) -> list:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def save_advisory(district: str, taluk: str, crop: str, season: str,
                  advisory_en: str, advisory_kn: str, run_id: int = 0):
    with _conn() as con:
        con.execute(
            """INSERT INTO advisories
               (created_at,district,taluk,crop,season,advisory_en,advisory_kn,agent_run_id)
               VALUES (?,?,?,?,?,?,?,?)""",
            (datetime.datetime.utcnow().isoformat(), district, taluk,
             crop, season, advisory_en, advisory_kn, run_id)
        )


def set_session(key: str, value):
    with _conn() as con:
        con.execute(
            """INSERT OR REPLACE INTO session_context (key, value, updated_at)
               VALUES (?, ?, ?)""",
            (key, json.dumps(value), datetime.datetime.utcnow().isoformat())
        )


def get_session(key: str, default=None):
    with _conn() as con:
        row = con.execute(
            "SELECT value FROM session_context WHERE key=?", (key,)
        ).fetchone()
        if row:
            return json.loads(row[0])
        return default


def log_guardrail_event(agent: str, check_type: str, input_text: str,
                        outcome: str, detail: str = ""):
    with _conn() as con:
        con.execute(
            """INSERT INTO guardrail_events
               (created_at,agent,check_type,input_text,outcome,detail)
               VALUES (?,?,?,?,?,?)""",
            (datetime.datetime.utcnow().isoformat(), agent, check_type,
             input_text[:500], outcome, detail)
        )


def get_guardrail_events(limit: int = 30) -> list:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM guardrail_events ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ── ChromaDB vector store ───────────────────────────────────────────────────

_chroma_client = None
_collection = None
_USE_CHROMA = False


def _init_chroma():
    global _chroma_client, _collection, _USE_CHROMA
    try:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path="./chroma_db")
        _collection = _chroma_client.get_or_create_collection(
            name="drought_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        # Seed if empty
        if _collection.count() == 0:
            _seed_knowledge()
        _USE_CHROMA = True
    except Exception as e:
        print(f"[VectorStore] ChromaDB unavailable, using keyword fallback: {e}")
        _USE_CHROMA = False


KNOWLEDGE_DOCS = [
    {
        "id": "pm_kisan",
        "text": "PM-KISAN scheme provides Rs 6000 per year to small and marginal farmers in three equal installments. Farmers must register through their local agriculture office or the PM-KISAN portal. During drought, timely disbursal is prioritized.",
        "topic": "government_scheme"
    },
    {
        "id": "crop_insurance",
        "text": "Pradhan Mantri Fasal Bima Yojana (PMFBY) covers crop losses due to drought, flood and other natural calamities. Premium is subsidized. Farmers must enroll before the crop season deadline. Claims are assessed using satellite and field data.",
        "topic": "crop_insurance"
    },
    {
        "id": "mgnrega_drought",
        "text": "MGNREGA provides 100 days of guaranteed wage employment per household. During drought, demand for work increases. District authorities should proactively plan water conservation works, check dam construction and borewell deepening under MGNREGA.",
        "topic": "relief_scheme"
    },
    {
        "id": "contingency_crops",
        "text": "During drought in Karnataka, recommended contingency crops include: short-duration pulses (cowpea, green gram), drought-tolerant millets (ragi, bajra, jowar), sunflower and groundnut for Rabi. Farmers should switch to low-water crops when rainfall deficit exceeds 30%.",
        "topic": "crop_advisory"
    },
    {
        "id": "water_conservation",
        "text": "Water conservation during drought: adopt drip irrigation to save 40-60% water; mulching reduces soil moisture loss; farm ponds help store excess rainwater; construct field bunds for water retention; use micro-irrigation for horticulture crops.",
        "topic": "water_advisory"
    },
    {
        "id": "fodder_scheme",
        "text": "During drought, the Karnataka government operates fodder banks and cattle camps. Livestock owners should register with the taluk panchayat. Subsidized fodder is provided to small and marginal farmers with livestock. Contact the Animal Husbandry Department.",
        "topic": "livestock_advisory"
    },
    {
        "id": "tanker_deployment",
        "text": "Drinking water tanker deployment is managed by the Karnataka Rural Water Supply and Sanitation Department. During severe drought, villages with less than 55 litres per capita per day supply are prioritized. Requests go through the Gram Panchayat to Taluk Panchayat.",
        "topic": "water_supply"
    },
    {
        "id": "drought_declaration",
        "text": "Karnataka drought declaration follows NDMA guidelines. KSNDMC monitors rainfall, NDVI, soil moisture and groundwater. Taluk-level drought is declared based on rainfall deviation, vegetation index and crop condition. This triggers relief under the State Disaster Response Fund (SDRF).",
        "topic": "drought_policy"
    },
    {
        "id": "soil_health",
        "text": "Soil moisture conservation: deep ploughing before monsoon helps retain moisture; applying 5 tonnes of FYM per hectare improves water holding capacity; green manuring with Dhaincha reduces moisture stress; raised bed cultivation reduces water requirement by 25-30%.",
        "topic": "soil_advisory"
    },
    {
        "id": "kharif_advisory",
        "text": "Kharif 2024 advisory for North Karnataka: delay sowing of cotton and tur dal if rainfall is below 50% of normal in June-July. Use pre-treated seeds with fungicide. Apply half dose of nitrogen at sowing. Irrigate at critical growth stages using available tank water. Consider crop diversification with short-duration cowpea.",
        "topic": "seasonal_advisory"
    },
]


def _seed_knowledge():
    docs = [d["text"] for d in KNOWLEDGE_DOCS]
    ids  = [d["id"]   for d in KNOWLEDGE_DOCS]
    metas = [{"topic": d["topic"]} for d in KNOWLEDGE_DOCS]
    _collection.add(documents=docs, ids=ids, metadatas=metas)


def query_knowledge(query_text: str, n_results: int = 3) -> list[dict]:
    """Return relevant knowledge chunks."""
    if _USE_CHROMA and _collection:
        try:
            results = _collection.query(
                query_texts=[query_text],
                n_results=min(n_results, _collection.count())
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            return [{"text": d, "topic": m.get("topic", "")} for d, m in zip(docs, metas)]
        except Exception:
            pass
    # Keyword fallback
    q_lower = query_text.lower()
    hits = []
    for doc in KNOWLEDGE_DOCS:
        if any(w in doc["text"].lower() for w in q_lower.split()):
            hits.append({"text": doc["text"], "topic": doc["topic"]})
        if len(hits) >= n_results:
            break
    return hits or [{"text": KNOWLEDGE_DOCS[0]["text"], "topic": "general"}]


# Initialize on import
init_db()
_init_chroma()
