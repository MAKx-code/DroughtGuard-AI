"""
DroughtGuard AI — Premium Environmental Intelligence Dashboard
Agentic Drought Early Warning & Farmer Advisory System for Karnataka
"""
import base64
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import datetime
from pathlib import Path

# ── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="DroughtGuard AI — Karnataka",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Core imports ─────────────────────────────────────────────────────────────
from config import (
    APP_NAME, APP_VERSION, DATA_MODE, DISTRICTS, DISTRICT_TALUKS,
    SEVERITY_COLORS, watsonx_available, gemini_available,
    groq_available, omniroute_available,
)
from data_service import get_dataset, get_district_summary, get_taluk_data
from drought_engine import calculate_risk, enrich_dataframe
from agents import run_agent1, run_agent2, run_agent3, run_full_assessment
from memory import (
    get_pending_alerts, get_all_alerts, approve_alert, reject_alert,
    get_recent_runs,
)
from monitoring import get_monitoring_data
from guardrails import guardrail_status
from llm_router import provider_status

# ── SVG background (base64 inline) ───────────────────────────────────────────
def _load_bg() -> str:
    p = Path("assets/bg_farmland.svg")
    if p.exists():
        data = base64.b64encode(p.read_bytes()).decode()
        return f"data:image/svg+xml;base64,{data}"
    return ""

BG_DATA = _load_bg()

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════
C_FOREST   = "#14291F"
C_SAGE_DK  = "#243D32"
C_MUTED    = "#527A60"
C_SAGE_LT  = "#A8D5A0"
C_AMBER    = "#D5A34C"
C_OFFWHITE = "#E8EFE5"
C_BORDER   = "#2E4A3A"
C_TEXT     = "#D4E8D4"
C_MUTED_TXT= "#7FA88A"
C_CARD_BG  = "rgba(20,41,31,0.82)"
C_CARD_BRD = "#2E4A3A"

SEV_COLORS = {
    "Low":      "#4caf77",
    "Moderate": "#D5A34C",
    "High":     "#E0783A",
    "Severe":   "#C03030",
}
SEV_BG = {
    "Low":      "rgba(76,175,119,0.15)",
    "Moderate": "rgba(213,163,76,0.15)",
    "High":     "rgba(224,120,58,0.15)",
    "Severe":   "rgba(192,48,48,0.18)",
}

# ── Chart theme ───────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(20,41,31,0.0)",
    plot_bgcolor="rgba(20,41,31,0.0)",
    font=dict(color=C_TEXT, family="Inter, system-ui, sans-serif", size=12),
    legend=dict(bgcolor="rgba(20,41,31,0.7)", bordercolor=C_BORDER, borderwidth=1,
                font=dict(color=C_TEXT)),
)
# Reusable axis defaults (apply manually to avoid duplicate kwarg conflicts)
AXIS_X = dict(gridcolor="rgba(82,122,96,0.2)", zerolinecolor="rgba(82,122,96,0.3)",
              tickfont=dict(color=C_MUTED_TXT))
AXIS_Y = dict(gridcolor="rgba(82,122,96,0.2)", zerolinecolor="rgba(82,122,96,0.3)",
              tickfont=dict(color=C_MUTED_TXT))

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
bg_style = (
    f"background-image: url('{BG_DATA}');"
    "background-size: cover; background-position: center; background-attachment: fixed;"
    if BG_DATA else f"background: {C_FOREST};"
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {{
  font-family: 'Inter', -apple-system, system-ui, sans-serif;
  color: {C_TEXT};
}}

/* ── App background ── */
.stApp {{
  {bg_style}
  background-color: {C_FOREST};
}}
.stApp::before {{
  content: '';
  position: fixed;
  inset: 0;
  background: rgba(13,25,18,0.78);
  z-index: 0;
  pointer-events: none;
}}
.main .block-container {{
  position: relative;
  z-index: 1;
  max-width: 1420px;
  padding: 1.5rem 2rem 3rem;
  background: transparent;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
  background: rgba(13,22,16,0.96) !important;
  border-right: 1px solid {C_BORDER} !important;
  backdrop-filter: blur(8px);
}}
/* Target only text nodes — never override input/svg/button with blanket * rule */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not([data-baseweb]),
[data-testid="stSidebar"] div:not([data-baseweb]) > label,
[data-testid="stSidebar"] .stMarkdown {{
  color: {C_TEXT};
}}
[data-testid="stSidebar"] .stRadio > div > label {{
  color: {C_MUTED_TXT} !important;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {{
  color: {C_TEXT} !important;
  font-size: 0.875rem;
  padding: 0.4rem 0.75rem;
  border-radius: 6px;
  transition: background 0.15s;
  display: block;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {{
  background: rgba(82,122,96,0.18);
}}
/* Keep radio button circles visible */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] input[type="radio"] {{
  accent-color: {C_SAGE_LT};
}}

/* ── Page header ── */
.pg-header {{
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
  padding-bottom: 0.875rem;
  border-bottom: 1px solid {C_BORDER};
}}
.pg-header h2 {{
  font-size: 1.5rem;
  font-weight: 600;
  color: {C_OFFWHITE};
  margin: 0;
  letter-spacing: -0.02em;
}}
.pg-header .pg-sub {{
  font-size: 0.8rem;
  color: {C_MUTED_TXT};
  margin-top: 0.1rem;
}}

/* ── KPI cards ── */
.kpi-row {{ display: flex; gap: 1rem; margin-bottom: 1.25rem; flex-wrap: wrap; }}
.kpi-card {{
  flex: 1; min-width: 160px;
  background: {C_CARD_BG};
  border: 1px solid {C_CARD_BRD};
  border-radius: 12px;
  padding: 1.1rem 1.25rem 0.9rem;
  backdrop-filter: blur(4px);
  position: relative;
  overflow: hidden;
}}
.kpi-card::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, {C_MUTED} 0%, {C_SAGE_LT} 100%);
  opacity: 0.6;
}}
.kpi-value {{
  font-size: 2.2rem;
  font-weight: 700;
  line-height: 1;
  color: {C_OFFWHITE};
  letter-spacing: -0.03em;
}}
.kpi-label {{
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: {C_MUTED_TXT};
  margin-top: 0.35rem;
}}
.kpi-delta {{
  font-size: 0.75rem;
  color: {C_SAGE_LT};
  margin-top: 0.2rem;
}}

/* ── Section header ── */
.sec-hdr {{
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: {C_MUTED_TXT};
  margin: 1.5rem 0 0.75rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}}
.sec-hdr::after {{
  content: '';
  flex: 1;
  height: 1px;
  background: {C_BORDER};
  opacity: 0.6;
}}

/* ── Severity badges ── */
.badge {{
  display: inline-block;
  padding: 0.18rem 0.7rem;
  border-radius: 20px;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}}
.badge-low      {{ background: rgba(76,175,119,0.18); color: #6ED49A; border: 1px solid rgba(76,175,119,0.3); }}
.badge-moderate {{ background: rgba(213,163,76,0.18);  color: #E8C06A; border: 1px solid rgba(213,163,76,0.3); }}
.badge-high     {{ background: rgba(224,120,58,0.18);  color: #F09060; border: 1px solid rgba(224,120,58,0.3); }}
.badge-severe   {{ background: rgba(192,48,48,0.2);    color: #EE7070; border: 1px solid rgba(192,48,48,0.35); }}

/* ── Content cards ── */
.content-card {{
  background: {C_CARD_BG};
  border: 1px solid {C_CARD_BRD};
  border-radius: 12px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 1rem;
  backdrop-filter: blur(4px);
}}

/* ── Agent cards ── */
.agent-card {{
  background: {C_CARD_BG};
  border: 1px solid {C_CARD_BRD};
  border-radius: 12px;
  padding: 1rem 1.25rem;
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  gap: 1rem;
  backdrop-filter: blur(4px);
}}
.agent-icon {{
  font-size: 1.75rem;
  line-height: 1;
  flex-shrink: 0;
}}
.agent-info .agent-name {{
  font-weight: 600;
  font-size: 0.95rem;
  color: {C_OFFWHITE};
}}
.agent-info .agent-provider {{
  font-size: 0.75rem;
  color: {C_MUTED_TXT};
  margin-top: 0.1rem;
}}
.agent-status-live {{
  margin-left: auto;
  background: rgba(76,175,119,0.18);
  border: 1px solid rgba(76,175,119,0.3);
  color: #6ED49A;
  padding: 0.2rem 0.65rem;
  border-radius: 20px;
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  flex-shrink: 0;
}}
.agent-status-demo {{
  margin-left: auto;
  background: rgba(213,163,76,0.15);
  border: 1px solid rgba(213,163,76,0.3);
  color: #E8C06A;
  padding: 0.2rem 0.65rem;
  border-radius: 20px;
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  flex-shrink: 0;
}}

/* ── Tool trace chips ── */
.tool-chip {{
  display: inline-block;
  background: rgba(82,122,96,0.2);
  border: 1px solid rgba(82,122,96,0.35);
  border-radius: 4px;
  padding: 0.15rem 0.5rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  color: {C_SAGE_LT};
  margin: 0.1rem 0.15rem;
}}

/* ── Advisory box ── */
.advisory-box {{
  background: rgba(20,41,31,0.6);
  border: 1px solid {C_BORDER};
  border-radius: 10px;
  padding: 1.1rem 1.25rem;
  font-size: 0.875rem;
  line-height: 1.75;
  color: {C_TEXT};
  white-space: pre-wrap;
  word-wrap: break-word;
  min-height: 120px;
}}

/* ── Data table ── */
[data-testid="stDataFrame"] {{
  border: 1px solid {C_CARD_BRD} !important;
  border-radius: 10px !important;
  overflow: hidden;
}}

/* ── Indicator row ── */
.ind-row {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.4rem 0;
  border-bottom: 1px solid rgba(46,74,58,0.4);
  font-size: 0.85rem;
}}
.ind-label {{ color: {C_MUTED_TXT}; }}
.ind-value {{ font-weight: 500; color: {C_OFFWHITE}; }}

/* ── Alert card ── */
.alert-high {{
  background: rgba(224,120,58,0.1);
  border: 1px solid rgba(224,120,58,0.3);
  border-left: 4px solid #E0783A;
  border-radius: 10px;
  padding: 0.9rem 1.1rem;
  margin-bottom: 0.75rem;
}}
.alert-severe {{
  background: rgba(192,48,48,0.12);
  border: 1px solid rgba(192,48,48,0.35);
  border-left: 4px solid #C03030;
  border-radius: 10px;
  padding: 0.9rem 1.1rem;
  margin-bottom: 0.75rem;
}}

/* ── Progress bar override ── */
.stProgress > div > div {{ background: {C_MUTED} !important; }}
.stProgress > div > div > div {{ background: {C_SAGE_LT} !important; }}

/* ── Demo banner ── */
.demo-banner {{
  background: rgba(213,163,76,0.12);
  border: 1px solid rgba(213,163,76,0.3);
  border-radius: 8px;
  padding: 0.5rem 1rem;
  font-size: 0.78rem;
  color: #E8C06A;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}}

/* ── Button overrides ── */
.stButton > button {{
  border-radius: 8px;
  font-weight: 500;
  font-size: 0.875rem;
  transition: all 0.15s;
}}
.stButton > button[kind="primary"] {{
  background: {C_MUTED} !important;
  border-color: {C_MUTED} !important;
  color: white !important;
}}
.stButton > button[kind="primary"]:hover {{
  background: #3d6a4a !important;
}}

/* ── Selectbox / input ── */
.stSelectbox > div > div,
.stTextInput > div > div > input {{
  background: rgba(20,41,31,0.8) !important;
  border-color: {C_BORDER} !important;
  color: {C_TEXT} !important;
  border-radius: 8px !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
  background: rgba(20,41,31,0.6);
  border-radius: 10px;
  padding: 0.25rem;
  gap: 0.25rem;
  border: 1px solid {C_BORDER};
}}
.stTabs [data-baseweb="tab"] {{
  background: transparent;
  color: {C_MUTED_TXT};
  border-radius: 7px;
  font-size: 0.82rem;
  padding: 0.4rem 0.9rem;
  border: none;
}}
.stTabs [aria-selected="true"] {{
  background: rgba(82,122,96,0.3) !important;
  color: {C_OFFWHITE} !important;
}}

/* ── Hide default header ── */
header[data-testid="stHeader"] {{ display: none; }}

/* ── Metric cards ── */
[data-testid="stMetric"] {{
  background: {C_CARD_BG};
  border: 1px solid {C_CARD_BRD};
  border-radius: 10px;
  padding: 0.75rem 1rem !important;
  backdrop-filter: blur(4px);
}}
[data-testid="stMetric"] label {{ color: {C_MUTED_TXT} !important; font-size: 0.75rem !important; }}
[data-testid="stMetric"] [data-testid="stMetricValue"] {{ color: {C_OFFWHITE} !important; }}

/* ── Spinner ── */
.stSpinner {{ color: {C_SAGE_LT} !important; }}

/* ── Footer ── */
.dg-footer {{
  text-align: center;
  color: {C_MUTED_TXT};
  font-size: 0.72rem;
  padding: 1.5rem 0 0.5rem;
  border-top: 1px solid {C_BORDER};
  margin-top: 2rem;
}}

/* ── Governance row ── */
.gov-row {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.65rem 0;
  border-bottom: 1px solid rgba(46,74,58,0.4);
  font-size: 0.85rem;
}}
.gov-label {{ color: {C_TEXT}; font-weight: 500; }}
.gov-active {{ color: #6ED49A; font-size: 0.75rem; font-weight: 600; }}
.gov-inactive {{ color: {C_MUTED_TXT}; font-size: 0.75rem; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 1.25rem 0 0.75rem;'>
      <div style='font-size:2.2rem; margin-bottom:0.4rem;'>🌿</div>
      <div style='font-size:1.05rem; font-weight:700; color:{C_OFFWHITE}; letter-spacing:-0.01em;'>DroughtGuard AI</div>
      <div style='font-size:0.68rem; color:{C_MUTED_TXT}; margin-top:0.2rem; letter-spacing:0.06em; text-transform:uppercase;'>Karnataka Early Warning</div>
    </div>
    <div style='height:1px; background:{C_BORDER}; margin: 0.5rem 0 1rem;'></div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        [
            "🗺️  Overview",
            "🔍  Drought Risk Explorer",
            "🤖  AI Agent Intelligence",
            "🌾  Farmer Advisory",
            "🏛️  Officer Command Center",
            "📊  Monitoring & Governance",
        ],
        label_visibility="collapsed",
    )

    st.markdown(f"<div style='height:1px; background:{C_BORDER}; margin:1rem 0 0.75rem;'></div>", unsafe_allow_html=True)

    # Provider status
    st.markdown(f"<div style='font-size:0.65rem; text-transform:uppercase; letter-spacing:0.12em; color:{C_MUTED_TXT}; margin-bottom:0.5rem;'>AI Providers</div>", unsafe_allow_html=True)
    prov = provider_status()
    prov_icons = {
        "IBM watsonx.ai (Granite)": ("🔷", "Granite"),
        "Google Gemini":            ("🔵", "Gemini"),
        "Groq":                     ("⚡", "Groq"),
        "OmniRoute Gateway":        ("🔀", "OmniRoute"),
    }
    for pname, pinfo in prov.items():
        ok = pinfo.get("configured", False)
        icon, short = prov_icons.get(pname, ("○", pname))
        dot = f"<span style='color:#6ED49A;'>●</span>" if ok else f"<span style='color:{C_AMBER};'>●</span>"
        status = "Live" if ok else "Demo"
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:0.5rem;padding:0.2rem 0;font-size:0.78rem;'>"
            f"{dot} <span style='color:{C_TEXT};'>{short}</span>"
            f"<span style='margin-left:auto;color:{'#6ED49A' if ok else C_AMBER};font-size:0.7rem;'>{status}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(f"""
    <div style='height:1px; background:{C_BORDER}; margin:0.75rem 0;'></div>
    <div style='text-align:center; font-size:0.65rem; color:{C_MUTED_TXT};'>
      v{APP_VERSION}<br>
      <span style='color:{C_AMBER}; font-size:0.62rem;'>⚠ {DATA_MODE}</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def load_enriched_data():
    df = get_dataset()
    df, risks = enrich_dataframe(df)
    return df, risks

@st.cache_data(ttl=300)
def load_district_summary():
    df = get_district_summary()
    df_risk, _ = enrich_dataframe(df)
    return df_risk

df_taluk, _ = load_enriched_data()
df_district  = load_district_summary()

def _badge(severity: str) -> str:
    cls = severity.lower() if severity else "low"
    return f'<span class="badge badge-{cls}">{severity}</span>'

def _demo_note():
    st.markdown(
        '<div class="demo-banner">⚠ <strong>Synthetic Demo Data</strong> — All values are generated for demonstration. Not official KSNDMC, IMD or government observations.</div>',
        unsafe_allow_html=True,
    )

def _pg_header(icon: str, title: str, sub: str = ""):
    sub_html = f'<div class="pg-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="pg-header"><span style="font-size:1.6rem;">{icon}</span>'
        f'<div><h2>{title}</h2>{sub_html}</div></div>',
        unsafe_allow_html=True,
    )

def _sec(label: str):
    st.markdown(f'<div class="sec-hdr">{label}</div>', unsafe_allow_html=True)

def _card_open(extra_style: str = ""):
    st.markdown(f'<div class="content-card" style="{extra_style}">', unsafe_allow_html=True)

def _card_close():
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    _pg_header("🗺️", "Karnataka Drought Overview",
               "Real-time risk intelligence across 8 districts · Kharif 2024")
    _demo_note()

    # ── KPI row ──────────────────────────────────────────────────────────────
    high_severe  = df_district[df_district["risk_severity"].isin(["High","Severe"])]
    avg_risk     = df_taluk["risk_score"].mean()
    critical_w   = df_taluk[df_taluk["drinking_water_status"] == "Critical"]["taluk"].nunique()
    sev_district = df_district.nlargest(1, "risk_score")
    top_name     = sev_district.iloc[0]["district"] if len(sev_district) else "—"
    top_sev      = sev_district.iloc[0]["risk_severity"] if len(sev_district) else "—"

    c1, c2, c3, c4 = st.columns(4)
    sev_color = SEV_COLORS.get(
        "Severe" if avg_risk>=75 else "High" if avg_risk>=55 else "Moderate" if avg_risk>=30 else "Low",
        C_SAGE_LT)
    kpis = [
        (c1, "8",           "Districts Monitored",       C_OFFWHITE,  "Karnataka coverage"),
        (c2, str(len(high_severe)), "High / Severe Risk", SEV_COLORS["High"], "Districts at elevated risk"),
        (c3, f"{avg_risk:.0f}",   "Avg Risk Score",       sev_color,   "0 – 100 scale"),
        (c4, str(critical_w),     "Critical Water Taluks",SEV_COLORS["Severe"], "Drinking water shortage"),
    ]
    for col, val, lbl, color, delta in kpis:
        with col:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-value" style="color:{color};">{val}</div>'
                f'<div class="kpi-label">{lbl}</div>'
                f'<div class="kpi-delta">{delta}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ── Map + Risk bar ────────────────────────────────────────────────────────
    col_map, col_right = st.columns([3, 2], gap="medium")

    with col_map:
        _sec("DISTRICT DROUGHT RISK MAP")
        fig_map = px.scatter_map(
            df_district,
            lat="lat", lon="lon",
            color="risk_severity",
            size="risk_score",
            hover_name="district",
            hover_data={
                "risk_score":             True,
                "rainfall_deviation_pct": ":.1f",
                "reservoir_storage_pct":  ":.1f",
                "lat": False, "lon": False,
            },
            color_discrete_map=SEV_COLORS,
            size_max=38,
            zoom=6.0,
            center={"lat": 15.3, "lon": 76.5},
            map_style="carto-darkmatter",
        )
        fig_map.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                title=dict(text="Severity", font=dict(color=C_MUTED_TXT, size=11)),
                bgcolor="rgba(13,22,16,0.85)",
                bordercolor=C_BORDER, borderwidth=1,
                font=dict(color=C_TEXT),
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_right:
        _sec("DISTRICT RISK SCORES")
        df_sorted = df_district.sort_values("risk_score", ascending=True)
        colors_bar = [SEV_COLORS.get(s, C_MUTED) for s in df_sorted["risk_severity"]]
        fig_bar = go.Figure(go.Bar(
            x=df_sorted["risk_score"],
            y=df_sorted["district"],
            orientation="h",
            marker=dict(color=colors_bar, opacity=0.9),
            text=df_sorted["risk_score"].round(1).astype(str),
            textposition="outside",
            textfont=dict(color=C_MUTED_TXT, size=11),
        ))
        fig_bar.update_layout(
            **CHART_LAYOUT,
            height=240,
            xaxis=dict(range=[0,105], gridcolor="rgba(82,122,96,0.15)",
                       tickfont=dict(color=C_MUTED_TXT), title=""),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=C_TEXT)),
            margin=dict(l=0, r=50, t=4, b=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Severity summary
        _sec("SEVERITY DISTRIBUTION")
        for sev in ["Severe","High","Moderate","Low"]:
            cnt = len(df_district[df_district["risk_severity"] == sev])
            if cnt == 0:
                continue
            pct = cnt / len(df_district) * 100
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.6rem;margin:0.3rem 0;">'
                f'{_badge(sev)}'
                f'<div style="flex:1;background:rgba(46,74,58,0.3);border-radius:4px;height:6px;">'
                f'<div style="width:{pct:.0f}%;background:{SEV_COLORS[sev]};height:6px;border-radius:4px;"></div>'
                f'</div>'
                f'<span style="font-size:0.78rem;color:{C_MUTED_TXT};min-width:16px;text-align:right;">{cnt}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Rainfall deficit chart ────────────────────────────────────────────────
    _sec("SEASONAL RAINFALL vs NORMAL (mm)")
    fig_rain = go.Figure()
    fig_rain.add_trace(go.Bar(
        name="Received", x=df_district["district"],
        y=df_district["rainfall_received_mm"].round(1),
        marker=dict(color=C_MUTED, opacity=0.85),
    ))
    fig_rain.add_trace(go.Bar(
        name="Normal", x=df_district["district"],
        y=df_district["normal_rainfall_mm"].round(1),
        marker=dict(color="rgba(168,213,160,0.25)"),
        marker_line=dict(color="rgba(168,213,160,0.5)", width=1),
    ))
    fig_rain.update_layout(
        **CHART_LAYOUT,
        barmode="group", height=240,
        yaxis=dict(title="mm", gridcolor="rgba(82,122,96,0.15)",
                   tickfont=dict(color=C_MUTED_TXT)),
        xaxis=dict(tickfont=dict(color=C_TEXT)),
        margin=dict(l=0, r=0, t=4, b=0),
    )
    st.plotly_chart(fig_rain, use_container_width=True)

    # ── Alert feed ────────────────────────────────────────────────────────────
    alerts = get_all_alerts(5)
    if alerts:
        _sec("RECENT ALERT FEED")
        for a in alerts:
            icon = "🔴" if a["severity"] == "Severe" else "🟠" if a["severity"] == "High" else "🟡"
            sc = "#EE7070" if a["status"] == "PENDING" else "#6ED49A" if a["status"] == "APPROVED" else C_MUTED_TXT
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.6rem;padding:0.4rem 0;'
                f'border-bottom:1px solid rgba(46,74,58,0.3);font-size:0.82rem;">'
                f'{icon} <strong style="color:{C_OFFWHITE};">{a["district"]} / {a["taluk"]}</strong>'
                f'&nbsp;{_badge(a["severity"])}'
                f'<span style="margin-left:auto;color:{sc};font-size:0.75rem;">{a["status"]}</span>'
                f'<span style="color:{C_MUTED_TXT};font-size:0.72rem;margin-left:0.5rem;">{a["created_at"][:16]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DROUGHT RISK EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif "Drought Risk" in page:
    _pg_header("🔍", "Drought Risk Explorer", "District and taluk-level risk indicators")
    _demo_note()

    col_d, col_t = st.columns(2)
    with col_d:
        sel_district = st.selectbox("District", DISTRICTS)
    with col_t:
        taluks = DISTRICT_TALUKS.get(sel_district, [sel_district])
        sel_taluk = st.selectbox("Taluk", taluks)

    row = get_taluk_data(sel_district, sel_taluk)
    if row is None:
        st.error(f"Data not found for {sel_taluk}, {sel_district}")
        st.stop()

    risk = calculate_risk(row)
    sev  = risk["severity"]
    sev_col = SEV_COLORS[sev]

    # ── Risk header cards ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl, color in [
        (c1, str(risk["score"]),   "Risk Score",   sev_col),
        (c2, sev,                  "Severity",      sev_col),
        (c3, risk["crop_stress"],  "Crop Stress",   SEV_COLORS.get(
            "Severe" if risk["crop_stress"]=="Critical" else
            "High" if risk["crop_stress"]=="High" else
            "Moderate" if risk["crop_stress"]=="Moderate" else "Low", C_SAGE_LT)),
        (c4, row["drinking_water_status"], "Water Status",
            "#EE7070" if row["drinking_water_status"]=="Critical" else
            "#F09060" if row["drinking_water_status"]=="Stressed" else "#6ED49A"),
    ]:
        with col:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-value" style="font-size:1.6rem;color:{color};">{val}</div>'
                f'<div class="kpi-label">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    col_charts, col_data = st.columns([3, 2], gap="medium")

    with col_charts:
        _sec("RISK FACTOR BREAKDOWN")
        factors = risk["factor_scores"]
        labels  = [k.replace("_", " ").title() for k in factors]
        vals    = list(factors.values())
        # Radar
        fig_radar = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor=SEV_BG.get(sev, "rgba(82,122,96,0.15)"),
            line=dict(color=sev_col, width=2.5),
            marker=dict(color=sev_col, size=6),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(20,41,31,0.0)",
                radialaxis=dict(visible=True, range=[0,100],
                                tickfont=dict(color=C_MUTED_TXT, size=10),
                                gridcolor="rgba(82,122,96,0.25)"),
                angularaxis=dict(tickfont=dict(color=C_TEXT, size=11),
                                 gridcolor="rgba(82,122,96,0.2)"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=20,r=20,t=20,b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        _sec("WEIGHTED CONTRIBUTION")
        wvals = list(risk["weighted_scores"].values())
        wlbls = [k.replace("_"," ").title() for k in risk["weighted_scores"]]
        wcolors = [sev_col if v > 3 else C_MUTED for v in wvals]
        fig_fb = go.Figure(go.Bar(
            x=wlbls, y=wvals,
            marker=dict(color=wcolors, opacity=0.85),
            text=[f"{v:.1f}" for v in wvals],
            textposition="outside",
            textfont=dict(color=C_MUTED_TXT, size=10),
        ))
        fig_fb.update_layout(
            **CHART_LAYOUT,
            height=200,
            yaxis=dict(title="Weighted pts", gridcolor="rgba(82,122,96,0.15)",
                       tickfont=dict(color=C_MUTED_TXT)),
            xaxis=dict(tickangle=-15, tickfont=dict(color=C_TEXT, size=10)),
            margin=dict(l=0, r=0, t=4, b=0),
        )
        st.plotly_chart(fig_fb, use_container_width=True)

    with col_data:
        _sec("TALUK INDICATORS")
        indicators = [
            ("Rainfall Received",  f"{row['rainfall_received_mm']:.0f} mm"),
            ("Normal Rainfall",    f"{row['normal_rainfall_mm']:.0f} mm"),
            ("Rainfall Deviation", f"{row['rainfall_deviation_pct']:.1f}%"),
            ("Dry Spell",          f"{row['dry_spell_days']} days"),
            ("Temperature",        f"{row['temperature_c']:.1f}°C"),
            ("Humidity",           f"{row['humidity_pct']:.0f}%"),
            ("Soil Moisture",      f"{row['soil_moisture_pct']:.0f}%"),
            ("Groundwater Depth",  f"{row['groundwater_depth_m']:.1f} m BGL"),
            ("Reservoir Storage",  f"{row['reservoir_storage_pct']:.0f}%"),
            ("ET Anomaly",         f"+{row['et_anomaly_mm_day']:.2f} mm/day"),
            ("NDVI",               f"{row['ndvi']:.3f}"),
            ("Main Crop",          row["main_crop"]),
            ("SM Farmers",         f"{row['sm_farmer_pct']:.0f}%"),
            ("Livestock Risk",     row["livestock_fodder_risk"]),
        ]
        html_rows = "".join(
            f'<div class="ind-row"><span class="ind-label">{l}</span>'
            f'<span class="ind-value">{v}</span></div>'
            for l, v in indicators
        )
        st.markdown(f'<div class="content-card" style="padding:0.75rem 1rem;">{html_rows}</div>',
                    unsafe_allow_html=True)

        _sec("TOP RISK DRIVERS")
        for i, f in enumerate(risk["top_factors"], 1):
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.5rem;padding:0.3rem 0;font-size:0.85rem;">'
                f'<span style="background:{sev_col};color:white;border-radius:50%;width:20px;height:20px;'
                f'display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">{i}</span>'
                f'<span style="color:{C_TEXT};">{f}</span></div>',
                unsafe_allow_html=True,
            )

        _sec("RECOMMENDED INTERVENTIONS")
        for action in risk["interventions"]:
            st.markdown(
                f'<div style="display:flex;gap:0.5rem;padding:0.25rem 0;font-size:0.82rem;color:{C_TEXT};">'
                f'<span style="color:{C_AMBER};flex-shrink:0;">›</span>{action}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(f'<div style="font-size:0.68rem;color:{C_MUTED_TXT};margin-top:0.75rem;">'
                    f'⚠ {risk["note"]}</div>', unsafe_allow_html=True)

    # ── All taluks table ──────────────────────────────────────────────────────
    _sec(f"ALL TALUKS — {sel_district.upper()}")
    dist_taluks = df_taluk[df_taluk["district"] == sel_district][[
        "taluk","risk_score","risk_severity","rainfall_deviation_pct",
        "soil_moisture_pct","groundwater_depth_m","reservoir_storage_pct","drinking_water_status"
    ]].sort_values("risk_score", ascending=False).copy()
    dist_taluks.columns = ["Taluk","Risk Score","Severity","Rainfall Dev %",
                            "Soil Moisture %","GW Depth m","Reservoir %","Water Status"]
    st.dataframe(
        dist_taluks.style.background_gradient(subset=["Risk Score"], cmap="RdYlGn_r"),
        use_container_width=True, hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — AI AGENT INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif "Agent Intelligence" in page:
    _pg_header("🤖", "AI Agent Intelligence",
               "Three-agent multi-LLM orchestration · ReAct pattern · MCP tools")

    # ── Agent status cards ────────────────────────────────────────────────────
    prov_st = provider_status()
    agents_meta = [
        ("🏔️", "Drought Intelligence Agent", "IBM Granite · watsonx.ai",
         "Agent 1", "IBM watsonx.ai (Granite)"),
        ("🌾", "Farmer Advisory Agent", "Google Gemini · Bilingual",
         "Agent 2", "Google Gemini"),
        ("⚡", "Alert & Relief Agent", "Groq · Fast inference",
         "Agent 3", "Groq"),
    ]
    for icon, name, provider_lbl, tag, prov_key in agents_meta:
        ok = prov_st.get(prov_key, {}).get("configured", False)
        status_cls = "agent-status-live" if ok else "agent-status-demo"
        status_txt = "Live" if ok else "Demo"
        st.markdown(
            f'<div class="agent-card">'
            f'<div class="agent-icon">{icon}</div>'
            f'<div class="agent-info">'
            f'<div class="agent-name">{name}</div>'
            f'<div class="agent-provider">{provider_lbl}</div>'
            f'</div>'
            f'<span class="{status_cls}">{status_txt}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    # ── Run controls ─────────────────────────────────────────────────────────
    col_d, col_t, col_s, _, col_btn = st.columns([2,2,1,0.2,1])
    with col_d:
        agent_district = st.selectbox("District", DISTRICTS, key="ag_dist")
    with col_t:
        agent_taluks = DISTRICT_TALUKS.get(agent_district, [agent_district])
        agent_taluk  = st.selectbox("Taluk", agent_taluks, key="ag_taluk")
    with col_s:
        agent_season = st.selectbox("Season", ["Kharif","Rabi"], key="ag_season")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_all = st.button("🚀 Run All Agents", type="primary", use_container_width=True)

    if run_all:
        with st.spinner("Orchestrating 3-agent pipeline…"):
            st.session_state["agent_results"] = run_full_assessment(
                agent_district, agent_taluk, season=agent_season)

    results = st.session_state.get("agent_results")
    if results:
        orch = results.get("orchestrator", {})
        sev_o = orch.get("overall_severity","—")
        st.markdown(
            f'<div class="content-card" style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">'
            f'<span style="color:{C_MUTED_TXT};font-size:0.78rem;">ORCHESTRATOR RESULT</span>'
            f'&nbsp;{_badge(sev_o)}&nbsp;'
            f'<span style="color:{C_OFFWHITE};font-weight:600;">{orch.get("risk_score",0):.0f}/100</span>'
            f'<span style="color:{C_MUTED_TXT};font-size:0.78rem;">·</span>'
            f'<span style="font-size:0.82rem;color:{C_TEXT};">MCP calls: <strong>{orch.get("total_tool_calls",0)}</strong></span>'
            f'<span style="font-size:0.82rem;color:{C_TEXT};">· Officer approval: '
            f'<strong style="color:{"#EE7070" if orch.get("requires_officer_approval") else "#6ED49A"};">'
            f'{"Required" if orch.get("requires_officer_approval") else "Not required"}</strong></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        tab1, tab2, tab3 = st.tabs([
            "🏔️ Agent 1 — Drought Intelligence",
            "🌾 Agent 2 — Farmer Advisory",
            "⚡ Agent 3 — Alert & Relief",
        ])

        def _agent_meta_bar(result: dict):
            demo = result.get("demo_mode", True)
            st.markdown(
                f'<div style="display:flex;flex-wrap:wrap;gap:1.25rem;'
                f'margin-bottom:0.75rem;font-size:0.78rem;color:{C_MUTED_TXT};">'
                f'<span>Provider: <strong style="color:{C_TEXT};">{result.get("provider","—")}</strong></span>'
                f'<span>Model: <code style="color:{C_SAGE_LT};background:rgba(82,122,96,0.2);'
                f'padding:0 4px;border-radius:3px;">{result.get("model","—")}</code></span>'
                f'<span>{"<span style=\"color:#E8C06A;\">⚠ DEMO MODE</span>" if demo else "<span style=\"color:#6ED49A;\">✓ LIVE</span>"}</span>'
                f'<span>Latency: <strong style="color:{C_TEXT};">{result.get("latency_ms",0):.0f} ms</strong></span>'
                f'<span>Tokens: <strong style="color:{C_TEXT};">{result.get("tokens",0)}</strong></span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            tc = result.get("tool_calls",[])
            if tc:
                chips = " ".join(f'<span class="tool-chip">{t}</span>' for t in tc)
                st.markdown(
                    f'<div style="margin-bottom:0.75rem;font-size:0.75rem;color:{C_MUTED_TXT};">'
                    f'MCP tools: {chips}</div>',
                    unsafe_allow_html=True,
                )

        with tab1:
            a1 = results.get("agent1", {})
            _agent_meta_bar(a1)
            if a1.get("error"):
                st.error(f"Agent error: {a1['error']}")
            st.markdown(
                f'<div class="advisory-box">{a1.get("assessment","No output generated.")}</div>',
                unsafe_allow_html=True,
            )

        with tab2:
            a2 = results.get("agent2", {})
            _agent_meta_bar(a2)
            if a2.get("error"):
                st.error(f"Agent error: {a2['error']}")
            lang_sel = st.radio("Language", ["English","Kannada"], horizontal=True, key="ag_lang2")
            text = a2.get("advisory_en" if lang_sel=="English" else "advisory_kn",
                          a2.get("full_text","No advisory generated."))
            st.markdown(f'<div class="advisory-box">{text}</div>', unsafe_allow_html=True)

        with tab3:
            a3 = results.get("agent3", {})
            _agent_meta_bar(a3)
            if a3.get("error"):
                st.error(f"Agent error: {a3['error']}")
            if a3.get("requires_approval"):
                st.warning("⚠️ This alert requires Officer approval before publication.")
            st.markdown(
                f'<div class="advisory-box">{a3.get("alert_text","No alert generated.")}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<div class="content-card" style="text-align:center;padding:2.5rem;">'
            f'<div style="font-size:2rem;margin-bottom:0.5rem;">🤖</div>'
            f'<div style="color:{C_OFFWHITE};font-weight:500;margin-bottom:0.3rem;">No assessment yet</div>'
            f'<div style="color:{C_MUTED_TXT};font-size:0.85rem;">Select a district and taluk above, then click Run All Agents.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── History ───────────────────────────────────────────────────────────────
    _sec("RECENT AGENT EXECUTIONS")
    runs = get_recent_runs(10)
    if runs:
        df_runs = pd.DataFrame(runs)[[
            "agent_name","provider","model_id","latency_ms","tokens_used","status","started_at"
        ]].copy()
        df_runs.columns = ["Agent","Provider","Model","Latency(ms)","Tokens","Status","Started"]
        df_runs["Latency(ms)"] = df_runs["Latency(ms)"].round(0)
        st.dataframe(df_runs, use_container_width=True, hide_index=True)
    else:
        st.markdown(f'<div style="color:{C_MUTED_TXT};font-size:0.85rem;">No runs recorded yet.</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FARMER ADVISORY
# ══════════════════════════════════════════════════════════════════════════════
elif "Farmer Advisory" in page:
    _pg_header("🌾", "Farmer Advisory", "AI-generated bilingual advisories · Kannada & English")
    _demo_note()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        fa_district = st.selectbox("District", DISTRICTS, key="fa_dist")
    with col2:
        fa_taluks = DISTRICT_TALUKS.get(fa_district,[fa_district])
        fa_taluk  = st.selectbox("Taluk", fa_taluks, key="fa_taluk")
    with col3:
        fa_season = st.selectbox("Season", ["Kharif","Rabi"], key="fa_season")
    with col4:
        fa_lang = st.selectbox("Display", ["English","Kannada","Both"], key="fa_lang")

    row_fa = get_taluk_data(fa_district, fa_taluk)
    default_crop = row_fa["main_crop"] if row_fa else "Cotton"
    crops = ["Cotton","Paddy","Tur Dal","Groundnut","Sunflower","Ragi","Sugarcane","Tomato","Coconut",default_crop]
    crops = list(dict.fromkeys(crops))
    fa_crop = st.selectbox("Main Crop", crops, index=crops.index(default_crop))

    fa_query = st.text_input(
        "Optional farmer question",
        placeholder="e.g. What should I do if my tur dal seedlings are wilting?",
    )

    if st.button("🌱 Generate Advisory", type="primary"):
        with st.spinner("Generating bilingual advisory with Gemini…"):
            st.session_state["advisory_result"] = run_agent2(
                fa_district, fa_taluk, fa_crop, fa_season,
                user_query=fa_query or None,
            )

    adv = st.session_state.get("advisory_result")
    if adv:
        demo = adv.get("demo_mode", True)
        st.markdown(
            f'<div class="content-card" style="display:flex;flex-wrap:wrap;gap:1rem;'
            f'align-items:center;margin-bottom:0.75rem;">'
            f'<span style="color:{C_MUTED_TXT};font-size:0.78rem;">Provider: <strong style="color:{C_TEXT};">{adv.get("provider","—")}</strong></span>'
            f'<span style="color:{C_MUTED_TXT};font-size:0.78rem;">Model: <code style="color:{C_SAGE_LT};'
            f'background:rgba(82,122,96,0.2);padding:0 4px;border-radius:3px;">{adv.get("model","—")}</code></span>'
            f'<span style="font-size:0.78rem;">{"<span style=\'color:#E8C06A;\'>⚠ DEMO</span>" if demo else "<span style=\'color:#6ED49A;\'>✓ LIVE</span>"}</span>'
            f'<span style="margin-left:auto;">{_badge(adv.get("risk_severity","—"))}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if adv.get("error"):
            st.error(f"Agent error: {adv['error']}")

        c_en, c_kn = (st.columns(2) if fa_lang == "Both" else
                      ([st.container(), None] if fa_lang == "English" else
                       [None, st.container()]))

        if fa_lang in ("English","Both") and c_en:
            with c_en:
                st.markdown(f'<div class="sec-hdr">ENGLISH ADVISORY</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="advisory-box">{adv.get("advisory_en","No advisory generated.")}</div>',
                    unsafe_allow_html=True,
                )

        if fa_lang in ("Kannada","Both") and c_kn:
            with c_kn:
                st.markdown(f'<div class="sec-hdr">ಕನ್ನಡ ಸಲಹೆ</div>', unsafe_allow_html=True)
                kn = adv.get("advisory_kn","")
                if kn:
                    st.markdown(f'<div class="advisory-box">{kn}</div>', unsafe_allow_html=True)
                else:
                    st.info("Kannada section not generated. Enable live Gemini for bilingual output.")

        tc = adv.get("tool_calls",[])
        if tc:
            chips = " ".join(f'<span class="tool-chip">{t}</span>' for t in tc)
            st.markdown(
                f'<div style="font-size:0.75rem;color:{C_MUTED_TXT};margin-top:0.5rem;">MCP tools: {chips}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<div class="content-card" style="text-align:center;padding:2rem;">'
            f'<div style="font-size:1.8rem;margin-bottom:0.4rem;">🌱</div>'
            f'<div style="color:{C_OFFWHITE};font-weight:500;">Select parameters and click Generate Advisory</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Chat query ────────────────────────────────────────────────────────────
    _sec("ASK DROUGHTGUARD AI")
    chat_q = st.text_input("Your question:", placeholder="What crops should I plant if rainfall fails in Kalaburagi?")
    if st.button("Ask", key="chat_ask"):
        if chat_q:
            with st.spinner("Generating response…"):
                resp = run_agent2(fa_district, fa_taluk, fa_crop, fa_season, user_query=chat_q)
                st.markdown(f'<div class="advisory-box">{resp.get("advisory_en","")}</div>',
                            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — OFFICER COMMAND CENTER
# ══════════════════════════════════════════════════════════════════════════════
elif "Officer" in page:
    _pg_header("🏛️", "Officer Command Center",
               "Human-in-the-Loop approval · High and Severe alerts require officer action")

    # ── Priority districts ────────────────────────────────────────────────────
    _sec("PRIORITY RISK LOCATIONS")
    top_d = df_district.nlargest(4,"risk_score")[[
        "district","risk_score","risk_severity","main_crop",
        "rainfall_deviation_pct","reservoir_storage_pct"
    ]].copy()
    top_d.columns = ["District","Risk Score","Severity","Main Crop","Rainfall Dev %","Reservoir %"]
    st.dataframe(
        top_d.style.background_gradient(subset=["Risk Score"], cmap="RdYlGn_r"),
        use_container_width=True, hide_index=True,
    )

    st.markdown("")
    # ── Generate recommendation ───────────────────────────────────────────────
    _sec("GENERATE RELIEF RECOMMENDATION")
    col_od, col_ot, col_btn2 = st.columns([2,2,1])
    with col_od:
        ofc_district = st.selectbox("District", DISTRICTS, key="ofc_dist")
    with col_ot:
        ofc_taluks = DISTRICT_TALUKS.get(ofc_district,[ofc_district])
        ofc_taluk  = st.selectbox("Taluk", ofc_taluks, key="ofc_taluk")
    with col_btn2:
        st.markdown("<br>", unsafe_allow_html=True)
        gen_alert = st.button("⚡ Generate Alert", type="primary", use_container_width=True)

    if gen_alert:
        with st.spinner("Agent 3 generating relief recommendation…"):
            st.session_state["relief_result"] = run_agent3(ofc_district, ofc_taluk)

    relief = st.session_state.get("relief_result")
    if relief:
        sev_r = relief.get("risk_severity","")
        demo_r = relief.get("demo_mode",True)
        alert_cls = "alert-severe" if sev_r == "Severe" else "alert-high"
        st.markdown(
            f'<div class="{alert_cls}">'
            f'<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.6rem;">'
            f'{_badge(sev_r)}'
            f'<strong style="color:{C_OFFWHITE};">{relief.get("district","—")} / {relief.get("taluk","—")}</strong>'
            f'<span style="margin-left:auto;font-size:0.75rem;color:{C_MUTED_TXT};">'
            f'{"⚠ DEMO" if demo_r else "✓ LIVE"} · {relief.get("latency_ms",0):.0f} ms</span>'
            f'</div>'
            f'{"<div style=\"color:#E8C06A;font-size:0.8rem;margin-bottom:0.5rem;\">⚠ PENDING OFFICER APPROVAL — Do not publish until approved.</div>" if relief.get("requires_approval") else ""}'
            f'<div style="font-size:0.85rem;line-height:1.65;white-space:pre-wrap;color:{C_TEXT};">'
            f'{relief.get("alert_text","No recommendation generated.")[:1200]}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if relief.get("error"):
            st.error(f"Agent error: {relief['error']}")

    # ── Pending approvals ─────────────────────────────────────────────────────
    _sec("PENDING APPROVAL QUEUE")
    pending = get_pending_alerts()
    if not pending:
        st.markdown(
            f'<div class="content-card" style="text-align:center;padding:1.5rem;color:{C_MUTED_TXT};">'
            f'✓ No alerts pending approval</div>',
            unsafe_allow_html=True,
        )
    else:
        for alert in pending:
            sev_a = alert["severity"]
            with st.expander(
                f"{'🔴' if sev_a=='Severe' else '🟠'} [{sev_a}] {alert['district']} / {alert['taluk']} · {alert['created_at'][:16]}"
            ):
                st.markdown(
                    f'<div class="advisory-box" style="max-height:200px;overflow-y:auto;">'
                    f'{alert["alert_text"][:800]}…</div>',
                    unsafe_allow_html=True,
                )
                ca, cr = st.columns(2)
                with ca:
                    if st.button(f"✅ Approve #{alert['id']}", key=f"approve_{alert['id']}", type="primary"):
                        approve_alert(alert["id"])
                        st.success(f"Alert #{alert['id']} approved.")
                        st.rerun()
                with cr:
                    reason = st.text_input("Rejection reason (optional)", key=f"reason_{alert['id']}")
                    if st.button(f"❌ Reject #{alert['id']}", key=f"reject_{alert['id']}"):
                        reject_alert(alert["id"], reason)
                        st.info(f"Alert #{alert['id']} rejected.")
                        st.rerun()

    # ── Decision history ──────────────────────────────────────────────────────
    _sec("DECISION HISTORY")
    decided = [a for a in get_all_alerts(20) if a["status"] != "PENDING"]
    if decided:
        df_hist = pd.DataFrame(decided)[["district","taluk","severity","status","approved_by","approved_at"]]
        df_hist.columns = ["District","Taluk","Severity","Status","Officer","Decided At"]
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
    else:
        st.markdown(f'<div style="color:{C_MUTED_TXT};font-size:0.85rem;">No decisions recorded yet.</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — MONITORING & GOVERNANCE
# ══════════════════════════════════════════════════════════════════════════════
elif "Monitoring" in page:
    _pg_header("📊", "Monitoring & Governance",
               "Agent observability · Guardrail events · Provider metrics · Model comparison")

    mon = get_monitoring_data()
    guard_st = guardrail_status()
    prov_st  = provider_status()

    # ── KPIs ──────────────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    mon_kpis = [
        (c1, str(mon["total_runs"]),      "Agent Runs",       C_OFFWHITE),
        (c2, str(mon["success_runs"]),    "Successful",       "#6ED49A"),
        (c3, str(mon["fail_runs"]),       "Failed",           "#EE7070" if mon["fail_runs"] else C_MUTED_TXT),
        (c4, f"{mon['avg_latency_ms']:.0f}", "Avg Latency ms", C_OFFWHITE),
        (c5, str(mon["total_tokens"]),    "Total Tokens",     C_AMBER),
    ]
    for col, val, lbl, color in mon_kpis:
        with col:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-value" style="font-size:1.6rem;color:{color};">{val}</div>'
                f'<div class="kpi-label">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    tab_p, tab_t, tab_g, tab_o, tab_e = st.tabs([
        "🔌 Providers",
        "🔧 MCP Tools",
        "🛡️ Guardrails",
        "📈 Observability",
        "⚖️ Model Comparison",
    ])

    with tab_p:
        _sec("PROVIDER CONFIGURATION & STATUS")
        for pname, pinfo in prov_st.items():
            ok = pinfo.get("configured",False)
            st.markdown(
                f'<div class="gov-row">'
                f'<span class="gov-label">{pname}</span>'
                f'<div style="display:flex;align-items:center;gap:1rem;">'
                f'<code style="font-size:0.72rem;color:{C_SAGE_LT};background:rgba(82,122,96,0.2);'
                f'padding:0 4px;border-radius:3px;">{pinfo.get("model","—")}</code>'
                f'<span class="{"gov-active" if ok else "gov-inactive"}">{"✓ Live" if ok else "⚠ Demo"}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    with tab_t:
        _sec("MCP TOOL INVOCATION STATISTICS")
        tool_counts = mon.get("tool_counts", {})
        if tool_counts:
            fig_tools = go.Figure(go.Bar(
                x=list(tool_counts.values()),
                y=list(tool_counts.keys()),
                orientation="h",
                marker=dict(color=C_MUTED, opacity=0.85),
                text=list(tool_counts.values()),
                textposition="outside",
                textfont=dict(color=C_MUTED_TXT, size=11),
            ))
            fig_tools.update_layout(
                **CHART_LAYOUT, height=220,
                xaxis=dict(title="Calls", gridcolor="rgba(82,122,96,0.15)", tickfont=dict(color=C_MUTED_TXT)),
                yaxis=dict(tickfont=dict(color=C_TEXT)),
                margin=dict(l=0,r=50,t=4,b=0),
            )
            st.plotly_chart(fig_tools, use_container_width=True)
        else:
            st.markdown(f'<div style="color:{C_MUTED_TXT};font-size:0.85rem;">No tool calls recorded yet.</div>',
                        unsafe_allow_html=True)
        _sec("ALLOWLISTED TOOLS")
        for t in guard_st["tool_allowlist"]["tools"]:
            st.markdown(
                f'<span class="tool-chip">{t}</span>',
                unsafe_allow_html=True,
            )

    with tab_g:
        _sec("GUARDRAIL LAYER STATUS")
        for layer, info in guard_st.items():
            avail = info.get("available", False)
            note  = info.get("note") or info.get("checks") or info.get("tools","")
            if isinstance(note, list): note = ", ".join(note)
            st.markdown(
                f'<div class="gov-row">'
                f'<div><span class="gov-label">{layer.replace("_"," ").title()}</span>'
                f'<div style="font-size:0.72rem;color:{C_MUTED_TXT};margin-top:0.1rem;">{note}</div></div>'
                f'<span class="{"gov-active" if avail else "gov-inactive"}">{"Active" if avail else "Inactive"}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        _sec("RECENT GUARDRAIL EVENTS")
        g_events = mon.get("guardrail_events",[])
        if g_events:
            df_g = pd.DataFrame(g_events)[["created_at","agent","check_type","outcome","detail"]]
            df_g.columns = ["Time","Agent","Check","Outcome","Detail"]
            st.dataframe(df_g, use_container_width=True, hide_index=True)
        else:
            st.markdown(f'<div style="color:{C_MUTED_TXT};font-size:0.85rem;">No events yet.</div>',
                        unsafe_allow_html=True)
        c_ge, c_gb = st.columns(2)
        c_ge.metric("Total Events", mon["guardrail_total"])
        c_gb.metric("Blocked / Flagged", mon["guardrail_blocked"])

    with tab_o:
        _sec("AGENT LATENCY")
        recent_runs = mon.get("recent_runs",[])
        if recent_runs:
            df_obs = pd.DataFrame(recent_runs)
            if "latency_ms" in df_obs.columns:
                fig_lat = go.Figure()
                for status, color in [("success","#6ED49A"),("error","#EE7070")]:
                    sub = df_obs[df_obs["status"]==status]
                    if len(sub):
                        fig_lat.add_trace(go.Bar(
                            name=status.title(),
                            x=sub["agent_name"], y=sub["latency_ms"],
                            marker=dict(color=color, opacity=0.8),
                        ))
                fig_lat.update_layout(
                    **CHART_LAYOUT, barmode="stack", height=220,
                    yaxis=dict(title="ms", gridcolor="rgba(82,122,96,0.15)", tickfont=dict(color=C_MUTED_TXT)),
                    xaxis=dict(tickangle=-15, tickfont=dict(color=C_TEXT, size=10)),
                    margin=dict(l=0,r=0,t=4,b=0),
                )
                st.plotly_chart(fig_lat, use_container_width=True)

        prov_counts = mon.get("provider_counts",{})
        if prov_counts:
            _sec("CALLS BY PROVIDER")
            fig_pie = go.Figure(go.Pie(
                labels=list(prov_counts.keys()),
                values=list(prov_counts.values()),
                hole=0.5,
                marker=dict(colors=[C_MUTED, C_AMBER, "#527A99", C_SAGE_LT],
                            line=dict(color=C_FOREST, width=2)),
                textfont=dict(color=C_TEXT),
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(bgcolor="rgba(13,22,16,0.7)", bordercolor=C_BORDER,
                            font=dict(color=C_TEXT)),
                height=220, margin=dict(l=0,r=0,t=0,b=0),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        ca2, cb2 = st.columns(2)
        ca2.metric("Pending Alerts",  mon["pending_alerts"])
        cb2.metric("Approved Alerts", mon["approved_alerts"])

    with tab_e:
        _sec("MODEL SELECTION RATIONALE")
        st.info(
            "The table below shows configured model identities and measured latency. "
            "Accuracy benchmarks require controlled evaluation data and are not fabricated here."
        )
        eval_rows = []
        for pname, pinfo in prov_st.items():
            eval_rows.append({
                "Provider":          pname,
                "Model":             pinfo.get("model","—"),
                "Status":            "Live" if pinfo.get("configured") else "Demo",
                "Role":              {
                    "IBM watsonx.ai (Granite)": "Drought Detection & Risk Assessment",
                    "Google Gemini":            "Bilingual Farmer Advisory (EN + KN)",
                    "Groq":                     "Fast Alert & Relief Recommendations",
                    "OmniRoute Gateway":        "Optional unified gateway",
                }.get(pname,"—"),
                "Selection Rationale": {
                    "IBM watsonx.ai (Granite)": "IBM Granite optimized for structured data reasoning; enterprise-grade watsonx.ai reliability",
                    "Google Gemini":            "Multilingual generation, Kannada language support, vision capability for future NDVI imagery",
                    "Groq":                     "Fastest token generation for real-time alerts; ideal for high-urgency responses",
                    "OmniRoute Gateway":        "Cost-optimized routing across providers; unified API management",
                }.get(pname,"—"),
            })
        st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)

        _sec("MEASURED LATENCY BY PROVIDER (THIS SESSION)")
        rr = mon.get("recent_runs",[])
        if rr:
            by_p: dict[str,list] = {}
            for r in rr:
                p = r.get("provider","unknown")
                if r.get("latency_ms"):
                    by_p.setdefault(p,[]).append(r["latency_ms"])
            if by_p:
                for p, lats in by_p.items():
                    avg_lat = sum(lats)/len(lats)
                    st.markdown(
                        f'<div class="ind-row"><span class="ind-label">{p}</span>'
                        f'<span class="ind-value">{avg_lat:.0f} ms avg ({len(lats)} calls)</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(f'<div style="color:{C_MUTED_TXT};font-size:0.85rem;">Run agents to generate latency data.</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="color:{C_MUTED_TXT};font-size:0.85rem;">No runs yet this session.</div>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f'<div class="dg-footer">'
    f'DroughtGuard AI · Karnataka Drought Early Warning System · Hackathon Prototype · '
    f'<span style="color:{C_AMBER};">⚠ All data is SYNTHETIC DEMO — not official government observations</span> · '
    f'Powered by IBM Bob · IBM Granite · Google Gemini · Groq'
    f'</div>',
    unsafe_allow_html=True,
)
