"""
app.py — Digital Twin Enabled Intelligent Decision Framework
           Using Edge Artificial Intelligence

Main Streamlit application: login, navigation, and all dashboard pages.
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="EdgeTwin AI · Digital Twin Framework",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Auto-refresh ─────────────────────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH = True
except ImportError:
    AUTO_REFRESH = False

# ─── Module imports ────────────────────────────────────────────────────────────
from modules.database import (
    initialize_db, insert_sensor_reading, insert_anomaly_result,
    insert_alert, insert_decision, get_recent_readings,
    get_all_alerts, update_alert_status, get_anomaly_history,
    get_sensor_history, get_recent_decisions, get_alert_counts, get_db_stats,
)
from modules.simulator import SensorSimulator, THRESHOLDS
from modules.edge_ai import EdgeAI
from modules.digital_twin import DigitalTwin, STATUS_COLORS, STATUS_ICONS
from modules.decision_engine import DecisionEngine

# ══════════════════════════════════════════════════════════════════════════════
#  CSS — Industrial Dark Theme
# ══════════════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #0a0e1a !important; color: #e2e8f0 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d1117 0%,#111827 100%) !important;
    border-right: 1px solid #1e2a3a !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebarNav"] { display: none; }

/* ── Main area ── */
[data-testid="stMain"] > div { padding-top: 1rem; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg,#00d4ff22,#7c3aed22) !important;
    border: 1px solid #00d4ff55 !important;
    color: #00d4ff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all .2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#00d4ff44,#7c3aed44) !important;
    border-color: #00d4ff !important;
    box-shadow: 0 0 12px #00d4ff55 !important;
    transform: translateY(-1px) !important;
}

/* ── Radio / Select ── */
.stRadio > div { gap: 4px !important; }
.stSelectbox > div > div { background: #111827 !important; border-color: #1e2a3a !important; }
.stSlider .stSlider { color: #00d4ff !important; }

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg,#111827,#1a2332) !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="metric-container"] label { color: #94a3b8 !important; font-size:.8rem !important; }
[data-testid="metric-container"] [data-testid="metric-delta"] { font-size:.75rem !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #111827 !important;
    border: 1px solid #1e2a3a !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── DataFrames ── */
.stDataFrame { border: 1px solid #1e2a3a !important; border-radius: 8px !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #111827 !important; border-radius: 8px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #94a3b8 !important; border-radius: 6px !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg,#00d4ff22,#7c3aed22) !important; color: #00d4ff !important; border-bottom: 2px solid #00d4ff !important; }

/* ── Divider ── */
hr { border-color: #1e2a3a !important; }

/* ── Custom cards ── */
.kpi-card {
    background: linear-gradient(135deg,#111827,#1a2332);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: transform .2s ease, box-shadow .2s ease;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 6px 32px rgba(0,212,255,.12); }

.kpi-label { color: #94a3b8; font-size: .78rem; font-weight: 500; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 6px; }
.kpi-value { font-size: 2rem; font-weight: 700; line-height: 1; }
.kpi-sub   { color: #64748b; font-size: .74rem; margin-top: 6px; }

.badge-normal   { display:inline-block; padding:4px 12px; border-radius:20px; font-size:.78rem; font-weight:600; background:rgba(16,185,129,.15); color:#10b981; border:1px solid #10b98144; }
.badge-warning  { display:inline-block; padding:4px 12px; border-radius:20px; font-size:.78rem; font-weight:600; background:rgba(245,158,11,.15); color:#f59e0b; border:1px solid #f59e0b44; }
.badge-critical { display:inline-block; padding:4px 12px; border-radius:20px; font-size:.78rem; font-weight:600; background:rgba(239,68,68,.15);  color:#ef4444; border:1px solid #ef444444; }
.badge-info     { display:inline-block; padding:4px 12px; border-radius:20px; font-size:.78rem; font-weight:600; background:rgba(0,212,255,.15);   color:#00d4ff; border:1px solid #00d4ff44; }
.badge-maint    { display:inline-block; padding:4px 12px; border-radius:20px; font-size:.78rem; font-weight:600; background:rgba(124,58,237,.15);  color:#7c3aed; border:1px solid #7c3aed44; }

.section-card {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
}
.section-title { font-size: 1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 4px; }
.section-sub   { font-size: .78rem; color: #64748b; }

.alert-item {
    padding: 12px 16px;
    border-radius: 10px;
    margin-bottom: 8px;
    border-left: 4px solid;
}
.alert-critical { background: rgba(239,68,68,.08);  border-color: #ef4444; }
.alert-warning  { background: rgba(245,158,11,.08); border-color: #f59e0b; }
.alert-info     { background: rgba(0,212,255,.08);  border-color: #00d4ff; }

.decision-flow {
    display: flex; flex-direction: column; gap: 0;
    align-items: center;
}
.flow-box {
    background: linear-gradient(135deg,#111827,#1a2332);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 14px 28px;
    text-align: center;
    width: 100%;
    position: relative;
}
.flow-arrow { text-align: center; color: #00d4ff; font-size: 1.4rem; line-height: 1.2; }

.twin-param {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 0; border-bottom: 1px solid #1e2a3a;
}
.twin-param:last-child { border-bottom: none; }
.twin-label { color: #94a3b8; font-size: .85rem; }
.twin-val   { font-size: 1.1rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

.rule-item {
    padding: 8px 14px; border-radius: 8px; margin-bottom: 6px;
    font-size: .83rem;
}
.rule-critical { background: rgba(239,68,68,.1);  color: #ef4444;  border-left: 3px solid #ef4444; }
.rule-warning  { background: rgba(245,158,11,.1); color: #f59e0b;  border-left: 3px solid #f59e0b; }
.rule-info     { background: rgba(0,212,255,.1);  color: #00d4ff;  border-left: 3px solid #00d4ff; }

/* Login page */
.login-container {
    max-width: 420px; margin: 60px auto;
    background: linear-gradient(135deg,#111827,#1a2332);
    border: 1px solid #1e3a5f; border-radius: 20px;
    padding: 48px 40px; box-shadow: 0 20px 80px rgba(0,0,0,.5);
}
.login-title { font-size: 1.6rem; font-weight: 700; color: #e2e8f0; margin-bottom: 6px; }
.login-sub   { color: #64748b; font-size: .85rem; margin-bottom: 32px; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00d4ff44; }
</style>
"""

# ══════════════════════════════════════════════════════════════════════════════
#  Users / Roles
# ══════════════════════════════════════════════════════════════════════════════
USERS = {
    "admin":       {"password": "admin123",  "role": "Admin",          "pages": "all"},
    "operator":    {"password": "op123",     "role": "Operator",       "pages": ["Overview","Live Monitoring","Digital Twin","Alerts","Architecture"]},
    "maintenance": {"password": "maint123",  "role": "Maintenance",    "pages": ["Overview","Digital Twin","Edge AI","Predictive Maintenance","Alerts","Historical Analytics"]},
    "manager":     {"password": "mgr123",    "role": "Decision Maker", "pages": ["Overview","Decision Engine","Historical Analytics","Architecture","Edge vs Cloud"]},
}

ALL_PAGES = [
    "Overview", "Live Monitoring", "Digital Twin", "Edge AI",
    "Predictive Maintenance", "Decision Engine", "Scenario Simulation",
    "Historical Analytics", "Alerts", "Architecture", "Edge vs Cloud",
]

# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0", family="Inter"),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="#1e2a3a", zerolinecolor="#1e2a3a"),
    yaxis=dict(gridcolor="#1e2a3a", zerolinecolor="#1e2a3a"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2a3a"),
)

SENSOR_COLORS = {
    "temperature": "#ef4444",
    "vibration":   "#f59e0b", 
    "humidity":    "#00d4ff",
    "energy":      "#10b981",
    "rpm":         "#7c3aed",
    "pressure":    "#ec4899",
}

SENSOR_UNITS = {
    "temperature": "°C", "vibration": "mm/s", "humidity": "%RH",
    "energy": "kWh", "rpm": "RPM", "pressure": "bar",
}

def sev_badge(sev: str) -> str:
    cls = {"CRITICAL": "badge-critical", "WARNING": "badge-warning",
           "INFO": "badge-info", "NORMAL": "badge-normal"}.get(sev, "badge-info")
    return f'<span class="{cls}">{sev}</span>'

def health_color(h: float) -> str:
    if h >= 70:  return "#10b981"
    if h >= 40:  return "#f59e0b"
    if h >= 20:  return "#ef4444"
    return "#7c3aed"

def gauge_fig(value, title, min_val=0, max_val=100, unit="", color="#00d4ff"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 13, "color": "#94a3b8"}},
        number={"suffix": unit, "font": {"size": 22, "color": color}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickcolor": "#1e2a3a"},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "#1a2332",
            "bordercolor": "#1e2a3a",
            "steps": [
                {"range": [min_val, max_val * 0.5], "color": "#0a0e1a"},
                {"range": [max_val * 0.5, max_val * 0.75], "color": "#1e2a3a"},
                {"range": [max_val * 0.75, max_val], "color": "#251a0a"},
            ],
            "threshold": {
                "line": {"color": "#ef4444", "width": 3},
                "thickness": 0.8,
                "value": max_val * 0.85,
            },
        },
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=200)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  Session State Init
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    initialize_db()
    if "logged_in"       not in st.session_state: st.session_state.logged_in     = False
    if "username"        not in st.session_state: st.session_state.username       = ""
    if "role"            not in st.session_state: st.session_state.role           = ""
    if "page"            not in st.session_state: st.session_state.page           = "Overview"
    if "sim_running"     not in st.session_state: st.session_state.sim_running    = True
    if "demo_mode"       not in st.session_state: st.session_state.demo_mode      = False
    if "scenario"        not in st.session_state: st.session_state.scenario       = "normal"
    if "readings"        not in st.session_state: st.session_state.readings       = []
    if "ai_results"      not in st.session_state: st.session_state.ai_results     = []
    if "decisions"       not in st.session_state: st.session_state.decisions      = []
    if "session_alerts"  not in st.session_state: st.session_state.session_alerts = []
    if "twin_state"      not in st.session_state: st.session_state.twin_state     = {}
    if "last_reading"    not in st.session_state: st.session_state.last_reading   = {}
    if "last_ai"         not in st.session_state: st.session_state.last_ai        = {}
    if "last_decision"   not in st.session_state: st.session_state.last_decision  = {}
    if "edge_latencies"  not in st.session_state: st.session_state.edge_latencies = []
    if "manual_overrides"not in st.session_state: st.session_state.manual_overrides = {}

    # Persistent objects (cached across reruns)
    if "simulator"       not in st.session_state:
        st.session_state.simulator      = SensorSimulator()
    if "edge_ai"         not in st.session_state:
        st.session_state.edge_ai        = EdgeAI()
    if "digital_twin"    not in st.session_state:
        st.session_state.digital_twin   = DigitalTwin()
    if "decision_engine" not in st.session_state:
        st.session_state.decision_engine= DecisionEngine()

# ══════════════════════════════════════════════════════════════════════════════
#  Simulation step
# ══════════════════════════════════════════════════════════════════════════════
def run_sim_step():
    if not st.session_state.sim_running:
        return

    sim = st.session_state.simulator
    ai  = st.session_state.edge_ai
    twin= st.session_state.digital_twin
    de  = st.session_state.decision_engine

    # Sync scenario with session state
    if sim.scenario != st.session_state.scenario:
        sim.set_scenario(st.session_state.scenario)

    # Generate reading
    if st.session_state.demo_mode:
        reading = sim.generate_demo_reading()
        st.session_state.scenario = sim.scenario
    else:
        reading = sim.generate_reading(
            manual_overrides=st.session_state.manual_overrides or None
        )

    # Edge AI inference
    ai_result = ai.predict(reading)

    # Digital Twin update
    twin.update(reading, ai_result)
    st.session_state.twin_state = twin.get_state()

    # Decision Engine
    decision = de.evaluate(reading, ai_result, twin.get_state())

    # Persist to DB
    rid = insert_sensor_reading(reading)
    insert_anomaly_result(rid, ai_result)
    insert_decision({
        "timestamp": decision["timestamp"],
        "action":    decision["action"],
        "reason":    decision["reason"],
        "confidence":decision["confidence"],
        "severity":  decision["severity"],
    })

    # Create alert for WARNING / CRITICAL
    if decision["severity"] in ("WARNING", "CRITICAL"):
        sensors_triggered = [r.split(":")[0].strip() for r in decision["rules_triggered"][:2]]
        alert = {
            "timestamp":     reading["timestamp"],
            "severity":      decision["severity"],
            "sensor":        ", ".join(sensors_triggered),
            "condition":     decision["reason"][:300],
            "ai_result":     f"Anomaly:{ai_result['is_anomaly']}  FP:{ai_result['failure_probability']*100:.1f}%",
            "recommendation":decision["action"],
            "status":        "open",
        }
        insert_alert(alert)
        st.session_state.session_alerts.insert(0, alert)
        if len(st.session_state.session_alerts) > 60:
            st.session_state.session_alerts.pop()

    # Track edge latency
    st.session_state.edge_latencies.append(ai_result.get("inference_ms", 0))
    if len(st.session_state.edge_latencies) > 200:
        st.session_state.edge_latencies.pop(0)

    # Append to rolling buffers
    for buf, item in [
        ("readings",    reading),
        ("ai_results",  ai_result),
        ("decisions",   decision),
    ]:
        st.session_state[buf].append(item)
        if len(st.session_state[buf]) > 150:
            st.session_state[buf].pop(0)

    st.session_state.last_reading  = reading
    st.session_state.last_ai       = ai_result
    st.session_state.last_decision = decision

# ══════════════════════════════════════════════════════════════════════════════
#  Login Page
# ══════════════════════════════════════════════════════════════════════════════
def show_login():
    st.markdown(CSS, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div class="login-container">
          <div style="text-align:center;margin-bottom:28px;">
            <div style="font-size:3rem;margin-bottom:12px;">🏭</div>
            <div class="login-title">EdgeTwin AI</div>
            <div class="login-sub">Digital Twin · Edge AI · Intelligent Decisions</div>
          </div>
        """, unsafe_allow_html=True)

        username = st.text_input("👤 Username", placeholder="e.g. admin")
        password = st.text_input("🔒 Password", type="password", placeholder="Password")

        if st.button("Sign In →", use_container_width=True):
            user = USERS.get(username)
            if user and user["password"] == password:
                st.session_state.logged_in = True
                st.session_state.username  = username
                st.session_state.role      = user["role"]
                st.rerun()
            else:
                st.error("Invalid credentials. Try admin / admin123")

        st.markdown("""
        <div style="margin-top:24px;padding-top:20px;border-top:1px solid #1e2a3a;">
          <div style="color:#64748b;font-size:.78rem;margin-bottom:10px;">Demo Credentials:</div>
          <table style="width:100%;font-size:.78rem;color:#94a3b8;border-collapse:collapse;">
            <tr><td style="padding:4px 0"><b style="color:#e2e8f0">admin</b></td><td>admin123</td><td><span class="badge-info">Admin</span></td></tr>
            <tr><td style="padding:4px 0"><b style="color:#e2e8f0">operator</b></td><td>op123</td><td><span class="badge-normal">Operator</span></td></tr>
            <tr><td style="padding:4px 0"><b style="color:#e2e8f0">maintenance</b></td><td>maint123</td><td><span class="badge-warning">Maintenance</span></td></tr>
            <tr><td style="padding:4px 0"><b style="color:#e2e8f0">manager</b></td><td>mgr123</td><td><span class="badge-maint">Manager</span></td></tr>
          </table>
        </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════
def show_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:20px 0 16px;border-bottom:1px solid #1e2a3a;margin-bottom:16px;">
          <div style="font-size:1.2rem;font-weight:700;color:#00d4ff;">🏭 EdgeTwin AI</div>
          <div style="font-size:.72rem;color:#64748b;margin-top:2px;">Digital Twin Framework v1.0</div>
        </div>
        """, unsafe_allow_html=True)

        # User info
        role_color = {"Admin":"#00d4ff","Operator":"#10b981","Maintenance":"#f59e0b","Decision Maker":"#7c3aed"}.get(st.session_state.role,"#00d4ff")
        st.markdown(f"""
        <div style="background:#1a2332;border:1px solid #1e2a3a;border-radius:10px;padding:12px 16px;margin-bottom:16px;">
          <div style="font-size:.72rem;color:#64748b;">Logged in as</div>
          <div style="font-weight:600;color:#e2e8f0;">{st.session_state.username}</div>
          <div style="font-size:.74rem;color:{role_color};margin-top:2px;">● {st.session_state.role}</div>
        </div>
        """, unsafe_allow_html=True)

        # Determine visible pages
        user_pages = USERS.get(st.session_state.username, {}).get("pages", [])
        visible = ALL_PAGES if user_pages == "all" else user_pages

        PAGE_ICONS = {
            "Overview": "⬡", "Live Monitoring": "📡", "Digital Twin": "🔷",
            "Edge AI": "🧠", "Predictive Maintenance": "🔧", "Decision Engine": "⚡",
            "Scenario Simulation": "🎮", "Historical Analytics": "📊",
            "Alerts": "🚨", "Architecture": "🗺", "Edge vs Cloud": "☁",
        }

        st.markdown('<div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;padding-left:4px;">Navigation</div>', unsafe_allow_html=True)
        for page in ALL_PAGES:
            if page not in visible:
                continue
            icon = PAGE_ICONS.get(page, "○")
            is_active = st.session_state.page == page
            style = "color:#00d4ff;background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.2);" if is_active else "color:#94a3b8;background:transparent;border:1px solid transparent;"
            if st.button(f"{icon}  {page}", key=f"nav_{page}", use_container_width=True):
                st.session_state.page = page
                st.rerun()

        st.markdown("---")

        # Simulation controls
        st.markdown('<div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px;">Simulation</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.sim_running:
                if st.button("⏸ Pause", use_container_width=True):
                    st.session_state.sim_running = False
                    st.rerun()
            else:
                if st.button("▶ Start", use_container_width=True):
                    st.session_state.sim_running = True
                    st.rerun()
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.simulator.reset()
                st.session_state.digital_twin.reset()
                st.session_state.readings.clear()
                st.session_state.ai_results.clear()
                st.session_state.decisions.clear()
                st.session_state.edge_latencies.clear()
                st.rerun()

        demo_mode = st.toggle("🎬 Demo Mode", value=st.session_state.demo_mode)
        if demo_mode != st.session_state.demo_mode:
            st.session_state.demo_mode = demo_mode
            if demo_mode:
                st.session_state.simulator.demo_step = 0

        # Status indicator
        sim_color = "#10b981" if st.session_state.sim_running else "#ef4444"
        sim_text  = "RUNNING" if st.session_state.sim_running else "PAUSED"
        st.markdown(f"""
        <div style="margin-top:12px;padding:10px 14px;background:#0d1117;border-radius:8px;border:1px solid #1e2a3a;text-align:center;">
          <span style="color:{sim_color};font-size:.7rem;font-weight:600;">● {sim_text}</span>
          &nbsp;·&nbsp;
          <span style="color:#64748b;font-size:.7rem;">{len(st.session_state.readings)} samples</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
def page_overview():
    st.markdown("## ⬡ Overview — System Dashboard")
    twin   = st.session_state.twin_state
    ai     = st.session_state.last_ai
    dec    = st.session_state.last_decision
    reading= st.session_state.last_reading

    if not twin:
        st.info("⏳ Simulation starting... please wait one moment.")
        return

    status = twin.get("status", "NORMAL")
    scol   = STATUS_COLORS.get(status, "#00d4ff")
    health = twin.get("health_score", 100)
    fp     = ai.get("failure_probability", 0) * 100

    # ── KPI Row ──
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Machine Status</div>
          <div class="kpi-value" style="font-size:1.1rem;color:{scol};">{STATUS_ICONS.get(status,'')} {status}</div>
          <div class="kpi-sub">Scenario: {reading.get('scenario','normal').replace('_',' ').title()}</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        hc = health_color(health)
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Machine Health</div>
          <div class="kpi-value" style="color:{hc};">{health:.1f}%</div>
          <div class="kpi-sub">Digital Twin sync ✓</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        tc = "#ef4444" if twin.get("temperature",70) > 85 else "#00d4ff"
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Temperature</div>
          <div class="kpi-value" style="color:{tc};">{twin.get('temperature',0):.1f}°C</div>
          <div class="kpi-sub">Threshold: 85°C warn / 95°C crit</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        vc = "#f59e0b" if twin.get("vibration",2) > 4 else "#10b981"
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Vibration</div>
          <div class="kpi-value" style="color:{vc};">{twin.get('vibration',0):.2f} mm/s</div>
          <div class="kpi-sub">Threshold: 4.0 warn / 6.0 crit</div>
        </div>""", unsafe_allow_html=True)
    with k5:
        ec = "#f59e0b" if twin.get("energy",20) > 35 else "#10b981"
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Energy</div>
          <div class="kpi-value" style="color:{ec};">{twin.get('energy',0):.1f} kWh</div>
          <div class="kpi-sub">Threshold: 35 warn / 45 crit</div>
        </div>""", unsafe_allow_html=True)
    with k6:
        fc = "#ef4444" if fp > 65 else ("#f59e0b" if fp > 35 else "#10b981")
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Failure Probability</div>
          <div class="kpi-value" style="color:{fc};">{fp:.1f}%</div>
          <div class="kpi-sub">Random Forest prediction</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts + Decision ──
    col_chart, col_dec = st.columns([2, 1])

    with col_chart:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Live Sensor Stream</div>', unsafe_allow_html=True)
        readings = st.session_state.readings
        if len(readings) >= 3:
            df = pd.DataFrame(readings[-60:])
            fig = make_subplots(rows=2, cols=3, subplot_titles=list(SENSOR_UNITS.keys()))
            positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
            for idx, (sensor, (row,col)) in enumerate(zip(SENSOR_UNITS.keys(), positions)):
                if sensor in df.columns:
                    fig.add_trace(go.Scatter(
                        y=df[sensor].values,
                        mode="lines",
                        line=dict(color=SENSOR_COLORS[sensor], width=2),
                        fill="tozeroy",
                        fillcolor=SENSOR_COLORS[sensor].replace("#", "rgba(").rstrip(")") + ",0.07)" if sensor else None,
                        name=sensor,
                        showlegend=False,
                    ), row=row, col=col)
                    # Threshold line
                    thr = THRESHOLDS.get(sensor, {}).get("warning")
                    if thr:
                        fig.add_hline(y=thr, line_dash="dash", line_color="#f59e0b44", row=row, col=col)

            fig.update_layout(**PLOTLY_LAYOUT, height=340)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Collecting data...")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_dec:
        # AI Recommendation
        sev = dec.get("severity", "INFO")
        sev_css = {"CRITICAL":"badge-critical","WARNING":"badge-warning","INFO":"badge-info"}.get(sev,"badge-info")
        action = dec.get("action","Continue Normal Operation")
        reason = dec.get("reason","")
        conf   = dec.get("confidence",0.99)*100

        st.markdown(f"""
        <div class="section-card" style="height:100%;">
          <div class="section-title">⚡ AI Recommendation</div>
          <div style="margin:16px 0;">
            <span class="{sev_css}">{sev}</span>
            <span style="color:#64748b;font-size:.74rem;margin-left:8px;">Confidence: {conf:.0f}%</span>
          </div>
          <div style="font-size:1rem;font-weight:600;color:#e2e8f0;margin-bottom:12px;">{action}</div>
          <div style="font-size:.82rem;color:#94a3b8;line-height:1.6;">{reason}</div>
          <hr>
          <div style="font-size:.78rem;color:#64748b;">
            <div>Anomaly Score: <b style="color:#f59e0b;">{ai.get('anomaly_score',0):.3f}</b></div>
            <div style="margin-top:4px;">Health Score: <b style="color:{health_color(health)};">{health:.1f}%</b></div>
            <div style="margin-top:4px;">Inference: <b style="color:#00d4ff;">{ai.get('inference_ms',0):.1f} ms</b></div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Recent alerts ──
    alerts_df = get_all_alerts(status="open")
    if not alerts_df.empty:
        st.markdown("#### 🚨 Open Alerts")
        for _, row in alerts_df.head(5).iterrows():
            cls = "alert-critical" if row.get("severity")=="CRITICAL" else ("alert-warning" if row.get("severity")=="WARNING" else "alert-info")
            st.markdown(f"""
            <div class="alert-item {cls}">
              <b style="color:#e2e8f0;">{row.get('severity','')} — {row.get('sensor','')}</b>
              <span style="float:right;color:#64748b;font-size:.75rem;">{str(row.get('timestamp',''))[:19]}</span><br>
              <span style="font-size:.83rem;color:#94a3b8;">{str(row.get('condition_desc',''))[:120]}</span>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — Live Monitoring
# ══════════════════════════════════════════════════════════════════════════════
def page_live_monitoring():
    st.markdown("## 📡 Live Monitoring")

    col_ctrl, col_scene = st.columns([2, 1])
    with col_ctrl:
        st.markdown("**Simulation Controls**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("▶ Start", use_container_width=True, disabled=st.session_state.sim_running):
                st.session_state.sim_running = True; st.rerun()
        with c2:
            if st.button("⏸ Pause", use_container_width=True, disabled=not st.session_state.sim_running):
                st.session_state.sim_running = False; st.rerun()
        with c3:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.readings.clear(); st.rerun()
        with c4:
            demo = st.toggle("Demo Mode", value=st.session_state.demo_mode)
            if demo != st.session_state.demo_mode:
                st.session_state.demo_mode = demo; st.rerun()

    with col_scene:
        scenario = st.selectbox(
            "Scenario",
            ["normal","overheating","excessive_vibration","high_energy","multiple_sensor_failure","critical_failure"],
            index=["normal","overheating","excessive_vibration","high_energy","multiple_sensor_failure","critical_failure"].index(st.session_state.scenario),
            format_func=lambda x: x.replace("_"," ").title()
        )
        if scenario != st.session_state.scenario:
            st.session_state.scenario = scenario
            st.session_state.simulator.set_scenario(scenario)

    # Manual sliders
    with st.expander("🎛 Manual Override Sliders"):
        sc1, sc2, sc3 = st.columns(3)
        overrides = {}
        with sc1:
            overrides["temperature"] = st.slider("Temperature (°C)", 50.0, 130.0, float(st.session_state.last_reading.get("temperature",70)))
            overrides["vibration"]   = st.slider("Vibration (mm/s)", 0.1, 16.0,  float(st.session_state.last_reading.get("vibration",2.0)))
        with sc2:
            overrides["humidity"]    = st.slider("Humidity (%RH)",   20.0, 100.0, float(st.session_state.last_reading.get("humidity",45.0)))
            overrides["energy"]      = st.slider("Energy (kWh)",      5.0,  80.0, float(st.session_state.last_reading.get("energy",20.0)))
        with sc3:
            overrides["rpm"]         = st.slider("RPM",              1000, 2500,  int(st.session_state.last_reading.get("rpm",1500)))
            overrides["pressure"]    = st.slider("Pressure (bar)",    1.0,  14.0, float(st.session_state.last_reading.get("pressure",4.5)))
        if st.button("Apply Overrides"):
            st.session_state.manual_overrides = overrides; st.rerun()
        if st.button("Clear Overrides"):
            st.session_state.manual_overrides = {}; st.rerun()

    # Live charts
    readings = st.session_state.readings
    if len(readings) < 3:
        st.info("⏳ Collecting data... simulation must be running.")
        return

    df = pd.DataFrame(readings[-80:])

    tab_names = ["Temperature", "Vibration", "Humidity", "Energy", "RPM", "Pressure", "All Sensors"]
    tabs = st.tabs(tab_names)

    for i, (sensor, unit) in enumerate(SENSOR_UNITS.items()):
        with tabs[i]:
            if sensor in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=df[sensor].values, mode="lines",
                    line=dict(color=SENSOR_COLORS[sensor], width=2.5),
                    fill="tozeroy",
                    fillcolor=SENSOR_COLORS[sensor] + "18",
                    name=sensor,
                ))
                w_thr = THRESHOLDS.get(sensor, {}).get("warning")
                c_thr = THRESHOLDS.get(sensor, {}).get("critical")
                if w_thr: fig.add_hline(y=w_thr, line_dash="dash", line_color="#f59e0b", annotation_text="Warning", annotation_font_size=10)
                if c_thr: fig.add_hline(y=c_thr, line_dash="dash", line_color="#ef4444", annotation_text="Critical", annotation_font_size=10)
                fig.update_layout(**PLOTLY_LAYOUT, height=320, yaxis_title=f"{sensor.title()} ({unit})", xaxis_title="Time (steps)")
                st.plotly_chart(fig, use_container_width=True)
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Current", f"{df[sensor].iloc[-1]:.2f} {unit}")
                mc2.metric("Average", f"{df[sensor].mean():.2f} {unit}")
                mc3.metric("Max",     f"{df[sensor].max():.2f} {unit}")

    with tabs[-1]:
        fig = go.Figure()
        for sensor, unit in SENSOR_UNITS.items():
            if sensor in df.columns:
                norm_vals = df[sensor] / df[sensor].max()
                fig.add_trace(go.Scatter(
                    y=norm_vals, mode="lines",
                    line=dict(color=SENSOR_COLORS[sensor], width=2),
                    name=f"{sensor} ({unit})",
                ))
        fig.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_title="Normalized Value", xaxis_title="Time (steps)")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — Digital Twin
# ══════════════════════════════════════════════════════════════════════════════
def page_digital_twin():
    st.markdown("## 🔷 Digital Twin — Virtual Machine Replica")

    twin = st.session_state.twin_state
    if not twin:
        st.info("⏳ Waiting for first data..."); return

    status = twin.get("status","NORMAL")
    scol   = STATUS_COLORS.get(status,"#00d4ff")
    health = twin.get("health_score",100)
    sync   = str(twin.get("last_sync",""))[:19]

    # Sync banner
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(0,212,255,.08),rgba(124,58,237,.08));
         border:1px solid rgba(0,212,255,.3);border-radius:12px;padding:14px 24px;
         display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;">
      <div>
        <span style="color:#00d4ff;font-weight:600;font-size:1.05rem;">🔄 Digital Twin Synchronization</span>
        <span style="background:rgba(16,185,129,.2);color:#10b981;border-radius:20px;padding:3px 12px;font-size:.75rem;font-weight:600;margin-left:12px;">LIVE</span>
      </div>
      <div style="text-align:right;font-size:.8rem;color:#64748b;">
        Last sync: <b style="color:#e2e8f0;">{sync}</b> &nbsp;·&nbsp;
        Sync count: <b style="color:#e2e8f0;">{twin.get('sync_count',0):,}</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_machine, col_stats = st.columns([3, 2])

    with col_machine:
        # Machine SVG visualization
        temp_c = "#ef4444" if twin.get("temperature",70)>85 else ("#f59e0b" if twin.get("temperature",70)>75 else "#10b981")
        vib_c  = "#ef4444" if twin.get("vibration",2)>6   else ("#f59e0b" if twin.get("vibration",2)>4   else "#10b981")
        pulse_anim = f"animation: pulse 1.5s infinite;" if status != "NORMAL" else ""

        machine_svg = f"""
        <style>
          @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:.5; }} }}
          @keyframes spin  {{ from {{ transform:rotate(0deg); }} to {{ transform:rotate(360deg); }} }}
          @keyframes flow  {{ 0% {{ stroke-dashoffset:100; }} 100% {{ stroke-dashoffset:0; }} }}
        </style>
        <div style="background:linear-gradient(135deg,#0d1117,#111827);border:1px solid {scol}44;
             border-radius:16px;padding:28px;position:relative;overflow:hidden;">
          <!-- Glow effect -->
          <div style="position:absolute;top:-40px;left:50%;transform:translateX(-50%);
               width:300px;height:200px;background:radial-gradient({scol}22,transparent 70%);
               pointer-events:none;"></div>

          <!-- Status header -->
          <div style="text-align:center;margin-bottom:20px;">
            <div style="font-size:.75rem;color:#64748b;text-transform:uppercase;letter-spacing:.1em;">Machine Status</div>
            <div style="font-size:1.8rem;font-weight:700;color:{scol};{pulse_anim}">{STATUS_ICONS.get(status,'')} {status}</div>
          </div>

          <svg viewBox="0 0 500 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-height:280px;">
            <!-- Machine body -->
            <rect x="120" y="80" width="260" height="150" rx="12" fill="#1a2332" stroke="{scol}" stroke-width="2"/>

            <!-- Cooling fins -->
            <rect x="130" y="90"  width="12" height="130" rx="3" fill="#0d1117" stroke="#1e3a5f" stroke-width="1"/>
            <rect x="148" y="90"  width="12" height="130" rx="3" fill="#0d1117" stroke="#1e3a5f" stroke-width="1"/>
            <rect x="166" y="90"  width="12" height="130" rx="3" fill="#0d1117" stroke="#1e3a5f" stroke-width="1"/>

            <!-- Motor body -->
            <circle cx="300" cy="155" r="60" fill="#111827" stroke="{scol}" stroke-width="2"/>
            <circle cx="300" cy="155" r="40" fill="#0d1117" stroke="#1e3a5f" stroke-width="1.5"/>
            <!-- Motor spindle (spinning) -->
            <g transform-origin="300 155" style="animation: spin {max(0.3,3-twin.get('rpm',1500)/1000):.1f}s linear infinite;">
              <line x1="300" y1="115" x2="300" y2="195" stroke="{scol}" stroke-width="3" opacity=".7"/>
              <line x1="260" y1="155" x2="340" y2="155" stroke="{scol}" stroke-width="3" opacity=".7"/>
              <circle cx="300" cy="155" r="8" fill="{scol}"/>
            </g>
            <circle cx="300" cy="155" r="4" fill="#e2e8f0"/>

            <!-- Temperature gauge -->
            <circle cx="160" cy="140" r="20" fill="#0d1117" stroke="{temp_c}" stroke-width="2"/>
            <text x="160" y="136" text-anchor="middle" fill="{temp_c}" font-size="7" font-family="Inter">TEMP</text>
            <text x="160" y="146" text-anchor="middle" fill="{temp_c}" font-size="9" font-weight="bold" font-family="Inter">{twin.get('temperature',70):.0f}</text>
            <text x="160" y="155" text-anchor="middle" fill="{temp_c}" font-size="6" font-family="Inter">°C</text>

            <!-- Shaft -->
            <rect x="360" y="148" width="60" height="14" rx="4" fill="#1a2332" stroke="#00d4ff" stroke-width="1.5"/>
            <text x="390" y="157" text-anchor="middle" fill="#00d4ff" font-size="7" font-family="Inter">SHAFT</text>

            <!-- Data flow lines (physical → digital) -->
            <path d="M80,155 Q100,155 120,155" stroke="#00d4ff" stroke-width="1.5" fill="none" stroke-dasharray="4,3"
                  style="animation:flow 1.5s linear infinite;"/>
            <text x="30" y="145" fill="#64748b" font-size="8" font-family="Inter">PHYSICAL</text>
            <text x="30" y="158" fill="#64748b" font-size="8" font-family="Inter">SENSOR</text>
            <circle cx="65" cy="155" r="6" fill="#00d4ff22" stroke="#00d4ff" stroke-width="1.5"/>

            <!-- Health bar -->
            <rect x="140" y="248" width="220" height="12" rx="6" fill="#0d1117" stroke="#1e2a3a" stroke-width="1"/>
            <rect x="140" y="248" width="{min(220, 220*health/100):.0f}" height="12" rx="6" fill="{health_color(health)}"/>
            <text x="255" y="278" text-anchor="middle" fill="#e2e8f0" font-size="9" font-family="Inter">Health Score: {health:.1f}%</text>

            <!-- Label -->
            <text x="255" y="76" text-anchor="middle" fill="{scol}" font-size="10" font-family="Inter" font-weight="bold">INDUSTRIAL MACHINE — DIGITAL TWIN</text>

            <!-- Vibration indicator -->
            <text x="390" y="130" fill="{vib_c}" font-size="8" font-family="Inter">VIB: {twin.get('vibration',2):.2f} mm/s</text>
          </svg>
        </div>
        """
        st.markdown(machine_svg, unsafe_allow_html=True)

    with col_stats:
        # Live parameters
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Live Parameters</div>', unsafe_allow_html=True)
        params = [
            ("🌡 Temperature", f"{twin.get('temperature',0):.1f}", "°C",  SENSOR_COLORS["temperature"]),
            ("〰 Vibration",   f"{twin.get('vibration',0):.2f}",   "mm/s",SENSOR_COLORS["vibration"]),
            ("💧 Humidity",    f"{twin.get('humidity',0):.1f}",     "%RH", SENSOR_COLORS["humidity"]),
            ("⚡ Energy",      f"{twin.get('energy',0):.1f}",       "kWh", SENSOR_COLORS["energy"]),
            ("🔄 RPM",         f"{twin.get('rpm',0):.0f}",          "RPM", SENSOR_COLORS["rpm"]),
            ("🔵 Pressure",    f"{twin.get('pressure',0):.1f}",     "bar", SENSOR_COLORS["pressure"]),
        ]
        for label, val, unit, color in params:
            st.markdown(f"""
            <div class="twin-param">
              <span class="twin-label">{label}</span>
              <span class="twin-val" style="color:{color};">{val} <span style="font-size:.7rem;color:#64748b;">{unit}</span></span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Prediction panel
        fp  = twin.get("failure_probability",0)*100
        anom= twin.get("anomaly_score",0)
        st.markdown(f"""
        <div class="section-card" style="margin-top:12px;">
          <div class="section-title">🧠 AI Predictions</div>
          <div style="margin-top:14px;">
            <div style="color:#64748b;font-size:.78rem;margin-bottom:4px;">Failure Probability</div>
            <div style="background:#0d1117;border-radius:8px;height:10px;overflow:hidden;">
              <div style="width:{fp:.1f}%;background:{health_color(100-fp)};height:100%;transition:width .5s;"></div>
            </div>
            <div style="font-weight:600;color:{health_color(100-fp)};margin-top:4px;">{fp:.1f}%</div>
          </div>
          <div style="margin-top:14px;">
            <div style="color:#64748b;font-size:.78rem;margin-bottom:4px;">Anomaly Score</div>
            <div style="background:#0d1117;border-radius:8px;height:10px;overflow:hidden;">
              <div style="width:{anom*100:.1f}%;background:#f59e0b;height:100%;transition:width .5s;"></div>
            </div>
            <div style="font-weight:600;color:#f59e0b;margin-top:4px;">{anom:.3f}</div>
          </div>
          <div style="margin-top:14px;">
            <div style="color:#64748b;font-size:.78rem;margin-bottom:4px;">Machine Health</div>
            <div style="background:#0d1117;border-radius:8px;height:10px;overflow:hidden;">
              <div style="width:{health:.1f}%;background:{health_color(health)};height:100%;transition:width .5s;"></div>
            </div>
            <div style="font-weight:600;color:{health_color(health)};margin-top:4px;">{health:.1f}%</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # Health trend chart
    dt_obj = st.session_state.digital_twin
    if len(dt_obj.history) >= 5:
        hist = dt_obj.history[-80:]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=[h["health_score"] for h in hist],
            mode="lines", name="Health Score",
            line=dict(color="#10b981", width=2.5),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.08)",
        ))
        fig.add_trace(go.Scatter(
            y=[h["failure_probability"]*100 for h in hist],
            mode="lines", name="Failure Prob %",
            line=dict(color="#ef4444", width=2, dash="dot"),
        ))
        fig.add_hline(y=70, line_dash="dash", line_color="#f59e0b55", annotation_text="Warning threshold")
        fig.update_layout(**PLOTLY_LAYOUT, height=250, yaxis_title="Score (%)", title="Health Score & Failure Probability — Digital Twin History")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — Edge AI
# ══════════════════════════════════════════════════════════════════════════════
def page_edge_ai():
    st.markdown("## 🧠 Edge AI Module — On-Device Inference")

    edge = st.session_state.edge_ai
    info = edge.get_model_info()
    ai   = st.session_state.last_ai

    # Model info
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="section-card" style="border-color:#00d4ff44;">
          <div class="section-title">Anomaly Detection Model</div>
          <div style="margin-top:12px;font-size:.85rem;color:#94a3b8;">
            <b style="color:#00d4ff;">Isolation Forest</b><br>
            150 estimators · contamination=0.04<br>
            Framework: scikit-learn<br>
            Model size: ~0.8 MB
          </div>
          <div style="margin-top:12px;"><span class="badge-normal">{'LOADED ✓' if info['model_loaded'] else 'FALLBACK'}</span></div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="section-card" style="border-color:#7c3aed44;">
          <div class="section-title">Failure Prediction Model</div>
          <div style="margin-top:12px;font-size:.85rem;color:#94a3b8;">
            <b style="color:#7c3aed;">Random Forest Classifier</b><br>
            150 estimators · depth=10<br>
            6 features · balanced classes<br>
            Outputs: failure probability
          </div>
          <div style="margin-top:12px;"><span class="badge-maint">{'LOADED ✓' if info['model_loaded'] else 'FALLBACK'}</span></div>
        </div>""", unsafe_allow_html=True)
    with c3:
        avg_lat = info['avg_latency_ms']
        st.markdown(f"""
        <div class="section-card" style="border-color:#10b98144;">
          <div class="section-title">Inference Statistics</div>
          <div style="margin-top:12px;font-size:.85rem;color:#94a3b8;">
            Total inferences: <b style="color:#10b981;">{info['inference_count']:,}</b><br>
            Avg latency: <b style="color:#10b981;">{avg_lat:.1f} ms</b><br>
            Inferences/sec: <b style="color:#10b981;">{1000/max(avg_lat,1):.0f}</b><br>
            Features: {', '.join(info['features'][:3])}...
          </div>
          <div style="margin-top:12px;"><span class="badge-info">EDGE PROCESSING</span></div>
        </div>""", unsafe_allow_html=True)

    # Edge processing pipeline
    st.markdown("### ⚙ Edge Processing Pipeline")
    col_flow, col_gauge = st.columns([1, 1])
    with col_flow:
        anom_score = ai.get("anomaly_score", 0)
        is_anom    = ai.get("is_anomaly", False)
        fp         = ai.get("failure_probability",0)

        step_color = lambda ok: "#10b981" if ok else "#ef4444"
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #1e2a3a;border-radius:14px;padding:28px;">
          <div style="display:flex;flex-direction:column;gap:0;align-items:center;">

            <div class="flow-box" style="border-color:#00d4ff44;background:rgba(0,212,255,.05);">
              <div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;">Step 1 — Data Ingestion</div>
              <div style="color:#00d4ff;font-weight:600;margin-top:4px;">📡 Sensor Reading Received</div>
              <div style="font-size:.76rem;color:#94a3b8;margin-top:4px;">
                T={st.session_state.last_reading.get('temperature',0):.1f}°C &nbsp;
                V={st.session_state.last_reading.get('vibration',0):.2f}mm/s &nbsp;
                E={st.session_state.last_reading.get('energy',0):.1f}kWh
              </div>
            </div>
            <div class="flow-arrow">↓</div>

            <div class="flow-box" style="border-color:#7c3aed44;background:rgba(124,58,237,.05);">
              <div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;">Step 2 — Preprocessing</div>
              <div style="color:#7c3aed;font-weight:600;margin-top:4px;">🔀 Normalize &amp; Scale Features</div>
              <div style="font-size:.76rem;color:#94a3b8;margin-top:4px;">Z-score normalization · Missing value fill · Clip outliers</div>
            </div>
            <div class="flow-arrow">↓</div>

            <div class="flow-box" style="border-color:#f59e0b44;background:rgba(245,158,11,.05);">
              <div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;">Step 3 — Edge AI Inference</div>
              <div style="color:#f59e0b;font-weight:600;margin-top:4px;">🧠 Isolation Forest + Random Forest</div>
              <div style="font-size:.76rem;color:#94a3b8;margin-top:4px;">
                Anomaly score: {anom_score:.3f} &nbsp;·&nbsp;
                Failure prob: {fp*100:.1f}% &nbsp;·&nbsp;
                Latency: {ai.get('inference_ms',0):.1f}ms
              </div>
            </div>
            <div class="flow-arrow">↓</div>

            <div class="flow-box" style="border-color:{step_color(is_anom)}44;background:rgba({'239,68,68' if is_anom else '16,185,129'},.05);">
              <div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;">Step 4 — Decision</div>
              <div style="color:{'#ef4444' if is_anom else '#10b981'};font-weight:600;margin-top:4px;">
                {'🚨 ANOMALY DETECTED' if is_anom else '✅ NORMAL OPERATION'}
              </div>
              <div style="font-size:.76rem;color:#94a3b8;margin-top:4px;">
                Health: {ai.get('health_score',100):.1f}% &nbsp;·&nbsp;
                Decision sent to Digital Twin + Decision Engine
              </div>
            </div>

          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_gauge:
        fig_anom = gauge_fig(anom_score*100, "Anomaly Score", 0, 100, "%",
                             "#ef4444" if anom_score>0.5 else "#f59e0b" if anom_score>0.3 else "#10b981")
        fig_fp   = gauge_fig(fp*100, "Failure Probability", 0, 100, "%",
                             "#ef4444" if fp>0.65 else "#f59e0b" if fp>0.35 else "#10b981")
        fig_hs   = gauge_fig(ai.get("health_score",100), "Machine Health", 0, 100, "%",
                             health_color(ai.get("health_score",100)))
        st.plotly_chart(fig_anom, use_container_width=True)
        st.plotly_chart(fig_fp,   use_container_width=True)
        st.plotly_chart(fig_hs,   use_container_width=True)

    # Anomaly score timeline
    ai_results = st.session_state.ai_results
    if len(ai_results) >= 5:
        fig = go.Figure()
        anom_scores = [r["anomaly_score"] for r in ai_results[-80:]]
        fail_probs  = [r["failure_probability"] for r in ai_results[-80:]]
        health_vals = [r["health_score"]/100 for r in ai_results[-80:]]

        fig.add_trace(go.Scatter(y=anom_scores, mode="lines", name="Anomaly Score", line=dict(color="#f59e0b",width=2)))
        fig.add_trace(go.Scatter(y=fail_probs,  mode="lines", name="Failure Prob",  line=dict(color="#ef4444",width=2,dash="dot")))
        fig.add_trace(go.Scatter(y=health_vals, mode="lines", name="Health/100",    line=dict(color="#10b981",width=2)))
        fig.add_hline(y=0.45, line_dash="dash", line_color="#f59e0b44", annotation_text="Anomaly threshold")
        fig.add_hline(y=0.65, line_dash="dash", line_color="#ef444444", annotation_text="Failure threshold")
        fig.update_layout(**PLOTLY_LAYOUT, height=280, yaxis_title="Score", title="Edge AI Output Timeline")
        st.plotly_chart(fig, use_container_width=True)

    # Model performance metrics (from training)
    if os.path.exists("data/training_data.csv"):
        st.markdown("### 📋 Model Evaluation (Training)")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Accuracy",  "~97-99%", "Synthetic dataset")
        mc2.metric("Precision", "~96-98%", "Balanced classes")
        mc3.metric("Recall",    "~95-97%", "Failure class")
        mc4.metric("F1-Score",  "~96-98%", "Weighted avg")
        st.caption("Note: Trained on synthetic data. Accuracy reflects dataset quality, not real-world conditions.")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — Predictive Maintenance
# ══════════════════════════════════════════════════════════════════════════════
def page_predictive_maintenance():
    st.markdown("## 🔧 Predictive Maintenance")

    ai    = st.session_state.last_ai
    twin  = st.session_state.twin_state
    dec   = st.session_state.last_decision

    health = ai.get("health_score", 100)
    fp     = ai.get("failure_probability", 0) * 100

    risk_level = "LOW" if fp < 25 else ("MEDIUM" if fp < 50 else ("HIGH" if fp < 75 else "CRITICAL"))
    risk_color = {"LOW":"#10b981","MEDIUM":"#f59e0b","HIGH":"#ef4444","CRITICAL":"#7c3aed"}.get(risk_level,"#10b981")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        fig = gauge_fig(health, "Machine Health Score", 0, 100, "%", health_color(health))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = gauge_fig(fp, "Failure Probability", 0, 100, "%", health_color(100-fp))
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        st.markdown(f"""
        <div class="section-card" style="text-align:center;padding:40px 20px;">
          <div style="font-size:.78rem;color:#64748b;margin-bottom:12px;">MAINTENANCE RISK</div>
          <div style="font-size:2.2rem;font-weight:800;color:{risk_color};">{risk_level}</div>
          <div style="font-size:.8rem;color:#64748b;margin-top:8px;">Failure Prob: {fp:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        # Estimated time to maintenance
        if fp >= 75:  ttm = "< 4 hours"
        elif fp >= 50:ttm = "4–24 hours"
        elif fp >= 25:ttm = "24–72 hours"
        else:         ttm = "> 168 hours"
        st.markdown(f"""
        <div class="section-card" style="text-align:center;padding:40px 20px;">
          <div style="font-size:.78rem;color:#64748b;margin-bottom:12px;">ESTIMATED MAINTENANCE IN</div>
          <div style="font-size:1.6rem;font-weight:700;color:#00d4ff;">{ttm}</div>
          <div style="font-size:.8rem;color:#64748b;margin-top:8px;">Based on AI prediction</div>
        </div>""", unsafe_allow_html=True)

    # Action recommendation
    action = dec.get("action","✅ Continue Normal Operation")
    reason = dec.get("reason","")
    conf   = dec.get("confidence",0.99)*100
    sev    = dec.get("severity","INFO")
    sev_css= {"CRITICAL":"badge-critical","WARNING":"badge-warning","INFO":"badge-info"}.get(sev,"badge-info")

    st.markdown(f"""
    <div class="section-card" style="border-color:{risk_color}44;margin-top:8px;">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
        <span class="{sev_css}">{sev}</span>
        <span style="font-size:1rem;font-weight:600;color:#e2e8f0;">Recommended Action</span>
        <span style="color:#64748b;font-size:.78rem;">Confidence: {conf:.0f}%</span>
      </div>
      <div style="font-size:1.05rem;font-weight:600;color:#e2e8f0;margin-bottom:10px;">{action}</div>
      <div style="font-size:.85rem;color:#94a3b8;line-height:1.7;">{reason}</div>
    </div>""", unsafe_allow_html=True)

    # Health degradation chart
    dt_obj = st.session_state.digital_twin
    if len(dt_obj.history) >= 5:
        fig = go.Figure()
        hist = dt_obj.history[-100:]
        hs   = [h["health_score"] for h in hist]
        fp_l = [h["failure_probability"]*100 for h in hist]

        fig.add_trace(go.Scatter(y=hs, name="Health Score (%)", line=dict(color="#10b981",width=2.5), fill="tozeroy", fillcolor="rgba(16,185,129,.07)"))
        fig.add_trace(go.Scatter(y=fp_l, name="Failure Probability (%)", line=dict(color="#ef4444",width=2,dash="dot")))
        fig.add_hrect(y0=0, y1=40, fillcolor="rgba(239,68,68,.05)", line_width=0, annotation_text="Critical Zone")
        fig.add_hrect(y0=40,y1=70, fillcolor="rgba(245,158,11,.04)", line_width=0, annotation_text="Warning Zone")
        fig.update_layout(**PLOTLY_LAYOUT, height=300, title="Machine Health Degradation Over Time")
        st.plotly_chart(fig, use_container_width=True)

    # Maintenance checklist
    st.markdown("### 📋 Maintenance Checklist")
    checks = [
        ("Motor Bearings", vib_ok := twin.get("vibration",2) < 4.0, f"{twin.get('vibration',2):.2f} mm/s"),
        ("Cooling System", temp_ok := twin.get("temperature",70) < 85, f"{twin.get('temperature',70):.1f}°C"),
        ("Electrical Load", energy_ok := twin.get("energy",20) < 35,  f"{twin.get('energy',20):.1f} kWh"),
        ("Shaft RPM",       rpm_ok   := twin.get("rpm",1500) < 1700,  f"{twin.get('rpm',1500):.0f} RPM"),
        ("Pressure Lines",  pres_ok  := twin.get("pressure",4.5) < 7, f"{twin.get('pressure',4.5):.1f} bar"),
        ("Humidity Control",hum_ok   := twin.get("humidity",45) < 70, f"{twin.get('humidity',45):.1f}%RH"),
    ]
    cc1, cc2 = st.columns(2)
    for i, (item, ok, val) in enumerate(checks):
        col = cc1 if i % 2 == 0 else cc2
        icon  = "✅" if ok else "❌"
        color = "#10b981" if ok else "#ef4444"
        col.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
             padding:10px 14px;border-radius:8px;margin-bottom:8px;
             background:rgba({'16,185,129' if ok else '239,68,68'},.07);
             border:1px solid rgba({'16,185,129' if ok else '239,68,68'},.2);">
          <span style="color:#e2e8f0;">{icon} {item}</span>
          <span style="color:{color};font-family:'JetBrains Mono',monospace;font-size:.85rem;">{val}</span>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — Decision Engine
# ══════════════════════════════════════════════════════════════════════════════
def page_decision_engine():
    st.markdown("## ⚡ Decision Engine — Intelligent Actions")

    dec  = st.session_state.last_decision
    ai   = st.session_state.last_ai
    read = st.session_state.last_reading

    if not dec:
        st.info("⏳ Awaiting first simulation tick..."); return

    col_inputs, col_arrow, col_engine, col_arrow2, col_output = st.columns([2,.3,2,.3,2])

    with col_inputs:
        st.markdown("""<div class="section-card" style="border-color:#00d4ff44;">
        <div class="section-title">📥 Inputs</div>""", unsafe_allow_html=True)
        inputs = dec.get("inputs", {})
        for k, v in inputs.items():
            label = k.replace("_"," ").title()
            if isinstance(v, float) and v > 0 and v <= 1 and "probability" in k:
                display = f"{v*100:.1f}%"
            elif isinstance(v, float):
                display = f"{v:.2f}"
            else:
                display = str(v)
            color = "#00d4ff"
            st.markdown(f"""
            <div class="twin-param">
              <span class="twin-label">{label}</span>
              <span class="twin-val" style="color:{color};">{display}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_arrow:
        st.markdown("<div style='padding-top:80px;text-align:center;font-size:2rem;color:#00d4ff;'>→</div>", unsafe_allow_html=True)

    with col_engine:
        sev = dec.get("severity","INFO")
        sev_col = {"CRITICAL":"#ef4444","WARNING":"#f59e0b","INFO":"#10b981"}.get(sev,"#00d4ff")
        st.markdown(f"""
        <div class="section-card" style="border-color:{sev_col}66;text-align:center;">
          <div class="section-title">⚙ Decision Engine</div>
          <div style="margin:20px 0;">
            <div style="font-size:.75rem;color:#64748b;margin-bottom:8px;">RULES EVALUATED</div>
            <div style="font-size:2rem;font-weight:700;color:{sev_col};">{len(dec.get('rules_triggered',[]))}</div>
          </div>
          <div style="margin-top:12px;">
            <div style="font-size:.75rem;color:#64748b;margin-bottom:4px;">SEVERITY LEVEL</div>
            <span class="{'badge-critical' if sev=='CRITICAL' else 'badge-warning' if sev=='WARNING' else 'badge-normal'}">{sev}</span>
          </div>
          <hr>
          <div style="font-size:.75rem;color:#64748b;margin-bottom:6px;">RULES TRIGGERED</div>""", unsafe_allow_html=True)
        rules = dec.get("rules_triggered",[])
        for rule in rules[:4]:
            cls = "rule-critical" if "CRITICAL" in rule or "ALERT" in rule else ("rule-warning" if "WARNING" in rule or "AI PREDICTION" in rule else "rule-info")
            st.markdown(f'<div class="rule-item {cls}">• {rule[:80]}</div>', unsafe_allow_html=True)
        if len(rules) > 4:
            st.markdown(f'<div style="color:#64748b;font-size:.75rem;margin-top:4px;">+{len(rules)-4} more rules...</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_arrow2:
        st.markdown("<div style='padding-top:80px;text-align:center;font-size:2rem;color:#00d4ff;'>→</div>", unsafe_allow_html=True)

    with col_output:
        conf = dec.get("confidence",0.99)*100
        action = dec.get("action","Continue Normal Operation")
        reason = dec.get("reason","")
        st.markdown(f"""
        <div class="section-card" style="border-color:{sev_col}44;">
          <div class="section-title">📤 Output — Recommendation</div>
          <div style="margin:16px 0;">
            <div style="font-size:.75rem;color:#64748b;margin-bottom:4px;">CONFIDENCE</div>
            <div style="font-size:1.8rem;font-weight:700;color:{sev_col};">{conf:.0f}%</div>
          </div>
          <div style="font-size:.95rem;font-weight:600;color:#e2e8f0;line-height:1.5;margin-bottom:12px;">{action}</div>
          <div style="font-size:.8rem;color:#94a3b8;line-height:1.6;">{reason}</div>
        </div>""", unsafe_allow_html=True)

    # Decision history
    st.markdown("### 📜 Decision History")
    dec_df = get_recent_decisions(20)
    if not dec_df.empty:
        for _, row in dec_df.head(8).iterrows():
            sev_r = row.get("severity","INFO")
            cls   = {"CRITICAL":"alert-critical","WARNING":"alert-warning"}.get(sev_r,"alert-info")
            st.markdown(f"""
            <div class="alert-item {cls}" style="margin-bottom:6px;">
              <span class="{'badge-critical' if sev_r=='CRITICAL' else 'badge-warning' if sev_r=='WARNING' else 'badge-info'}">{sev_r}</span>
              <span style="color:#64748b;font-size:.74rem;float:right;">{str(row.get('timestamp',''))[:19]}</span><br>
              <b style="color:#e2e8f0;">{str(row.get('action',''))[:80]}</b><br>
              <span style="font-size:.78rem;color:#94a3b8;">{str(row.get('reason',''))[:120]}</span>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 7 — Scenario Simulation
# ══════════════════════════════════════════════════════════════════════════════
def page_scenario_simulation():
    st.markdown("## 🎮 Scenario Simulation — Live Demonstration")
    st.markdown('<div style="color:#94a3b8;font-size:.9rem;margin-bottom:20px;">Select a scenario to instantly change machine conditions. Edge AI and Digital Twin will react in real-time.</div>', unsafe_allow_html=True)

    scenarios = {
        "normal":                 ("✅ Normal Operation",        "#10b981", "All sensors within safe operating range. Machine running optimally."),
        "overheating":            ("🌡 Overheating",             "#ef4444", "Temperature rising rapidly. Cooling system failure simulation."),
        "excessive_vibration":    ("〰 Excessive Vibration",      "#f59e0b", "Mechanical fault — bearing wear or shaft imbalance detected."),
        "high_energy":            ("⚡ High Energy Consumption",  "#7c3aed", "Electrical overload — check power factor and load balancing."),
        "multiple_sensor_failure":("⚠ Multiple Sensor Failure",  "#f59e0b", "Several parameters simultaneously exceeding thresholds."),
        "critical_failure":       ("🚨 Critical Machine Failure", "#ef4444", "Catastrophic failure imminent — emergency shutdown required."),
    }

    current = st.session_state.scenario
    cols = st.columns(3)
    for i, (key, (label, color, desc)) in enumerate(scenarios.items()):
        with cols[i % 3]:
            is_active = current == key
            border = f"2px solid {color}" if is_active else f"1px solid {color}44"
            bg     = f"rgba({','.join(str(int(int(color[1:3],16))) + ',' + str(int(int(color[3:5],16))) + ',' + str(int(int(color[5:7],16)))) if False else ''},0)" 

            st.markdown(f"""
            <div style="background:{'rgba(0,0,0,0)' if not is_active else color+'11'};
                 border:{border};border-radius:14px;padding:20px;margin-bottom:12px;
                 transition:all .3s;{'box-shadow:0 0 20px '+color+'44;' if is_active else ''}">
              <div style="font-size:1.05rem;font-weight:600;color:{color};margin-bottom:6px;">{label}</div>
              <div style="font-size:.8rem;color:#94a3b8;margin-bottom:14px;">{desc}</div>
              {'<div style="font-size:.72rem;background:'+color+'22;color:'+color+';border-radius:20px;padding:3px 10px;display:inline-block;">● ACTIVE</div>' if is_active else ''}
            </div>""", unsafe_allow_html=True)
            if not is_active:
                if st.button(f"▶ Activate", key=f"sc_{key}", use_container_width=True):
                    st.session_state.scenario = key
                    st.session_state.simulator.set_scenario(key)
                    st.session_state.sim_running = True
                    st.rerun()
            else:
                st.markdown('<div style="text-align:center;color:#64748b;font-size:.8rem;padding:8px;">Currently Active</div>', unsafe_allow_html=True)

    # Live reaction panel
    st.markdown("---")
    st.markdown("### 🔄 Live System Reaction")

    twin = st.session_state.twin_state
    ai   = st.session_state.last_ai
    dec  = st.session_state.last_decision

    if twin and ai and dec:
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""
            <div class="section-card" style="text-align:center;">
              <div style="color:#64748b;font-size:.75rem;margin-bottom:8px;">MACHINE STATUS</div>
              <div style="font-size:1.3rem;font-weight:700;color:{STATUS_COLORS.get(twin.get('status','NORMAL'),'#10b981')};">
                {STATUS_ICONS.get(twin.get('status','NORMAL'),'')} {twin.get('status','NORMAL')}
              </div>
            </div>""", unsafe_allow_html=True)
        with r2:
            anom = ai.get("anomaly_score",0)
            st.markdown(f"""
            <div class="section-card" style="text-align:center;">
              <div style="color:#64748b;font-size:.75rem;margin-bottom:8px;">AI ANOMALY SCORE</div>
              <div style="font-size:1.3rem;font-weight:700;color:{'#ef4444' if anom>0.5 else '#f59e0b' if anom>0.3 else '#10b981'};">
                {anom:.3f}
              </div>
            </div>""", unsafe_allow_html=True)
        with r3:
            fp = ai.get("failure_probability",0)*100
            st.markdown(f"""
            <div class="section-card" style="text-align:center;">
              <div style="color:#64748b;font-size:.75rem;margin-bottom:8px;">FAILURE PROBABILITY</div>
              <div style="font-size:1.3rem;font-weight:700;color:{health_color(100-fp)};">
                {fp:.1f}%
              </div>
            </div>""", unsafe_allow_html=True)
        with r4:
            hs = twin.get("health_score",100)
            st.markdown(f"""
            <div class="section-card" style="text-align:center;">
              <div style="color:#64748b;font-size:.75rem;margin-bottom:8px;">MACHINE HEALTH</div>
              <div style="font-size:1.3rem;font-weight:700;color:{health_color(hs)};">
                {hs:.1f}%
              </div>
            </div>""", unsafe_allow_html=True)

        # Sensor bars
        st.markdown("#### 📊 Current Sensor Values")
        sb1, sb2 = st.columns(2)
        sensors_a = [("temperature","Temperature","°C",130),("vibration","Vibration","mm/s",16),("energy","Energy","kWh",80)]
        sensors_b = [("humidity","Humidity","%RH",100),("rpm","RPM","RPM",2500),("pressure","Pressure","bar",14)]
        for col, sensors in [(sb1,sensors_a),(sb2,sensors_b)]:
            with col:
                for key, label, unit, max_v in sensors:
                    val = twin.get(key, 0)
                    pct = min(100, val/max_v*100)
                    w_thr = THRESHOLDS.get(key,{}).get("warning",999)
                    c_thr = THRESHOLDS.get(key,{}).get("critical",999)
                    bar_c = "#ef4444" if val >= c_thr else ("#f59e0b" if val >= w_thr else "#10b981")
                    st.markdown(f"""
                    <div style="margin-bottom:14px;">
                      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="color:#94a3b8;font-size:.82rem;">{label}</span>
                        <span style="color:{bar_c};font-family:'JetBrains Mono',monospace;font-size:.85rem;">{val:.2f} {unit}</span>
                      </div>
                      <div style="background:#0d1117;border-radius:6px;height:8px;">
                        <div style="width:{pct:.1f}%;background:{bar_c};height:100%;border-radius:6px;transition:width .5s;"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

        # Decision
        sev = dec.get("severity","INFO")
        sev_css = {"CRITICAL":"badge-critical","WARNING":"badge-warning","INFO":"badge-info"}.get(sev,"badge-info")
        st.markdown(f"""
        <div class="section-card" style="margin-top:8px;border-color:{'#ef4444' if sev=='CRITICAL' else '#f59e0b' if sev=='WARNING' else '#10b981'}44;">
          <span class="{sev_css}">{sev}</span>
          <span style="font-size:1rem;font-weight:600;color:#e2e8f0;margin-left:12px;">{dec.get('action','')}</span>
          <div style="font-size:.82rem;color:#94a3b8;margin-top:8px;">{dec.get('reason','')}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 8 — Historical Analytics
# ══════════════════════════════════════════════════════════════════════════════
def page_historical_analytics():
    st.markdown("## 📊 Historical Analytics — Cloud Layer")
    st.markdown('<div style="color:#94a3b8;font-size:.85rem;margin-bottom:16px;">Historical data stored in SQLite (local cloud simulation). Processed summaries, trend analysis, and anomaly frequency.</div>', unsafe_allow_html=True)

    df = get_sensor_history(limit=500)
    anom_df = get_anomaly_history(limit=500)

    if df.empty:
        st.info("No historical data yet. Run the simulation for a while."); return

    # Stats row
    db_stats = get_db_stats()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Readings",  f"{db_stats['total_readings']:,}")
    s2.metric("Total Alerts",    f"{db_stats['total_alerts']:,}")
    s3.metric("Anomalies Logged",f"{db_stats['total_anomalies']:,}")
    s4.metric("Decisions Made",  f"{db_stats['total_decisions']:,}")

    tab1, tab2, tab3 = st.tabs(["📈 Sensor Trends", "🔍 Anomaly Analysis", "📋 Raw Data"])

    with tab1:
        sensors_to_plot = st.multiselect(
            "Select sensors", list(SENSOR_UNITS.keys()),
            default=["temperature","vibration","energy"]
        )
        if sensors_to_plot:
            fig = go.Figure()
            for sensor in sensors_to_plot:
                if sensor in df.columns:
                    fig.add_trace(go.Scatter(
                        y=df[sensor], mode="lines", name=sensor,
                        line=dict(color=SENSOR_COLORS[sensor], width=1.8),
                    ))
                    # Add threshold lines
                    w = THRESHOLDS.get(sensor,{}).get("warning")
                    if w: fig.add_hline(y=w, line_dash="dash", line_color=SENSOR_COLORS[sensor]+"66")
            fig.update_layout(**PLOTLY_LAYOUT, height=350, title="Sensor History (All Recorded Data)")
            st.plotly_chart(fig, use_container_width=True)

        # Stats table
        st.markdown("**Summary Statistics**")
        numeric_cols = [c for c in SENSOR_UNITS.keys() if c in df.columns]
        st.dataframe(
            df[numeric_cols].describe().round(3).T,
            use_container_width=True,
        )

    with tab2:
        if not anom_df.empty:
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Anomaly Score Over Time","Health Score Over Time"])
            fig.add_trace(go.Scatter(y=anom_df["anomaly_score"], mode="lines",
                         line=dict(color="#f59e0b",width=1.8), name="Anomaly Score"), row=1, col=1)
            fig.add_trace(go.Scatter(y=anom_df["health_score"], mode="lines",
                         line=dict(color="#10b981",width=1.8), name="Health Score"), row=1, col=2)
            fig.update_layout(**PLOTLY_LAYOUT, height=300)
            st.plotly_chart(fig, use_container_width=True)

            # Anomaly frequency by scenario
            if "scenario" in df.columns:
                scen_counts = df.groupby("scenario").size().reset_index(name="count")
                fig2 = px.bar(scen_counts, x="scenario", y="count",
                              color="count", color_continuous_scale=["#10b981","#f59e0b","#ef4444"],
                              title="Readings by Scenario")
                fig2.update_layout(**PLOTLY_LAYOUT, height=280)
                st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.dataframe(df.tail(100), use_container_width=True, height=400)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 9 — Alerts
# ══════════════════════════════════════════════════════════════════════════════
def page_alerts():
    st.markdown("## 🚨 Alert Management System")

    filter_status = st.selectbox("Filter by Status", ["All","open","acknowledged","resolved"])
    status_param  = None if filter_status == "All" else filter_status
    alerts_df     = get_all_alerts(status_param)

    counts = get_alert_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🚨 Critical Open",  counts.get("CRITICAL",0))
    c2.metric("⚠ Warning Open",   counts.get("WARNING",0))
    c3.metric("ℹ Info Open",       counts.get("INFO",0))
    c4.metric("📋 Total Shown",    len(alerts_df))

    if alerts_df.empty:
        st.success("✅ No alerts matching this filter.")
        return

    for _, row in alerts_df.iterrows():
        sev = row.get("severity","INFO")
        cls = {"CRITICAL":"alert-critical","WARNING":"alert-warning"}.get(sev,"alert-info")
        sev_css = {"CRITICAL":"badge-critical","WARNING":"badge-warning"}.get(sev,"badge-info")
        status_val = row.get("status","open")

        with st.expander(f"{'🚨' if sev=='CRITICAL' else '⚠' if sev=='WARNING' else 'ℹ'} [{sev}]  {str(row.get('condition_desc',''))[:80]}  —  {str(row.get('timestamp',''))[:19]}"):
            sc1, sc2 = st.columns([3,1])
            with sc1:
                st.markdown(f"""
                <div class="alert-item {cls}">
                  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                    <span class="{sev_css}">{sev}</span>
                    <span style="color:#64748b;font-size:.78rem;">{str(row.get('timestamp',''))[:19]}</span>
                  </div>
                  <div><b style="color:#e2e8f0;">Sensor:</b> <span style="color:#94a3b8;">{row.get('sensor','')}</span></div>
                  <div><b style="color:#e2e8f0;">Condition:</b> <span style="color:#94a3b8;">{row.get('condition_desc','')}</span></div>
                  <div><b style="color:#e2e8f0;">AI Result:</b> <span style="color:#94a3b8;">{row.get('ai_result','')}</span></div>
                  <div><b style="color:#e2e8f0;">Recommendation:</b> <span style="color:#00d4ff;">{row.get('recommendation','')}</span></div>
                </div>""", unsafe_allow_html=True)
            with sc2:
                st.markdown(f"**Status:** `{status_val}`")
                alert_id = int(row.get("id",0))
                if status_val == "open":
                    if st.button("✅ Acknowledge", key=f"ack_{alert_id}"):
                        update_alert_status(alert_id, "acknowledged"); st.rerun()
                    if st.button("🔒 Resolve", key=f"res_{alert_id}"):
                        update_alert_status(alert_id, "resolved"); st.rerun()
                elif status_val == "acknowledged":
                    if st.button("🔒 Resolve", key=f"res2_{alert_id}"):
                        update_alert_status(alert_id, "resolved"); st.rerun()
                else:
                    st.markdown("✅ Resolved")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 10 — Architecture
# ══════════════════════════════════════════════════════════════════════════════
def page_architecture():
    st.markdown("## 🗺 System Architecture — 3-Layer Edge AI Framework")

    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1117,#111827);border:1px solid #1e2a3a;border-radius:16px;padding:32px;font-family:'Inter',sans-serif;">

      <!-- Title -->
      <div style="text-align:center;margin-bottom:32px;">
        <div style="font-size:1.2rem;font-weight:700;color:#e2e8f0;">A Digital Twin Enabled Intelligent Decision Framework</div>
        <div style="font-size:.85rem;color:#64748b;margin-top:6px;">Using Edge Artificial Intelligence · Real-time · Predictive · Edge Enabled</div>
      </div>

      <!-- Communication bar -->
      <div style="background:#00d4ff0a;border:1px solid #00d4ff22;border-radius:8px;padding:8px 16px;text-align:center;margin-bottom:24px;">
        <span style="color:#00d4ff;font-size:.8rem;font-weight:600;">📡 COMMUNICATION: MQTT / HTTP / 5G / Wi-Fi / Ethernet &nbsp;&nbsp;&nbsp; 🔒 SECURITY: Encryption · Authentication · Authorization</span>
      </div>

      <!-- Three columns -->
      <div style="display:flex;gap:20px;align-items:flex-start;">

        <!-- Physical Layer -->
        <div style="flex:1;background:rgba(16,185,129,.05);border:1px solid rgba(16,185,129,.3);border-radius:12px;padding:20px;">
          <div style="color:#10b981;font-weight:700;font-size:.95rem;text-align:center;margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid rgba(16,185,129,.2);">
            🏭 PHYSICAL LAYER
          </div>
          <div style="color:#94a3b8;font-size:.82rem;line-height:2;">
            <div>📦 Physical Assets</div>
            <div>🌡 Temperature Sensor</div>
            <div>〰 Vibration Sensor</div>
            <div>💧 Humidity Sensor</div>
            <div>⚡ Energy Meter</div>
            <div>🔄 RPM / Motor Sensor</div>
            <div>🔵 Pressure Sensor</div>
            <div>📷 Camera Feed</div>
            <div style="margin-top:12px;background:rgba(16,185,129,.1);border-radius:6px;padding:6px 10px;">
              <span style="color:#10b981;font-size:.75rem;">Data Acquisition (IoT Sensors, PLC, OPC UA)</span>
            </div>
          </div>
        </div>

        <div style="padding-top:60px;color:#00d4ff;font-size:1.5rem;text-align:center;">→</div>

        <!-- Edge Layer -->
        <div style="flex:2;background:rgba(0,212,255,.04);border:1px solid rgba(0,212,255,.3);border-radius:12px;padding:20px;">
          <div style="color:#00d4ff;font-weight:700;font-size:.95rem;text-align:center;margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid rgba(0,212,255,.2);">
            🔷 EDGE LAYER — Edge Gateway / Edge Device
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">

            <div style="background:rgba(0,0,0,.3);border:1px solid #1e2a3a;border-radius:8px;padding:12px;">
              <div style="color:#00d4ff;font-size:.8rem;font-weight:600;margin-bottom:6px;">📥 Data Ingestion</div>
              <div style="color:#64748b;font-size:.75rem;line-height:1.8;">Protocol Handling<br>MQTT/HTTP/OPC UA<br>Data Filtering &amp; Preprocessing</div>
            </div>

            <div style="background:rgba(0,0,0,.3);border:1px solid #1e2a3a;border-radius:8px;padding:12px;">
              <div style="color:#f59e0b;font-size:.8rem;font-weight:600;margin-bottom:6px;">🧠 Edge AI Module</div>
              <div style="color:#64748b;font-size:.75rem;line-height:1.8;">Anomaly Detection<br>Predictive Maintenance<br>Real-time Analytics<br>TFLite / ONNX / Edge ML</div>
            </div>

            <div style="background:rgba(0,0,0,.3);border:1px solid #1e2a3a;border-radius:8px;padding:12px;">
              <div style="color:#7c3aed;font-size:.8rem;font-weight:600;margin-bottom:6px;">🔷 Digital Twin Module</div>
              <div style="color:#64748b;font-size:.75rem;line-height:1.8;">Real-time Data Sync<br>Virtual Model Update<br>Simulation &amp; Prediction</div>
            </div>

            <div style="background:rgba(0,0,0,.3);border:1px solid #1e2a3a;border-radius:8px;padding:12px;">
              <div style="color:#ef4444;font-size:.8rem;font-weight:600;margin-bottom:6px;">⚡ Decision Engine</div>
              <div style="color:#64748b;font-size:.75rem;line-height:1.8;">Rule Engine<br>AI Inference<br>Decision Logic<br>Alert Generation</div>
            </div>

          </div>
          <div style="margin-top:12px;background:rgba(0,0,0,.3);border:1px solid #1e2a3a;border-radius:8px;padding:10px;text-align:center;">
            <span style="color:#64748b;font-size:.75rem;">🗄 Local Storage — Historical Data · Model Repository · Alert Log</span>
          </div>
        </div>

        <div style="padding-top:60px;color:#00d4ff;font-size:1.5rem;text-align:center;">→</div>

        <!-- Cloud Layer -->
        <div style="flex:1.2;background:rgba(124,58,237,.04);border:1px solid rgba(124,58,237,.3);border-radius:12px;padding:20px;">
          <div style="color:#7c3aed;font-weight:700;font-size:.95rem;text-align:center;margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid rgba(124,58,237,.2);">
            ☁ CLOUD LAYER
          </div>
          <div style="color:#94a3b8;font-size:.82rem;line-height:2;">
            <div>🗄 Data Storage</div>
            <div>📊 Advanced Analytics</div>
            <div>🤖 ML Model Training</div>
            <div>📈 Dashboard / Visualization</div>
            <div>👥 User Management &amp; Security</div>
            <div style="margin-top:12px;background:rgba(124,58,237,.1);border-radius:6px;padding:6px 10px;">
              <span style="color:#7c3aed;font-size:.75rem;">AWS / Azure / GCP / Databases</span>
            </div>
          </div>
        </div>

        <div style="padding-top:60px;color:#00d4ff;font-size:1.5rem;text-align:center;">→</div>

        <!-- End Users -->
        <div style="flex:1;background:rgba(245,158,11,.04);border:1px solid rgba(245,158,11,.3);border-radius:12px;padding:20px;">
          <div style="color:#f59e0b;font-weight:700;font-size:.95rem;text-align:center;margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid rgba(245,158,11,.2);">
            👥 END USERS
          </div>
          <div style="color:#94a3b8;font-size:.82rem;line-height:2.2;">
            <div>⚙ Industry Operators</div>
            <div>🔧 Maintenance Team</div>
            <div>📋 Decision Makers</div>
            <div>💻 IT / Support Team</div>
          </div>
          <div style="margin-top:14px;font-size:.75rem;color:#64748b;">
            <div style="color:#f59e0b;font-weight:600;margin-bottom:6px;">End User Benefits:</div>
            <div>✅ Real-time visibility</div>
            <div>✅ Reduced downtime</div>
            <div>✅ Optimized operations</div>
            <div>✅ Cost savings</div>
            <div>✅ Improved safety</div>
          </div>
        </div>

      </div><!-- end flex -->

      <!-- Bottom: Module summary -->
      <div style="margin-top:24px;padding-top:20px;border-top:1px solid #1e2a3a;">
        <div style="color:#64748b;font-size:.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px;">Application Areas</div>
        <div style="display:flex;gap:12px;flex-wrap:wrap;">
          <span style="background:#1a2332;border:1px solid #1e2a3a;border-radius:8px;padding:6px 14px;font-size:.78rem;color:#94a3b8;">🏭 Smart Manufacturing</span>
          <span style="background:#1a2332;border:1px solid #1e2a3a;border-radius:8px;padding:6px 14px;font-size:.78rem;color:#94a3b8;">🏙 Smart Cities</span>
          <span style="background:#1a2332;border:1px solid #1e2a3a;border-radius:8px;padding:6px 14px;font-size:.78rem;color:#94a3b8;">⚕ Healthcare</span>
          <span style="background:#1a2332;border:1px solid #1e2a3a;border-radius:8px;padding:6px 14px;font-size:.78rem;color:#94a3b8;">⚡ Energy Management</span>
          <span style="background:#1a2332;border:1px solid #1e2a3a;border-radius:8px;padding:6px 14px;font-size:.78rem;color:#94a3b8;">🚗 Transportation</span>
          <span style="background:#1a2332;border:1px solid #1e2a3a;border-radius:8px;padding:6px 14px;font-size:.78rem;color:#94a3b8;">🌾 Agriculture</span>
        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 11 — Edge vs Cloud
# ══════════════════════════════════════════════════════════════════════════════
def page_edge_vs_cloud():
    st.markdown("## ☁ Edge vs Cloud — Why Edge AI?")

    # Latencies from actual inference
    edge_lats = st.session_state.edge_latencies
    avg_edge  = float(np.mean(edge_lats)) if edge_lats else 12.0
    avg_cloud = avg_edge * 18 + np.random.uniform(80, 120)   # simulated cloud RTT

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(0,212,255,.06),rgba(0,212,255,.02));border:2px solid #00d4ff44;border-radius:16px;padding:28px;text-align:center;">
          <div style="font-size:1.1rem;font-weight:700;color:#00d4ff;margin-bottom:20px;">⚡ WITH EDGE AI</div>
          <div style="font-size:.85rem;color:#94a3b8;margin-bottom:20px;line-height:1.8;">
            Sensor → <b style="color:#00d4ff;">Edge Node</b> → AI Inference → Decision<br>
            ↓ (Only summaries sent)<br>
            Cloud Storage
          </div>
          <div style="font-size:2.8rem;font-weight:800;color:#00d4ff;">{avg_edge:.1f} ms</div>
          <div style="color:#64748b;font-size:.8rem;margin-top:6px;">Average Edge Inference Latency</div>
          <div style="margin-top:20px;display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:.8rem;text-align:left;">
            <div style="background:rgba(16,185,129,.1);border-radius:8px;padding:10px;color:#10b981;">✅ Low latency</div>
            <div style="background:rgba(16,185,129,.1);border-radius:8px;padding:10px;color:#10b981;">✅ Offline capable</div>
            <div style="background:rgba(16,185,129,.1);border-radius:8px;padding:10px;color:#10b981;">✅ Data privacy</div>
            <div style="background:rgba(16,185,129,.1);border-radius:8px;padding:10px;color:#10b981;">✅ 78% less bandwidth</div>
          </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(239,68,68,.06),rgba(239,68,68,.02));border:2px solid #ef444444;border-radius:16px;padding:28px;text-align:center;">
          <div style="font-size:1.1rem;font-weight:700;color:#ef4444;margin-bottom:20px;">🐌 WITHOUT EDGE AI</div>
          <div style="font-size:.85rem;color:#94a3b8;margin-bottom:20px;line-height:1.8;">
            Sensor → Network → <b style="color:#ef4444;">Cloud Server</b> → AI → Decision<br>
            ↓ (All raw data sent)<br>
            Response back to device
          </div>
          <div style="font-size:2.8rem;font-weight:800;color:#ef4444;">{avg_cloud:.0f} ms</div>
          <div style="color:#64748b;font-size:.8rem;margin-top:6px;">Estimated Cloud Round-Trip Latency</div>
          <div style="margin-top:20px;display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:.8rem;text-align:left;">
            <div style="background:rgba(239,68,68,.1);border-radius:8px;padding:10px;color:#ef4444;">❌ High latency</div>
            <div style="background:rgba(239,68,68,.1);border-radius:8px;padding:10px;color:#ef4444;">❌ Requires internet</div>
            <div style="background:rgba(239,68,68,.1);border-radius:8px;padding:10px;color:#ef4444;">❌ Privacy concerns</div>
            <div style="background:rgba(239,68,68,.1);border-radius:8px;padding:10px;color:#ef4444;">❌ High bandwidth cost</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # Latency comparison chart
    if len(edge_lats) > 10:
        fig = go.Figure()
        cloud_sim = [l * np.random.uniform(15, 22) + np.random.uniform(60,120) for l in edge_lats[-60:]]
        fig.add_trace(go.Scatter(y=edge_lats[-60:], mode="lines", name="Edge Latency (ms)",
                     line=dict(color="#00d4ff", width=2), fill="tozeroy", fillcolor="rgba(0,212,255,.07)"))
        fig.add_trace(go.Scatter(y=cloud_sim, mode="lines", name="Simulated Cloud Latency (ms)",
                     line=dict(color="#ef4444", width=2, dash="dot")))
        fig.update_layout(**PLOTLY_LAYOUT, height=280, title="Edge vs Cloud Latency Comparison — Live Data",
                         yaxis_title="Latency (ms)")
        st.plotly_chart(fig, use_container_width=True)

    speedup = avg_cloud / max(avg_edge, 0.1)
    bw_save = 78.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Latency Speedup", f"{speedup:.0f}×", f"Edge is {speedup:.0f}x faster")
    m2.metric("Bandwidth Saved", f"{bw_save:.0f}%", "Only summaries to cloud")
    m3.metric("Edge Inferences", f"{st.session_state.edge_ai.inference_count:,}", "All processed locally")

    # Data flow explanation
    st.markdown("""
    <div class="section-card" style="margin-top:16px;">
      <div class="section-title">🔄 Data Flow Comparison</div>
      <div style="margin-top:16px;display:grid;grid-template-columns:1fr 1fr;gap:20px;">
        <div>
          <div style="color:#00d4ff;font-weight:600;margin-bottom:10px;">Edge AI Data Flow</div>
          <div style="font-size:.82rem;color:#94a3b8;line-height:2.2;">
            1️⃣ Sensor generates raw data (100 readings/min)<br>
            2️⃣ Edge node filters + normalizes data locally<br>
            3️⃣ AI model runs inference locally (&lt;15ms)<br>
            4️⃣ Decision made immediately at the edge<br>
            5️⃣ Only summaries/alerts sent to cloud (~22% of data)
          </div>
        </div>
        <div>
          <div style="color:#ef4444;font-weight:600;margin-bottom:10px;">Cloud-Only Data Flow</div>
          <div style="font-size:.82rem;color:#94a3b8;line-height:2.2;">
            1️⃣ Sensor generates raw data (100 readings/min)<br>
            2️⃣ ALL raw data sent to cloud (100% bandwidth)<br>
            3️⃣ Cloud processes data (100-300ms latency)<br>
            4️⃣ Decision sent back to device (network delay)<br>
            5️⃣ Action taken 200-400ms after event occurred
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════════════════════════════════════
PAGE_FUNCTIONS = {
    "Overview":               page_overview,
    "Live Monitoring":        page_live_monitoring,
    "Digital Twin":           page_digital_twin,
    "Edge AI":                page_edge_ai,
    "Predictive Maintenance": page_predictive_maintenance,
    "Decision Engine":        page_decision_engine,
    "Scenario Simulation":    page_scenario_simulation,
    "Historical Analytics":   page_historical_analytics,
    "Alerts":                 page_alerts,
    "Architecture":           page_architecture,
    "Edge vs Cloud":          page_edge_vs_cloud,
}

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    if not st.session_state.logged_in:
        show_login()
        return

    # Auto-refresh for live simulation (every 2 seconds on live pages)
    live_pages = {"Overview","Live Monitoring","Digital Twin","Edge AI","Predictive Maintenance",
                  "Decision Engine","Scenario Simulation"}
    if AUTO_REFRESH and st.session_state.page in live_pages and st.session_state.sim_running:
        st_autorefresh(interval=2000, key="autorefresh")

    show_sidebar()

    # Run one simulation step per page load (auto-refresh triggers this)
    run_sim_step()

    # Render the active page
    page_fn = PAGE_FUNCTIONS.get(st.session_state.page, page_overview)
    page_fn()


if __name__ == "__main__":
    main()
