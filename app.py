import sys
from pathlib import Path

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

import data as dmod
import model as mmod

st.set_page_config(
    page_title="Tasty Bytes · Recipe Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def _load_app_data():
    _df = dmod.load_cleaned()
    _model_info = mmod.train_all(_df)
    return _df, _model_info


df, MODEL_INFO = _load_app_data()

# ── Palette ───────────────────────────────────────────────────────────────────
ORANGE    = "#f97316"
ORANGE_DK = "#ea6d0a"
TEAL      = "#06b6d4"
CYAN      = "#38bdf8"
BLUE      = "#2563eb"
BLUE_DK   = "#2f4270"
GREEN     = "#22c55e"
RED       = "#ef4444"
YELLOW    = "#eab308"
PURPLE    = "#a855f7"
BG        = "#0b0d1a"
CARD      = "#151729"
CARD2     = "#1a1d2e"
TEXT      = "#f1f5f9"
MUTED     = "#94a3b8"
DIM       = "#475569"
TRACK     = "rgba(255,255,255,0.07)"
GRID      = "rgba(255,255,255,0.06)"

# ── Derived constants ─────────────────────────────────────────────────────────
NUMERIC_FEATURES   = dmod.NUMERIC_FEATURES
NUTRITION_FEATURES = dmod.NUTRITION_FEATURES
FEATURE_COLUMNS    = dmod.NUMERIC_FEATURES + ["category"]
MODEL_NAMES        = mmod.MODEL_NAMES
LR                 = MODEL_NAMES[0]
high_rate          = float(df["high_traffic_label"].mean())
n_recipes          = len(df)
n_categories       = df["category"].nunique()
median_calories    = float(df["calories"].median())
LR_RESULTS         = MODEL_INFO["results"][LR]
LR_RECALL          = LR_RESULTS["Recall"]
LR_PRECISION       = LR_RESULTS["Precision"]
LR_AUC             = LR_RESULTS["ROC AUC"]
LR_F1              = LR_RESULTS["F1 Score"]
LR_FPR             = LR_RESULTS["False Positive Rate"]
RECALL_TARGET      = 0.80
MODEL_ON_TARGET    = LR_RECALL >= RECALL_TARGET

# Pre-compute category stats once
_rates       = df.groupby("category")["high_traffic_label"].mean()
BEST_CAT     = _rates.idxmax()
BEST_PCT     = float(_rates.max() * 100)
WORST_CAT    = _rates.idxmin()
WORST_PCT    = float(_rates.min() * 100)

# ── Chart layout helpers ──────────────────────────────────────────────────────
def _cl(h=360, t=44, b=14, l=10, r=10):
    return dict(
        height=h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=11),
        margin=dict(l=l, r=r, t=t, b=b),
    )


def _axis(grid=True, zero=True, tick_color=None):
    gc = "rgba(255,255,255,0.06)"
    return dict(
        gridcolor=gc if grid else "rgba(0,0,0,0)",
        zerolinecolor="rgba(255,255,255,0.10)" if zero else "rgba(0,0,0,0)",
        linecolor="rgba(255,255,255,0.08)",
        tickfont=dict(color=tick_color or MUTED, size=11),
    )


# ── CSS ───────────────────────────────────────────────────────────────────────
def apply_css():
    _css = f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
/* ── Reset & Root ──────────────────────────── */
.stApp {{
  background: {BG} !important;
  color: {TEXT};
  font-family: "Inter", "Segoe UI", system-ui, sans-serif;
}}
.block-container {{
  padding-top: 0 !important;
  padding-bottom: 3rem;
  max-width: 1700px;
}}
h1, h2, h3, h4, h5 {{ color: {TEXT} !important; font-weight: 700; margin: 0; }}

/* ── Sidebar ────────────────────────────────── */
div[data-testid="stSidebar"] {{
  background: #0d0f1e !important;
  border-right: 1px solid rgba(255,255,255,0.06);
}}
div[data-testid="stSidebar"] * {{ color: {MUTED} !important; }}
div[data-testid="stSidebar"] .stRadio > label {{
  font-size: 0.68rem !important;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: {DIM} !important;
  padding-bottom: 0.25rem;
}}
div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {{
  display: flex !important;
  align-items: center !important;
  gap: 0.6rem !important;
  padding: 0.58rem 0.85rem !important;
  border-radius: 8px !important;
  cursor: pointer !important;
  transition: background 0.15s !important;
  color: {MUTED} !important;
  font-size: 0.91rem !important;
  font-weight: 500 !important;
  margin-bottom: 2px !important;
}}
div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {{
  background: rgba(249,115,22,0.10) !important;
  color: {TEXT} !important;
}}
div[data-testid="stSidebar"] .stRadio label:has(input[type="radio"]:checked),
div[data-testid="stSidebar"] .stRadio [data-checked="true"],
div[data-testid="stSidebar"] .stRadio [aria-checked="true"] {{
  background: linear-gradient(135deg, {ORANGE} 0%, {ORANGE_DK} 100%) !important;
  box-shadow: 0 6px 18px rgba(249,115,22,0.35) !important;
}}
div[data-testid="stSidebar"] .stRadio label:has(input[type="radio"]:checked) *,
div[data-testid="stSidebar"] .stRadio [data-checked="true"] *,
div[data-testid="stSidebar"] .stRadio [aria-checked="true"] * {{
  color: #fff !important;
  font-weight: 700 !important;
}}

/* ── Top Banner ─────────────────────────────── */
.top-banner {{
  background: linear-gradient(135deg, {CARD2} 0%, {CARD} 100%);
  border-bottom: 1px solid rgba(255,255,255,0.07);
  padding: 1.1rem 1.8rem 1rem 3.7rem;
  margin-bottom: 1.4rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.75rem;
}}
.tb-left {{ display: flex; flex-direction: column; }}
.tb-title {{ font-size: 1.45rem; font-weight: 800; color: {TEXT} !important; letter-spacing: -0.02em; }}
.tb-sub   {{ font-size: 0.83rem; color: {MUTED}; margin-top: 2px; }}
.tb-right {{ display: flex; align-items: center; gap: 0.65rem; flex-wrap: wrap; }}
.tb-pill  {{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 999px;
  padding: 0.28rem 0.8rem;
  font-size: 0.77rem;
  color: {MUTED};
  white-space: nowrap;
}}
.tb-badge {{
  background: rgba(249,115,22,0.15);
  border: 1px solid rgba(249,115,22,0.35);
  color: {ORANGE};
  border-radius: 8px;
  padding: 0.32rem 0.85rem;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.03em;
}}

/* ── KPI Cards ──────────────────────────────── */
.kpi-card {{
  background: {CARD};
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 12px;
  padding: 1rem 1.2rem 0.9rem;
  position: relative;
  overflow: hidden;
  height: 100%;
}}
.kpi-card::after {{
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: var(--kpi-accent, {ORANGE});
  border-radius: 12px 12px 0 0;
}}
.kpi-label {{
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: {MUTED};
  margin-bottom: 0.38rem;
  font-weight: 600;
}}
.kpi-value {{
  font-size: 1.85rem;
  font-weight: 800;
  color: {TEXT};
  line-height: 1.1;
  letter-spacing: -0.02em;
}}
.kpi-delta {{
  font-size: 0.77rem;
  margin-top: 0.32rem;
  display: flex;
  align-items: center;
  gap: 0.28rem;
  font-weight: 600;
}}
.kpi-delta.up      {{ color: {GREEN}; }}
.kpi-delta.down    {{ color: {RED}; }}
.kpi-delta.warn    {{ color: {YELLOW}; }}
.kpi-delta.neutral {{ color: {MUTED}; }}
.kpi-sub {{
  font-size: 0.74rem;
  color: {DIM};
  margin-top: 0.1rem;
}}

/* ── Panel ──────────────────────────────────── */
.panel {{
  background: {CARD};
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 14px;
  padding: 1.2rem 1.4rem;
  margin-bottom: 1.2rem;
}}

/* ── Section headers ────────────────────────── */
.sec-eyebrow {{
  font-size: 0.67rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: {ORANGE};
  font-weight: 700;
  margin-bottom: 0.2rem;
}}
.sec-heading {{
  font-size: 1.1rem;
  font-weight: 800;
  color: {TEXT};
  margin-bottom: 0.18rem;
}}
.sec-sub {{
  font-size: 0.84rem;
  color: {MUTED};
  margin-bottom: 0.9rem;
}}

/* ── Insight badges ─────────────────────────── */
.insight-badge {{
  display: flex;
  align-items: flex-start;
  gap: 0.7rem;
  background: {CARD2};
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 10px;
  padding: 0.8rem 0.95rem;
  margin-bottom: 0.65rem;
}}
.insight-badge.orange {{ border-left: 3px solid {ORANGE}; }}
.insight-badge.green  {{ border-left: 3px solid {GREEN}; }}
.insight-badge.teal   {{ border-left: 3px solid {TEAL}; }}
.insight-badge.red    {{ border-left: 3px solid {RED}; }}
.insight-badge.yellow {{ border-left: 3px solid {YELLOW}; }}
.ib-icon  {{ font-size: 1.15rem; flex-shrink: 0; margin-top: 1px; }}
.ib-title {{ font-size: 0.88rem; font-weight: 700; color: {TEXT}; }}
.ib-sub   {{ font-size: 0.80rem; color: {MUTED}; margin-top: 3px; line-height: 1.45; }}

/* ── Progress bars ──────────────────────────── */
.prog-row {{ margin-bottom: 0.75rem; }}
.prog-header {{ display: flex; justify-content: space-between; margin-bottom: 5px; }}
.prog-name {{ font-size: 0.82rem; color: {TEXT}; font-weight: 600; }}
.prog-pct  {{ font-size: 0.80rem; color: {MUTED}; }}
.prog-track {{ background: rgba(255,255,255,0.07); border-radius: 999px; height: 7px; overflow: hidden; }}
.prog-fill  {{ height: 100%; border-radius: 999px; transition: width 0.5s; }}

/* ── Bottom summary strip ───────────────────── */
.bottom-strip {{
  background: {CARD};
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 14px;
  padding: 1.05rem 1.5rem;
  display: flex;
  gap: 0;
  align-items: stretch;
  margin-top: 0.5rem;
}}
.bs-item {{
  flex: 1;
  min-width: 130px;
  padding: 0 1.2rem;
  border-right: 1px solid rgba(255,255,255,0.07);
}}
.bs-item:first-child {{ padding-left: 0; }}
.bs-item:last-child  {{ border-right: none; padding-right: 0; }}
.bs-label {{ font-size: 0.69rem; text-transform: uppercase; letter-spacing: 0.08em; color: {MUTED}; margin-bottom: 4px; }}
.bs-value {{ font-size: 1.12rem; font-weight: 800; color: {TEXT}; }}
.bs-sub   {{ font-size: 0.74rem; color: {DIM}; margin-top: 2px; }}

/* ── Sidebar target card ────────────────────── */
.target-card {{
  background: linear-gradient(135deg, rgba(249,115,22,0.14) 0%, rgba(249,115,22,0.04) 100%);
  border: 1px solid rgba(249,115,22,0.25);
  border-radius: 12px;
  padding: 1rem 1.1rem;
}}
.tc-eyebrow {{ font-size: 0.67rem; text-transform: uppercase; letter-spacing: 0.1em; color: {ORANGE} !important; font-weight: 700; }}
.tc-value   {{ font-size: 1.5rem; font-weight: 800; color: {TEXT} !important; margin: 0.28rem 0 0.1rem; }}
.tc-sub     {{ font-size: 0.78rem; color: {MUTED} !important; margin-bottom: 0.55rem; }}
.tc-status-on  {{ font-size: 0.8rem; color: {GREEN} !important; font-weight: 700; }}
.tc-status-off {{ font-size: 0.8rem; color: {YELLOW} !important; font-weight: 700; }}

/* ── Stat table row ─────────────────────────── */
.stat-row {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.55rem 0;
  border-bottom: 1px solid rgba(255,255,255,0.05);
}}
.stat-row:last-child {{ border-bottom: none; }}
.stat-name  {{ font-size: 0.85rem; color: {MUTED}; }}
.stat-value {{ font-size: 0.9rem; font-weight: 700; color: {TEXT}; }}
.stat-badge {{
  font-size: 0.73rem;
  font-weight: 700;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
}}
.stat-badge.green  {{ background: rgba(34,197,94,0.15); color: {GREEN}; }}
.stat-badge.orange {{ background: rgba(249,115,22,0.15); color: {ORANGE}; }}
.stat-badge.teal   {{ background: rgba(6,182,212,0.15); color: {TEAL}; }}
.stat-badge.yellow {{ background: rgba(234,179,8,0.15); color: {YELLOW}; }}

/* ── Tabs ────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
  background: {CARD2};
  border-radius: 10px;
  padding: 3px;
  gap: 3px;
  border: 1px solid rgba(255,255,255,0.07);
}}
.stTabs [data-baseweb="tab"] {{
  background: transparent;
  border-radius: 8px;
  color: {MUTED};
  font-size: 0.87rem;
  font-weight: 600;
  padding: 0.4rem 1.1rem;
}}
.stTabs [aria-selected="true"] {{
  background: {ORANGE} !important;
  color: #fff !important;
}}

/* ── DataFrames ──────────────────────────────── */
div[data-testid="stDataFrame"] {{
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.07);
}}

/* ── Form inputs ─────────────────────────────── */
.stNumberInput > div > div > input {{
  background: {CARD2} !important;
  border-color: rgba(255,255,255,0.1) !important;
  color: {TEXT} !important;
  border-radius: 8px !important;
}}
.stSelectbox > div > div {{
  background: {CARD2} !important;
  border-color: rgba(255,255,255,0.1) !important;
  color: {TEXT} !important;
  border-radius: 8px !important;
}}

/* ── Buttons ─────────────────────────────────── */
.stButton > button, .stDownloadButton > button {{
  background: {ORANGE} !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 700 !important;
  padding: 0.45rem 1.4rem !important;
  letter-spacing: 0.02em !important;
}}
.stButton > button:hover {{ background: {ORANGE_DK} !important; }}

/* ── Hide chrome ─────────────────────────────── */
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu {{ visibility: hidden; }}
footer    {{ visibility: hidden; }}
div[data-testid="stToolbar"] {{ visibility: hidden; }}

/* ── Sidebar expand control ──────────────────── */
/* Streamlit nests stExpandSidebarButton inside stToolbar, so the rule above
   hides it and the sidebar can never be reopened. Re-show and theme it. */
div[data-testid="stToolbar"] [data-testid="stExpandSidebarButton"],
div[data-testid="stToolbar"] [data-testid="stExpandSidebarButton"] * {{ visibility: visible; }}
[data-testid="stExpandSidebarButton"] {{
  width: 2.4rem !important;
  height: 2.4rem !important;
  margin-left: 0.55rem !important;
  border-radius: 10px !important;
  background: {CARD} !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  color: {ORANGE} !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  box-shadow: 0 6px 18px rgba(0,0,0,0.38);
  transition: background 0.15s ease, transform 0.15s ease;
}}
[data-testid="stExpandSidebarButton"]:hover {{
  background: {CARD2} !important;
  transform: translateY(-1px);
}}
[data-testid="stExpandSidebarButton"] svg,
[data-testid="stExpandSidebarButton"] span {{
  color: {ORANGE} !important;
  fill: {ORANGE} !important;
  stroke: {ORANGE} !important;
}}

/* ── Selectbox label colour ──────────────────── */
.stSelectbox label, .stNumberInput label {{ color: {MUTED} !important; font-size: 0.82rem !important; }}

/* ── Export PDF button ───────────────────────── */
.export-btn {{
  background: linear-gradient(135deg, {ORANGE} 0%, {ORANGE_DK} 100%);
  color: #fff !important;
  border: none;
  border-radius: 8px;
  font-weight: 700;
  font-size: 0.82rem;
  padding: 0.52rem 1.25rem;
  cursor: pointer;
  letter-spacing: 0.03em;
  box-shadow: 0 6px 18px rgba(249,115,22,0.35);
  transition: transform 0.12s, box-shadow 0.12s;
  font-family: "Inter", "Segoe UI", system-ui, sans-serif;
}}
.export-btn:hover {{ transform: translateY(-1px); box-shadow: 0 8px 22px rgba(249,115,22,0.45); }}

/* ── Ring KPI cards ──────────────────────────── */
.ring-kpi {{
  background: {CARD};
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 12px;
  padding: 0;
  overflow: hidden;
  height: 100%;
  display: flex;
  flex-direction: column;
  box-shadow: 0 8px 24px rgba(0,0,0,0.22);
}}
.ring-kpi:hover {{ border-color: rgba(255,255,255,0.14); }}
.ring-body {{
  flex: 1;
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.15rem 0.85rem 0.8rem;
  min-width: 0;
}}
.ring-meta {{ flex: 1; min-width: 0; }}
.ring-svg {{ flex-shrink: 0; }}
.ring-value {{ font-size: 1.6rem; font-weight: 800; color: {TEXT}; line-height: 1.05; letter-spacing: -0.02em; }}
.ring-label {{ font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; color: {MUTED}; font-weight: 600; margin-bottom: 0.3rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.ring-delta {{ font-size: 0.72rem; font-weight: 600; margin-top: 0.24rem; }}
.ring-sub {{ font-size: 0.7rem; color: {DIM}; margin-top: 0.08rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

/* ── Data table ──────────────────────────────── */
.dtable {{ width: 100%; border-collapse: collapse; }}
.dtable th {{
  text-align: left;
  font-size: 0.64rem;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: {DIM};
  font-weight: 700;
  padding: 0.45rem 0.6rem;
  border-bottom: 1px solid rgba(255,255,255,0.09);
  white-space: nowrap;
}}
.dtable td {{
  padding: 0.55rem 0.6rem;
  font-size: 0.82rem;
  color: {TEXT};
  border-bottom: 1px solid rgba(255,255,255,0.05);
  white-space: nowrap;
}}
.dtable tr:hover td {{ background: rgba(255,255,255,0.03); }}
.dtable .muted {{ color: {MUTED}; }}
.dtable .tk-cat {{ font-weight: 700; color: {TEXT}; }}
.dtable .tk-rate {{ font-weight: 800; }}
.dtable .up   {{ color: {GREEN} !important; font-weight: 700; }}
.dtable .down {{ color: {RED} !important; font-weight: 700; }}
.dtable .share {{ font-size: 0.76rem; color: {MUTED}; }}

/* ── Scrollbar ───────────────────────────────── */
::-webkit-scrollbar {{ width: 9px; height: 9px; }}
::-webkit-scrollbar-track {{ background: rgba(255,255,255,0.04); }}
::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.14); border-radius: 999px; }}
::-webkit-scrollbar-thumb:hover {{ background: rgba(249,115,22,0.5); }}

/* ── Print (Export PDF) ──────────────────────── */
@media print {{
  @page {{ size: A4 landscape; margin: 8mm; }}
  /* the dashboard is dark-themed, so backgrounds must be forced or the
     light text prints invisible on white paper */
  html, body, .stApp, [data-testid="stAppViewContainer"], .main,
  [data-testid="stMainBlockContainer"], [data-testid="stVerticalBlock"] {{
    background: {BG} !important;
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
  }}
  * {{ -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }}
  div[data-testid="stSidebar"],
  header[data-testid="stHeader"],
  nav[data-testid="stToolbar"],
  footer, #MainMenu, .export-btn, [data-testid="stExpandSidebarButton"] {{
    display: none !important;
  }}
  div[data-testid="stAppViewContainer"] .main .block-container {{ max-width: 100% !important; padding: 0 !important; }}
  .stTabs [data-baseweb="tab"] {{ color: #fff !important; }}
  .dtable tr, .kpi-card, .ring-kpi, .panel, .top-banner {{ break-inside: avoid; }}
}}
</style>"""
    st.html(_css)


# ── UI helpers ────────────────────────────────────────────────────────────────
def kpi_card(label, value, delta="", delta_cls="neutral", sub="", accent=ORANGE):
    delta_colors = {"up": GREEN, "down": RED, "warn": YELLOW, "neutral": MUTED}
    d_color = delta_colors.get(delta_cls, MUTED)
    delta_html = (
        f'<div style="font-size:0.77rem;font-weight:600;color:{d_color};'
        f'margin-top:0.32rem">{delta}</div>'
    ) if delta else ""
    sub_html = (
        f'<div style="font-size:0.74rem;color:{DIM};margin-top:0.1rem">{sub}</div>'
    ) if sub else ""
    return (
        f'<div style="background:{CARD};border:1px solid rgba(255,255,255,0.07);'
        f'border-radius:12px;padding:0 1.2rem 1rem;overflow:hidden;">'
        # coloured top bar as a real div
        f'<div style="height:3px;background:{accent};border-radius:12px 12px 0 0;'
        f'margin:0 -1.2rem 0.9rem;"></div>'
        f'<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.09em;'
        f'color:{MUTED};margin-bottom:0.38rem;font-weight:600">{label}</div>'
        f'<div style="font-size:1.85rem;font-weight:800;color:{TEXT};line-height:1.1;'
        f'letter-spacing:-0.02em">{value}</div>'
        f'{delta_html}{sub_html}</div>'
    )


def ring_svg(pct, color, size=38, stroke=4.5, track=TRACK):
    pct = max(0.0, min(float(pct), 100.0))
    r = (size - stroke) / 2
    circ = 2 * math.pi * r
    filled = circ * pct / 100.0
    c = size / 2
    return (
        f'<svg class="ring-svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="{track}" stroke-width="{stroke}"/>'
        f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-dasharray="{filled:.2f} {circ:.2f}" '
        f'transform="rotate(-90 {c} {c})"/>'
        f'</svg>'
    )


RING_DELTA_COLORS = {"up": GREEN, "down": RED, "warn": YELLOW, "neutral": MUTED}


def ring_kpi(label, value, pct, color=ORANGE, delta="", delta_cls="neutral", sub=""):
    d_color = RING_DELTA_COLORS.get(delta_cls, MUTED)
    delta_html = (
        f'<div class="ring-delta" style="color:{d_color}">{delta}</div>'
    ) if delta else ""
    sub_html = (
        f'<div class="ring-sub">{sub}</div>'
    ) if sub else ""
    return (
        f'<div class="ring-kpi">'
        f'<div style="height:3px;background:linear-gradient(90deg,{color},{color}00);'
        f'border-radius:12px 12px 0 0;"></div>'
        f'<div class="ring-body">'
        f'<div class="ring-meta">'
        f'<div class="ring-label">{label}</div>'
        f'<div class="ring-value">{value}</div>'
        f'{delta_html}{sub_html}'
        f'</div>'
        f'{ring_svg(pct, color)}'
        f'</div></div>'
    )


def prog_bar(name, pct, color=ORANGE, pct_label=None):
    lbl = pct_label or f"{pct:.1f}%"
    return (
        f'<div class="prog-row">'
        f'<div class="prog-header"><span class="prog-name">{name}</span>'
        f'<span class="prog-pct">{lbl}</span></div>'
        f'<div class="prog-track"><div class="prog-fill" '
        f'style="width:{min(pct,100):.1f}%;background:{color}"></div></div></div>'
    )


def insight_badge(icon, title, sub, kind="orange"):
    return (
        f'<div class="insight-badge {kind}">'
        f'<div class="ib-icon">{icon}</div>'
        f'<div><div class="ib-title">{title}</div>'
        f'<div class="ib-sub">{sub}</div></div></div>'
    )


def stat_row(name, value, badge=None, badge_cls="teal"):
    badge_html = f'<span class="stat-badge {badge_cls}">{badge}</span>' if badge else ""
    return (
        f'<div class="stat-row">'
        f'<span class="stat-name">{name}</span>'
        f'<div style="display:flex;align-items:center;gap:0.5rem">'
        f'<span class="stat-value">{value}</span>{badge_html}</div></div>'
    )


def section_header(eyebrow, heading, sub=""):
    sub_html = f'<div class="sec-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="sec-eyebrow">{eyebrow}</div>'
        f'<div class="sec-heading">{heading}</div>{sub_html}',
        unsafe_allow_html=True,
    )


def panel_open():
    st.markdown('<div class="panel">', unsafe_allow_html=True)


def panel_close():
    st.markdown("</div>", unsafe_allow_html=True)


# ── Charts ────────────────────────────────────────────────────────────────────
def traffic_donut(data=None):
    data = df if data is None else data
    counts = data["high_traffic"].value_counts().reindex(["high", "low"]).fillna(0)
    rate = float(counts.get("high", 0) / max(counts.sum(), 1) * 100)
    fig = go.Figure(go.Pie(
        labels=["High Traffic", "Low Traffic"],
        values=[counts.get("high", 0), counts.get("low", 0)],
        hole=0.68,
        marker=dict(
            colors=[ORANGE, CARD2],
            line=dict(color=CARD, width=3),
        ),
        textinfo="percent",
        textfont=dict(color=TEXT, size=11),
        sort=False,
        hovertemplate="%{label}: <b>%{value:,} recipes</b> (%{percent})<extra></extra>",
    ))
    layout = _cl(h=300, t=34)
    layout.update(
        showlegend=False,
        annotations=[dict(
            text=f"<b style='color:{TEXT}'>{rate:.1f}%</b>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=22, color=TEXT),
        ), dict(
            text="High Traffic", x=0.5, y=0.38, showarrow=False,
            font=dict(size=10, color=MUTED),
        )],
    )
    fig.update_layout(**layout)
    return fig


def category_target_bars(data=None, top_n=12):
    data = df if data is None else data
    baseline = high_rate * 100
    counts = data["category"].value_counts().head(top_n).index
    rates = data[data["category"].isin(counts)].groupby("category")["high_traffic_label"].mean() * 100
    cats = rates.sort_values().index.tolist()
    act = rates.sort_values().values
    tgt = np.full(len(cats), baseline)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cats, x=tgt, orientation="h",
        name="Catalogue Baseline",
        marker=dict(color=BLUE_DK, line=dict(color="rgba(0,0,0,0)", width=0)),
        text=[f"{baseline:.1f}%" for _ in cats],
        textposition="outside",
        textfont=dict(color=DIM, size=9),
        hovertemplate="%{y}: baseline <b>%{x:.1f}%</b><extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=cats, x=act, orientation="h",
        name="Current View Rate",
        marker=dict(
            color=[ORANGE if v >= baseline else CYAN for v in act],
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        text=[f"{v:.1f}%" for v in act],
        textposition="outside",
        textfont=dict(color=MUTED, size=9),
        hovertemplate="%{y}: <b>%{x:.1f}% high traffic</b><extra></extra>",
    ))
    layout = _cl(h=60 * max(len(cats), 5) + 90, t=56)
    layout.update(
        barmode="group",
        bargap=0.18,
        title=dict(text="Regional Sales · Target vs Actual", font=dict(color=TEXT, size=13), x=0),
        xaxis=dict(title="High-Traffic Rate (%)", **_axis()),
        yaxis=dict(**_axis(grid=False, zero=False, tick_color=TEXT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(color=MUTED), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_layout(**layout)
    return fig


def band_mix_100(data=None):
    data = df if data is None else data
    try:
        bands = pd.qcut(data["calories"], 3, labels=["Light", "Moderate", "Rich"])
    except (ValueError, IndexError):
        bands = pd.cut(data["calories"], 3, labels=["Light", "Moderate", "Rich"])
    tmp = data.assign(band=bands)
    top = tmp["category"].value_counts().head(8).index
    sub = tmp[tmp["category"].isin(top)]
    tab = pd.crosstab(sub["category"], sub["band"], normalize="index") * 100
    tab = tab.reindex(columns=["Light", "Moderate", "Rich"]).fillna(0)
    cat_order = sub.groupby("category").size().sort_values().index
    tab = tab.reindex(cat_order)

    COLS = [CYAN, BLUE, ORANGE]
    fig = go.Figure()
    for col_name, col in zip(["Light", "Moderate", "Rich"], COLS):
        fig.add_trace(go.Bar(
            y=tab.index, x=tab[col_name], orientation="h", name=f"{col_name} Calories",
            marker=dict(color=col, line=dict(color=CARD, width=1.5)),
            text=[f"{v:.0f}%" if v >= 4 else "" for v in tab[col_name]],
            textposition="inside",
            textfont=dict(color=TEXT, size=9),
            hovertemplate="%{y} · %{x:.0f}% in <b>" + col_name + " band</b><extra></extra>",
        ))
    layout = _cl(h=60 * max(len(tab), 5) + 90, t=56)
    layout.update(
        barmode="stack",
        bargap=0.18,
        title=dict(text="Price-Band Mix · Calorie Bands by Category", font=dict(color=TEXT, size=13), x=0),
        xaxis=dict(title="Share of Category (%)", **_axis()),
        yaxis=dict(**_axis(grid=False, zero=False, tick_color=TEXT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(color=MUTED), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_layout(**layout)
    return fig


def calorie_line(data=None):
    data = df if data is None else data
    try:
        q = pd.qcut(data["calories"].rank(method="first"), 10)
    except (ValueError, IndexError):
        q = pd.cut(data["calories"], 10)
    grp = data.assign(bin=q).groupby("bin", observed=False)["high_traffic_label"].mean() * 100
    labels = []
    for b in grp.index:
        if hasattr(b, "mid"):
            labels.append(f"{b.mid:.0f}")
        else:
            labels.append(str(b))
    grp.index = labels
    peak_i = int(grp.idxmax())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grp.index, y=grp.values,
        mode="lines+markers",
        line=dict(color=CYAN, width=3),
        marker=dict(size=7, color=CYAN, line=dict(color=CARD, width=2)),
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.06)",
        hovertemplate="~%{x} kcal: <b>%{y:.1f}%</b><extra></extra>",
        name="Traffic Rate",
    ))
    fig.add_trace(go.Scatter(
        x=[peak_i], y=[float(grp.max())],
        mode="markers",
        marker=dict(size=14, color=ORANGE, symbol="diamond",
                    line=dict(color="#fff", width=1.5)),
        hovertemplate="<b>Peak {y:.1f}%</b> at ~%{x} kcal<extra></extra>",
        name="Peak",
        showlegend=False,
    ))
    fig.add_annotation(
        x=peak_i, y=float(grp.max()),
        text=f"Peak {grp.max():.1f}%",
        showarrow=False,
        yshift=18,
        font=dict(color=ORANGE, size=10),
    )
    layout = _cl(h=440, t=50)
    layout.update(
        title=dict(text="Revenue Trajectory · Traffic Rate across Calorie Bands", font=dict(color=TEXT, size=13), x=0),
        xaxis=dict(title="Avg. Calories (kcal band)", **_axis(grid=False, zero=False, tick_color=MUTED)),
        yaxis=dict(title="High-Traffic Rate (%)", range=[0, max(grp.max() * 1.25, 10)], **_axis()),
    )
    fig.update_layout(**layout)
    return fig


def category_bars(metric="count", top_n=12):
    if metric == "count":
        data   = df["category"].value_counts().head(top_n).iloc[::-1]
        title  = "Recipe Volume by Category"
        x_lbl  = "Number of Recipes"
        hover  = "%{y}: <b>%{x} recipes</b><extra></extra>"
        base_c = TEAL
    else:
        rates  = df.groupby("category")["high_traffic_label"].mean() * 100
        data   = rates.sort_values().tail(top_n)
        title  = "High-Traffic Rate by Category (%)"
        x_lbl  = "High-Traffic Rate (%)"
        hover  = "%{y}: <b>%{x:.1f}%</b><extra></extra>"
        base_c = ORANGE

    avg = float(data.mean())
    colors = [ORANGE if v >= avg else base_c for v in data.values]

    fig = go.Figure(go.Bar(
        x=data.values, y=data.index,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        text=[f"{v:.0f}" if metric == "count" else f"{v:.1f}%" for v in data.values],
        textposition="outside",
        textfont=dict(color=MUTED, size=10),
        hovertemplate=hover,
    ))
    avg_lbl = f"Avg {avg:.0f}" if metric == "count" else f"Avg {avg:.1f}%"
    fig.add_vline(
        x=avg,
        line=dict(dash="dot", color=YELLOW, width=1.5),
        annotation_text=avg_lbl,
        annotation_font_color=YELLOW,
        annotation_font_size=10,
        annotation_position="top right",
    )
    layout = _cl(h=400, t=50)
    layout.update(
        title=dict(text=title, font=dict(color=TEXT, size=13), x=0),
        xaxis=dict(title=x_lbl, **_axis()),
        yaxis=dict(**_axis(grid=False, zero=False, tick_color=TEXT)),
    )
    fig.update_layout(**layout)
    return fig


def servings_popularity():
    grp = df.groupby("servings")["high_traffic_label"].mean().reset_index()
    grp["rate"] = grp["high_traffic_label"] * 100
    grp["srv_str"] = grp["servings"].astype(int).astype(str) + " servings"

    fig = go.Figure(go.Bar(
        x=grp["srv_str"],
        y=grp["rate"],
        marker=dict(
            color=grp["rate"],
            colorscale=[[0, CARD2], [0.5, TEAL], [1, ORANGE]],
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        text=[f"{v:.1f}%" for v in grp["rate"]],
        textposition="outside",
        textfont=dict(color=MUTED, size=11),
        hovertemplate="%{x}: <b>%{y:.1f}% high traffic</b><extra></extra>",
    ))
    avg = float(grp["rate"].mean())
    fig.add_hline(y=avg, line=dict(dash="dot", color=YELLOW, width=1.5),
                  annotation_text=f"Avg {avg:.1f}%",
                  annotation_font_color=YELLOW, annotation_font_size=10)
    layout = _cl(h=300)
    layout.update(
        title=dict(text="High-Traffic Rate by Serving Size", font=dict(color=TEXT, size=13), x=0),
        xaxis=dict(title="Serving Size", **_axis(grid=False, zero=False)),
        yaxis=dict(title="High-Traffic Rate (%)", **_axis()),
    )
    fig.update_layout(**layout)
    return fig


def calories_traffic_box():
    fig = px.box(
        df, x="high_traffic", y="calories",
        color="high_traffic",
        color_discrete_map={"high": GREEN, "low": TEAL},
        labels={"high_traffic": "Traffic Type", "calories": "Calories (kcal)"},
        points=False,
        notched=True,
    )
    layout = _cl(h=360)
    layout.update(
        showlegend=False,
        title=dict(text="Calorie Distribution · High vs Low Traffic", font=dict(color=TEXT, size=13), x=0),
        xaxis=dict(title="Traffic Type", **_axis(grid=False, zero=False)),
        yaxis=dict(title="Calories (kcal)", **_axis()),
    )
    fig.update_layout(**layout)
    return fig


def nutrition_by_category():
    grp = df.groupby("category")[NUTRITION_FEATURES].mean()
    COLS = [ORANGE, TEAL, PURPLE, GREEN]
    fig = go.Figure()
    for feat, col in zip(NUTRITION_FEATURES, COLS):
        fig.add_trace(go.Bar(
            x=grp.index, y=grp[feat],
            name=feat.capitalize(),
            marker=dict(color=col, line=dict(color="rgba(0,0,0,0)", width=0)),
            hovertemplate=f"%{{x}} — {feat}: <b>%{{y:.1f}}</b><extra></extra>",
        ))
    layout = _cl(h=380, t=55)
    layout.update(
        barmode="group",
        title=dict(text="Mean Nutrition Profile by Category", font=dict(color=TEXT, size=13), x=0),
        xaxis=dict(title="Category", **_axis(grid=False, zero=False), tickangle=45),
        yaxis=dict(title="Mean Value", **_axis()),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    font=dict(color=MUTED), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_layout(**layout)
    return fig


def nutrition_compare():
    grp = df.groupby("high_traffic")[NUTRITION_FEATURES].mean().reindex(["low", "high"])
    grp.index = ["Low Traffic", "High Traffic"]
    COLS = [ORANGE, TEAL, PURPLE, GREEN]
    fig = go.Figure()
    for feat, col in zip(NUTRITION_FEATURES, COLS):
        fig.add_trace(go.Bar(
            x=grp.index, y=grp[feat],
            name=feat.capitalize(),
            text=[f"{v:.1f}" for v in grp[feat]],
            textposition="outside",
            textfont=dict(color=MUTED, size=10),
            marker=dict(color=col, line=dict(color="rgba(0,0,0,0)", width=0)),
            hovertemplate=f"{feat}: <b>%{{y:.1f}}</b><extra></extra>",
        ))
    layout = _cl(h=340, t=55)
    layout.update(
        barmode="group",
        title=dict(text="Mean Nutrition: High vs Low Traffic", font=dict(color=TEXT, size=13), x=0),
        xaxis=dict(**_axis(grid=False, zero=False)),
        yaxis=dict(title="Mean Value", **_axis()),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    font=dict(color=MUTED), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_layout(**layout)
    return fig


def feature_importance_fig():
    imp = MODEL_INFO["feature_importance"]
    top = imp.head(10)[::-1]
    fig = go.Figure(go.Bar(
        x=top["Importance"],
        y=top["Feature"],
        orientation="h",
        marker=dict(
            color=top["Importance"],
            colorscale=[[0, TEAL], [1, ORANGE]],
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        text=[f"{v:.4f}" for v in top["Importance"]],
        textposition="outside",
        textfont=dict(color=MUTED, size=10),
        hovertemplate="%{y}: <b>%{x:.4f}</b><extra></extra>",
    ))
    layout = _cl(h=400, t=50)
    layout.update(
        title=dict(text="Top 10 Feature Importances · Random Forest", font=dict(color=TEXT, size=13), x=0),
        xaxis=dict(title="Importance Score", **_axis()),
        yaxis=dict(**_axis(grid=False, zero=False, tick_color=TEXT)),
    )
    fig.update_layout(**layout)
    return fig


def model_compare_fig():
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    COLS = [ORANGE, TEAL]
    fig = go.Figure()
    for m, col in zip(MODEL_NAMES, COLS):
        res = MODEL_INFO["results"][m]
        fig.add_trace(go.Bar(
            x=metrics,
            y=[res[k] for k in metrics],
            name=m,
            text=[f"{res[k]:.3f}" for k in metrics],
            textposition="outside",
            textfont=dict(color=MUTED, size=10),
            marker=dict(color=col, line=dict(color="rgba(0,0,0,0)", width=0)),
            hovertemplate="%{x}: <b>%{y:.3f}</b><extra></extra>",
        ))
    fig.add_hline(
        y=RECALL_TARGET,
        line=dict(dash="dot", color=YELLOW, width=1.5),
        annotation_text=f"Recall target {RECALL_TARGET:.0%}",
        annotation_font_color=YELLOW,
        annotation_font_size=10,
        annotation_position="right",
    )
    layout = _cl(h=400, t=55)
    layout.update(
        barmode="group",
        title=dict(text="Model Performance Comparison", font=dict(color=TEXT, size=13), x=0),
        yaxis=dict(range=[0, 1.15], tickformat=".1%", title="Score", **_axis()),
        xaxis=dict(title="Metric", **_axis(grid=False, zero=False)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    font=dict(color=MUTED), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_layout(**layout)
    return fig


def confusion_fig(cm, title):
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=mmod.CONFUSION_LABELS,
        y=mmod.CONFUSION_LABELS,
        colorscale=[[0, CARD2], [0.4, "#7c3a0a"], [1, ORANGE]],
        text=cm,
        texttemplate="<b>%{text}</b>",
        textfont=dict(size=20, color=TEXT),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: <b>%{z}</b><extra></extra>",
        showscale=False,
    ))
    layout = _cl(h=340)
    layout.update(
        title=dict(text=title, font=dict(color=TEXT, size=13), x=0),
        xaxis=dict(title="Predicted Label", **_axis(grid=False, zero=False)),
        yaxis=dict(title="Actual Label", autorange="reversed",
                   **_axis(grid=False, zero=False)),
    )
    fig.update_layout(**layout)
    return fig


def roc_fig():
    COLS = [ORANGE, TEAL]
    fig = go.Figure()
    for m, col in zip(MODEL_NAMES, COLS):
        data = MODEL_INFO["roc_data"][m]
        fig.add_trace(go.Scatter(
            x=data["fpr"], y=data["tpr"],
            mode="lines",
            name=f"{m}  (AUC = {data['auc']:.3f})",
            line=dict(width=2.5, color=col),
            hovertemplate="FPR: %{x:.3f}  TPR: %{y:.3f}<extra></extra>",
            fill="tozeroy",
            fillcolor=col.replace(")", ",0.07)").replace("rgb", "rgba") if col.startswith("rgb") else
                       f"rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},0.06)",
        ))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dot", color=DIM, width=1.5))
    fig.add_annotation(x=0.62, y=0.35, text="Random Classifier",
                       showarrow=False, font=dict(color=DIM, size=10))
    layout = _cl(h=400, t=55)
    layout.update(
        title=dict(text="ROC Curves · Model Comparison", font=dict(color=TEXT, size=13), x=0),
        xaxis=dict(title="False Positive Rate", range=[-0.02, 1.02], **_axis()),
        yaxis=dict(title="True Positive Rate", range=[-0.02, 1.02], **_axis()),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    font=dict(color=MUTED), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_layout(**layout)
    return fig


def dist_histogram(col):
    fig = px.histogram(
        df, x=col, nbins=30,
        color_discrete_sequence=[TEAL],
        labels={col: col.capitalize(), "count": "Recipes"},
    )
    fig.update_traces(
        marker=dict(line=dict(color=CARD, width=0.5)),
        hovertemplate=f"{col}: %{{x}}<br>Count: <b>%{{y}}</b><extra></extra>",
    )
    layout = _cl(h=320)
    layout.update(
        showlegend=False,
        title=dict(text=f"Distribution of {col.capitalize()}", font=dict(color=TEXT, size=13), x=0),
        xaxis=dict(title=col.capitalize(), **_axis(grid=False, zero=False)),
        yaxis=dict(title="Recipe Count", **_axis()),
        bargap=0.05,
    )
    fig.update_layout(**layout)
    return fig


# ── Page renderers ────────────────────────────────────────────────────────────
def render_overview():
    baseline = high_rate * 100

    # ── Header filter bar ──────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    cat_f = f1.selectbox("Category · Segment", ["All"] + sorted(df["category"].unique()), key="ov_cat")
    srv_f = f2.selectbox("Servings · Period", ["All", 2, 4, 6], key="ov_srv")
    trf_f = f3.selectbox("Traffic · Region", ["All", "High", "Low"], key="ov_trf")

    # ── Filtered view ──────────────────────────────────────────────────────
    view = df
    if cat_f != "All":
        view = view[view["category"] == str(cat_f).lower()]
    if srv_f != "All":
        view = view[view["servings"] == int(srv_f)]
    if trf_f != "All":
        view = view[view["high_traffic"] == str(trf_f).lower()]

    if view.empty:
        st.warning("No recipes match the current filter combination — showing full catalogue.")
        view = df

    n_view    = len(view)
    rate_view = float(view["high_traffic_label"].mean() * 100)
    n_hi      = int(view["high_traffic_label"].sum())
    n_cat_view = view["category"].nunique()
    med_cal   = float(view["calories"].median())

    # ── KPI ribbon (7 compact ring cards) ──────────────────────────────────
    share_view  = n_view / n_recipes * 100
    rate_diff   = rate_view - baseline
    r_cls, r_arrow = ("up", "▲") if rate_diff >= 0 else ("down", "▼")
    recall_diff = LR_RECALL - RECALL_TARGET
    rec_cls, rec_arrow = ("up", "▲") if recall_diff >= 0 else ("down", "▼")

    kpis = [
        ring_kpi("High-Traffic Rate", f"{rate_view:.1f}%", rate_view, ORANGE,
                 delta=f"{r_arrow} {abs(rate_diff):.1f} pp vs baseline", delta_cls=r_cls,
                 sub=f"{n_hi:,} of {n_view:,} recipes"),
        ring_kpi("Model Recall", f"{LR_RECALL*100:.1f}%", LR_RECALL * 100,
                 GREEN if MODEL_ON_TARGET else YELLOW,
                 delta=f"{rec_arrow} {abs(recall_diff)*100:.1f} pp vs target", delta_cls=rec_cls,
                 sub=f"target ≥ {RECALL_TARGET:.0%}"),
        ring_kpi("Precision", f"{LR_PRECISION*100:.1f}%", LR_PRECISION * 100, CYAN,
                 delta="Correctly flagged", delta_cls="neutral",
                 sub="Logistic Regression"),
        ring_kpi("ROC AUC", f"{LR_AUC:.3f}", LR_AUC * 100, BLUE,
                 delta="Discriminative power", delta_cls="neutral",
                 sub="full dataset"),
        ring_kpi("Total Recipes", f"{n_view:,}", share_view, BLUE_DK,
                 delta=f"{share_view:.0f}% of catalogue", delta_cls="neutral",
                 sub="cleaned dataset"),
        ring_kpi("Categories", f"{n_cat_view}", n_cat_view / n_categories * 100, PURPLE,
                 delta=f"{n_cat_view / n_categories * 100:.0f}% of catalogue", delta_cls="neutral",
                 sub="food segments"),
        ring_kpi("Median Calories", f"{med_cal:,.0f} kcal",
                 med_cal / float(df["calories"].max()) * 100, TEAL,
                 delta="per recipe", delta_cls="neutral",
                 sub="after outlier removal"),
    ]
    cols = st.columns(7)
    for col, card in zip(cols, kpis):
        col.markdown(card, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Row 2 · Main charts ────────────────────────────────────────────────
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        panel_open()
        section_header("REGIONAL SALES · TARGET VS ACTUAL",
                       "High-Traffic Rate by Category",
                       "Dark = catalogue baseline; cyan/orange = current filter view. Orange exceeds baseline.")
        st.plotly_chart(category_target_bars(view), width='stretch')
        panel_close()

    with c2:
        panel_open()
        section_header("GLOBAL MARKET SHARE", "Traffic Split")
        st.plotly_chart(traffic_donut(view), width='stretch')
        st.markdown(
            prog_bar(f"High Traffic ({n_hi:,})", rate_view, ORANGE) +
            prog_bar(f"Low Traffic  ({n_view - n_hi:,})", 100 - rate_view, CYAN),
            unsafe_allow_html=True,
        )
        panel_close()

    with c3:
        panel_open()
        section_header("PRICE-BAND MIX", "Calorie Bands by Category",
                       "100% stacked share of light / moderate / rich calorie bands per category.")
        st.plotly_chart(band_mix_100(view), width='stretch')
        panel_close()

    # ── Row 3 · Detailed performance ───────────────────────────────────────
    cat_stat = view.groupby("category").agg(
        hi=("high_traffic_label", "sum"),
        rec=("high_traffic_label", "size"),
    )
    cat_stat["rate"]     = cat_stat["hi"] / cat_stat["rec"] * 100
    cat_stat["diff"]     = cat_stat["rate"] - baseline
    cat_stat["share"]    = cat_stat["rec"] / n_view * 100
    cat_stat             = cat_stat.sort_values("rate", ascending=False)

    c1, c2, c3 = st.columns([1.25, 1, 1])

    with c1:
        panel_open()
        section_header("YOY PRODUCT LINE PERFORMANCE", "Category Scorecard",
                       "▲/▼ show movement vs the catalogue baseline; share = % of recipes in the view.")
        body = ""
        for name, r in cat_stat.iterrows():
            up = r["diff"] >= 0
            vcol = "up" if up else "down"
            body += (
                f'<tr><td class="tk-cat">{str(name).capitalize()}</td>'
                f'<td class="muted">{int(r["rec"]):,}</td>'
                f'<td class="tk-rate">{r["rate"]:.1f}%</td>'
                f'<td><span class="{vcol}">{"▲" if up else "▼"} {abs(r["diff"]):.1f} pp</span></td>'
                f'<td class="share">{r["share"]:.1f}%</td></tr>'
            )
        st.markdown(
            '<div style="max-height:360px;overflow-y:auto;">'
            '<table class="dtable">'
            '<tr><th>Category</th><th>Recipes</th><th>High-Traffic</th><th>Δ vs Baseline</th><th>Share</th></tr>'
            f'{body}</table></div>',
            unsafe_allow_html=True,
        )
        panel_close()

    with c2:
        panel_open()
        section_header("REVENUE TRAJECTORY", "Traffic Rate across Calorie Bands",
                       "Sequential calorie bands — orange diamond marks the peak.")
        st.plotly_chart(calorie_line(view), width='stretch')
        panel_close()

    with c3:
        panel_open()
        section_header("RETAIL CHANNELS", "High-Traffic Share by Category")
        ct = view[view["high_traffic_label"] == 1]
        if len(ct):
            counts = ct["category"].value_counts()
            top5   = counts.head(5)
            tot    = counts.sum()
            GRAD = [ORANGE, CYAN, TEAL, PURPLE, YELLOW]
            bars = "".join(
                prog_bar(str(cat).capitalize(), v / tot * 100, GRAD[i],
                         pct_label=f"{v / tot * 100:.1f}%")
                for i, (cat, v) in enumerate(top5.items())
            )
        else:
            bars = '<div class="sec-sub">No high-traffic recipes in the current view.</div>'
        st.markdown(bars, unsafe_allow_html=True)

        st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
        section_header("INTELLIGENCE ALERTS", "Live View Signals")
        top_cat  = str(cat_stat["rate"].idxmax()).capitalize()
        top_pct  = float(cat_stat["rate"].max())
        low_cat  = str(cat_stat["rate"].idxmin()).capitalize()
        low_pct  = float(cat_stat["rate"].min())
        st.markdown(
            insight_badge("🚀", f"{top_cat} leads this view",
                          f"{top_pct:.1f}% high-traffic rate vs {baseline:.1f}% baseline — feature it on the homepage.",
                          "orange") +
            insight_badge("⚠️", f"{low_cat} needs attention",
                          f"Only {low_pct:.1f}% high-traffic rate. Review presentation before promoting.",
                          "red") +
            insight_badge("🎯", "Model ready for deployment",
                          f"Recall {LR_RECALL*100:.1f}% vs the {RECALL_TARGET:.0%} target.",
                          "green"),
            unsafe_allow_html=True,
        )
        panel_close()

    # ── Row 4 · Footer insights ────────────────────────────────────────────
    fast_cat = cat_stat["diff"].idxmax()
    fast_p   = float(cat_stat["diff"].max())
    on_lbl   = ("✓ On Target" if MODEL_ON_TARGET else "⚠ Below Target")
    st.markdown(f"""
<div class="bottom-strip">
  <div class="bs-item">
    <div class="bs-label">🟠 Current Run Rate</div>
    <div class="bs-value">{rate_view:.1f}%</div>
    <div class="bs-sub">high-traffic · baseline {baseline:.1f}%</div>
  </div>
  <div class="bs-item">
    <div class="bs-label">⭐ Top Category</div>
    <div class="bs-value" style="color:{GREEN}">{top_cat}</div>
    <div class="bs-sub">{top_pct:.1f}% high-traffic rate</div>
  </div>
  <div class="bs-item">
    <div class="bs-label">🚀 Fastest Growth</div>
    <div class="bs-value" style="color:{CYAN}">{str(fast_cat).capitalize()}</div>
    <div class="bs-sub">+{fast_p:.1f} pp vs baseline</div>
  </div>
  <div class="bs-item">
    <div class="bs-label">🎯 Model Recall</div>
    <div class="bs-value" style="color:{'#22c55e' if MODEL_ON_TARGET else '#eab308'}">{LR_RECALL*100:.1f}%</div>
    <div class="bs-sub">Target ≥ {RECALL_TARGET:.0%} — {on_lbl}</div>
  </div>
  <div class="bs-item">
    <div class="bs-label">⚠️ Watchlist</div>
    <div class="bs-value" style="color:{RED}">{low_cat}</div>
    <div class="bs-sub">{low_pct:.1f}% — below baseline</div>
  </div>
</div>""", unsafe_allow_html=True)


def render_data():
    raw      = dmod.load_raw()
    raw_rows = len(raw)
    n_removed = raw_rows - len(df)
    pct_removed = n_removed / raw_rows * 100

    # ── Cleaning stats KPIs ────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Raw Rows",          f"{raw_rows:,}",
                delta="Before cleaning", delta_cls="neutral",
                sub="source CSV", accent=MUTED), unsafe_allow_html=True)
    c2.markdown(kpi_card("Cleaned Rows",      f"{len(df):,}",
                delta="▼ After outlier removal", delta_cls="neutral",
                sub="analysis-ready", accent=TEAL), unsafe_allow_html=True)
    c3.markdown(kpi_card("Outliers Removed",  f"{n_removed:,}",
                delta=f"▼ {pct_removed:.1f}% of raw", delta_cls="warn",
                sub="z-score > 3", accent=YELLOW), unsafe_allow_html=True)
    c4.markdown(kpi_card("Missing Imputed",
                f"{int(raw[dmod.NUTRITION_FEATURES].isna().any(axis=1).sum()):,}",
                delta="Category-median fill", delta_cls="neutral",
                sub="nutrition values", accent=ORANGE), unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Distribution + category ────────────────────────────────────────────
    c1, c2 = st.columns([1.4, 1])
    with c1:
        panel_open()
        section_header("UNIVARIATE ANALYSIS", "Feature Distributions",
                        "Select a numeric feature to inspect its distribution.")
        dist_col = st.selectbox("Variable", NUTRITION_FEATURES, key="dist_var")
        st.plotly_chart(dist_histogram(dist_col), width='stretch')
        panel_close()
    with c2:
        panel_open()
        section_header("CATEGORY COUNT", "Recipes per Category")
        st.plotly_chart(category_bars(metric="count", top_n=10), width='stretch')
        panel_close()

    # ── Bivariate ──────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        panel_open()
        section_header("BIVARIATE · NUTRITION", "Nutrition Averages: High vs Low Traffic",
                        "Grouped bars show mean values for each nutrient split by traffic class.")
        st.plotly_chart(nutrition_compare(), width='stretch')
        panel_close()
    with c2:
        panel_open()
        section_header("BIVARIATE · CALORIES", "Calorie Distribution by Traffic Type",
                        "Notched box plots — overlapping notches indicate no significant median difference.")
        st.plotly_chart(calories_traffic_box(), width='stretch')
        panel_close()

    # ── Nutrition by category ──────────────────────────────────────────────
    panel_open()
    section_header("NUTRITION PROFILE", "Mean Nutrition Values by Category",
                   "Compare calories, carbohydrate, sugar, and protein across all food categories.")
    st.plotly_chart(nutrition_by_category(), width='stretch')
    panel_close()


def render_insights():
    baseline = high_rate * 100

    # ── KPI row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Baseline Rate",    f"{baseline:.1f}%",
                delta="Catalogue average", delta_cls="neutral",
                sub="high-traffic recipes", accent=MUTED), unsafe_allow_html=True)
    c2.markdown(kpi_card("Best Category",    BEST_CAT.capitalize(),
                delta=f"▲ {BEST_PCT:.1f}% high traffic", delta_cls="up",
                sub=f"+{BEST_PCT - baseline:.1f} pp vs baseline", accent=GREEN), unsafe_allow_html=True)
    c3.markdown(kpi_card("Weakest Category", WORST_CAT.capitalize(),
                delta=f"▼ {WORST_PCT:.1f}% high traffic", delta_cls="down",
                sub=f"{WORST_PCT - baseline:.1f} pp vs baseline", accent=RED), unsafe_allow_html=True)
    c4.markdown(kpi_card("Category Spread",
                f"{BEST_PCT - WORST_PCT:.1f} pp",
                delta="Best minus worst", delta_cls="warn",
                sub="popularity range", accent=ORANGE), unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Charts ─────────────────────────────────────────────────────────────
    c1, c2 = st.columns([1.4, 1])
    with c1:
        panel_open()
        section_header("CATEGORY PERFORMANCE", "High-Traffic Rate by Category",
                        "Orange bars exceed the catalogue average (dotted line).")
        st.plotly_chart(category_bars(metric="rate"), width='stretch')
        panel_close()
    with c2:
        panel_open()
        section_header("SERVING SIZE ANALYSIS", "Popularity Rate by Serving Size",
                        "Which serving-size group attracts the most traffic?")
        st.plotly_chart(servings_popularity(), width='stretch')
        panel_close()

    # ── Intelligence alerts / recommendations ──────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        panel_open()
        section_header("INTELLIGENCE ALERTS", "Key Findings from Data Analysis")
        st.markdown(
            insight_badge("🚀", "Homepage Prioritisation",
                          f"{BEST_CAT.capitalize()} leads at {BEST_PCT:.1f}% popularity — "
                          f"feature it prominently on the homepage for maximum traffic impact.",
                          "orange") +
            insight_badge("⚠️", f"Review {WORST_CAT.capitalize()}",
                          f"Only {WORST_PCT:.1f}% high-traffic rate vs {baseline:.1f}% baseline. "
                          f"Review presentation, imagery, and recipe selection before promoting.",
                          "red") +
            insight_badge("🍽️", "Serving Size Sweet Spot",
                          "Recipes designed for small groups (2–4 servings) consistently attract "
                          "higher traffic — prioritise these in homepage slots.",
                          "teal") +
            insight_badge("📈", "Model Confidence",
                          f"Logistic Regression achieves {LR_RECALL*100:.1f}% recall vs the "
                          f"{RECALL_TARGET:.0%} business target — safe to deploy for homepage selection.",
                          "green"),
            unsafe_allow_html=True,
        )
        panel_close()

    with c2:
        panel_open()
        section_header("BUSINESS RECOMMENDATIONS", "Actionable Decisions")
        st.markdown(
            insight_badge("🎯", "Nutrition-Driven Selection",
                          "High-protein, moderate-calorie recipes show the strongest engagement signal. "
                          "Use the predictor tool to pre-screen new recipes before publishing.",
                          "orange") +
            insight_badge("📊", "KPI Monitoring",
                          f"Track Recall (current {LR_RECALL*100:.1f}%) and Precision "
                          f"({LR_PRECISION*100:.1f}%) monthly. Retrain the model as new recipe "
                          "data accumulates to maintain performance.",
                          "teal") +
            insight_badge("🌟", "Content Strategy",
                          f"The {int(BEST_PCT - baseline):.0f} pp gap between top and average categories "
                          "signals that food-type matters as much as recipe quality. "
                          "Diversify the homepage to include high-performing categories daily.",
                          "green") +
            insight_badge("🔄", "Low-Traffic Recovery",
                          "Consider A/B testing new recipe photography and titles for "
                          f"{WORST_CAT.capitalize()} before concluding the category underperforms.",
                          "yellow"),
            unsafe_allow_html=True,
        )
        panel_close()

    # ── Bottom summary ─────────────────────────────────────────────────────
    st.markdown(f"""
<div class="bottom-strip">
  <div class="bs-item">
    <div class="bs-label">📌 Catalogue Baseline</div>
    <div class="bs-value">{baseline:.1f}%</div>
    <div class="bs-sub">Average high-traffic rate</div>
  </div>
  <div class="bs-item">
    <div class="bs-label">⭐ Best Category</div>
    <div class="bs-value" style="color:#22c55e">{BEST_CAT.capitalize()}</div>
    <div class="bs-sub">{BEST_PCT:.1f}% · +{BEST_PCT-baseline:.1f} pp above baseline</div>
  </div>
  <div class="bs-item">
    <div class="bs-label">⚠️ Weakest Category</div>
    <div class="bs-value" style="color:#ef4444">{WORST_CAT.capitalize()}</div>
    <div class="bs-sub">{WORST_PCT:.1f}% · {WORST_PCT-baseline:.1f} pp below baseline</div>
  </div>
  <div class="bs-item">
    <div class="bs-label">📈 Model Recommendation</div>
    <div class="bs-value">{"Deploy ✓" if MODEL_ON_TARGET else "Review ⚠"}</div>
    <div class="bs-sub">Recall {"≥" if MODEL_ON_TARGET else "<"} {RECALL_TARGET:.0%} target</div>
  </div>
</div>""", unsafe_allow_html=True)


def render_models():
    # ── KPI row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (lbl, val, d, dcls, sub, acc) in zip([c1, c2, c3, c4, c5], [
        ("LR Recall",    f"{LR_RESULTS['Recall']*100:.1f}%",
         f"{'▲' if MODEL_ON_TARGET else '▼'} {'On Target' if MODEL_ON_TARGET else 'Below Target'}",
         "up" if MODEL_ON_TARGET else "down", f"target ≥ {RECALL_TARGET:.0%}", GREEN if MODEL_ON_TARGET else YELLOW),
        ("LR Precision", f"{LR_RESULTS['Precision']*100:.1f}%",
         "Correct flags", "neutral", "Logistic Regression", TEAL),
        ("LR ROC AUC",   f"{LR_RESULTS['ROC AUC']:.3f}",
         "Discriminative power", "neutral", "Logistic Regression", ORANGE),
        ("RF Recall",    f"{MODEL_INFO['results'][MODEL_NAMES[-1]]['Recall']*100:.1f}%",
         "Random Forest", "neutral", "comparison model", PURPLE),
        ("RF ROC AUC",   f"{MODEL_INFO['results'][MODEL_NAMES[-1]]['ROC AUC']:.3f}",
         "Random Forest AUC", "neutral", "comparison model", PURPLE),
    ]):
        col.markdown(kpi_card(lbl, val, d, dcls, sub, acc), unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Model comparison chart ─────────────────────────────────────────────
    panel_open()
    section_header("PERFORMANCE COMPARISON", "All Metrics · Both Models",
                   "Dotted line shows the 80% Recall business target.")
    st.plotly_chart(model_compare_fig(), width='stretch')
    panel_close()

    # ── Confusion + ROC tabs ───────────────────────────────────────────────
    tab1, tab2 = st.tabs(["Confusion Matrix", "ROC Curves"])
    with tab1:
        c1, c2 = st.columns([0.3, 1])
        with c1:
            model = st.selectbox("Select model", MODEL_NAMES, key="cm_model")
        panel_open()
        section_header("CONFUSION MATRIX", f"{model} — Prediction Breakdown",
                       "Rows = actual labels, Columns = predicted labels.")
        st.plotly_chart(
            confusion_fig(MODEL_INFO["results"][model]["confusion_matrix"],
                          f"Confusion Matrix · {model}"),
            width='stretch',
        )
        panel_close()
    with tab2:
        panel_open()
        section_header("ROC CURVES", "True vs False Positive Rate",
                       "Area under the curve (AUC) measures overall discriminative ability. "
                       "Higher = better.")
        st.plotly_chart(roc_fig(), width='stretch')
        panel_close()

    # ── Feature importance + metrics table ─────────────────────────────────
    c1, c2 = st.columns([1.4, 1])
    with c1:
        panel_open()
        section_header("FEATURE IMPORTANCE", "Top 10 Predictors · Random Forest",
                       "Colour gradient: teal (lower importance) → orange (highest importance).")
        st.plotly_chart(feature_importance_fig(), width='stretch')
        panel_close()

    with c2:
        panel_open()
        section_header("METRICS TABLE", "Full Scorecard — Both Models")
        metric_keys = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "False Positive Rate"]
        rows = []
        for k in metric_keys:
            rows.append({
                "Metric": k,
                **{m: MODEL_INFO["results"][m][k] for m in MODEL_NAMES},
            })
        mdf = pd.DataFrame(rows).set_index("Metric")
        fmt = {m: "{:.3f}" for m in MODEL_NAMES}
        st.dataframe(mdf.style.format(fmt).background_gradient(
            cmap="YlOrRd", axis=1), width='stretch')
        panel_close()


def render_predict():
    section_header(
        "INTERACTIVE PREDICTOR", "Recipe Traffic Estimator",
        "Enter recipe attributes below. The Logistic Regression model will estimate "
        "the probability of attracting high site traffic.",
    )
    pipe = MODEL_INFO["results"][LR]["model"]

    default = {
        "calories":     float(df["calories"].median()),
        "carbohydrate": float(df["carbohydrate"].median()),
        "sugar":        float(df["sugar"].median()),
        "protein":      float(df["protein"].median()),
        "servings":     4,
        "category":     df["category"].mode()[0],
    }

    panel_open()
    with st.form("predictor"):
        c1, c2, c3 = st.columns(3)
        with c1:
            calories     = st.number_input("Calories (kcal)", 0.0, 3000.0, default["calories"], 5.0)
            carbohydrate = st.number_input("Carbohydrate (g)", 0.0, 500.0, default["carbohydrate"], 1.0)
        with c2:
            sugar   = st.number_input("Sugar (g)",   0.0, 300.0, default["sugar"],   1.0)
            protein = st.number_input("Protein (g)", 0.0, 200.0, default["protein"], 1.0)
        with c3:
            servings = st.selectbox("Servings", [1, 2, 4, 6], index=2)
            category = st.selectbox("Category", sorted(df["category"].unique()))
        submitted = st.form_submit_button("▶  Run Prediction")
    panel_close()

    if submitted:
        row = pd.DataFrame([{
            "calories": calories, "carbohydrate": carbohydrate,
            "sugar": sugar, "protein": protein,
            "servings": servings, "category": category,
        }])
        proba  = float(pipe.predict_proba(row)[:, 1][0])
        pred   = 1 if proba >= 0.5 else 0
        verdict = "High Traffic" if pred == 1 else "Low Traffic"
        v_col   = GREEN if pred == 1 else RED
        conf    = proba if pred == 1 else (1 - proba)

        panel_open()
        c1, c2, c3 = st.columns([1, 1, 1.6])
        with c1:
            st.markdown(
                kpi_card("Prediction", verdict,
                         delta=f"Confidence {conf*100:.1f}%",
                         delta_cls="up" if pred == 1 else "down",
                         sub="Logistic Regression", accent=v_col),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                kpi_card("High-Traffic Probability", f"{proba*100:.1f}%",
                         delta="≥ 50% → High Traffic prediction", delta_cls="neutral",
                         sub=f"Recipe: {category.capitalize()}, {servings} srv",
                         accent=ORANGE),
                unsafe_allow_html=True,
            )
        with c3:
            gage = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 32, "color": TEXT}},
                title={"text": "High-Traffic Probability", "font": {"size": 12, "color": MUTED}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickfont": {"color": MUTED}},
                    "bar":  {"color": v_col, "thickness": 0.28},
                    "bgcolor": "rgba(255,255,255,0.03)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50],   "color": "rgba(255,255,255,0.03)"},
                        {"range": [50, 100], "color": "rgba(34,197,94,0.06)"},
                    ],
                    "threshold": {
                        "line":  {"color": ORANGE, "width": 2.5},
                        "thickness": 0.9, "value": 50,
                    },
                },
            ))
            gage.update_layout(
                height=230,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=MUTED),
            )
            st.plotly_chart(gage, width='stretch')
        panel_close()

        # ── Interpretation badges ───────────────────────────────────────────
        panel_open()
        section_header("PREDICTION INTERPRETATION", "What This Means for Operations")
        if pred == 1:
            st.markdown(
                insight_badge("✅", "Recommend for Homepage",
                              f"This recipe has a {proba*100:.1f}% estimated probability of attracting "
                              f"high traffic — above the 50% decision threshold. "
                              f"Safe to feature on the homepage.", "green") +
                insight_badge("📌", "Category Signal",
                              f"{category.capitalize()} has a {_rates.get(category, 0)*100:.1f}% "
                              f"historical high-traffic rate in this dataset, "
                              f"supporting this prediction.", "teal"),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                insight_badge("⚠️", "Not Recommended for Homepage",
                              f"Estimated probability is {proba*100:.1f}% — below the 50% threshold. "
                              f"Consider adjusting the recipe's nutritional profile or targeting "
                              f"a different serving size.", "red") +
                insight_badge("💡", "Improvement Suggestion",
                              f"Recipes with higher protein and moderate calories tend to perform better. "
                              f"Consider revising this recipe or selecting a different category.", "orange"),
                unsafe_allow_html=True,
            )
        panel_close()


# ── Sidebar ───────────────────────────────────────────────────────────────────
apply_css()

with st.sidebar:
    # Logo / brand
    st.markdown(f"""
<div style="background:linear-gradient(135deg,{ORANGE} 0%,{ORANGE_DK} 100%);
            border-radius:10px;padding:0.85rem 1rem;margin-bottom:1.2rem;
            display:flex;align-items:center;gap:0.7rem;">
  <div style="font-size:1.5rem;">🍽️</div>
  <div>
    <div style="font-size:0.95rem;font-weight:800;color:#fff !important;">TASTY BYTES</div>
    <div style="font-size:0.72rem;color:rgba(255,255,255,0.75) !important;
                text-transform:uppercase;letter-spacing:0.08em;">Recipe Intelligence</div>
  </div>
</div>""", unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATE",
        ["📊  Overview Dashboard",
         "🔍  Data & EDA",
         "💡  Business Insights",
         "🤖  Model Performance",
         "🎯  Predictor"],
    )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='border-top:1px solid rgba(255,255,255,0.07);margin:0.5rem 0 0.75rem'></div>",
                unsafe_allow_html=True)

    # Model target card
    on_status = "✓ Status: On Target" if MODEL_ON_TARGET else "⚠ Status: Below Target"
    on_color  = GREEN if MODEL_ON_TARGET else YELLOW
    st.markdown(f"""
<div class="target-card">
  <div class="tc-eyebrow">BUSINESS TARGET</div>
  <div class="tc-value">{RECALL_TARGET:.0%}</div>
  <div class="tc-sub">Minimum Recall — High-Traffic Recipes</div>
  {prog_bar("Current Recall", LR_RECALL * 100,
             GREEN if MODEL_ON_TARGET else YELLOW,
             pct_label=f"{LR_RECALL*100:.1f}%")}
  <div class="{'tc-status-on' if MODEL_ON_TARGET else 'tc-status-off'}">{on_status}</div>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="margin-top:1rem;font-size:0.75rem;color:{DIM};text-align:center;">
  Data: recipe_site_traffic_2212.csv<br>
  Models: {' · '.join(MODEL_NAMES)}
</div>""", unsafe_allow_html=True)


# ── Top banner ────────────────────────────────────────────────────────────────
badge_txt = "Model: On Target ✓" if MODEL_ON_TARGET else "Model: Review Needed ⚠"
badge_col = GREEN if MODEL_ON_TARGET else YELLOW
st.html(f"""
<div class="top-banner">
  <div class="tb-left">
    <div class="tb-title">RECIPE SITE TRAFFIC DASHBOARD</div>
    <div class="tb-sub">Tasty Bytes · Predictive Analytics for Homepage Recipe Selection · FY2026</div>
  </div>
  <div class="tb-right">
    <span class="tb-pill">📁 {n_recipes:,} Recipes</span>
    <span class="tb-pill">🏷️ {n_categories} Categories</span>
    <span class="tb-pill">🎯 Recall Target: {RECALL_TARGET:.0%}</span>
    <span class="tb-badge" style="color:{badge_col};border-color:{badge_col}33;
          background:{badge_col}18">{badge_txt}</span>
    <button class="export-btn" type="button">⬇&nbsp; Export PDF</button>
  </div>
</div>
<script>
if (!window.__tbPrintWired) {{
  window.__tbPrintWired = true;
  document.addEventListener('click', function (ev) {{
    var t = ev.target;
    if (t && t.closest && t.closest('.export-btn')) {{
      ev.preventDefault();
      window.print();
    }}
  }});
}}
</script>""", unsafe_allow_javascript=True)


# ── Router ────────────────────────────────────────────────────────────────────
if   "Overview"   in page: render_overview()
elif "Data"       in page: render_data()
elif "Insights"   in page: render_insights()
elif "Model"      in page: render_models()
else:                      render_predict()
