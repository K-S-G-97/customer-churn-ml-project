"""Streamlit app: customer churn risk dashboard (English / Thai / Burmese)."""
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from app.translations import LABELS, OPTION_LABELS, SECTION_NAMES, TEXT
from src.costs import CHURN_COST, OFFER_COST, SAVE_RATE, cost_breakdown
from src.data import load_data
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CYAN, VIOLET, ROSE, AMBER, GREEN = "#22D3EE", "#A78BFA", "#F43F5E", "#F59E0B", "#22C55E"
RISK_COLORS = {"high": ROSE, "medium": AMBER, "low": GREEN}

# code: (emoji flag, language name, country code for the flag image)
FLAGS = {"en": ("🇬🇧", "English", "gb"), "th": ("🇹🇭", "ไทย", "th"), "my": ("🇲🇲", "မြန်မာ", "mm")}

st.set_page_config(page_title="Customer Churn Risk", page_icon="📉", layout="wide")


# =====================================================================
# Dashboard styling: black + cyan, with hover effects
# =====================================================================
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&display=swap');

.stApp {
  background-image:
    radial-gradient(circle at 100% 0%, rgba(34, 211, 238, 0.22), transparent 40%),
    radial-gradient(circle at 0% 100%, rgba(167, 139, 250, 0.10), transparent 35%),
    linear-gradient(rgba(34, 211, 238, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(34, 211, 238, 0.05) 1px, transparent 1px);
  background-size: 100% 100%, 100% 100%, 48px 48px, 48px 48px;
  background-attachment: fixed;
}

/* ---------- Title ---------- */
.dash-title {font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 2.6rem; line-height: 1.15;
  margin: 0; transition: text-shadow 0.3s ease;}
.dash-title:hover {text-shadow: 0 0 24px rgba(34, 211, 238, 0.55);}
.dash-sub {opacity: 0.75; margin: 6px 0 0;}

/* ---------- Language flag (circle, sits right next to the dropdown) ---------- */
.flag {width: 32px; height: 32px; border-radius: 50%; object-fit: cover; display: block; margin-left: auto;
  box-shadow: 0 0 0 2px #22D3EE, 0 0 12px rgba(34, 211, 238, 0.5); transition: transform 0.25s ease;}
.flag:hover {transform: scale(1.15) rotate(-8deg);}

/* ---------- Result pills ---------- */
.pills {display: flex; flex-wrap: wrap; gap: 10px; justify-content: flex-end; margin-top: 8px;}
.pill {display: flex; border: 1px solid #22D3EE; font-family: 'Poppins', sans-serif; font-size: 0.9rem;
  transition: transform 0.2s ease, box-shadow 0.2s ease;}
.pill b {background: #22D3EE; color: #041014; padding: 7px 16px; font-weight: 600;}
.pill span {padding: 7px 16px; background: rgba(128, 128, 128, 0.10); font-weight: 500;}
.pill:hover {transform: translateY(-3px); box-shadow: 0 0 18px rgba(34, 211, 238, 0.55);}

/* ---------- Panel title tab ---------- */
.panel-title {display: inline-block; background: #22D3EE; color: #041014; font-family: 'Poppins', sans-serif;
  font-weight: 600; font-size: 0.85rem; letter-spacing: 0.06em; text-transform: uppercase;
  padding: 7px 18px; margin-bottom: 6px; transition: box-shadow 0.2s ease;}
.panel-title:hover {box-shadow: 0 0 16px rgba(34, 211, 238, 0.6);}

/* ---------- KPI cards ---------- */
.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 8px 0 18px;}
.kpi {border: 1px solid rgba(34, 211, 238, 0.45); background: rgba(128, 128, 128, 0.06);
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;}
.kpi:hover {transform: translateY(-4px); border-color: #22D3EE;
  box-shadow: 0 0 0 1px #22D3EE, 0 12px 32px rgba(34, 211, 238, 0.25);}
.kpi-head {background: #22D3EE; color: #041014; text-align: center; font-family: 'Poppins', sans-serif;
  font-weight: 600; font-size: 0.82rem; letter-spacing: 0.05em; text-transform: uppercase; padding: 8px 10px;}
.kpi-body {padding: 16px 18px 6px; display: flex; align-items: center; justify-content: center; gap: 14px;}
.kpi-arrow {color: #22D3EE; font-size: 1.9rem; font-weight: 700; line-height: 1; transition: transform 0.2s ease;}
.kpi:hover .kpi-arrow {transform: translateX(6px);}
.kpi-value {font-family: 'Poppins', sans-serif; font-size: 1.75rem; font-weight: 600;}
.kpi-sub {font-size: 0.8rem; opacity: 0.7; text-align: center; padding: 4px 12px 14px;}

/* ---------- Risk gauge ---------- */
.risk-body {display: flex; align-items: center; gap: 22px; padding: 18px 20px;}
.gauge {position: relative; width: 120px; height: 120px; flex-shrink: 0; transition: transform 0.3s ease;}
.kpi:hover .gauge {transform: rotate(-6deg) scale(1.04);}
.gauge-ring {position: absolute; inset: 0; border-radius: 50%;
  -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 12px), #000 calc(100% - 11px));
  mask: radial-gradient(farthest-side, transparent calc(100% - 12px), #000 calc(100% - 11px));}
.gauge-label {position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  font-family: 'Poppins', sans-serif; font-size: 1.8rem; font-weight: 700;}
.risk-level {font-family: 'Poppins', sans-serif; font-size: 1.5rem; font-weight: 700; color: var(--accent);}
.risk-decision {font-size: 1.05rem; margin-top: 4px;}
.risk-note {font-size: 0.85rem; opacity: 0.7; margin-top: 6px;}

/* ---------- Hover effects on Streamlit's own widgets ----------
   These use Streamlit's internal names. If a future version renames them,
   the rule is simply ignored and nothing breaks. */
[data-testid="stVerticalBlockBorderWrapper"] {border-color: rgba(34, 211, 238, 0.35) !important;
  background: rgba(128, 128, 128, 0.04); transition: border-color 0.2s ease, box-shadow 0.2s ease;}
[data-testid="stVerticalBlockBorderWrapper"]:hover {border-color: #22D3EE !important;
  box-shadow: 0 0 24px rgba(34, 211, 238, 0.18);}

[data-baseweb="select"] > div, [data-baseweb="input"] {transition: border-color 0.2s ease, box-shadow 0.2s ease;}
[data-baseweb="select"] > div:hover, [data-baseweb="input"]:hover {border-color: #22D3EE !important;
  box-shadow: 0 0 12px rgba(34, 211, 238, 0.35);}

[data-testid="stButtonGroup"] button {transition: color 0.15s ease, background 0.15s ease, transform 0.15s ease;}
[data-testid="stButtonGroup"] button:hover {color: #22D3EE !important; border-color: #22D3EE !important;
  background: rgba(34, 211, 238, 0.10) !important; transform: translateY(-1px);}

[data-testid="stSlider"] [role="slider"] {transition: box-shadow 0.2s ease;}
[data-testid="stSlider"] [role="slider"]:hover {box-shadow: 0 0 0 7px rgba(34, 211, 238, 0.25);}

[data-baseweb="tab"] {transition: color 0.2s ease;}
[data-baseweb="tab"]:hover {color: #22D3EE !important;}

[data-testid="stDataFrame"] {transition: box-shadow 0.2s ease;}
[data-testid="stDataFrame"]:hover {box-shadow: 0 0 18px rgba(34, 211, 238, 0.2);}

@media (max-width: 640px) {
  .dash-title {font-size: 1.9rem;}
  .pills {justify-content: flex-start;}
  .risk-body {flex-direction: column; text-align: center;}
}
</style>
""")


# =====================================================================
# Load model and data (cached so it only happens once)
# =====================================================================
@st.cache_resource
def load_model():
    bundle = joblib.load(ROOT / "models" / "churn_model.joblib")
    return bundle["model"], bundle["threshold"]


@st.cache_data
def load_dataset():
    df = load_data(ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, stratify=df[TARGET], random_state=42
    )
    return df, X_train, X_test, y_train, y_test


model, THRESHOLD = load_model()
df, X_train, X_test, y_train, y_test = load_dataset()
prep = model.named_steps["prep"]
clf = model.named_steps["model"]


# =====================================================================
# Header: language dropdown with flag, title, result pills
# =====================================================================
h_left, h_right = st.columns([3, 2], vertical_alignment="bottom")

with h_right:
    _, flag_col, select_col = st.columns([5, 1, 3], vertical_alignment="center")
    with select_col:
        lang = st.selectbox(
            "Language", list(FLAGS),
            format_func=lambda code: f"{FLAGS[code][0]}  {FLAGS[code][1]}",
            label_visibility="collapsed", key="language",
        )
    with flag_col:
        st.html(f'<img class="flag" src="https://flagcdn.com/w80/{FLAGS[lang][2]}.png" alt="{FLAGS[lang][1]} flag">')


def tr(key, **values):
    """Return the text for `key` in the current language, filling in any {placeholders}."""
    return TEXT[lang][key].format(**values)


def opt(value):
    """Translate a data value (like 'Yes') for display. The model still receives the English value."""
    return OPTION_LABELS[lang].get(value, value)


pills = "".join(
    f'<div class="pill"><b>{label}</b><span>{value}</span></div>'
    for label, value in [
        (tr("pill_threshold"), f"{THRESHOLD:.2f}"),
        (tr("pill_recall"), "93%"),
        (tr("pill_saving"), "12.7%"),
    ]
)
with h_right:
    st.html(f'<div class="pills">{pills}</div>')
    st.caption(tr("theme_hint"))
with h_left:
    st.html(f'<div class="dash-title">{tr("title")}</div><p class="dash-sub">{tr("hero_sub")}</p>')


# =====================================================================
# Helpers
# =====================================================================
feature_names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
train_mean = np.asarray(prep.transform(X_train)).mean(axis=0)

SECTIONS = {
    "customer": ["SeniorCitizen", "Partner", "Dependents"],
    "account": ["tenure", "Contract", "PaymentMethod", "PaperlessBilling", "MonthlyCharges"],
    "services": ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                 "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"],
}


def original_feature(name):
    for f in FEATURES:
        if name == f or name.startswith(f + "_"):
            return f
    return name


def explain(customer):
    x = np.asarray(prep.transform(customer))[0]
    contrib = pd.Series(clf.coef_[0] * (x - train_mean), index=feature_names)
    return contrib.groupby(contrib.index.map(original_feature)).sum()


def input_widget(col):
    """Show the right input for each feature and return the chosen (English) value."""
    label = LABELS[lang].get(col, col)
    if col == "tenure":
        return st.slider(label, 0, 72, 12, key=col)
    if col == "MonthlyCharges":
        return st.slider(label, 18.0, 120.0, 70.0, key=col)
    if col == "SeniorCitizen":
        picked = st.segmented_control(label, ["No", "Yes"], default="No", format_func=opt, key=col)
        return 1 if picked == "Yes" else 0
    options = sorted(df[col].unique())
    if len(options) <= 3:
        picked = st.segmented_control(label, options, default=options[0], format_func=opt, key=col)
        return picked if picked is not None else options[0]
    return st.selectbox(label, options, format_func=opt, key=col)


def panel_title(text):
    """Cyan title tab at the top of a panel."""
    st.html(f'<div class="panel-title">{text}</div>')


def kpi_cards(items):
    """Grid of KPI cards. `items` is a list of (header, value, subtitle)."""
    cards = "".join(
        f'<div class="kpi"><div class="kpi-head">{head}</div>'
        f'<div class="kpi-body"><span class="kpi-arrow">➜</span><span class="kpi-value">{value}</span></div>'
        f'<div class="kpi-sub">{sub}</div></div>'
        for head, value, sub in items
    )
    return f'<div class="kpis">{cards}</div>'


def risk_card(proba):
    """KPI-style card with a circular risk gauge."""
    level_key = "high" if proba >= 0.5 else "medium" if proba >= THRESHOLD else "low"
    color = RISK_COLORS[level_key]
    decision = tr("send_offer") if proba >= THRESHOLD else tr("no_offer")
    return (
        f'<div class="kpi" style="--accent:{color}"><div class="kpi-head">{tr("panel_risk")}</div>'
        f'<div class="risk-body"><div class="gauge">'
        f'<div class="gauge-ring" style="background:conic-gradient({color} {proba * 100:.1f}%, rgba(128,128,128,0.2) 0)"></div>'
        f'<div class="gauge-label">{proba:.0%}</div></div>'
        f'<div><div class="risk-level">{tr(level_key)}</div>'
        f'<div class="risk-decision">{decision}</div>'
        f'<div class="risk-note">{tr("risk_note", t=THRESHOLD)}</div></div></div></div>'
    )


# =====================================================================
# Tabs
# =====================================================================
tab_predict, tab_impact, tab_about = st.tabs([tr("tab_predict"), tr("tab_impact"), tr("tab_about")])

# ---------- Tab 1: Predict ----------
with tab_predict:
    inputs = {}
    for column, (section, features) in zip(st.columns(3), SECTIONS.items()):
        with column:
            with st.container(border=True):
                panel_title(SECTION_NAMES[lang][section])
                for f in features:
                    inputs[f] = input_widget(f)

    customer = pd.DataFrame([inputs])[FEATURES]
    proba = float(model.predict_proba(customer)[0, 1])

    left, right = st.columns([2, 3])
    with left:
        st.html(risk_card(proba))
        st.caption(tr("threshold_caption", t=THRESHOLD))
    with right:
        with st.container(border=True):
            panel_title(tr("panel_factors"))
            contrib = explain(customer)
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6)
            chart_df = top.rename("impact").rename_axis("feature").reset_index()
            chart_df["feature"] = chart_df["feature"].map(lambda f: LABELS[lang].get(f, f))
            chart_df["direction"] = np.where(chart_df["impact"] > 0, tr("toward_churn"), tr("toward_staying"))
            chart = alt.Chart(chart_df).mark_bar(cornerRadius=3).encode(
                x=alt.X("impact:Q", title=tr("push_axis")),
                y=alt.Y("feature:N", sort="-x", title=None,
                        axis=alt.Axis(labelOverlap=False, labelFontSize=13, labelLimit=220)),
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(domain=[tr("toward_churn"), tr("toward_staying")], range=[ROSE, CYAN]),
                    legend=alt.Legend(title=None, orient="bottom"),
                ),
                tooltip=["feature", alt.Tooltip("impact:Q", format=".2f")],
            ).properties(height=280)
            st.altair_chart(chart, width="stretch")
            st.caption(tr("factors_caption"))

# ---------- Tab 2: Business impact ----------
with tab_impact:
    st.caption(tr("impact_caption"))

    with st.container(border=True):
        panel_title(tr("panel_assumptions"))
        c1, c2, c3, c4 = st.columns(4)
        offer = c1.number_input(tr("offer_cost"), min_value=1, max_value=200, value=OFFER_COST, key="offer")
        churn = c2.number_input(tr("churn_cost"), min_value=10, max_value=5000, value=CHURN_COST, key="churn")
        save = c3.slider(tr("save_rate"), 0.05, 1.0, SAVE_RATE, 0.05, key="save")
        user_t = c4.slider(tr("threshold"), 0.01, 0.95, float(THRESHOLD), 0.01, key="user_t")

    test_proba = model.predict_proba(X_test)[:, 1]

    def cost(th):
        return cost_breakdown(y_test, test_proba, th, offer, churn, save)

    thresholds = np.round(np.arange(0.01, 0.96, 0.01), 2)
    curve = pd.DataFrame([cost(th) for th in thresholds]).set_index("threshold")
    best_t = float(curve["total_cost"].idxmin())
    r, base, best = cost(user_t), cost(0.5), cost(best_t)

    st.html(kpi_cards([
        (tr("kpi_total"), f"${r['total_cost']:,.0f}", tr("sub_total", v=f"${base['total_cost']:,.0f}")),
        (tr("kpi_caught"), f"{r['TP']} / {r['TP'] + r['FN']}", tr("sub_caught", v=f"{r['TP'] / (r['TP'] + r['FN']):.0%}")),
        (tr("kpi_false"), f"{r['FP']}", tr("sub_false", v=f"${r['FP'] * offer:,.0f}")),
        (tr("kpi_best"), f"{best_t:.2f}", tr("sub_best", v=f"${best['total_cost']:,.0f}")),
    ]))

    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            panel_title(tr("panel_curve"))
            lines = curve[["total_cost", "offer_cost", "churn_cost"]].rename(columns={
                "total_cost": tr("line_total"), "offer_cost": tr("line_offer"), "churn_cost": tr("line_churn"),
            })
            st.line_chart(lines, color=[CYAN, VIOLET, ROSE], height=320)
    with right:
        with st.container(border=True):
            panel_title(tr("panel_strategies"))
            strategies = {
                tr("s_nothing"): cost(1.01),
                tr("s_everyone"): cost(0.0),
                tr("s_050"): base,
                tr("s_yours", t=user_t): r,
            }
            table = pd.DataFrame(strategies).T[["TP", "FP", "FN", "total_cost"]]
            table = table.rename(columns={"total_cost": tr("col_total")})
            st.dataframe(table.style.format({tr("col_total"): "${:,.0f}"}), width="stretch")

# ---------- Tab 3: About ----------
with tab_about:
    with st.container(border=True):
        panel_title(tr("about_header"))
        st.markdown(tr("about_body", threshold=THRESHOLD, offer=OFFER_COST, churn=CHURN_COST, save=SAVE_RATE))