"""Streamlit app: customer churn risk and business impact."""
import altair as alt
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.costs import CHURN_COST, OFFER_COST, SAVE_RATE, cost_breakdown
from src.data import load_data
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

st.set_page_config(page_title="Customer Churn Risk", page_icon="📉", layout="wide")


# ---------- Load model and data (cached so it only happens once) ----------
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


# ---------- Explanation: each feature's push vs an average customer ----------
feature_names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
train_mean = np.asarray(prep.transform(X_train)).mean(axis=0)


def original_feature(name):
    for f in FEATURES:
        if name == f or name.startswith(f + "_"):
            return f
    return name


def explain(customer):
    x = np.asarray(prep.transform(customer))[0]
    contrib = pd.Series(clf.coef_[0] * (x - train_mean), index=feature_names)
    return contrib.groupby(contrib.index.map(original_feature)).sum()


# ---------- Layout ----------
st.title("📉 Customer Churn Risk")

st.caption("Logistic Regression model with a cost-based decision threshold · IBM Telco dataset")

tab_predict, tab_impact, tab_about = st.tabs(["🔮 Predict", "💰 Business impact", "ℹ️ About"])

# ---------- Tab 1: Predict ----------
with tab_predict:
    st.subheader("Predict churn risk for one customer")
    cols = st.columns(3)
    inputs = {}
    with cols[0]:
        inputs["tenure"] = st.slider("Tenure (months)", 0, 72, 12)
        inputs["MonthlyCharges"] = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0)
        inputs["SeniorCitizen"] = 1 if st.selectbox("Senior citizen", ["No", "Yes"]) == "Yes" else 0
    for i, col in enumerate(CATEGORICAL_FEATURES):
        with cols[(i + 1) % 3]:
            inputs[col] = st.selectbox(col, sorted(df[col].unique()))

    customer = pd.DataFrame([inputs])[FEATURES]
    proba = model.predict_proba(customer)[0, 1]

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Churn probability", f"{proba:.0%}")
    m2.metric("Decision", "📨 Send retention offer" if proba >= THRESHOLD else "✅ No offer needed")
    st.caption(f"An offer is sent when the probability is at least {THRESHOLD:.2f} (chosen to minimize business cost).")

    st.markdown("**Top factors for this customer** (compared with an average customer)")
    contrib = explain(customer)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6)
    chart_df = top.rename("impact").rename_axis("feature").reset_index()
    chart_df["direction"] = np.where(chart_df["impact"] > 0, "Toward churn", "Toward staying")
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("impact:Q", title="Push on churn risk"),
        y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelFontSize=13, labelLimit=200)),
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(domain=["Toward churn", "Toward staying"], range=["#e4572e", "#4c9be8"]),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
        tooltip=["feature", alt.Tooltip("impact:Q", format=".2f")],
    ).properties(height=320)
    st.altair_chart(chart, width="stretch")
    st.caption("Positive = pushes toward churn · Negative = pushes toward staying")


# ---------- Tab 2: Business impact ----------
with tab_impact:
    st.subheader("How the decision threshold affects business cost")
    st.caption("Calculated on the 1,409 test-set customers the model never saw during training.")

    c1, c2, c3 = st.columns(3)
    offer = c1.number_input("Offer cost ($)", min_value=1, max_value=200, value=OFFER_COST)
    churn = c2.number_input("Lost customer cost ($)", min_value=10, max_value=5000, value=CHURN_COST)
    save = c3.slider("Offer success rate", 0.05, 1.0, SAVE_RATE, 0.05)
    t = st.slider("Decision threshold", 0.01, 0.95, float(THRESHOLD), 0.01)

    test_proba = model.predict_proba(X_test)[:, 1]

    def cost(th):
        return cost_breakdown(y_test, test_proba, th, offer, churn, save)

    thresholds = np.round(np.arange(0.01, 0.96, 0.01), 2)
    curve = pd.DataFrame([cost(th) for th in thresholds]).set_index("threshold")
    best_t = float(curve["total_cost"].idxmin())
    r = cost(t)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total cost at your threshold", f"${r['total_cost']:,.0f}")
    k2.metric("Churners caught", f"{r['TP']} / {r['TP'] + r['FN']}")
    k3.metric("False alarms", f"{r['FP']}")
    k4.metric("Cheapest threshold for these costs", f"{best_t:.2f}")

    st.line_chart(curve[["total_cost", "offer_cost", "churn_cost"]])

    strategies = {
        "Do nothing": cost(1.01),
        "Offer to everyone": cost(0.0),

        "Threshold 0.50": cost(0.5),
        f"Your threshold ({t:.2f})": r,
    }
    table = pd.DataFrame(strategies).T[["TP", "FP", "FN", "total_cost"]]
    st.dataframe(table.style.format({"total_cost": "${:,.0f}"}), width="stretch")

# ---------- Tab 3: About ----------
with tab_about:
    st.subheader("About this project")
    st.markdown(f"""
**Goal:** identify customers likely to churn and choose the decision threshold that minimizes business cost.

**Key finding:** at the default 0.5 threshold, the model costs *more* than offering everyone a discount.
With a cost-based threshold of **{THRESHOLD:.2f}**, it becomes the cheapest strategy, saving **12.7%** vs the
default threshold on unseen test data and catching **93%** of churners instead of 55%.

**Approach:** EDA → leakage audit → stratified split → pipeline (scaling + one-hot) →
model comparison with 5-fold CV (Logistic Regression, class weights, SMOTE, XGBoost) →
cost-based threshold → SHAP explanations.

**Assumptions:** offer cost ${OFFER_COST}, lost customer ${CHURN_COST}, offer saves {SAVE_RATE:.0%} of targeted churners.
Change them in the *Business impact* tab.

**Data:** IBM Telco Customer Churn (fictional sample dataset, 7,043 customers).
""")