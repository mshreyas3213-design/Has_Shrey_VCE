"""
Digital Twin Dashboard - Type 2 Diabetes glucose-spike early warning
Run:  python -m streamlit run app.py

Needs the files created by generate_data.py and train_model.py.
All data is SYNTHETIC. This is a research prototype, not medical advice.
"""
import json
import os
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from train_model import EHR_FEATURES, WEARABLE_FEATURES, add_features

st.set_page_config(page_title="Digital Twin - Diabetes", page_icon="🩺", layout="wide")

FEATURES = EHR_FEATURES + WEARABLE_FEATURES
SPIKE = 180

FRIENDLY = {
    "glucose": "Current glucose", "glucose_lag_15m": "Glucose 15 min ago",
    "glucose_lag_30m": "Glucose 30 min ago", "glucose_lag_60m": "Glucose 1 h ago",
    "glucose_lag_120m": "Glucose 2 h ago", "glucose_delta_30m": "Glucose change (30 min)",
    "glucose_delta_60m": "Glucose change (1 h)", "glucose_roll_mean_2h": "Avg glucose (2 h)",
    "glucose_roll_std_2h": "Glucose variability (2 h)", "glucose_roll_max_2h": "Peak glucose (2 h)",
    "heart_rate": "Heart rate", "hrv_rmssd": "HRV", "steps": "Steps (now)",
    "steps_last_1h": "Steps (last hour)", "hrv_mean_2h": "Avg HRV (2 h)",
    "hr_mean_2h": "Avg heart rate (2 h)", "sleep_stage": "Sleep stage",
    "meal_carbs_g": "Carbs just eaten", "carbs_last_2h": "Carbs eaten (2 h)",
    "carbs_last_4h": "Carbs eaten (4 h)", "sleep_hours_24h": "Sleep (last 24 h)",
    "deep_sleep_hours_24h": "Deep sleep (24 h)", "hour_sin": "Time of day", "hour_cos": "Time of day (2)",
    "age": "Age", "sex_male": "Sex (male)", "bmi": "BMI", "years_since_dx": "Years since diagnosis",
    "hba1c": "HbA1c", "fasting_glucose": "Fasting glucose", "family_history": "Family history",
    "hypertension": "Hypertension", "on_insulin": "On insulin", "on_metformin": "On metformin",
    "ldl": "LDL", "egfr": "eGFR (kidney)", "genetic_risk": "Genetic risk score",
}

# ---------------------------------------------------------------- loading
if not (os.path.exists("data/wearables.csv") and os.path.exists("models/model.pkl")):
    st.error("Missing files. Run `python generate_data.py` and then `python train_model.py` first.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")["model"]


@st.cache_data
def load_raw():
    ehr = pd.read_csv("data/ehr.csv")
    ehr["sex_male"] = (ehr["sex"] == "M").astype(int)
    wear = pd.read_csv("data/wearables.csv", parse_dates=["timestamp"])
    return ehr, wear.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)


@st.cache_data
def test_patient_ids():
    if os.path.exists("outputs/predictions.csv"):
        p = pd.read_csv("outputs/predictions.csv", usecols=["patient_id", "is_test_patient"])
        return sorted(p.loc[p["is_test_patient"] == 1, "patient_id"].unique())
    return []


@st.cache_data
def patient_frame(pid):
    model = load_model()
    ehr, wear = load_raw()
    d = wear[wear["patient_id"] == pid].reset_index(drop=True)
    d = d.merge(ehr.drop(columns=["sex"]), on="patient_id", how="left")
    d = add_features(d).dropna(subset=WEARABLE_FEATURES).reset_index(drop=True)
    d["risk"] = model.predict_proba(d[FEATURES])[:, 1]
    return d


model = load_model()
ehr_all, _ = load_raw()

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🩺 Digital Twin")
st.sidebar.caption("Virtual patient monitor - synthetic data")

tests = test_patient_ids()
scope = st.sidebar.radio("Patients", [f"Unseen test patients ({len(tests)})", "All patients"]
                         if tests else ["All patients"])
ids = tests if (tests and scope.startswith("Unseen")) else list(ehr_all["patient_id"])
pid = st.sidebar.selectbox("Select patient", ids)
threshold = st.sidebar.slider("Alert threshold (risk)", 0.10, 0.90, 0.50, 0.05,
                              help="Lower = catch more spikes but more false alarms.")
window_h = st.sidebar.select_slider("History window (hours)", [6, 12, 24, 48], value=24)
show_truth = st.sidebar.checkbox("Show what actually happened next (hidden from model)", True)

d = patient_frame(pid)
prof = ehr_all[ehr_all["patient_id"] == pid].iloc[0]

t_min = d["timestamp"].iloc[0] + timedelta(hours=12)
t_max = d["timestamp"].iloc[-1]
sub = d[d["timestamp"] >= t_min]
# default view: an "early warning" moment (glucose still below 170, risk high but not saturated)
early = sub[(sub["glucose"] < 170) & (sub["risk"] >= threshold) & (sub["risk"] < 0.95)]
pool = early if len(early) else sub
default_t = pool.loc[(pool["risk"] - 0.75).abs().idxmin(), "timestamp"]
now_t = st.sidebar.slider("Time ('now')", min_value=t_min.to_pydatetime(),
                          max_value=t_max.to_pydatetime(), value=default_t.to_pydatetime(),
                          step=timedelta(minutes=15), format="DD MMM, HH:mm", key=f"t_{pid}")
now_t = pd.Timestamp(now_t)
i = int(np.argmin(np.abs((d["timestamp"] - now_t).values.astype("int64"))))
row = d.iloc[[i]]
r = row.iloc[0]
risk = float(r["risk"])
level = "HIGH" if risk >= threshold else ("MODERATE" if risk >= threshold * 0.5 else "LOW")
color = {"HIGH": "#d62728", "MODERATE": "#ff9f1c", "LOW": "#2ca02c"}[level]

# ---------------------------------------------------------------- header
st.title("Digital Twin - Glucose Spike Early Warning")
st.caption(f"Patient **{pid}** | Type 2 Diabetes | Predicts glucose > {SPIKE} mg/dL in the next 2 hours "
           f"| Synthetic data - research prototype, not medical advice")

tab1, tab2, tab3 = st.tabs(["📡 Live monitor", "🧪 What-if twin", "ℹ️ About the model"])

# ================================================================ TAB 1
with tab1:
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Age / Sex", f"{int(prof['age'])} / {prof['sex']}")
    p2.metric("BMI", f"{prof['bmi']:.1f}")
    p3.metric("HbA1c", f"{prof['hba1c']:.1f} %")
    p4.metric("Years since diagnosis", f"{prof['years_since_dx']:.1f}")
    p5.metric("Medication", ("Insulin" if prof["on_insulin"] else "") +
              (" + " if prof["on_insulin"] and prof["on_metformin"] else "") +
              ("Metformin" if prof["on_metformin"] else "") or "None")

    left, right = st.columns([1, 2.4])

    with left:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=risk * 100, number={"suffix": "%"},
            title={"text": "Spike risk (next 2 h)"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                   "steps": [{"range": [0, threshold * 50], "color": "#e8f5e9"},
                             {"range": [threshold * 50, threshold * 100], "color": "#fff3e0"},
                             {"range": [threshold * 100, 100], "color": "#fdecea"}],
                   "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100}}))
        gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(gauge, width="stretch")
        st.markdown(f"<div style='text-align:center;font-size:1.4rem;font-weight:700;color:{color}'>"
                    f"{'🚨' if level == 'HIGH' else '⚠️' if level == 'MODERATE' else '✅'} {level} RISK</div>",
                    unsafe_allow_html=True)
        st.write("")
        c1, c2 = st.columns(2)
        c1.metric("Glucose now", f"{r['glucose']:.0f} mg/dL",
                  f"{r['glucose_delta_30m']:+.0f} vs 30 min", delta_color="inverse")
        c2.metric("Peak last 2 h", f"{r['glucose_roll_max_2h']:.0f} mg/dL")
        c3, c4 = st.columns(2)
        c3.metric("Heart rate", f"{r['heart_rate']:.0f} bpm")
        c4.metric("HRV", f"{r['hrv_rmssd']:.0f} ms")
        c5, c6 = st.columns(2)
        c5.metric("Steps (last hour)", f"{r['steps_last_1h']:.0f}")
        c6.metric("Sleep (last 24 h)", f"{r['sleep_hours_24h']:.1f} h")

    with right:
        hist = d[(d["timestamp"] <= now_t) & (d["timestamp"] >= now_t - timedelta(hours=window_h))]
        fut = d[(d["timestamp"] >= now_t) & (d["timestamp"] <= now_t + timedelta(hours=2))]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=hist["timestamp"], y=hist["glucose"], name="Glucose (CGM)",
                                 line=dict(color="#1f77b4", width=2.5)), secondary_y=False)
        if show_truth and len(fut):
            fig.add_trace(go.Scatter(x=fut["timestamp"], y=fut["glucose"], name="Actual next 2 h",
                                     line=dict(color="gray", dash="dot", width=2)), secondary_y=False)
        meals = hist[hist["meal_carbs_g"] > 0]
        fig.add_trace(go.Scatter(x=meals["timestamp"], y=meals["glucose"], mode="markers+text",
                                 name="Meal", marker=dict(symbol="diamond", size=11, color="#2ca02c"),
                                 text=[f"{c:.0f}g" for c in meals["meal_carbs_g"]],
                                 textposition="top center"), secondary_y=False)
        fig.add_trace(go.Scatter(x=hist["timestamp"], y=hist["risk"] * 100, name="Predicted risk %",
                                 line=dict(color="#ff7f0e", width=1.8), opacity=0.8), secondary_y=True)
        fig.add_hline(y=SPIKE, line_dash="dash", line_color="red",
                      annotation_text="180 mg/dL spike", secondary_y=False)
        fig.add_hline(y=threshold * 100, line_dash="dot", line_color="#ff7f0e",
                      annotation_text="alert threshold", secondary_y=True)
        fig.add_shape(type="line", x0=now_t, x1=now_t, y0=0, y1=1, yref="paper",
                      line=dict(color="black", width=2))
        fig.add_annotation(x=now_t, y=1.06, yref="paper", text="now", showarrow=False)
        fig.update_yaxes(title_text="Glucose (mg/dL)", secondary_y=False)
        fig.update_yaxes(title_text="Risk (%)", range=[0, 100], secondary_y=True)
        fig.update_layout(height=470, margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", y=-0.15), hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

    # ---- explainability + recommendation
    ex1, ex2 = st.columns([1.3, 1])
    with ex1:
        st.subheader("Why this risk score?")
        contrib = model.predict(row[FEATURES], pred_contrib=True)[0][:-1]
        s = pd.Series(contrib, index=FEATURES)
        top = s.reindex(s.abs().sort_values(ascending=False).index).head(7)[::-1]
        bar = go.Figure(go.Bar(
            x=top.values, y=[FRIENDLY.get(k, k) for k in top.index], orientation="h",
            marker_color=["#d62728" if v > 0 else "#2ca02c" for v in top.values]))
        bar.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="← lowers risk | raises risk →")
        st.plotly_chart(bar, width="stretch")
    with ex2:
        st.subheader("Suggested next step")
        st.caption("Conceptual clinical decision support - for clinician review")
        if level == "HIGH":
            st.error("**High spike risk in the next 2 hours.**\n\n"
                     "- Notify the care team / patient app\n"
                     "- Suggest a 10-15 min light walk and review of recent carbohydrate intake\n"
                     "- Recheck glucose in 30 minutes\n"
                     "- Clinician to review if alerts repeat daily")
        elif level == "MODERATE":
            st.warning("**Rising risk.** Keep monitoring; consider a short walk after the next meal.")
        else:
            st.success("**Low risk.** No action needed. Continue routine monitoring.")
        with st.expander("Wearable signals (selected window)"):
            sig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                subplot_titles=("Heart rate (bpm)", "HRV (ms)", "Steps / 15 min"))
            sig.add_trace(go.Scatter(x=hist["timestamp"], y=hist["heart_rate"], line=dict(color="#d62728")), 1, 1)
            sig.add_trace(go.Scatter(x=hist["timestamp"], y=hist["hrv_rmssd"], line=dict(color="#9467bd")), 2, 1)
            sig.add_trace(go.Bar(x=hist["timestamp"], y=hist["steps"], marker_color="#2ca02c"), 3, 1)
            sig.update_layout(height=420, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(sig, width="stretch")

# ================================================================ TAB 2
with tab2:
    st.subheader("What-if simulation on this virtual patient")
    st.caption("Change long-term patient factors and see how the twin's predicted risk responds. "
               "Approximate, feature-level simulation using the trained model (synthetic data).")
    w1, w2 = st.columns(2)
    new_hba1c = w1.slider("Long-term glucose control (HbA1c %)", 5.5, 12.0, float(prof["hba1c"]), 0.1)
    new_bmi = w2.slider("Body weight (BMI)", 18.0, 45.0, float(prof["bmi"]), 0.5)

    sim = row.copy()
    sim["hba1c"] = new_hba1c
    sim["bmi"] = new_bmi
    base_risk = risk
    sim_risk = float(model.predict_proba(sim[FEATURES])[:, 1][0])

    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline risk", f"{base_risk:.0%}")
    m2.metric("Scenario risk", f"{sim_risk:.0%}", f"{(sim_risk - base_risk) * 100:+.0f} pts",
              delta_color="inverse")
    m3.metric("Scenario level", "HIGH" if sim_risk >= threshold else
              "MODERATE" if sim_risk >= threshold * 0.5 else "LOW")
    cmp = go.Figure(go.Bar(x=["Baseline", "Scenario"], y=[base_risk * 100, sim_risk * 100],
                           marker_color=["#7f7f7f", "#d62728" if sim_risk >= threshold else "#2ca02c"]))
    cmp.add_hline(y=threshold * 100, line_dash="dot", annotation_text="alert threshold")
    cmp.update_layout(height=320, yaxis=dict(range=[0, 100], title="Spike risk (%)"),
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(cmp, width="stretch")
    st.info("Try: lower HbA1c by 1 point (better long-term control) or reduce BMI by a few points and "
            "watch the 2-hour spike risk fall. In this synthetic dataset HbA1c is the strongest driver, "
            "so the effect is larger than it would be in real patients.")

# ================================================================ TAB 3
with tab3:
    st.subheader("About this prototype")
    st.markdown(
        "**Digital Twin** = a virtual replica of a patient that fuses two data streams:\n"
        "- **Static / historical (EHR):** age, sex, BMI, HbA1c, years since diagnosis, medication, "
        "kidney function, family history, genetic risk score\n"
        "- **Dynamic / real-time (wearables + CGM):** glucose, heart rate, HRV, steps, sleep stages, meals\n\n"
        "**Model:** LightGBM classifier predicting whether glucose will exceed 180 mg/dL within 2 hours. "
        "Features use only past data. Evaluated on patients never seen in training.\n\n"
        "**Data:** fully synthetic (no real patient data), in line with the challenge's sandbox rules.")
    if os.path.exists("outputs/metrics.json"):
        m = json.load(open("outputs/metrics.json"))
        st.markdown("**Test results (unseen patients)**")
        ab = pd.DataFrame(m["ablation"]).T.rename(columns={"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"})
        st.dataframe(ab, width="stretch")
        a, b = st.columns(2)
        a.metric("Precision @0.5", m["at_threshold_0.50"]["precision"])
        b.metric("Recall @0.5", m["at_threshold_0.50"]["recall"])
    ci1, ci2 = st.columns(2)
    if os.path.exists("outputs/feature_importance.png"):
        ci1.image("outputs/feature_importance.png", caption="Top features")
    if os.path.exists("outputs/roc_curve.png"):
        ci2.image("outputs/roc_curve.png", caption="ROC curve")
    st.warning("Limitations: synthetic data means scores show the pipeline works, not clinical accuracy. "
               "Real deployment would need validation on real, consented data and clinical oversight.")