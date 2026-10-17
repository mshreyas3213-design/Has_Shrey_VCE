# [TODO: Project Title, e.g. "GlucoTwin: A Digital Twin for Early Glucose-Spike Warning in Type 2 Diabetes"]

> Submission for the **Happiest Health Digital Twin Challenge 2026**
> (Reimagining & Reforming Healthcare in India Summit 2026)

---

## 1. Team Details

| Role | Name | Email | Phone |
|---|---|---|---|
| Team Leader | [TODO] | [TODO] | [TODO] |
| Member | [TODO] | [TODO] | [TODO] |

**College / Incubator:** [TODO: Full college name]
**Team Name:** Has_Shrey

## 2. Links

| Item | Link |
|---|---|
| Demo video (20+ min, unlisted YouTube) | [TODO: paste link] |
| Presentation (PDF) | [`docs/presentation.pdf`](docs/presentation.pdf) |
| Architecture diagram (PDF) | [`docs/architecture.pdf`](docs/architecture.pdf) |

---

## 3. Problem Statement and Healthcare Use Case

[TODO - Business section, to be written by the team's healthcare/business lead. Suggested points:]

- India has one of the largest populations of people with Type 2 Diabetes; much of the care is reactive (patients act only after a high-glucose episode or hospital visit).
- Repeated glucose spikes drive long-term complications (heart, kidney, eyes, nerves) and avoidable hospital costs.
- **Use case:** a *Digital Twin* of each patient that continuously fuses their health record with wearable and CGM data and warns the care team **up to 2 hours before** a glucose spike, so a small action (a short walk, meal adjustment, clinician call) can prevent it.
- **Users:** endocrinologists / physicians, diabetes care teams, hospitals, remote-monitoring programmes, and patients.
- **Expected impact:** [TODO: fewer emergency visits, better time-in-range, lower cost, scalable remote care].

## 4. Solution Overview

The prototype builds a virtual patient model that **fuses two data streams**:

| Stream | Content | Source |
|---|---|---|
| **Static / historical (EHR)** | Age, sex, BMI, HbA1c, fasting glucose, LDL, eGFR, years since diabetes diagnosis, hypertension, medication (insulin / metformin), family history, synthetic genetic risk score | Synthetic (`generate_data.py`) |
| **Dynamic / real-time** | Continuous glucose (CGM), heart rate, heart-rate variability (HRV), step count, sleep stages, logged meals; one reading every 15 min for 14 days | Synthetic (`generate_data.py`) |

**Predicted adverse event:** glucose rising **above 180 mg/dL at any time in the next 2 hours.**

A **doctor-facing dashboard** shows each virtual patient's live risk, glucose trend, an explanation of why the alert fired, and a what-if simulation.

## 5. Architecture

```
 Synthetic EHR  ─────────┐
 (static, per patient)   │
                         ├─►  Feature engineering  ─►  LightGBM classifier  ─►  2-hour spike risk
 Synthetic wearables ────┘    (past data only:         (trained on unseen-        │
 (CGM, HR, HRV, steps,         lags, trends,            patient split)            ▼
  sleep, meals)                rolling stats)                              Doctor dashboard
                                                                           (Streamlit: risk gauge,
                                                                            glucose chart, explanation,
                                                                            what-if twin)
```

Full diagram: [`docs/architecture.pdf`](docs/architecture.pdf)

## 6. Technical Stack, AI/ML Model and Framework

- **Language:** Python 3
- **Data:** pandas, numpy
- **Model:** LightGBM gradient-boosted trees (binary classifier)
- **Evaluation:** scikit-learn (ROC-AUC, PR-AUC, precision, recall)
- **Dashboard:** Streamlit + Plotly
- **Explainability:** LightGBM per-prediction feature contributions (shown in the dashboard)

**Key design choices**
- Features use **only past data**, so there is no leakage of future values.
- Train / validation / test split is **by patient** (136 / 24 / 40 patients), so the model is tested on people it has never seen.
- Ablation study proves the value of **fusing** the two streams.

### Results (on 40 unseen test patients)

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| EHR only | 0.837 | 0.617 |
| Wearables only | 0.970 | 0.923 |
| **Fused (EHR + wearables)** | **0.975** | **0.931** |

At alert threshold 0.50: precision **0.87**, recall **0.81**. At threshold 0.30: recall rises to **0.89** with precision 0.78 (the threshold is adjustable in the dashboard).

Charts: [`outputs/feature_importance.png`](outputs/feature_importance.png), [`outputs/roc_curve.png`](outputs/roc_curve.png)

> **Honest note:** all data is synthetic and the spike behaviour comes from our simulator, so these scores demonstrate that the pipeline works end to end, not clinical accuracy. Validation on real, consented data would be required before any clinical use.

## 7. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) regenerate the synthetic data and retrain the model
python generate_data.py
python train_model.py

# 3. Launch the doctor dashboard
python -m streamlit run app.py
```

The repository already contains the generated data (`data/`), the trained model (`models/`) and outputs (`outputs/`), so step 3 works immediately.

## 8. Repository Structure

```
├── generate_data.py     # synthetic EHR + wearable/CGM data generator
├── train_model.py       # feature engineering, training, evaluation, ablation
├── app.py               # Streamlit doctor dashboard
├── data/                # ehr.csv, wearables.csv (synthetic)
├── models/              # trained model (model.pkl)
├── outputs/             # metrics, charts, predictions
├── docs/                # presentation and architecture diagram (PDF)
├── requirements.txt
└── LICENSE
```

## 9. Data and Privacy

No real patient data is used. Both streams are **fully synthetic**, in line with the challenge's sandbox rules (DPDP Act / HIPAA considerations).

## 10. Limitations and Future Scope

- Replace synthetic data with open datasets (e.g. MIMIC-IV, public CGM datasets) and Synthea-generated records.
- Extend to hypertension and cardiovascular-risk twins.
- Personalised models per patient; real-time streaming from consumer wearables.
- Clinical validation and integration with hospital EHR systems.

## 11. Open-Source License

Released under the **MIT License**, see [`LICENSE`](LICENSE).
