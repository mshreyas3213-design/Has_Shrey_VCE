# GlucoTwin: A Digital Twin for Early Glucose-Spike Warning in Type 2 Diabetes
Submission for the Happiest Health Digital Twin Challenge 2026 (Reimagining & Reforming Healthcare in India Summit 2026)

## Abstract
Diabetes care is predominantly reactive, with interventions often occurring only after a hyperglycemic event. This project presents GlucoTwin, a proof-of-concept Digital Twin for patients with Type 2 Diabetes that shifts care from reactive to proactive. The system builds a virtual patient model by fusing static Electronic Health Records (EHR) with dynamic, real-time time-series data from wearables (Continuous Glucose Monitors, heart rate, sleep, and activity). Using a LightGBM gradient-boosted classifier, the model predicts the likelihood of a glucose spike (>180 mg/dL) up to two hours in advance. Evaluated on 40 unseen synthetic patients, the fused model achieved a ROC-AUC of 0.975 and PR-AUC of 0.931, demonstrating that combining historical and real-time streams significantly outperforms single-source models. The predictions are served via a Streamlit doctor-facing dashboard that provides explainable risk scores and what-if simulations, enabling timely, preventative clinical interventions.

## 1. Team Details
| Role | Name | Email | Phone |
| :--- | :--- | :--- | :--- |
| Team Leader | Shreyas M | mshreyas3213@gmail.com | +91 9618603169 |
| Member | Ambati Hasanthi | hasanthi136@gmail.com | +91 63037 34007 |

**College / Incubator:** Vasavi College of Engineering 
**Team Name:** Has_Shrey

## 2. Links
| Item | Link |
| :--- | :--- |
| Demo video (20+ min, unlisted YouTube) | **[INSERT YOUTUBE LINK HERE]** |
| Presentation (PDF) | docs/presentation.pdf |
| Architecture diagram (PDF) | docs/architecture.pdf |

## 3. Problem Statement and Healthcare Use Case
India has one of the largest populations of people with Type 2 Diabetes; much of the care is reactive (patients act only after a high-glucose episode or hospital visit). Repeated glucose spikes drive long-term complications (heart, kidney, eyes, nerves) and avoidable hospital costs.

**Use case:** A Digital Twin of each patient that continuously fuses their health record with wearable and CGM data and warns the care team up to 2 hours before a glucose spike, so a small action (a short walk, meal adjustment, clinician call) can prevent it.

**Users:** Endocrinologists / physicians, diabetes care teams, hospitals, remote-monitoring programmes, and patients.

**Expected impact:** Fewer emergency visits, better time-in-range, lower cost, and scalable remote care.

## 4. Solution Overview
The prototype builds a virtual patient model that fuses two data streams:

| Stream | Content | Source |
| :--- | :--- | :--- |
| **Static / historical (EHR)** | Age, sex, BMI, HbA1c, fasting glucose, LDL, eGFR, years since diabetes diagnosis, hypertension, medication (insulin / metformin), family history, synthetic genetic risk score | Synthetic (generate_data.py) |
| **Dynamic / real-time** | Continuous glucose (CGM), heart rate, heart-rate variability (HRV), step count, sleep stages, logged meals; one reading every 15 min for 14 days | Synthetic (generate_data.py) |

**Predicted adverse event:** Glucose rising above 180 mg/dL at any time in the next 2 hours.

A doctor-facing dashboard shows each virtual patient's live risk, glucose trend, an explanation of why the alert fired, and a what-if simulation.

## 5. Architecture
```text
 Synthetic EHR  ─────────┐
 (static, per patient)   │
                         ├─►  Feature engineering  ─►  LightGBM classifier  ─►  2-hour spike risk
 Synthetic wearables ────┘    (past data only:         (trained on unseen-        │
 (CGM, HR, HRV, steps,         lags, trends,            patient split)            ▼
  sleep, meals)                rolling stats)                              Doctor dashboard
                                                                           (Streamlit: risk gauge,
                                                                            glucose chart, explanation,
                                                                            what-if twin)
