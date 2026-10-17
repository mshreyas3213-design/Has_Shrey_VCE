"""
Synthetic data generator - Digital Twin Challenge (Type 2 Diabetes)

Creates two files in ./data:
  ehr.csv         -> static/historical data, one row per virtual patient
  wearables.csv   -> time-series data (every 15 min for 14 days), one row per patient per timestamp

Target label (in wearables.csv):
  spike_next_2h = 1 if glucose goes above 180 mg/dL at any point in the next 2 hours

All data is SYNTHETIC. No real patient data is used (complies with the challenge's sandbox rules).
Run:  python generate_data.py
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N_PATIENTS = 200
DAYS = 14
STEP_MIN = 15                      # sampling interval in minutes
STEPS_PER_DAY = 24 * 60 // STEP_MIN  # 96
HORIZON = 2 * 60 // STEP_MIN       # 2 hours = 8 steps
SPIKE_THRESHOLD = 180              # mg/dL

rng = np.random.default_rng(SEED)
os.makedirs("data", exist_ok=True)

# ------------------------------------------------------------------
# 1. STATIC DATA (synthetic EHR)
# ------------------------------------------------------------------
def make_ehr(n):
    age = rng.integers(30, 80, n)
    sex = rng.choice(["M", "F"], n)
    bmi = np.clip(rng.normal(28.5, 4.5, n), 18.5, 45).round(1)
    years_dx = np.clip(rng.gamma(2.0, 3.0, n), 0.2, 30).round(1)
    # HbA1c rises a bit with BMI and years since diagnosis
    hba1c = np.clip(rng.normal(7.2, 1.0, n) + 0.03 * (bmi - 28) + 0.04 * years_dx, 5.8, 12.0).round(1)
    fasting_glucose = np.clip(28.7 * hba1c - 46.7 - 55 + rng.normal(0, 12, n), 80, 280).round(0)
    family_history = rng.choice([0, 1], n, p=[0.45, 0.55])
    hypertension = (rng.random(n) < (0.25 + 0.01 * (age - 30))).astype(int)
    on_insulin = (rng.random(n) < np.clip(0.05 + 0.04 * years_dx, 0, 0.7)).astype(int)
    on_metformin = (rng.random(n) < 0.85).astype(int)
    ldl = np.clip(rng.normal(115, 30, n), 50, 220).round(0)
    egfr = np.clip(rng.normal(88 - 0.4 * (age - 30), 12, n), 25, 120).round(0)
    # Simple "genetic risk marker" score (0-1), e.g. polygenic risk placeholder
    genetic_risk = np.clip(rng.beta(2, 3, n) + 0.15 * family_history, 0, 1).round(2)
    return pd.DataFrame({
        "patient_id": [f"P{i:03d}" for i in range(1, n + 1)],
        "age": age, "sex": sex, "bmi": bmi, "years_since_dx": years_dx,
        "hba1c": hba1c, "fasting_glucose": fasting_glucose,
        "family_history": family_history, "hypertension": hypertension,
        "on_insulin": on_insulin, "on_metformin": on_metformin,
        "ldl": ldl, "egfr": egfr, "genetic_risk": genetic_risk,
    })


# ------------------------------------------------------------------
# 2. TIME-SERIES (synthetic wearable + CGM data) for one patient
# ------------------------------------------------------------------
def simulate_patient(p):
    n = DAYS * STEPS_PER_DAY
    t = np.arange(n)
    hour = (t % STEPS_PER_DAY) * STEP_MIN / 60.0

    # Insulin resistance factor: higher HbA1c / BMI / genetic risk -> bigger meal spikes
    resist = 0.55 + 0.12 * (p.hba1c - 6) + 0.012 * (p.bmi - 25) + 0.35 * p.genetic_risk
    resist *= (0.85 if p.on_insulin else 1.0) * (0.9 if p.on_metformin else 1.0)
    base = 28.7 * p.hba1c - 46.7           # estimated average glucose (ADA formula)
    base = base * 0.72                     # fasting baseline sits below the daily average

    # --- activity (steps) ---
    awake = ((hour >= 6.5) & (hour <= 22.5)).astype(float)
    activity_level = rng.uniform(0.6, 1.4)
    steps = rng.poisson(awake * activity_level * 140 * rng.gamma(2, 0.5, n))
    steps = steps.astype(float)

    # --- meals (carbs in grams, logged by app) ---
    carbs = np.zeros(n)
    for d in range(DAYS):
        for mean_h, mean_c in [(8.0, 55), (13.2, 80), (16.8, 25), (20.2, 75)]:
            if rng.random() < 0.93:  # sometimes skipped
                h = rng.normal(mean_h, 0.5)
                idx = d * STEPS_PER_DAY + int(h * 60 / STEP_MIN)
                if 0 <= idx < n:
                    carbs[idx] += max(10, rng.normal(mean_c, 18))

    # --- meal response: rise peaks ~60-75 min, decays over ~3 h ---
    kernel_t = np.arange(0, 14) * STEP_MIN        # minutes after meal
    tau = 70.0
    kernel = (kernel_t / tau) * np.exp(1 - kernel_t / tau)
    meal_effect = np.convolve(carbs, kernel)[:n] * resist * 0.8

    # --- exercise lowers glucose (walking after meals, general activity) ---
    act_smooth = np.convolve(steps, np.ones(6) / 6, mode="same")
    exercise_effect = np.clip(act_smooth - 120, 0, None) * 0.08

    # --- sleep: nightly sleep duration, poor sleep raises next-day glucose ---
    sleep_stage = np.zeros(n, dtype=int)  # 0=awake, 1=light, 2=deep, 3=REM
    sleep_hours = np.zeros(DAYS)
    for d in range(DAYS):
        bed = rng.normal(22.8, 0.8)
        dur = np.clip(rng.normal(6.6, 1.1), 3.5, 9.0)
        sleep_hours[d] = dur
        for k in range(int(dur * 60 / STEP_MIN)):
            idx = d * STEPS_PER_DAY + int(bed * 60 / STEP_MIN) + k
            if idx < n:
                sleep_stage[idx] = rng.choice([1, 2, 3, 0], p=[0.5, 0.2, 0.22, 0.08])
    sleep_penalty = np.repeat(np.clip(7 - sleep_hours, 0, 4), STEPS_PER_DAY) * 4.0
    steps[sleep_stage > 0] = 0

    # --- dawn effect + noise (AR(1)) ---
    dawn = 10 * np.exp(-((hour - 6.5) ** 2) / 2.0)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 3.2)

    glucose = base + meal_effect - exercise_effect + sleep_penalty + dawn + noise
    glucose = np.clip(glucose, 55, 400)

    # --- heart rate & HRV (lower HRV when glucose is high / poor sleep / older) ---
    hr = (66 + 0.1 * (p.bmi - 25) + 10 * (steps > 100) + 0.03 * (glucose - 120)
          - 6 * (sleep_stage > 0) + rng.normal(0, 3, n))
    hrv = np.clip(55 - 0.35 * (p.age - 30) - 0.12 * (glucose - 120) - 2.0 * (steps > 100)
                  + 8 * (sleep_stage == 2) + rng.normal(0, 5, n), 8, 120)

    # --- label: any glucose > threshold within the next 2 hours ---
    future_max = pd.Series(glucose[::-1]).rolling(HORIZON, min_periods=1).max().values[::-1]
    future_max = np.roll(future_max, -1)    # window starts at t+1
    future_max[-HORIZON:] = np.nan          # not enough future data at the end
    label = (future_max > SPIKE_THRESHOLD).astype(float)
    label[-HORIZON:] = np.nan

    start = pd.Timestamp("2026-09-01 00:00:00")
    ts = start + pd.to_timedelta(t * STEP_MIN, unit="m")
    return pd.DataFrame({
        "patient_id": p.patient_id, "timestamp": ts,
        "glucose": glucose.round(1), "heart_rate": hr.round(0), "hrv_rmssd": hrv.round(1),
        "steps": steps.astype(int), "sleep_stage": sleep_stage,
        "meal_carbs_g": carbs.round(0), "spike_next_2h": label,
    })


if __name__ == "__main__":
    ehr = make_ehr(N_PATIENTS)
    ehr.to_csv("data/ehr.csv", index=False)

    frames = [simulate_patient(p) for p in ehr.itertuples()]
    wear = pd.concat(frames, ignore_index=True).dropna(subset=["spike_next_2h"])
    wear["spike_next_2h"] = wear["spike_next_2h"].astype(int)
    wear.to_csv("data/wearables.csv", index=False)

    print(f"Patients: {len(ehr)}")
    print(f"Wearable rows: {len(wear):,}")
    print(f"Spike rate (label=1): {wear['spike_next_2h'].mean():.1%}")
    print(f"Glucose mean/min/max: {wear['glucose'].mean():.0f} / {wear['glucose'].min():.0f} / {wear['glucose'].max():.0f}")
    print("Saved: data/ehr.csv, data/wearables.csv")