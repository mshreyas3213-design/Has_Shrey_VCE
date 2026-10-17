"""
Train the Digital Twin glucose-spike predictor.

Fuses two data streams:
  - Static EHR   (data/ehr.csv)
  - Wearable/CGM (data/wearables.csv)
and predicts: will glucose exceed 180 mg/dL within the next 2 hours?

Outputs (in ./models and ./outputs):
  models/model.pkl            trained LightGBM model + feature list
  outputs/metrics.json        test-set metrics (for README / slides)
  outputs/feature_importance.png, outputs/roc_curve.png
  outputs/predictions.csv     risk score for every row (used by the dashboard)

Run:  python train_model.py
"""
import json
import os
import warnings

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import GroupShuffleSplit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")  # hide library deprecation notices

SEED = 42
STEP_MIN = 15
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

EHR_FEATURES = ["age", "sex_male", "bmi", "years_since_dx", "hba1c", "fasting_glucose",
                "family_history", "hypertension", "on_insulin", "on_metformin",
                "ldl", "egfr", "genetic_risk"]


# ------------------------------------------------------------------
# 1. Load and merge the two streams
# ------------------------------------------------------------------
def load_data():
    ehr = pd.read_csv("data/ehr.csv")
    wear = pd.read_csv("data/wearables.csv", parse_dates=["timestamp"])
    ehr["sex_male"] = (ehr["sex"] == "M").astype(int)
    df = wear.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    df = df.merge(ehr.drop(columns=["sex"]), on="patient_id", how="left")
    return df


# ------------------------------------------------------------------
# 2. Feature engineering (only uses PAST data -> no leakage)
# ------------------------------------------------------------------
def add_features(df):
    g = df.groupby("patient_id", group_keys=False)
    out = df.copy()

    # glucose history
    for lag in (1, 2, 4, 8):
        out[f"glucose_lag_{lag * STEP_MIN}m"] = g["glucose"].shift(lag)
    out["glucose_delta_30m"] = out["glucose"] - out["glucose_lag_30m"]
    out["glucose_delta_60m"] = out["glucose"] - out["glucose_lag_60m"]
    out["glucose_roll_mean_2h"] = g["glucose"].transform(lambda s: s.rolling(8, min_periods=1).mean())
    out["glucose_roll_std_2h"] = g["glucose"].transform(lambda s: s.rolling(8, min_periods=2).std())
    out["glucose_roll_max_2h"] = g["glucose"].transform(lambda s: s.rolling(8, min_periods=1).max())

    # activity, heart, recovery
    out["steps_last_1h"] = g["steps"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    out["hrv_mean_2h"] = g["hrv_rmssd"].transform(lambda s: s.rolling(8, min_periods=1).mean())
    out["hr_mean_2h"] = g["heart_rate"].transform(lambda s: s.rolling(8, min_periods=1).mean())

    # meals (logged by the app) and sleep over the last 24 h
    out["carbs_last_2h"] = g["meal_carbs_g"].transform(lambda s: s.rolling(8, min_periods=1).sum())
    out["carbs_last_4h"] = g["meal_carbs_g"].transform(lambda s: s.rolling(16, min_periods=1).sum())
    out["sleep_hours_24h"] = g["sleep_stage"].transform(
        lambda s: (s > 0).astype(float).rolling(96, min_periods=1).sum() * STEP_MIN / 60)
    out["deep_sleep_hours_24h"] = g["sleep_stage"].transform(
        lambda s: (s == 2).astype(float).rolling(96, min_periods=1).sum() * STEP_MIN / 60)

    # time of day
    hour = out["timestamp"].dt.hour + out["timestamp"].dt.minute / 60
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return out


WEARABLE_FEATURES = [
    "glucose", "glucose_lag_15m", "glucose_lag_30m", "glucose_lag_60m", "glucose_lag_120m",
    "glucose_delta_30m", "glucose_delta_60m", "glucose_roll_mean_2h", "glucose_roll_std_2h",
    "glucose_roll_max_2h", "heart_rate", "hrv_rmssd", "steps", "steps_last_1h",
    "hrv_mean_2h", "hr_mean_2h", "sleep_stage", "meal_carbs_g", "carbs_last_2h",
    "carbs_last_4h", "sleep_hours_24h", "deep_sleep_hours_24h", "hour_sin", "hour_cos",
]
FUSED_FEATURES = EHR_FEATURES + WEARABLE_FEATURES


# ------------------------------------------------------------------
# 3. Train / evaluate helper
# ------------------------------------------------------------------
PARAMS = dict(objective="binary", n_estimators=600, learning_rate=0.05, num_leaves=31,
              min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=1.0, random_state=SEED, verbose=-1)


def fit_eval(train, valid, test, features, label="spike_next_2h"):
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(train[features], train[label],
              eval_set=[(valid[features], valid[label])],
              callbacks=[lgb.early_stopping(40, verbose=False)])
    proba = model.predict_proba(test[features])[:, 1]
    return model, proba


def metrics_at(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"threshold": thr,
            "precision": round(float(precision_score(y, pred, zero_division=0)), 3),
            "recall": round(float(recall_score(y, pred)), 3),
            "f1": round(float(f1_score(y, pred)), 3),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


# ------------------------------------------------------------------
# 4. Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading data and building features ...")
    df = add_features(load_data()).dropna(subset=WEARABLE_FEATURES).reset_index(drop=True)
    label = "spike_next_2h"
    print(f"Rows: {len(df):,} | Patients: {df['patient_id'].nunique()} | Positive rate: {df[label].mean():.1%}")

    # --- Split BY PATIENT so the model is tested on people it has never seen ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(df, groups=df["patient_id"]))
    train_full, test = df.iloc[train_idx], df.iloc[test_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tr_idx, va_idx = next(gss2.split(train_full, groups=train_full["patient_id"]))
    train, valid = train_full.iloc[tr_idx], train_full.iloc[va_idx]
    print(f"Train patients: {train['patient_id'].nunique()} | Valid: {valid['patient_id'].nunique()} | "
          f"Test: {test['patient_id'].nunique()} (no overlap)")

    # --- Ablation: does FUSING the two streams help? ---
    results = {}
    for name, feats in [("EHR only", EHR_FEATURES),
                        ("Wearables only", WEARABLE_FEATURES),
                        ("Fused (EHR + wearables)", FUSED_FEATURES)]:
        m, p = fit_eval(train, valid, test, feats)
        results[name] = {"roc_auc": round(float(roc_auc_score(test[label], p)), 3),
                         "pr_auc": round(float(average_precision_score(test[label], p)), 3)}
        if name.startswith("Fused"):
            model, proba = m, p
        print(f"{name:28s} ROC-AUC {results[name]['roc_auc']:.3f} | PR-AUC {results[name]['pr_auc']:.3f}")

    # --- Detailed metrics for the fused model ---
    y = test[label].values
    report = {
        "ablation": results,
        "at_threshold_0.50": metrics_at(y, proba, 0.5),
        "at_threshold_0.30_high_recall": metrics_at(y, proba, 0.3),
        "test_rows": int(len(test)), "test_patients": int(test["patient_id"].nunique()),
        "test_positive_rate": round(float(y.mean()), 3),
        "best_iteration": int(model.best_iteration_ or PARAMS["n_estimators"]),
        "note": "All data is synthetic; metrics show pipeline behaviour, not clinical accuracy.",
    }
    print("\nFused model on UNSEEN patients:")
    for k in ("at_threshold_0.50", "at_threshold_0.30_high_recall"):
        print(f"  {k}: {report[k]}")

    with open("outputs/metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    # --- Plots ---
    imp = pd.Series(model.feature_importances_, index=FUSED_FEATURES).sort_values().tail(15)
    plt.figure(figsize=(8, 6))
    imp.plot(kind="barh", color="#305496")
    plt.title("Top 15 features (fused model)")
    plt.xlabel("Importance (split count)")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    for name, feats in [("EHR only", EHR_FEATURES), ("Wearables only", WEARABLE_FEATURES),
                        ("Fused", FUSED_FEATURES)]:
        m, p = fit_eval(train, valid, test, feats)
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC on unseen patients")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/roc_curve.png", dpi=150)
    plt.close()

    # --- Save model + risk scores for the dashboard ---
    joblib.dump({"model": model, "features": FUSED_FEATURES}, "models/model.pkl")
    df["risk_score"] = model.predict_proba(df[FUSED_FEATURES])[:, 1]
    df["is_test_patient"] = df["patient_id"].isin(test["patient_id"].unique()).astype(int)
    keep = ["patient_id", "timestamp", "glucose", "heart_rate", "hrv_rmssd", "steps",
            "sleep_stage", "meal_carbs_g", "spike_next_2h", "risk_score", "is_test_patient"]
    df[keep].to_csv("outputs/predictions.csv", index=False)
    print("\nSaved: models/model.pkl, outputs/metrics.json, outputs/feature_importance.png, "
          "outputs/roc_curve.png, outputs/predictions.csv")