"""
train_models.py — One-time script to:
  1. Generate a realistic synthetic dataset
  2. Train Isolation Forest for anomaly detection
  3. Train Random Forest for failure probability prediction
  4. Save models + normalization params to disk
  5. Initialize the SQLite database

Run before starting the Streamlit app:
  python train_models.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import sys

# Ensure modules are importable from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.makedirs("models", exist_ok=True)
os.makedirs("data",   exist_ok=True)

FEATURE_COLS = ["temperature", "vibration", "humidity", "energy", "rpm", "pressure"]

print()
print("=" * 65)
print("  Digital Twin Edge AI — Model Training Pipeline")
print("=" * 65)

# ─── 1. Synthetic dataset generation ───────────────────────────────────────────
print("\n[1/5]  Generating synthetic training dataset ...")

np.random.seed(42)

N_NORMAL  = 6000
N_WARN    = 1500
N_FAILURE = 1000

def make_samples(n, mu, sigma, label):
    data = {col: np.random.normal(mu[col], sigma[col], n) for col in FEATURE_COLS}
    data["machine_failure"] = np.full(n, label, dtype=int)
    return pd.DataFrame(data)

# Normal operating data
normal_df = make_samples(
    N_NORMAL,
    mu    = dict(temperature=70, vibration=2.0, humidity=45, energy=20, rpm=1500, pressure=4.5),
    sigma = dict(temperature=5,  vibration=0.4, humidity=8,  energy=4,  rpm=45,   pressure=0.5),
    label = 0,
)

# Warning-range data (borderline — still label=0)
warn_df = make_samples(
    N_WARN,
    mu    = dict(temperature=83, vibration=3.8, humidity=65, energy=32, rpm=1680, pressure=6.5),
    sigma = dict(temperature=4,  vibration=0.5, humidity=7,  energy=4,  rpm=50,   pressure=0.6),
    label = 0,
)

# Pre-failure / failure data
failure_df = make_samples(
    N_FAILURE,
    mu    = dict(temperature=96, vibration=7.5, humidity=78, energy=47, rpm=1870, pressure=8.8),
    sigma = dict(temperature=7,  vibration=1.5, humidity=10, energy=7,  rpm=100,  pressure=1.0),
    label = 1,
)

df = pd.concat([normal_df, warn_df, failure_df], ignore_index=True).sample(frac=1, random_state=42)
df.to_csv("data/training_data.csv", index=False)
print(f"   [OK] {len(df):,} samples  |  Normal: {N_NORMAL+N_WARN:,}  |  Failure: {N_FAILURE:,}")

# ─── 2. Compute normalization params from NORMAL data only ─────────────────────
print("\n[2/5]  Computing normalization parameters ...")

X_normal = normal_df[FEATURE_COLS]
means = X_normal.mean().to_dict()
stds  = X_normal.std().to_dict()
norm_params = {"means": means, "stds": stds}
joblib.dump(norm_params, "models/norm_params.pkl")
print("   [OK] Saved  models/norm_params.pkl")
print(f"        Means : { {k: round(v,2) for k,v in means.items()} }")
print(f"        Stds  : { {k: round(v,2) for k,v in stds.items()} }")

def scale(df_in):
    df_s = df_in[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        df_s[col] = (df_s[col] - means[col]) / max(stds[col], 1e-6)
    return df_s.values

# ─── 3. Train Isolation Forest ─────────────────────────────────────────────────
print("\n[3/5]  Training Isolation Forest (Anomaly Detection) ...")

X_normal_scaled = scale(normal_df)
iso = IsolationForest(
    n_estimators=150,
    contamination=0.04,
    max_samples="auto",
    random_state=42,
    n_jobs=-1,
)
iso.fit(X_normal_scaled)

# Quick evaluation on full dataset
X_all_scaled    = scale(df)
preds           = iso.predict(X_all_scaled)
n_anomalies     = (preds == -1).sum()
print(f"   [OK] Trained on {len(X_normal_scaled):,} normal samples")
print(f"        Anomalies flagged in full dataset: {n_anomalies} / {len(preds)}")

joblib.dump(iso, "models/anomaly_model.pkl")
print("   [OK] Saved  models/anomaly_model.pkl")

# ─── 4. Train Random Forest ────────────────────────────────────────────────────
print("\n[4/5]  Training Random Forest (Failure Prediction) ...")

X = scale(df)
y = df["machine_failure"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n   -- Classification Report ------------------------------")
print(classification_report(y_test, y_pred, target_names=["Normal", "Failure"]))

cm = confusion_matrix(y_test, y_pred)
print(f"   Confusion Matrix:\n      TN={cm[0,0]}  FP={cm[0,1]}\n      FN={cm[1,0]}  TP={cm[1,1]}")
print(f"\n   [OK] Accuracy: {acc:.4f}")

# Feature importance
importances = dict(zip(FEATURE_COLS, rf.feature_importances_))
print("\n   Feature importances:")
for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
    bar = "=" * int(imp * 40)
    print(f"      {feat:14s}  {bar:40s}  {imp:.3f}")

joblib.dump(rf, "models/failure_model.pkl")
print("\n   [OK] Saved  models/failure_model.pkl")

# ─── 5. Initialise database ────────────────────────────────────────────────────
print("\n[5/5]  Initialising SQLite database ...")
from modules.database import initialize_db
initialize_db()
print("   [OK] Database ready at  data/digital_twin.db")

# ─── Done ──────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  [OK] All models trained and saved successfully!")
print()
print("  Next step -- start the dashboard:")
print("     streamlit run app.py")
print("=" * 65)
print()
