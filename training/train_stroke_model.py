import pandas as pd
import numpy as np
import pickle
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ── Step 1: Load stroke-specific data ────────────────────────────────────────
print("Loading stroke rehabilitation dataset...")
df = pd.read_csv('data/stroke_appointments.csv')
print(f"Dataset shape: {df.shape}")
print(f"No-show rate: {df['no_show'].mean():.1%}")

# ── Step 2: Engineer additional features ─────────────────────────────────────
print("\nEngineering features...")

# High distance + no caregiver = very high risk
df['isolated_far_patient'] = (
    (df['distance_km'] > 10) &
    (df['has_caregiver'] == 0)
).astype(int)

# Severe stroke + early session = very motivated
df['motivated_patient'] = (
    (df['stroke_severity'] == 2) &
    (df['session_number'] <= 5)
).astype(int)

# SMS + good history = engaged
df['engaged_patient'] = (
    (df['sms_received'] == 1) &
    (df['prior_attendance_rate'] > 0.7)
).astype(int)

# Financial struggle + public transport = double risk
df['access_difficulty'] = (
    (df['financial_constraint'] == 1) &
    (df['transport_type'] == 1)
).astype(int)

# Morning appointment = lower risk
df['is_morning'] = (df['appointment_hour'] < 12).astype(int)

# Long wait = higher risk
df['long_wait'] = (df['days_in_advance'] > 14).astype(int)

# Late stage patient (may feel recovered, skips sessions)
df['late_stage'] = (df['session_number'] > 12).astype(int)

# Distance squared (non-linear effect — very far = much worse)
df['distance_squared'] = (df['distance_km'] ** 2) / 100

# Attendance x severity interaction
df['attendance_x_severity'] = (
    df['prior_attendance_rate'] * (df['stroke_severity'] + 1)
)

# Age group
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 40, 55, 65, 90],
    labels=[0, 1, 2, 3],
    include_lowest=True
).astype(float).fillna(0).astype(int)

print("Features engineered successfully")

# ── Step 3: Define features ───────────────────────────────────────────────────
FEATURES = [
    # Core patient features
    'age',
    'gender_binary',
    'stroke_severity',
    'days_since_stroke',
    'session_number',
    # Access features (most important for Nigeria)
    'distance_km',
    'has_caregiver',
    'transport_type',
    'financial_constraint',
    # Appointment features
    'day_of_week',
    'appointment_hour',
    'days_in_advance',
    'sms_received',
    # History
    'prior_attendance_rate',
    # Engineered features
    'isolated_far_patient',
    'motivated_patient',
    'engaged_patient',
    'access_difficulty',
    'is_morning',
    'long_wait',
    'late_stage',
    'distance_squared',
    'attendance_x_severity',
    'age_group'
]

TARGET = 'no_show'

X = df[FEATURES]
y = df[TARGET]

print(f"\nTotal features: {len(FEATURES)}")

# ── Step 4: Split data ────────────────────────────────────────────────────────
print("Splitting data (70/15/15)...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Training:   {X_train.shape[0]} rows")
print(f"Validation: {X_val.shape[0]} rows")
print(f"Test:       {X_test.shape[0]} rows")

# ── Step 5: Hyperparameter search ────────────────────────────────────────────
print("\nRunning hyperparameter search...")
print("This will take 2-3 minutes...")

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale = neg / pos
print(f"Class scale weight: {scale:.2f}")

param_grid = {
    'max_depth':        [3, 4, 5, 6, 7],
    'learning_rate':    [0.01, 0.05, 0.1, 0.15],
    'n_estimators':     [300, 400, 500, 600],
    'subsample':        [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma':            [0, 0.1, 0.2, 0.3],
    'reg_alpha':        [0, 0.1, 0.5],
    'reg_lambda':       [1, 1.5, 2]
}

base_model = XGBClassifier(
    scale_pos_weight=scale,
    random_state=42,
    eval_metric='auc',
    verbosity=0
)

search = RandomizedSearchCV(
    base_model,
    param_distributions=param_grid,
    n_iter=30,
    scoring='roc_auc',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

print(f"\nBest cross-validation AUC-ROC: {search.best_score_:.4f}")
print(f"Best parameters:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")

# ── Step 6: Train final model ─────────────────────────────────────────────────
print("\nTraining final model with best parameters...")

best_params = search.best_params_.copy()
best_params['scale_pos_weight']    = scale
best_params['random_state']        = 42
best_params['eval_metric']         = 'auc'
best_params['verbosity']           = 0
best_params['early_stopping_rounds'] = 30

final_model = XGBClassifier(**best_params)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"Best iteration: {final_model.best_iteration}")

# ── Step 7: Find optimal threshold ───────────────────────────────────────────
print("\nFinding optimal threshold...")
val_proba = final_model.predict_proba(X_val)[:, 1]
best_threshold = 0.5
best_f1 = 0

for threshold in np.arange(0.1, 0.9, 0.01):
    preds = (val_proba >= threshold).astype(int)
    score = f1_score(y_val, preds, zero_division=0)
    if score > best_f1:
        best_f1 = score
        best_threshold = round(threshold, 2)

print(f"Optimal threshold: {best_threshold}")

# ── Step 8: Evaluate ─────────────────────────────────────────────────────────
print("\nEvaluating on test set...")

y_pred_proba = final_model.predict_proba(X_test)[:, 1]
y_pred       = (y_pred_proba >= best_threshold).astype(int)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
auc_roc   = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*50)
print("    STROKE MODEL PERFORMANCE RESULTS")
print("="*50)
print(f"  Accuracy:   {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"  Precision:  {precision:.4f}  ({precision*100:.1f}%)")
print(f"  Recall:     {recall:.4f}  ({recall*100:.1f}%)")
print(f"  F1-Score:   {f1:.4f}  ({f1*100:.1f}%)")
print(f"  AUC-ROC:    {auc_roc:.4f}  ({auc_roc*100:.1f}%)")
print(f"  Threshold:  {best_threshold}")
print("="*50)

if auc_roc >= 0.85:
    print("  ✓ AUC-ROC target PASSED (>= 0.85) 🎉")
elif auc_roc >= 0.75:
    print("  ~ AUC-ROC is good (>= 0.75)")
else:
    print("  ✗ AUC-ROC below target")

# ── Step 9: Confusion matrix ──────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"                  Predicted")
print(f"                  Attended    No-show")
print(f"Actual  Attended  [ {cm[0][0]:4d}        {cm[0][1]:4d} ]")
print(f"        No-show   [ {cm[1][0]:4d}        {cm[1][1]:4d} ]")

tn, fp, fn, tp = cm.ravel()
print(f"\n  True Positives  (correctly caught no-shows): {tp}")
print(f"  False Positives (wrongly flagged as no-show): {fp}")
print(f"  True Negatives  (correctly predicted attendance): {tn}")
print(f"  False Negatives (missed no-shows):             {fn}")

# ── Step 10: Feature importance ───────────────────────────────────────────────
print("\nTop 12 Most Important Features:")
importance = pd.DataFrame({
    'feature':    FEATURES,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False).head(12)

for _, row in importance.iterrows():
    bar = '█' * int(row['importance'] * 100)
    print(f"  {row['feature']:<28} {row['importance']:.4f}  {bar}")

# ── Step 11: Sample predictions ───────────────────────────────────────────────
print("\nSample predictions on 5 test patients:")
print("-" * 60)
sample = X_test.head(5).copy()
sample_proba = final_model.predict_proba(sample)[:, 1]

for i, (idx, row) in enumerate(sample.iterrows()):
    risk = "HIGH RISK ⚠" if sample_proba[i] >= best_threshold else "LOW RISK  ✓"
    print(f"  Patient {i+1}: score={sample_proba[i]:.3f}  {risk}")
    print(f"    Age={int(row['age'])}, Severity={int(row['stroke_severity'])}, "
          f"Distance={row['distance_km']:.1f}km, "
          f"Caregiver={int(row['has_caregiver'])}, "
          f"Prior attendance={row['prior_attendance_rate']:.2f}")

# ── Step 12: Save everything ──────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)

with open('models/xgboost_noshow.pkl', 'wb') as f:
    pickle.dump(final_model, f)

with open('models/features.pkl', 'wb') as f:
    pickle.dump(FEATURES, f)

with open('models/threshold.pkl', 'wb') as f:
    pickle.dump(best_threshold, f)

print(f"\n✓ Model saved to     models/xgboost_noshow.pkl")
print(f"✓ Features saved to  models/features.pkl")
print(f"✓ Threshold saved to models/threshold.pkl")
print(f"\n✓ Done! Ready to build the prediction API.")
