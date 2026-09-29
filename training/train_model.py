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

# ── Step 1: Load cleaned data ─────────────────────────────────────────────────
print("Loading cleaned dataset...")
df = pd.read_csv('data/appointments_cleaned.csv')
print(f"Dataset shape: {df.shape}")

# ── Step 2: Engineer additional features ─────────────────────────────────────
print("\nEngineering additional features...")

# Patients with chronic conditions may be more committed to attendance
df['chronic_condition'] = (
    (df['hypertension'] == 1) |
    (df['diabetes'] == 1)
).astype(int)

# High risk profile — older + no SMS + booked far in advance
df['high_risk_profile'] = (
    (df['age'] > 50) &
    (df['sms_received'] == 0) &
    (df['days_in_advance'] > 20)
).astype(int)

# SMS effectiveness — did they get a reminder and have good history?
df['engaged_patient'] = (
    (df['sms_received'] == 1) &
    (df['prior_attendance_rate'] > 0.7)
).astype(int)

# Booking urgency — same day or next day booking = more likely to attend
df['urgent_booking'] = (df['days_in_advance'] <= 1).astype(int)

# Long wait — booked more than 30 days out = higher no-show risk
df['long_wait'] = (df['days_in_advance'] > 30).astype(int)

# Weekend appointment (5=Saturday, 6=Sunday)
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Morning appointment (before 12pm) = lower no-show risk
df['is_morning'] = (df['appointment_hour'] < 12).astype(int)

# Young adult with no chronic condition = higher no-show risk
df['low_commitment_risk'] = (
    (df['age'] < 35) &
    (df['chronic_condition'] == 0) &
    (df['prior_attendance_rate'] <= 0.5)
).astype(int)

# Prior attendance interaction with days in advance
df['attendance_x_advance'] = (
    df['prior_attendance_rate'] * df['days_in_advance']
)

# Age groups
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 18, 35, 50, 65, 115],
    labels=[0, 1, 2, 3, 4],
    include_lowest=True
).astype(float).fillna(0).astype(int)


print("Additional features engineered successfully")

# ── Step 3: Define features ───────────────────────────────────────────────────
FEATURES = [
    # Original features
    'age',
    'gender_binary',
    'scholarship',
    'hypertension',
    'diabetes',
    'alcoholism',
    'handicap',
    'sms_received',
    'days_in_advance',
    'day_of_week',
    'appointment_hour',
    'prior_attendance_rate',
    # New engineered features
    'chronic_condition',
    'high_risk_profile',
    'engaged_patient',
    'urgent_booking',
    'long_wait',
    'is_weekend',
    'is_morning',
    'low_commitment_risk',
    'attendance_x_advance',
    'age_group'
]

TARGET = 'no_show'

X = df[FEATURES]
y = df[TARGET]

print(f"\nTotal features: {len(FEATURES)}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# ── Step 4: Split data ────────────────────────────────────────────────────────
print("\nSplitting data (70/15/15)...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Training:   {X_train.shape[0]} rows")
print(f"Validation: {X_val.shape[0]} rows")
print(f"Test:       {X_test.shape[0]} rows")

# ── Step 5: Hyperparameter tuning ─────────────────────────────────────────────
print("\nRunning hyperparameter search...")
print("Testing 20 different parameter combinations...")
print("This will take 3-5 minutes...")

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale = neg / pos

param_grid = {
    'max_depth':        [3, 4, 5, 6],
    'learning_rate':    [0.05, 0.1, 0.15, 0.2],
    'n_estimators':     [200, 300, 400, 500],
    'subsample':        [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma':            [0, 0.1, 0.2]
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
    n_iter=20,              # test 20 random combinations
    scoring='roc_auc',      # optimize for AUC-ROC
    cv=3,                   # 3-fold cross validation
    random_state=42,
    n_jobs=-1,              # use all CPU cores
    verbose=1
)

search.fit(X_train, y_train)

print(f"\nBest parameters found:")
for param, value in search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nBest cross-validation AUC-ROC: {search.best_score_:.4f}")

# ── Step 6: Train final model with best parameters ───────────────────────────
print("\nTraining final model with best parameters...")

best_params = search.best_params_
best_params['scale_pos_weight'] = scale
best_params['random_state']     = 42
best_params['eval_metric']      = 'auc'
best_params['verbosity']        = 0
best_params['early_stopping_rounds'] = 20

final_model = XGBClassifier(**best_params)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# ── Step 7: Find optimal threshold ───────────────────────────────────────────
print("\nFinding optimal classification threshold...")

val_proba = final_model.predict_proba(X_val)[:, 1]
best_threshold = 0.5
best_f1 = 0

for threshold in np.arange(0.1, 0.9, 0.05):
    preds = (val_proba >= threshold).astype(int)
    score = f1_score(y_val, preds)
    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

print(f"Optimal threshold: {best_threshold:.2f}")

# ── Step 8: Evaluate on test set ─────────────────────────────────────────────
print("\nEvaluating on test set...")

y_pred_proba = final_model.predict_proba(X_test)[:, 1]
y_pred       = (y_pred_proba >= best_threshold).astype(int)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
auc_roc   = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*45)
print("     FINAL XGBOOST MODEL PERFORMANCE")
print("="*45)
print(f"  Accuracy:   {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"  Precision:  {precision:.4f}  ({precision*100:.1f}%)")
print(f"  Recall:     {recall:.4f}  ({recall*100:.1f}%)")
print(f"  F1-Score:   {f1:.4f}  ({f1*100:.1f}%)")
print(f"  AUC-ROC:    {auc_roc:.4f}  ({auc_roc*100:.1f}%)")
print(f"  Threshold:  {best_threshold:.2f}")
print("="*45)

if auc_roc >= 0.85:
    print(f"  ✓ AUC-ROC target PASSED (>= 0.85)")
elif auc_roc >= 0.75:
    print(f"  ~ AUC-ROC acceptable (>= 0.75) — good enough for project")
else:
    print(f"  ✗ AUC-ROC below target — needs more work")

# ── Step 9: Confusion matrix ──────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"                  Predicted")
print(f"                  Show     No-show")
print(f"Actual  Show    [ {cm[0][0]:5d}    {cm[0][1]:5d} ]")
print(f"        No-show [ {cm[1][0]:5d}    {cm[1][1]:5d} ]")

# ── Step 10: Feature importance ───────────────────────────────────────────────
print("\nTop 10 Most Important Features:")
importance = pd.DataFrame({
    'feature':    FEATURES,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

for _, row in importance.iterrows():
    bar = '█' * int(row['importance'] * 100)
    print(f"  {row['feature']:<28} {row['importance']:.4f}  {bar}")

# ── Step 11: Save everything ──────────────────────────────────────────────────
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
print(f"\n✓ Training complete! Ready to build the prediction API.")
