import pandas as pd
import numpy as np
import os

# Set seed so results are reproducible
np.random.seed(42)

N = 50000  # number of synthetic patient appointments
print(f"Generating {N} synthetic stroke rehabilitation appointments...")

# ── Patient characteristics ───────────────────────────────────────────────────

# Age: stroke patients typically 40-80 years old
age = np.random.normal(loc=58, scale=12, size=N).clip(18, 90).astype(int)

# Gender
gender = np.random.choice([0, 1], size=N, p=[0.45, 0.55])

# Stroke severity: 0=mild, 1=moderate, 2=severe
stroke_severity = np.random.choice([0, 1, 2], size=N, p=[0.40, 0.40, 0.20])

# Days since stroke event
days_since_stroke = np.random.exponential(scale=60, size=N).clip(1, 365).astype(int)

# Session number in treatment plan (1st session to 20th)
session_number = np.random.randint(1, 21, size=N)

# Distance from clinic in km (Nigerian context — many patients travel far)
distance_km = np.random.exponential(scale=8, size=N).clip(0.5, 50)

# Caregiver support
has_caregiver = np.random.choice([0, 1], size=N, p=[0.40, 0.60])

# Transportation type: 0=private, 1=public, 2=ambulance/clinic
transport_type = np.random.choice([0, 1, 2], size=N, p=[0.30, 0.60, 0.10])

# Financial constraint proxy (scholarship/low income)
financial_constraint = np.random.choice([0, 1], size=N, p=[0.65, 0.35])

# Day of week (0=Monday to 4=Friday, clinics closed weekends)
day_of_week = np.random.choice([0, 1, 2, 3, 4], size=N)

# Appointment hour (8am to 4pm)
appointment_hour = np.random.choice(range(8, 17), size=N)

# Days booked in advance
days_in_advance = np.random.exponential(scale=7, size=N).clip(0, 60).astype(int)

# SMS reminder sent
sms_received = np.random.choice([0, 1], size=N, p=[0.40, 0.60])

# ── Calculate prior attendance rate ──────────────────────────────────────────
# Simulate realistic attendance history
# Patients with more sessions have more history
prior_attendance_base = np.random.beta(a=3, b=1.5, size=N)  # skewed toward higher attendance
prior_attendance_rate = np.where(
    session_number == 1,
    0.5,  # no history for first session
    prior_attendance_base
)

# ── Generate no-show outcome ──────────────────────────────────────────────────
# Build no-show probability from real risk factors
# Each factor contributes based on clinical evidence

no_show_prob = np.zeros(N)

# Base rate for stroke rehab in Nigeria ~30%
no_show_prob += 0.30

# Severe stroke = more motivated to attend (they know they need it)
no_show_prob -= 0.08 * (stroke_severity == 2)
no_show_prob += 0.05 * (stroke_severity == 0)  # mild = less urgent feeling

# Far distance = much higher no-show risk
no_show_prob += 0.015 * distance_km

# No caregiver = higher risk
no_show_prob += 0.12 * (has_caregiver == 0)

# Public transport = higher risk (unreliable in Nigeria)
no_show_prob += 0.08 * (transport_type == 1)

# Financial constraint = higher risk
no_show_prob += 0.10 * (financial_constraint == 1)

# Prior attendance strongly predicts future attendance
no_show_prob -= 0.30 * prior_attendance_rate

# SMS reminder reduces no-show
no_show_prob -= 0.08 * sms_received

# Later sessions = patient may feel recovered enough, skips
no_show_prob += 0.008 * session_number

# Far in advance booking = higher no-show
no_show_prob += 0.004 * days_in_advance

# Morning appointments = lower no-show
no_show_prob -= 0.05 * (appointment_hour < 12)

# Monday = higher no-show (weekend forgot)
no_show_prob += 0.04 * (day_of_week == 0)

# Clip to valid probability range
no_show_prob = no_show_prob.clip(0.02, 0.98)

# Generate actual outcome
no_show = (np.random.random(N) < no_show_prob).astype(int)

print(f"Synthetic no-show rate: {no_show.mean():.1%}")

# ── Assemble dataframe ────────────────────────────────────────────────────────
df = pd.DataFrame({
    'age':                   age,
    'gender_binary':         gender,
    'stroke_severity':       stroke_severity,
    'days_since_stroke':     days_since_stroke,
    'session_number':        session_number,
    'distance_km':           distance_km.round(2),
    'has_caregiver':         has_caregiver,
    'transport_type':        transport_type,
    'financial_constraint':  financial_constraint,
    'day_of_week':           day_of_week,
    'appointment_hour':      appointment_hour,
    'days_in_advance':       days_in_advance,
    'sms_received':          sms_received,
    'prior_attendance_rate': prior_attendance_rate.round(4),
    'no_show':               no_show
})

print(f"\nDataset shape: {df.shape}")
print(f"\nFeature summary:")
print(df.describe().round(2))
print(f"\nNo-show distribution:")
print(df['no_show'].value_counts())

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs('data', exist_ok=True)
df.to_csv('data/stroke_appointments.csv', index=False)
print(f"\n✓ Synthetic stroke data saved to data/stroke_appointments.csv")
print(f"✓ Ready to retrain model on stroke-specific features!")
