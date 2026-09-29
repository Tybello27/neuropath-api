import pandas as pd
import numpy as np

# ── Step 1: Load raw data ─────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('data/KaggleV2-May-2016.csv')

# ── Step 2: Clean column names ────────────────────────────────────────────────
df = df.rename(columns={
    'PatientId':      'patient_id',
    'AppointmentID':  'appointment_id',
    'Gender':         'gender',
    'ScheduledDay':   'scheduled_day',
    'AppointmentDay': 'appointment_day',
    'Age':            'age',
    'Neighbourhood':  'neighbourhood',
    'Scholarship':    'scholarship',
    'Hipertension':   'hypertension',
    'Diabetes':       'diabetes',
    'Alcoholism':     'alcoholism',
    'Handcap':        'handicap',
    'SMS_received':   'sms_received',
    'No-show':        'no_show'
})

# ── Step 3: Convert target variable to binary ─────────────────────────────────
df['no_show'] = df['no_show'].map({'Yes': 1, 'No': 0})
print(f"No-show rate: {df['no_show'].mean():.1%}")

# ── Step 4: Convert dates ─────────────────────────────────────────────────────
df['scheduled_day']   = pd.to_datetime(df['scheduled_day'])
df['appointment_day'] = pd.to_datetime(df['appointment_day'])

# ── Step 5: Engineer features ─────────────────────────────────────────────────

# Days booked in advance
df['days_in_advance'] = (
    df['appointment_day'] - df['scheduled_day']
).dt.days

# Remove impossible values
df = df[df['days_in_advance'] >= 0]
print(f"Rows after cleaning: {len(df)}")

# Day of week (0=Monday, 6=Sunday)
df['day_of_week'] = df['appointment_day'].dt.dayofweek

# Hour appointment was scheduled
df['appointment_hour'] = df['scheduled_day'].dt.hour

# Gender to binary
df['gender_binary'] = df['gender'].map({'F': 0, 'M': 1})

# Remove invalid ages
df = df[(df['age'] >= 0) & (df['age'] <= 115)]

# ── Step 6: Fast prior attendance rate ───────────────────────────────────────
# This version uses pandas groupby instead of a slow loop
# It runs in seconds instead of minutes
print("Calculating prior attendance rates (fast version)...")

# Sort by appointment day so history is in order
df = df.sort_values(['patient_id', 'appointment_day']).reset_index(drop=True)

# For each patient, calculate the cumulative attendance rate
# using only PAST appointments (shift by 1 so current row not included)
df['attended'] = 1 - df['no_show']

# cumsum gives running total of attended appointments
# cumcount gives running total of all appointments
df['cumulative_attended'] = df.groupby('patient_id')['attended'].cumsum().shift(1)
df['cumulative_total']    = df.groupby('patient_id')['attended'].cumcount()

# Prior attendance rate = attended so far / total so far
# For first appointment (cumulative_total = 0), use 0.5 as neutral default
df['prior_attendance_rate'] = np.where(
    df['cumulative_total'] == 0,
    0.5,
    df['cumulative_attended'] / df['cumulative_total']
)

# Fill any remaining NaN with 0.5
df['prior_attendance_rate'] = df['prior_attendance_rate'].fillna(0.5)

print(f"Prior attendance rate sample:")
print(df['prior_attendance_rate'].describe())

# ── Step 7: Select final features ────────────────────────────────────────────
FEATURES = [
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
    'prior_attendance_rate'
]

TARGET = 'no_show'

df_final = df[FEATURES + [TARGET]].copy()

print(f"\nFinal dataset shape: {df_final.shape}")
print(f"\nFeature summary:")
print(df_final.describe().round(2))

# ── Step 8: Save cleaned dataset ─────────────────────────────────────────────
df_final.to_csv('data/appointments_cleaned.csv', index=False)
print("\n✓ Cleaned dataset saved to data/appointments_cleaned.csv")
print("✓ Ready for model training!")