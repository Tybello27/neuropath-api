from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.supabase_client import create_patient, update_patient_risk

app = Flask(__name__)

# ── Load model, features and threshold on startup ────────────────────────────
print("Loading model...")

with open('models/xgboost_noshow.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/features.pkl', 'rb') as f:
    FEATURES = pickle.load(f)

with open('models/threshold.pkl', 'rb') as f:
    THRESHOLD = pickle.load(f)

print(f"Model loaded successfully")
print(f"Features: {FEATURES}")
print(f"Threshold: {THRESHOLD}")

# ── Health check endpoint ─────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'running',
        'model':  'XGBoost No-Show Predictor',
        'features': len(FEATURES)
    })

# ── Main prediction endpoint ──────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # ── Validate required fields ──────────────────────────────────────────
        required_fields = [
            'age', 'gender_binary', 'stroke_severity',
            'days_since_stroke', 'session_number',
            'distance_km', 'has_caregiver', 'transport_type',
            'financial_constraint', 'day_of_week',
            'appointment_hour', 'days_in_advance',
            'sms_received', 'prior_attendance_rate'
        ]

        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({
                'error': f'Missing fields: {missing}'
            }), 400

        # ── Engineer the same features as training ────────────────────────────
        distance_km           = float(data['distance_km'])
        has_caregiver         = int(data['has_caregiver'])
        stroke_severity       = int(data['stroke_severity'])
        session_number        = int(data['session_number'])
        prior_attendance_rate = float(data['prior_attendance_rate'])
        sms_received          = int(data['sms_received'])
        financial_constraint  = int(data['financial_constraint'])
        transport_type        = int(data['transport_type'])
        appointment_hour      = int(data['appointment_hour'])
        days_in_advance       = int(data['days_in_advance'])
        day_of_week           = int(data['day_of_week'])
        age                   = int(data['age'])

        # Engineered features (must match train_stroke_model.py exactly)
        isolated_far_patient  = int(distance_km > 10 and has_caregiver == 0)
        motivated_patient     = int(stroke_severity == 2 and session_number <= 5)
        engaged_patient       = int(sms_received == 1 and prior_attendance_rate > 0.7)
        access_difficulty     = int(financial_constraint == 1 and transport_type == 1)
        is_morning            = int(appointment_hour < 12)
        long_wait             = int(days_in_advance > 14)
        late_stage            = int(session_number > 12)
        distance_squared      = (distance_km ** 2) / 100
        attendance_x_severity = prior_attendance_rate * (stroke_severity + 1)

        # Age group
        if age <= 40:
            age_group = 0
        elif age <= 55:
            age_group = 1
        elif age <= 65:
            age_group = 2
        else:
            age_group = 3

        # ── Build feature vector in exact same order as training ──────────────
        feature_vector = [[
            age,
            int(data['gender_binary']),
            stroke_severity,
            int(data['days_since_stroke']),
            session_number,
            distance_km,
            has_caregiver,
            transport_type,
            financial_constraint,
            day_of_week,
            appointment_hour,
            days_in_advance,
            sms_received,
            prior_attendance_rate,
            isolated_far_patient,
            motivated_patient,
            engaged_patient,
            access_difficulty,
            is_morning,
            long_wait,
            late_stage,
            distance_squared,
            attendance_x_severity,
            age_group
        ]]

        # ── Make prediction ───────────────────────────────────────────────────
        risk_score = float(model.predict_proba(feature_vector)[0][1])
        risk_level = 'high' if risk_score >= THRESHOLD else 'low'

        # ── Build response ────────────────────────────────────────────────────
        response = {
            'risk_score':  round(risk_score, 4),
            'risk_level':  risk_level,
            'threshold':   round(THRESHOLD, 2),
            'high_risk':   risk_level == 'high',
            'recommendation': {
                'slot_preference': 'morning' if risk_level == 'high' else 'any',
                'reminders':       3 if risk_level == 'high' else 1,
                'reminder_type':   'voice' if risk_level == 'high' else 'text'
            }
        }

        print(f"Prediction: patient scored {risk_score:.4f} → {risk_level} risk")

        # ── Save to Supabase if patient data provided ─────────────────────────
        if 'name' in data and 'phone' in data:
            try:
                # Create patient record
                patient = create_patient(
                    name=data.get('name'),
                    phone=data.get('phone'),
                    age=data['age'],
                    gender='M' if data['gender_binary'] == 1 else 'F',
                    stroke_severity=data['stroke_severity'],
                    days_since_stroke=data['days_since_stroke'],
                    distance_km=data['distance_km'],
                    has_caregiver=bool(data['has_caregiver']),
                    transport_type=data['transport_type'],
                    financial_constraint=bool(data['financial_constraint'])
                )

                if patient:
                    # Update with risk score
                    update_patient_risk(
                        patient['id'],
                        risk_score,
                        risk_level
                    )
                    response['patient_id'] = patient['id']
                    print(f"Patient saved to Supabase: {patient['id']}")

            except Exception as db_error:
                print(f"Database save error: {str(db_error)}")
                # Don't fail the prediction if DB save fails

        return jsonify(response)


    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ── Batch prediction endpoint ─────────────────────────────────────────────────
# Useful when scheduling multiple patients at once
@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        patients = data.get('patients', [])

        if not patients:
            return jsonify({'error': 'No patients provided'}), 400

        results = []
        for patient in patients:
            # Call single predict logic for each patient
            with app.test_request_context(
                '/predict',
                method='POST',
                json=patient
            ):
                response = predict()
                result = response.get_json()
                result['patient_id'] = patient.get('patient_id', 'unknown')
                results.append(result)

        return jsonify({
            'total':    len(results),
            'results':  results,
            'high_risk_count': sum(1 for r in results if r.get('high_risk'))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*45)
    print("  Stroke Rehab No-Show Prediction API")
    print("  Running on http://localhost:5001")
    print("="*45 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=True)
