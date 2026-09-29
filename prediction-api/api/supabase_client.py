from supabase import create_client
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file")

# Create Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print(f"Connected to Supabase: {SUPABASE_URL}")


# ── Patient functions ─────────────────────────────────────────────────────────
def create_patient(name, phone, age, gender, stroke_severity,
                   days_since_stroke, distance_km, has_caregiver,
                   transport_type, financial_constraint):
    response = supabase.table('patients').insert({
        'name':                 name,
        'phone':                phone,
        'age':                  age,
        'gender':               gender,
        'stroke_severity':      stroke_severity,
        'days_since_stroke':    days_since_stroke,
        'distance_km':          distance_km,
        'has_caregiver':        has_caregiver,
        'transport_type':       transport_type,
        'financial_constraint': financial_constraint
    }).execute()
    return response.data[0] if response.data else None


def get_patient(patient_id):
    response = supabase.table('patients')\
        .select('*')\
        .eq('id', patient_id)\
        .execute()
    return response.data[0] if response.data else None


def get_all_patients():
    response = supabase.table('patients')\
        .select('*')\
        .order('created_at', desc=True)\
        .execute()
    return response.data


def update_patient_risk(patient_id, risk_score, risk_level):
    response = supabase.table('patients')\
        .update({
            'risk_score': risk_score,
            'risk_level': risk_level
        })\
        .eq('id', patient_id)\
        .execute()
    return response.data


# ── Appointment functions ─────────────────────────────────────────────────────
def create_appointment(patient_id, therapist_id, room,
                       day, hour, risk_score):
    # Convert day and hour to a proper timestamp
    day_map = {
        'monday': '2026-03-16',
        'tuesday': '2026-03-17',
        'wednesday': '2026-03-18',
        'thursday': '2026-03-19',
        'friday': '2026-03-20'
    }
    date_str = day_map.get(day, '2026-03-16')
    scheduled_time = f"{date_str}T{hour:02d}:00:00"

    response = supabase.table('appointments').insert({
        'patient_id':     patient_id,
        'therapist_id':   therapist_id,
        'room':           room,
        'day':            day,
        'hour':           hour,
        'scheduled_time': scheduled_time,
        'risk_score':     risk_score
    }).execute()
    return response.data[0] if response.data else None


def get_all_appointments():
    response = supabase.table('appointments')\
        .select('*, patients(name, phone, risk_level)')\
        .order('scheduled_time')\
        .execute()
    return response.data


def get_upcoming_appointments(risk_level=None):
    query = supabase.table('appointments')\
        .select('*, patients(name, phone, risk_level, risk_score)')\
        .is_('attended', 'null')\
        .order('scheduled_time')
    if risk_level:
        query = query.eq('patients.risk_level', risk_level)
    return query.execute().data


def mark_attendance(appointment_id, attended):
    response = supabase.table('appointments')\
        .update({'attended': attended})\
        .eq('id', appointment_id)\
        .execute()
    return response.data


def mark_reminder_sent(appointment_id):
    response = supabase.table('appointments')\
        .update({
            'reminder_sent':  True,
            'reminder_count': supabase.table('appointments')
                .select('reminder_count')
                .eq('id', appointment_id)
                .execute().data[0]['reminder_count'] + 1
        })\
        .eq('id', appointment_id)\
        .execute()
    return response.data


# ── Therapist functions ───────────────────────────────────────────────────────
def get_all_therapists():
    response = supabase.table('therapists')\
        .select('*')\
        .execute()
    return response.data


def get_all_rooms():
    response = supabase.table('rooms')\
        .select('*')\
        .execute()
    return response.data


# ── Reminder functions ────────────────────────────────────────────────────────
def log_reminder(appointment_id, patient_id,
                 reminder_type, message_type):
    response = supabase.table('reminders').insert({
        'appointment_id': appointment_id,
        'patient_id':     patient_id,
        'reminder_type':  reminder_type,
        'message_type':   message_type,
        'delivered':      True
    }).execute()
    return response.data


# ── Statistics ────────────────────────────────────────────────────────────────
def get_stats():
    total_patients    = len(get_all_patients())
    all_appointments  = get_all_appointments()
    total_appointments = len(all_appointments)
    attended          = sum(1 for a in all_appointments
                           if a.get('attended') is True)
    no_shows          = sum(1 for a in all_appointments
                           if a.get('attended') is False)
    attendance_rate   = (attended / total_appointments * 100
                        if total_appointments > 0 else 0)

    return {
        'total_patients':    total_patients,
        'total_appointments': total_appointments,
        'attended':          attended,
        'no_shows':          no_shows,
        'attendance_rate':   round(attendance_rate, 1)
    }