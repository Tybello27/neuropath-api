from flask import Flask, request, jsonify
import os
import json
import clingo

app = Flask(__name__)


# â”€â”€ Health check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running', 'service': 'ASP Scheduler'})


def build_facts(patients, therapists, rooms, days):
    facts = []

    for p in patients:
        patient_id = p['id'].replace('-', '_')
        risk       = p['risk_level']
        facts.append(f"patient({patient_id}, {risk}).")

    for t in therapists:
        facts.append(f"therapist({t['id']}).")
        facts.append(
            f"therapist_hours({t['id']}, {t['start_hour']}, {t['end_hour']})."
        )

    for r in rooms:
        facts.append(f"room({r}).")

    for d in days:
        facts.append(f"day({d}).")

    for hour in range(8, 18):
        facts.append(f"time_slot({hour}).")

    return '\n'.join(facts)


@app.route('/schedule', methods=['POST'])
def schedule():
    try:
        data = request.get_json()

        required = ['patients', 'therapists', 'rooms', 'days']
        missing  = [f for f in required if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        patients   = data['patients']
        therapists = data['therapists']
        rooms      = data['rooms']
        days       = data['days']

        print(f"Scheduling {len(patients)} patients with "
              f"{len(therapists)} therapists, "
              f"{len(rooms)} rooms, "
              f"{len(days)} days")

        facts = build_facts(patients, therapists, rooms, days)

        rules_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),

            'scheduler.lp'
        )

        print(f"Loading rules from: {rules_path}")

        with open(rules_path, 'r') as f:
            rules = f.read()

        full_program = facts + '\n\n' + rules

        print("Running Clingo solver...")

        schedule_result = []
        solution_found  = False

        ctl = clingo.Control()
        ctl.add('base', [], full_program)
        ctl.ground([('base', [])])

        with ctl.solve(yield_=True) as handle:
            for model in handle:
                solution_found = True
                for atom in model.symbols(shown=True):
                    if atom.name == 'assigned' and len(atom.arguments) == 5:
                        patient_id   = str(atom.arguments[0])
                        therapist_id = str(atom.arguments[1])
                        room         = str(atom.arguments[2])
                        day          = str(atom.arguments[3])
                        hour         = int(str(atom.arguments[4]))
                        schedule_result.append({
                            'patient_id':   patient_id,
                            'therapist_id': therapist_id,
                            'room':         room,
                            'day':          day,
                            'hour':         hour,
                            'time':         f"{hour:02d}:00"
                        })
                break

        if solution_found:

            day_order = ['monday', 'tuesday', 'wednesday',
                         'thursday', 'friday']
            schedule_result.sort(
                key=lambda x: (
                    day_order.index(x['day'])
                    if x['day'] in day_order else 99,
                    x['hour']
                )
            )

            high_risk_ids = [
                p['id'].replace('-', '_')
                for p in patients
                if p['risk_level'] == 'high'
            ]

            morning_high_risk = sum(
                1 for s in schedule_result
                if s['patient_id'] in high_risk_ids
                and s['hour'] < 12
            )

            print(f"Schedule generated successfully!")
            print(f"Total appointments: {len(schedule_result)}")
            print(f"High risk in morning: "
                  f"{morning_high_risk}/{len(high_risk_ids)}")

            print("\nGenerated Schedule:")
            print("-" * 55)
            for s in schedule_result:
                risk_label = (
                    "HIGH RISK"
                    if s['patient_id'] in high_risk_ids
                    else "low risk "
                )
                print(f"  {s['day']:<12} {s['time']}  "
                      f"{s['patient_id']:<15} "
                      f"{s['therapist_id']:<15} "
                      f"{s['room']:<10} "
                      f"{risk_label}")
            print("-" * 55)

            return jsonify({
                'status':   'success',
                'schedule': schedule_result,
                'stats': {
                    'total_scheduled':   len(schedule_result),
                    'high_risk_morning': morning_high_risk,
                    'total_high_risk':   len(high_risk_ids),
                    'solver_status':     'SATISFIABLE'
                }
            })

        else:
            return jsonify({
                'status':  'error',
                'message': 'No valid schedule found â€” try adding more therapists or rooms'
            }), 500

    except FileNotFoundError:
        return jsonify({
            'status':  'error',
            'message': 'scheduler.lp not found â€” make sure asp/scheduler.lp exists'
        }), 500

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 45)
    print("  Stroke Rehab ASP Scheduling API")
    print("  Running on http://localhost:5002")
    print("=" * 45 + "\n")
    app.run(host='0.0.0.0', port=5002, debug=True)
