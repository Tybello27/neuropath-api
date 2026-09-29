% ============================================================
% ASP SCHEDULING RULES FOR STROKE REHABILITATION CLINIC
% Bello Toyyibah - LASUSTECH Final Year Project
% ============================================================

% ── GENERATE: Assign each patient exactly one slot ──────────
% For every patient, pick exactly one combination of
% therapist, room, day and hour
1 { assigned(Patient, Therapist, Room, Day, Hour) :
    therapist(Therapist),
    room(Room),
    day(Day),
    time_slot(Hour)
} 1 :- patient(Patient, _).

% ── HARD CONSTRAINTS ────────────────────────────────────────
% These must NEVER be violated

% A therapist cannot see two patients at the same time
:- assigned(P1, T, R1, D, H),
   assigned(P2, T, R2, D, H),
   P1 != P2.

% A room cannot have two patients at the same time
:- assigned(P1, T1, R, D, H),
   assigned(P2, T2, R, D, H),
   P1 != P2.

% Therapist can only work during their available hours
:- assigned(P, T, R, D, H),
   therapist_hours(T, Start, End),
   H < Start.

:- assigned(P, T, R, D, H),
   therapist_hours(T, Start, End),
   H >= End.

% ── SOFT CONSTRAINTS ────────────────────────────────────────
% These are preferences — violated only when unavoidable
% The number in square brackets is the penalty for violating

% High risk patients STRONGLY prefer morning slots (8-12)
% Penalty of 10 for each violation
:~ assigned(P, T, R, D, H),
   patient(P, high),
   H >= 12.
   [10, P]

% Low risk patients prefer afternoon slots (12 onwards)
% Penalty of 2 for each violation
:~ assigned(P, T, R, D, H),
   patient(P, low),
   H < 12.
   [2, P]

% Spread appointments across the week (avoid clustering)
% Penalty of 1 for each patient on the same day as another
:~ assigned(P1, T1, R1, D, H1),
   assigned(P2, T2, R2, D, H2),
   P1 != P2,
   P1 < P2.
   [1, D]

% ── OUTPUT ──────────────────────────────────────────────────
% Only show the assignment results
#show assigned/5.