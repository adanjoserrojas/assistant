# FXX01 — Learning Gym Feature Implementation Plan

## 1. Objective

- Extend the existing calendar assistant so it can:
  - Receive gym check-in and checkout commands from a phone.
  - Persist scheduled-versus-actual gym behavior.
  - Maintain the workout rotation without repeating workouts.
  - Learn a preferred gym start time from historical behavior.
  - Learn how much calendar time to reserve for each workout.
  - Rank only calendar slots already proven valid by the deterministic scheduler.
  - Fall back safely to the current hard-coded `GYM` configuration.
- Keep the existing system invariant:
  - The LLM may interpret calendar semantics.
  - The LLM must not select timestamps.
  - The deterministic scheduler must generate feasible timestamps.
  - The validator must remain the final gate before any calendar write.

---

## 2. Final Technical Decisions

### 2.1 Machine-learning decision

- Do **not** build a neural network for this feature.
- Use this progression:
  1. Hard-coded configuration while data is collected.
  2. Recency-weighted statistics after enough completed sessions exist.
  3. Scikit-learn logistic regression after enough labeled scheduling outcomes exist.
- Use logistic regression to estimate:
  - `P(attend | candidate time, calendar context, workout context)`
- Do not use the model to invent a timestamp.
- Use the model only to score candidate timestamps produced by `scheduler.py`.
- Estimate duration separately:
  - MVP: per-workout 75th percentile of completed-session durations.
  - Later alternative: scikit-learn quantile regression if the duration data becomes sufficiently large and varied.

### 2.2 Library decision

- Use **neither PyTorch nor scikit-learn during the data-collection phase**.
- Use Python standard-library statistics for the first learned preference profile.
- Use **scikit-learn for model training** when the dataset meets the training gate.
- Do not use PyTorch in this feature’s production path.
- Do not add PyTorch to the current Lambda deployment package.
- Keep PyTorch installed locally only if you want it for separate deep-learning practice.

### 2.3 Why scikit-learn is the correct library

- The data is:
  - Tabular.
  - Small.
  - Mostly numeric, categorical, and Boolean.
  - Produced by one user.
- Scikit-learn directly supports:
  - Logistic regression.
  - One-hot encoding.
  - Numeric scaling.
  - Pipelines.
  - Time-ordered validation.
  - Probability metrics.
  - Quantile regression.
- PyTorch would require manually building:
  - Tensor datasets.
  - Data loaders.
  - Neural-network layers.
  - Loss functions.
  - Training loops.
  - Optimizer logic.
  - Early stopping.
  - Model serialization.
- That additional flexibility does not provide an accuracy benefit for this dataset.

### 2.4 AWS decision

- Use:
  - API Gateway HTTP API.
  - Lambda for command handling.
  - DynamoDB for gym state, sessions, decisions, and learned profiles.
  - S3 for versioned trained model artifacts.
  - EventBridge Scheduler for daily reconciliation and optional weekly retraining.
  - CloudWatch Logs for operational diagnostics.
- Do not use:
  - RDS for the MVP.
  - Stored procedures.
  - `SELECT *` CSV exports.
  - Five-minute location polling.

---

## 3. Current Repository Baseline

The current repository follows this pipeline:

```text
EventBridge 6:00 AM
    -> agent.py
    -> Google Calendar read
    -> Bedrock semantic classification
    -> scheduler.py
    -> validator.py
    -> Google Calendar write
```

Current relevant behavior:

- `agent.py`
  - Orchestrates `READ -> ANALYZE -> SCHEDULE -> VALIDATE -> WRITE`.
  - Builds activities from configuration.
  - Writes nothing when validation fails.
- `config.py`
  - Contains the hard-coded `GYM` duration and time window.
- `models.py`
  - Contains `CalendarEvent`, `Activity`, and `ScheduledActivity` dataclasses.
- `scheduler.py`
  - Generates 15-minute candidate timestamps.
  - Rejects candidates that do not fit inside a free window.
  - Scores candidates by distance from `preferred_start`.
  - Applies the meal-to-gym gap penalty.
- `validator.py`
  - Checks duration against the static configuration.
  - Checks time windows and overlaps.

Required integration correction:

- A learned gym duration can differ from `config.GYM["duration"]`.
- Therefore, the validator cannot continue treating the static duration as the only valid duration.
- The validator must validate against the resolved `Activity.duration_minutes`, with separate hard safety bounds.

---

## 4. Target Architecture

```text
iPhone Shortcut
    -> HTTPS POST /gym/commands
    -> API Gateway HTTP API
    -> GymCommand Lambda
    -> DynamoDB
         - gym state
         - gym sessions
         - scheduling decisions
         - learned preference profile

EventBridge 6:00 AM
    -> Existing agent Lambda
    -> Reconcile yesterday
    -> Load workout state
    -> Load learned gym profile/model
    -> Read Google Calendar
    -> Bedrock semantic classification
    -> Deterministic candidate generation
    -> Learned candidate scoring
    -> Validator
    -> Google Calendar write
    -> Save gym scheduling decision to DynamoDB

EventBridge weekly, only after the training gate
    -> GymTraining Lambda or local training command
    -> Read labeled decisions from DynamoDB
    -> Build feature matrix
    -> Train and evaluate logistic regression
    -> Export lightweight JSON model
    -> Store versioned artifact in S3
    -> Update active-model metadata in DynamoDB
```

---

## 5. Repository Layout

Add the following structure without moving the existing modules during the first implementation:

```text
assistant/
├── agent.py
├── calendar_client.py
├── config.py
├── deploy.py
├── llm_client.py
├── models.py
├── scheduler.py
├── validator.py
├── requirements.txt
├── planning.md
│
├── gym/
│   ├── __init__.py
│   ├── config.py
│   ├── domain.py
│   ├── commands.py
│   ├── repository.py
│   ├── rotation.py
│   ├── reconciliation.py
│   ├── preferences.py
│   ├── features.py
│   ├── scoring.py
│   ├── training.py
│   └── artifacts.py
│
├── handlers/
│   ├── __init__.py
│   ├── gym_command_handler.py
│   └── gym_training_handler.py
│
├── infrastructure/
│   ├── template.yaml
│   └── policies/
│       ├── calendar-agent-policy.json
│       ├── gym-command-policy.json
│       └── gym-training-policy.json
│
└── test/
    ├── test_gym_commands.py
    ├── test_gym_repository.py
    ├── test_gym_rotation.py
    ├── test_gym_reconciliation.py
    ├── test_gym_preferences.py
    ├── test_gym_features.py
    ├── test_gym_scoring.py
    ├── test_gym_training.py
    ├── test_scheduler_gym_learning.py
    └── test_validator_dynamic_duration.py
```

File responsibilities:

- `gym/domain.py`
  - Gym-specific dataclasses and enums.
- `gym/commands.py`
  - Parse and execute `START`, `STOP`, `SKIP`, and `STATUS` commands.
- `gym/repository.py`
  - DynamoDB access behind an interface.
- `gym/rotation.py`
  - Workout-cycle state transitions.
- `gym/reconciliation.py`
  - Resolve stale and missed sessions.
- `gym/preferences.py`
  - Compute non-ML learned start time and duration.
- `gym/features.py`
  - Produce one deterministic ordered feature vector for training and inference.
- `gym/scoring.py`
  - Load the active model and score candidates.
- `gym/training.py`
  - Train, evaluate, and export the scikit-learn model.
- `gym/artifacts.py`
  - Read and write S3 model artifacts and active-model metadata.
- `handlers/gym_command_handler.py`
  - API Gateway Lambda entry point.
- `handlers/gym_training_handler.py`
  - Optional scheduled training entry point.

---

## 6. Configuration

### 6.1 Preserve the fallback configuration

Keep the current values as safe defaults:

```json
{
  "duration": 90,
  "earliest": "07:00",
  "preferred": "17:30",
  "latest": "22:00"
}
```

### 6.2 Add gym-specific settings

Add configuration values equivalent to:

```text
GYM_MIN_DURATION_MINUTES = 45
GYM_MAX_DURATION_MINUTES = 120
GYM_DURATION_QUANTILE = 0.75
GYM_START_GRANULARITY_MINUTES = 15
GYM_PROFILE_MIN_COMPLETED_SESSIONS = 5
GYM_PROFILE_MIN_ELIGIBLE_OUTCOMES = 20
GYM_MODEL_MIN_ELIGIBLE_OUTCOMES = 60
GYM_MODEL_MIN_POSITIVE_LABELS = 10
GYM_MODEL_MIN_NEGATIVE_LABELS = 10
GYM_MODEL_RETRAIN_NEW_ROWS = 10
GYM_MODEL_MAX_AGE_DAYS = 14
GYM_RECENCY_HALF_LIFE_DAYS = 30
GYM_COMMAND_SECRET_ID = "calendar-agent/gym-command-secret"
GYM_TABLE_NAME = "calendar-assistant-data"
GYM_MODEL_BUCKET = "<account>-calendar-assistant-models"
GYM_MODEL_KEY = "gym/attendance/active-model.json"
```

### 6.3 Workout sequence

Define one explicit sequence:

```text
WORKOUT_SEQUENCE = [
    "Chest-Triceps",
    "Back-Biceps",
    "Sharms",
    "Rest"
]
```

Rules:

- Do not describe this as an eight-day rotation unless eight sequence positions exist.
- Repeated workout names are allowed when their positions are intentional.
- The stored `cycle_index` is the source of truth.
- The command text must not arbitrarily change the cycle position.

---

## 7. Domain Model

### 7.1 `GymSession`

Required fields:

```text
session_id
user_id
status
workout
cycle_index
location_code
timezone
scheduled_start
scheduled_duration_minutes
checkin_at
checkout_at
actual_duration_minutes
source
created_at
updated_at
request_ids
```

Allowed statuses:

```text
scheduled
active
completed
missed
cancelled
incomplete
rest_completed
```

Rules:

- `duration = 0` must not represent missing data.
- Missing values remain null.
- `actual_duration_minutes` exists only when both timestamps are valid.
- `rest_completed` is not a negative attendance example.
- `cancelled` requires a reason code.

### 7.2 `GymDecision`

Required fields:

```text
decision_id
user_id
local_date
timezone
workout
cycle_index
recommended_start
recommended_duration_minutes
selected_start
selected_duration_minutes
candidate_count
candidate_start_times
calendar_busy_minutes
longest_free_window_minutes
gap_before_minutes
gap_after_minutes
preference_source
model_version
created_at
outcome
outcome_reason
session_id
```

Possible `preference_source` values:

```text
hardcoded
statistical_profile
logistic_model
fallback
```

### 7.3 `GymState`

Required fields:

```text
user_id
next_workout_index
active_session_id
last_completed_session_id
last_completed_at
version
updated_at
```

Use optimistic concurrency:

- Increment `version` on every mutation.
- Use a DynamoDB conditional expression when changing the state.
- Reject or retry stale writes.

### 7.4 `GymPreferenceProfile`

Required fields:

```text
user_id
preferred_start_minutes
preferred_start_hhmm
default_duration_minutes
duration_by_workout
completed_session_count
eligible_outcome_count
computed_at
source_window_start
source_window_end
schema_version
```

### 7.5 `GymModelMetadata`

Required fields:

```text
model_name
active_version
artifact_s3_key
feature_schema_version
training_row_count
positive_label_count
negative_label_count
trained_at
validation_log_loss
baseline_log_loss
validation_brier_score
status
```

---

## 8. DynamoDB Single-Table Design

Table:

```text
calendar-assistant-data
```

Keys:

```text
PK: string
SK: string
```

Items:

```text
PK = USER#ADAN
SK = GYM_STATE

PK = USER#ADAN
SK = GYM_SESSION#<ISO_TIMESTAMP>#<SESSION_ID>

PK = USER#ADAN
SK = GYM_DECISION#<YYYY-MM-DD>

PK = USER#ADAN
SK = GYM_PROFILE

PK = MODEL#GYM_ATTENDANCE
SK = VERSION#<VERSION>

PK = MODEL#GYM_ATTENDANCE
SK = ACTIVE
```

Optional index for session lookup:

```text
GSI1PK = SESSION#<SESSION_ID>
GSI1SK = USER#ADAN
```

Access patterns:

- Read current gym state.
- Read active session by ID.
- Query sessions by date range.
- Query decisions by date range.
- Read the active preference profile.
- Read active model metadata.
- Write one session event idempotently.

Do not use a table scan in the production path.

---

## 9. Phone Command API

### 9.1 Endpoint

```text
POST /gym/commands
```

### 9.2 Request headers

```text
Content-Type: application/json
X-Gym-Command-Secret: <secret>
X-Request-Id: <UUID generated by phone shortcut>
```

### 9.3 Request body

Start:

```json
{
  "command": "START",
  "workout": "Chest-Triceps",
  "location": "CRUNCH"
}
```

Stop:

```json
{
  "command": "STOP"
}
```

Skip:

```json
{
  "command": "SKIP",
  "reason": "sick"
}
```

Status:

```json
{
  "command": "STATUS"
}
```

### 9.4 Response contract

Return:

```text
request_id
accepted
command
message
session_id
current_workout
next_workout
server_timestamp
```

### 9.5 Authentication

MVP:

- Store a random high-entropy secret in AWS Secrets Manager.
- Add the secret to the iPhone Shortcut header.
- Compare secrets with constant-time comparison.
- Never log the supplied secret.
- Rate-limit the API Gateway route.

Later improvement:

- Replace the static secret with Cognito or signed requests only if multiple users are introduced.

---

## 10. iPhone Shortcut

Create two primary shortcuts.

### 10.1 `Gym Start`

Steps:

1. Ask for workout from a fixed menu:
   - Chest-Triceps.
   - Back-Biceps.
   - Sharms.
2. Ask for location from a fixed menu:
   - CRUNCH.
   - UCF.
3. Generate a UUID.
4. Build the JSON request.
5. Call `POST /gym/commands`.
6. Add the command secret and request ID headers.
7. Display the returned confirmation.

### 10.2 `Gym Stop`

Steps:

1. Generate a UUID.
2. Send the `STOP` request.
3. Display calculated duration and next workout.

### 10.3 Optional shortcut

- `Gym Status`
  - Display active session.
  - Display today’s scheduled gym time.
  - Display next workout.

Do not build natural-language command parsing in the MVP.

---

## 11. Command State Machine

### 11.1 `START`

Execution order:

1. Validate authentication.
2. Validate request schema.
3. Check whether `X-Request-Id` was already processed.
4. Read `GYM_STATE` with strongly consistent read.
5. Reject when `active_session_id` already exists.
6. Resolve the expected workout from `next_workout_index`.
7. Compare the supplied workout with the expected workout.
8. On mismatch:
   - Reject by default.
   - Return the expected workout.
9. Create a `GymSession` with status `active`.
10. Set `checkin_at` from the Lambda server timestamp.
11. Store the active session ID in `GYM_STATE` conditionally.
12. Persist the request ID.
13. Return confirmation.

### 11.2 `STOP`

Execution order:

1. Validate authentication and idempotency.
2. Read `GYM_STATE`.
3. Reject when no active session exists.
4. Load the active session.
5. Set `checkout_at` from the server timestamp.
6. Calculate actual duration.
7. Reject clearly invalid durations:
   - Negative.
   - Greater than a configurable maximum such as six hours.
8. Mark the session `completed`.
9. Advance the workout cycle.
10. Clear `active_session_id`.
11. Update `last_completed_session_id` and `last_completed_at`.
12. Recompute the statistical preference profile.
13. Return duration and next workout.

### 11.3 `SKIP`

Execution order:

1. Require a reason.
2. Reject when a session is already active.
3. Record the scheduled opportunity as cancelled.
4. Apply reason-specific rotation behavior.

Recommended reason behavior:

```text
rest              -> advance only when current sequence entry is Rest
sick              -> do not advance
injury            -> do not advance
travel            -> do not advance
intentional_skip  -> do not advance
schedule_conflict -> do not advance
```

### 11.4 `STATUS`

Return:

- Active session, when present.
- Current workout.
- Next workout.
- Most recent completed session.
- Statistical preferred time.
- Active model version.

---

## 12. Workout Rotation

Implement rotation as a pure function.

Inputs:

```text
current_index
event_type
session_status
```

Outputs:

```text
next_index
advanced: true|false
reason
```

Advancement rules:

- Completed workout:
  - Advance one position.
- Completed rest entry:
  - Advance one position.
- Missed workout:
  - Do not advance.
- Cancelled workout:
  - Do not advance by default.
- Active session:
  - Do not advance until checkout completes.
- Duplicate command:
  - Do not advance.
- Calendar event creation:
  - Do not advance.

Test the cycle as a state machine independently from AWS.

---

## 13. Daily Reconciliation

Run reconciliation before scheduling the current day.

Steps:

1. Determine yesterday in `America/New_York`.
2. Read yesterday’s `GymDecision`.
3. Read any linked session.
4. Apply one outcome:

```text
checkin exists and checkout exists -> completed
checkin exists and checkout missing -> incomplete
no checkin and expected workout     -> missed
rest sequence entry                 -> rest_completed
explicit cancellation               -> cancelled
no decision                         -> no outcome
```

5. For an `incomplete` session:
   - Do not invent an exact duration.
   - Mark it excluded from duration training.
   - Count it as attended only when a valid check-in exists.
6. Update the linked `GymDecision.outcome`.
7. Do not overwrite an already finalized outcome.
8. Emit one structured CloudWatch log entry.

---

## 14. Integrating Gym State into `agent.py`

Modify the daily flow in this order:

```text
READ GYM STATE
    -> RECONCILE YESTERDAY
    -> RESOLVE TODAY'S WORKOUT ENTRY
    -> LOAD GYM PREFERENCE/MODEL
    -> READ CALENDAR
    -> LLM ANALYSIS
    -> DETERMINE REQUIRED ACTIVITIES
    -> BUILD DYNAMIC ACTIVITIES
    -> GENERATE CANDIDATES
    -> SCORE CANDIDATES
    -> VALIDATE
    -> WRITE CALENDAR
    -> SAVE GYM DECISION
```

Detailed steps:

1. Add `gym_repository` initialization.
2. Run reconciliation before resolving today’s workout.
3. Read `GYM_STATE`.
4. Resolve the current workout sequence entry.
5. If the entry is `Rest`:
   - Remove gym from required activities.
   - Store a rest decision if needed.
6. If gym is already semantically satisfied by an existing calendar event:
   - Do not create a second gym event.
   - Record the existing event as the selected gym decision when its timestamps are usable.
7. Resolve dynamic preference:
   - Model when valid.
   - Statistical profile when valid.
   - Static config otherwise.
8. Build the gym `Activity` with dynamic preferred start and duration.
9. Schedule normally.
10. Validate normally using the resolved activity constraints.
11. After a successful calendar write, store a `GymDecision`.
12. If no gym candidate fits:
   - Store an unplaced decision.
   - Do not force a conflicting event.

---

## 15. Scheduler Refactor

### 15.1 Preserve candidate generation

Do not change:

- Free-window calculation.
- 15-minute grid generation.
- Requirement that the full duration fits.
- Existing overlap protection.
- Existing meal-to-gym penalty.

### 15.2 Expose gym candidates

Refactor so the caller can obtain:

```text
candidate start
candidate end
base preference-distance score
meal-gap penalty
free-window context
```

Possible interface:

```text
generate_candidates(activity, events, day, timezone)
score_candidate(activity, candidate, context, gym_scorer=None)
select_best_candidate(scored_candidates)
```

### 15.3 Combined scoring rule

For non-gym activities:

```text
final_score = existing deterministic score
```

For gym with no valid model:

```text
final_score = distance_from_statistical_or_static_preference
            + existing_meal_gap_penalty
```

For gym with a valid model:

```text
attendance_probability = model(candidate_features)

final_score = -log(max(attendance_probability, epsilon)) * probability_weight
            + distance_from_learned_preference * preference_weight
            + existing_meal_gap_penalty
```

Rules:

- Lower score remains better to preserve current scheduler behavior.
- Clamp probability away from zero before taking a logarithm.
- Keep deterministic tie-breaking toward the earlier candidate.
- The ML model cannot make an invalid candidate valid.

---

## 16. Validator Refactor

Problem:

- The current validator compares generated duration with the static config duration.
- A learned 75-minute gym activity would fail when the static value is 90 minutes.

Required change:

1. Pass the resolved `Activity` definitions into `validate_schedule`.
2. Build a map by activity name.
3. Validate generated duration against the matching resolved activity.
4. Add gym hard bounds:
   - Minimum 45 minutes.
   - Maximum 120 minutes.
5. Continue validating:
   - Start before end.
   - Configured earliest/latest boundaries.
   - Day boundaries.
   - Generated-to-generated overlap.
   - Generated-to-existing overlap.
6. Fail closed:
   - Any validation problem writes zero events.

---

## 17. Data Collection Phase

### 17.1 No ML library required

During this phase:

- Do not import PyTorch.
- Do not import scikit-learn.
- Do not create tensors.
- Persist normalized Python values to DynamoDB.

### 17.2 Eligible scheduling outcome

A row is eligible for attendance training only when:

- A workout was expected.
- At least one valid candidate existed.
- A gym time was selected or recognized from an existing event.
- The final outcome is known.
- The outcome was not a rest day.
- The outcome was not excluded for an external reason such as illness or injury.

### 17.3 Attendance label

Use:

```text
1 -> valid gym check-in occurred
0 -> gym was scheduled but no valid check-in occurred
```

Do not use checkout completion as the attendance target.

- Checkout is needed for duration.
- Check-in is sufficient to prove attendance.

### 17.4 Duration target

Use only sessions where:

- Check-in exists.
- Checkout exists.
- Duration passes sanity checks.
- Session was not manually corrected without audit metadata.

### 17.5 Selection-bias warning

- The model can only learn from times the current policy selected.
- Record the full candidate set on every decision.
- Do not label unselected candidates as missed.
- Defer deliberate exploration until the baseline system is stable.

---

## 18. Statistical Preference Phase

Activate after:

```text
at least 5 completed sessions for a global estimate
and
at least 20 eligible outcomes for a stable adaptive profile
```

### 18.1 Preferred start-time calculation

Steps:

1. Query recent completed sessions.
2. Convert each check-in time to minutes after local midnight.
3. Assign recency weights using the configured half-life.
4. Calculate a weighted median.
5. Round to the nearest 15 minutes.
6. Clamp inside `GYM.earliest` and `GYM.latest`.
7. Store the result in `GymPreferenceProfile`.

Weekday specialization:

- Use weekday-specific preference only after at least five completed sessions exist for that weekday.
- Otherwise use the global preference.

### 18.2 Duration calculation

Steps:

1. Query valid completed sessions.
2. Group by workout.
3. Calculate the 75th percentile for each workout.
4. Round upward to the nearest five minutes.
5. Clamp to 45–120 minutes.
6. Use the global 75th percentile when a workout has fewer than five valid completions.
7. Use static 90 minutes when the global dataset is still insufficient.

### 18.3 Recalculation schedule

- Recalculate after every completed session.
- The operation is cheap and deterministic.
- Store the source-window timestamps for auditing.

---

## 19. Logistic Regression Training Gate

Train only when all conditions are true:

```text
eligible outcomes >= 60
positive labels >= 10
negative labels >= 10
data spans >= 6 weeks
at least 3 distinct weekdays contain outcomes
feature schema version is supported
```

If any condition fails:

- Do not train.
- Keep using the statistical profile.
- Log the unmet conditions.

Why both label classes are required:

- Logistic regression cannot learn attendance discrimination when every label is identical.
- A model trained on nearly all positive outcomes may produce misleading probabilities.

---

## 20. Feature Schema

Define an immutable ordered feature schema.

### 20.1 Numeric features

```text
start_time_sin
start_time_cos
scheduled_duration_minutes
calendar_busy_minutes
longest_free_window_minutes
gap_before_minutes
gap_after_minutes
days_since_previous_completed_session
recent_7_day_completed_count
recent_14_day_missed_count
```

### 20.2 Boolean features

```text
is_weekend
previous_opportunity_attended
existing_semantic_gym_event
is_ucf_location
is_crunch_location
```

### 20.3 Categorical features

```text
weekday
workout
location_code
```

Encode categorical values using fixed categories:

```text
weekday:
Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday

workout:
Chest-Triceps, Back-Biceps, Sharms

location_code:
CRUNCH, UCF, UNKNOWN
```

### 20.4 Do not include

- Raw user text.
- Full event titles.
- Exact addresses.
- Future outcome information.
- Checkout-derived duration when predicting attendance.
- Any feature calculated after the recommendation was made.

This prevents target leakage.

---

## 21. Scikit-learn Training Pipeline

Install for the training environment:

```text
scikit-learn
pandas
numpy
```

Do not automatically add these dependencies to the current calendar-agent zip Lambda.

Training sequence:

1. Query eligible `GymDecision` records in chronological order.
2. Join the linked outcome/session data.
3. Build one row per historical selected gym decision.
4. Validate required columns and null behavior.
5. Sort by decision timestamp.
6. Reserve the newest 20% as a chronological holdout set.
7. Fit preprocessing only on the training portion.
8. Use:
   - `ColumnTransformer`.
   - `OneHotEncoder(handle_unknown="ignore")`.
   - `StandardScaler` for numeric features.
   - `LogisticRegression` with L2 regularization.
9. Start with conservative defaults rather than hyperparameter search.
10. Evaluate on the chronological holdout.
11. Compare with a constant base-rate probability baseline.
12. Promote only when the model beats the baseline.

Recommended first estimator:

```text
LogisticRegression(
    regularization = L2,
    solver = lbfgs,
    class weighting = evaluate before enabling,
    max iterations = sufficient for convergence
)
```

Do not randomly shuffle the entire dataset before evaluation.

---

## 22. Evaluation

### 22.1 Metrics

Track:

```text
log loss
Brier score
ROC AUC only when both classes exist in the holdout
precision and recall at the attendance threshold
calibration by probability bucket
```

Primary promotion metric:

- Chronological holdout log loss.

Baseline:

- Predict the training-set attendance rate for every holdout row.

Promotion rule:

```text
model_log_loss < baseline_log_loss
and
no feature-schema mismatch
and
no training warning indicating non-convergence
and
holdout contains both classes
```

Additional safeguard:

- Do not replace a functioning active model when the new model is worse.
- Mark the candidate model `rejected` with metrics retained for audit.

### 22.2 Time-ordered cross-validation

After the dataset becomes larger:

- Add `TimeSeriesSplit` for diagnostics.
- Keep final promotion based on the newest untouched holdout window.
- Never train on future decisions and test on older decisions.

---

## 23. Lightweight Model Artifact

Do not require scikit-learn inside the daily scheduler Lambda.

Export a JSON artifact containing:

```text
model_name
model_version
feature_schema_version
trained_at
training_row_count
intercept
ordered_feature_names
ordered_coefficients
numeric_means
numeric_scales
categorical_encoding_map
validation_metrics
```

Inference formula:

```text
z = intercept + sum(feature[i] * coefficient[i])
probability = 1 / (1 + exp(-z))
```

Benefits:

- Small artifact.
- Fast cold start.
- No PyTorch runtime.
- No scikit-learn runtime.
- No pickle or joblib compatibility requirement in the calendar agent.
- Easy model inspection and rollback.

Rules:

- Training uses scikit-learn.
- Inference uses deterministic pure Python.
- The feature encoder must be shared logically between both environments.
- Add a golden-vector test proving training-side and inference-side features are identical.

---

## 24. S3 Artifact Layout

```text
s3://<bucket>/gym/attendance/versions/<version>/model.json
s3://<bucket>/gym/attendance/versions/<version>/metrics.json
s3://<bucket>/gym/attendance/active-model.json
```

Promotion sequence:

1. Write versioned model.
2. Write versioned metrics.
3. Update DynamoDB model metadata.
4. Replace the `active-model.json` pointer last.
5. Retain at least the previous five accepted versions.

Rollback:

- Change the active pointer to a prior accepted version.
- Do not retrain during rollback.

---

## 25. Inference and Candidate Ranking

For every valid gym candidate:

1. Build features using only information available at scheduling time.
2. Load the active JSON artifact once per Lambda execution environment.
3. Verify `feature_schema_version`.
4. Compute attendance probability.
5. Store probability with the candidate for debugging.
6. Combine it with deterministic penalties.
7. Select the minimum final score.
8. Record:
   - Candidate count.
   - Candidate starts.
   - Selected candidate.
   - Selected probability.
   - Model version.
   - Preference source.

Fallback order:

```text
valid active logistic model
    -> statistical profile
    -> hard-coded GYM configuration
```

Fallback triggers:

- S3 unavailable.
- Artifact malformed.
- Feature schema mismatch.
- Model older than allowed.
- No candidate can be encoded.
- Model returns NaN or infinite probability.
- Training gate no longer considered valid.

The scheduler must continue operating after any ML failure.

---

## 26. Duration Prediction

### 26.1 MVP

Use per-workout 75th percentile.

Reason:

- The calendar needs a reservation long enough to finish most sessions.
- The mean can under-allocate when durations have a long upper tail.
- The median may still be slightly aggressive for calendar reservation.

### 26.2 Future quantile regression

Consider `QuantileRegressor` only when:

```text
completed duration rows >= 60
at least 15 rows exist for each common workout or workout is encoded categorically
calendar-context variation exists
chronological validation beats the percentile baseline
```

Possible target:

```text
actual_duration_minutes
```

Possible quantile:

```text
0.75
```

Do not build a second model until the percentile approach has been measured.

---

## 27. AWS Infrastructure Steps

### Step 1 — DynamoDB

1. Create `calendar-assistant-data`.
2. Partition key: `PK` string.
3. Sort key: `SK` string.
4. Use on-demand billing.
5. Enable point-in-time recovery.
6. Add the optional session-ID GSI only when required.
7. Add resource tags.

### Step 2 — S3

1. Create a private model bucket.
2. Block all public access.
3. Enable versioning.
4. Enable server-side encryption.
5. Add a lifecycle rule only after retention behavior is understood.

### Step 3 — Gym command Lambda

1. Create Python Lambda.
2. Handler: `handlers.gym_command_handler.lambda_handler`.
3. Grant only:
   - DynamoDB item read/write on the assistant table.
   - Secrets Manager read for the command secret.
   - CloudWatch logging.
4. Set environment variables.
5. Set timeout conservatively, such as 10 seconds.
6. Set reserved concurrency to a small value because this is a single-user API.

### Step 4 — API Gateway HTTP API

1. Create HTTP API.
2. Create `POST /gym/commands`.
3. Integrate with the gym command Lambda.
4. Enable access logging without sensitive headers.
5. Configure throttling.
6. Deploy a `prod` stage.
7. Test with a manually generated request before building the Shortcut.

### Step 5 — Existing calendar-agent Lambda permissions

Add least-privilege access for:

- Read/write relevant DynamoDB items.
- Read active gym model artifact from S3.
- Existing Calendar secret and Bedrock permissions remain unchanged.

### Step 6 — Training environment

MVP choice:

- Run `python -m gym.training` locally when the training gate is first reached.

Later scheduled choice:

- Use a separate container-image Lambda containing scikit-learn, pandas, and NumPy.
- Do not enlarge the current calendar-agent Lambda package.

### Step 7 — EventBridge

Existing schedule:

- Keep the 6:00 AM daily scheduler.

Optional training schedule:

```text
weekly, Sunday evening, America/New_York
```

Training handler behavior:

- Exit successfully without training when the gate is unmet.
- Train only when at least 10 new labeled rows exist or the model is stale.

---

## 28. Local Development Sequence

Implement in this order.

### Phase A — Pure domain behavior

1. Create `gym/domain.py`.
2. Create gym enums and dataclasses.
3. Create `gym/rotation.py`.
4. Write rotation tests.
5. Create a repository protocol/interface.
6. Create an in-memory repository for tests.

Exit criteria:

- Rotation rules pass without AWS.
- Duplicate state transitions are idempotent.

### Phase B — Command behavior

1. Create command parser.
2. Implement `START` with in-memory repository.
3. Implement `STOP`.
4. Implement `SKIP`.
5. Implement `STATUS`.
6. Add request-ID idempotency.
7. Add state-version conditional logic.
8. Test all invalid transitions.

Exit criteria:

- A complete local start/stop session produces correct duration and rotation.

### Phase C — DynamoDB

1. Create DynamoDB repository implementation.
2. Keep domain behavior independent of `boto3`.
3. Add integration tests using mocked AWS or a test table.
4. Confirm strongly consistent state reads where required.
5. Confirm conditional writes reject stale state versions.

Exit criteria:

- Commands work against DynamoDB without the phone API.

### Phase D — Phone API

1. Add Lambda handler.
2. Add request authentication.
3. Add API Gateway route.
4. Test with `curl` or an API client.
5. Create iPhone Shortcuts.
6. Verify server timestamps are stored in UTC plus local timezone context.

Exit criteria:

- Phone start and stop commands create one correct session.

### Phase E — Daily scheduler integration

1. Add gym-state loading to `agent.py`.
2. Add rest-day exclusion.
3. Add dynamic gym activity resolution.
4. Refactor scheduler candidate exposure.
5. Refactor validator dynamic duration.
6. Save a `GymDecision` after successful scheduling.
7. Add reconciliation before daily scheduling.

Exit criteria:

- Existing meal tests still pass.
- Gym rest days create no gym event.
- Dynamic gym duration passes validation.
- Rerunning the same day remains idempotent.

### Phase F — Statistical learning

1. Implement completed-session query.
2. Implement weighted-median start time.
3. Implement per-workout 75th-percentile duration.
4. Store `GymPreferenceProfile`.
5. Add profile fallback hierarchy.
6. Add deterministic tests with fixed datasets.

Exit criteria:

- Profile activates only after thresholds.
- Outputs remain clamped inside safety bounds.

### Phase G — Scikit-learn trainer

1. Install scikit-learn in the training environment.
2. Build the decision/session dataset.
3. Add feature-schema validation.
4. Add chronological split.
5. Add preprocessing pipeline.
6. Train logistic regression.
7. Calculate metrics and baseline.
8. Reject inferior models.
9. Export accepted model to JSON.
10. Upload versioned artifacts to S3.

Exit criteria:

- Training is repeatable with fixed input data.
- Exported JSON reproduces scikit-learn probabilities within a strict tolerance.

### Phase H — Production model scoring

1. Add S3 artifact loading.
2. Cache artifact per warm Lambda instance.
3. Add pure-Python sigmoid inference.
4. Add model candidate scoring.
5. Record model version in every decision.
6. Test every fallback path.

Exit criteria:

- Model failure never prevents deterministic scheduling.
- Invalid candidates remain impossible to select.

---

## 29. Testing Matrix

### Command tests

- Valid start.
- Duplicate start request ID.
- Start while already active.
- Start with wrong workout.
- Stop active session.
- Duplicate stop.
- Stop with no active session.
- Skip with no reason.
- Status with and without active session.

### Rotation tests

- Completed workout advances once.
- Missed workout does not advance.
- Cancelled workout does not advance.
- Rest entry advances correctly.
- Duplicate completion does not advance twice.
- Sequence wraps to index zero.

### Data tests

- Null duration remains null.
- Rest days are excluded from attendance labels.
- Illness/injury exclusions are honored.
- Incomplete sessions are excluded from duration training.
- All timestamps retain timezone meaning.

### Scheduler tests

- Static fallback preserves current behavior.
- Statistical preference changes ranking.
- Logistic probability changes ranking only among valid candidates.
- Meal-gap penalty remains effective.
- No free slot returns `unplaced`.
- Existing gym event prevents duplicate scheduling.

### Validator tests

- Dynamic duration accepted.
- Duration below hard minimum rejected.
- Duration above hard maximum rejected.
- Existing overlap rejected.
- Generated overlap rejected.
- Start outside gym bounds rejected.

### Model tests

- Training gate rejects too few rows.
- Training gate rejects one-class data.
- Chronological split preserves order.
- Unknown categories are handled.
- Feature leakage columns are absent.
- JSON inference matches scikit-learn output.
- Worse model is not promoted.
- Schema mismatch causes fallback.

### Operational tests

- DynamoDB temporary failure.
- S3 temporary failure.
- Secrets Manager failure.
- Bedrock failure.
- Google Calendar failure.
- Lambda retry of the same command.
- Daylight-saving-time transition.

---

## 30. Observability

Use structured logs.

Command log fields:

```text
request_id
command
accepted
rejection_reason
session_id
state_version
latency_ms
```

Scheduling log fields:

```text
local_date
workout
preference_source
candidate_count
selected_start
selected_duration
model_version
fallback_reason
validation_passed
```

Training log fields:

```text
training_row_count
positive_count
negative_count
feature_schema_version
candidate_model_version
baseline_log_loss
model_log_loss
promoted
rejection_reason
```

Do not log:

- Command secrets.
- Google credentials.
- Full authorization headers.
- Raw private calendar descriptions unless already intentionally supported.

Optional CloudWatch alarms:

- Command Lambda errors.
- Daily scheduler errors.
- Consecutive days without a successful scheduler invocation.
- Model artifact load failures.
- DynamoDB conditional-write conflict spike.

---

## 31. Security and Privacy

- Store the phone command secret in Secrets Manager.
- Use least-privilege IAM per Lambda.
- Encrypt DynamoDB and S3 with AWS-managed encryption initially.
- Block public S3 access.
- Do not store exact gym coordinates in the MVP.
- Store `CRUNCH`, `UCF`, or `UNKNOWN` location codes.
- Treat timestamps and routine data as personal behavioral data.
- Add a deletion script capable of deleting all gym items for the user.
- Keep audit metadata for manual corrections.
- Never deserialize untrusted pickle/joblib artifacts in the daily agent.

---

## 32. Deployment and Rollback

Deployment order:

1. Merge domain and test-only modules.
2. Deploy DynamoDB table.
3. Deploy command Lambda.
4. Deploy API Gateway route.
5. Test phone logging for several sessions.
6. Deploy scheduler integration in hard-coded mode.
7. Enable statistical profile after validation.
8. Collect enough eligible outcomes.
9. Train the first logistic model locally.
10. Deploy JSON scoring in shadow mode.
11. Compare shadow ranking with deterministic ranking.
12. Enable model ranking only after shadow results are reasonable.

Shadow mode:

- Compute model scores.
- Record what the model would select.
- Continue booking the statistical-profile selection.
- Use this to identify feature or calibration problems safely.

Rollback switches:

```text
GYM_LEARNING_ENABLED=false
GYM_MODEL_SCORING_ENABLED=false
GYM_COMMAND_API_ENABLED=false
```

Rollback behavior:

- Disabling learning returns immediately to static `config.GYM` behavior.
- Historical data remains intact.
- The existing scheduler remains functional.

---

## 33. Definition of Done

### MVP complete

- Phone can send `START` and `STOP`.
- DynamoDB stores one normalized session.
- Duplicate commands are idempotent.
- Workout rotation is correct.
- Rest day prevents gym scheduling.
- Daily decisions record scheduled-versus-actual behavior.
- Dynamic duration does not break validation.
- Static fallback always works.

### Statistical learning complete

- Preferred time updates from completed sessions.
- Duration updates by workout.
- Thresholds prevent unstable estimates.
- Learned values stay within hard bounds.

### ML complete

- Dataset passes the training gate.
- Scikit-learn logistic regression trains chronologically.
- Model beats the constant-probability baseline.
- JSON inference matches scikit-learn.
- Model scores only valid scheduler candidates.
- S3/model failures trigger deterministic fallback.
- Model version and decision context are auditable.

---

## 34. Immediate Implementation Checklist

1. Create a feature branch.
2. Preserve the current passing test suite.
3. Add `gym/domain.py`.
4. Add `gym/rotation.py` and tests.
5. Add repository protocol and in-memory repository.
6. Implement local `START`, `STOP`, `SKIP`, and `STATUS` behavior.
7. Add DynamoDB table and repository.
8. Add command Lambda and API Gateway.
9. Build iPhone `Gym Start` and `Gym Stop` Shortcuts.
10. Collect real sessions while the scheduler remains hard-coded.
11. Add daily reconciliation.
12. Add `GymDecision` persistence.
13. Refactor validator to accept dynamic duration.
14. Add statistical preference profile.
15. Wait for the logistic-regression training gate.
16. Install scikit-learn in a separate training environment.
17. Train and evaluate logistic regression.
18. Export a JSON model.
19. Run model scoring in shadow mode.
20. Enable model ranking only after validation.

---

## 35. Tooling Answer

Use this exact policy:

```text
Data collection and statistical profile:
    Python standard library; neither PyTorch nor scikit-learn required.

Attendance model training:
    scikit-learn.

Daily production inference:
    Pure Python reading a lightweight JSON coefficient artifact.

Neural-network experimentation:
    PyTorch, but outside this feature's production path.
```

Do not uninstall PyTorch merely because it is not used here.

Do not let the fact that PyTorch is already downloaded influence the architecture. The model and data requirements should determine the library.

---

## 36. Reference Material

- Existing assistant repository:
  - https://github.com/adanjoserrojas/assistant
- Scikit-learn logistic regression:
  - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- Scikit-learn pipelines and column transformations:
  - https://scikit-learn.org/stable/modules/compose.html
- Scikit-learn time-series split:
  - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Scikit-learn quantile regression:
  - https://scikit-learn.org/stable/modules/linear_model.html#quantile-regression
- Scikit-learn model persistence considerations:
  - https://scikit-learn.org/stable/model_persistence.html
- PyTorch documentation:
  - https://docs.pytorch.org/docs/stable/
- AWS API Gateway, Lambda, and DynamoDB serverless API tutorial:
  - https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-dynamo-db.html
