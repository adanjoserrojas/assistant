# Assistant

Reads your Google Calendar every morning at 6:00 AM, finds free time, and schedules breakfast, lunch, dinner, and gym around what's already there.

One LLM call classifies existing events (does "Lunch with Alex" already cover lunch?). Deterministic Python does all time arithmetic and calendar writes. The LLM never picks a timestamp.

```
EventBridge 6AM -> Lambda -> Google Calendar (read)
                          -> Bedrock (classify)
                          -> scheduler -> validator -> Google Calendar (write)
```

Gym is a second loop. An iPhone Shortcut logs each session to DynamoDB; those logs train a model that decides how long the next session should be and when to put it. See [Gym ML pipeline](#gym-ml-pipeline).

## Requirements

- Python 3.13
- Google Cloud project with Calendar API enabled
- AWS account with Bedrock model access

## 1. Google service account

1. GCP Console -> IAM & Admin -> Service Accounts -> Create.
2. Keys -> Add Key -> JSON. Save as `service-account.json` in the repo root.
3. Copy the service account email (`...@....iam.gserviceaccount.com`).
4. Google Calendar -> Settings for my calendar -> Share with specific people -> add that email -> **Make changes to events**.

No OAuth consent screen, no browser flow. Do not use `primary` as the calendar ID — it resolves to the service account's own empty calendar.

## 2. Local setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -r requirements.txt
```

```bash
setx CALENDAR_ID you@example.com   # Windows, reopen terminal after
export CALENDAR_ID=you@example.com # macOS/Linux
```

`config.py` reads `TABLE_NAME` at import with no default, so it must be set before anything imports it — including the tests.

Verify the calendar connection:

```bash
python calendar_client.py         # prints today's events
python agent.py --dry-run         # prints the plan, writes nothing
python agent.py                   # writes to the calendar
python -m pytest test -q          # 103 tests, no network
```

## 3. Bedrock

Bedrock console -> Model access -> enable a model. Default is `deepseek.v3.2` (`ON_DEMAND`, no inference profile, no use-case form). Override with `BEDROCK_MODEL_ID`.

The agent runs without the LLM — if the call fails it schedules all four activities and logs a warning. It never blocks the schedule.

## 4. AWS deploy

**Secret** — Secrets Manager -> Store a new secret -> Other -> Plaintext -> paste all of `service-account.json`. Name it `calendar-agent/google-sa`.

**Execution role** — IAM -> Roles -> Create -> Lambda. Attach `AWSLambdaBasicExecutionRole` plus:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "secretsmanager:GetSecretValue",
      "Resource": "arn:aws:secretsmanager:<region>:<account>:secret:calendar-agent/google-sa-*"
    },
    {
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": "arn:aws:bedrock:<region>::foundation-model/deepseek.v3.2"
    }
  ]
}
```

The trailing `-*` is required — Secrets Manager appends a random suffix to the ARN.

**Package**

```bash
python deploy/deploy.py            # build/calendar-agent.zip (~7.5 MB)
python deploy/deploy.py --upload   # needs lambda:UpdateFunctionCode
```

**Function** — create, then upload the zip:

| Setting | Value |
|---|---|
| Runtime | Python 3.13 |
| Architecture | x86_64 |
| Handler | `agent.lambda_handler` |
| Timeout | 2 min (default 3s will time out) |
| Memory | 512 MB (CPU scales with memory) |
| Role | the role above |

Environment variables:

```
CALENDAR_ID         = you@example.com
GOOGLE_SA_SECRET_ID = calendar-agent/google-sa
```

Test payloads: `{"dry_run": true}` first, then `{}`, then `{}` again — the third must write 0 events.

**Schedule** — EventBridge -> Scheduler -> Create schedule. Recurring, cron `cron(0 6 * * ? *)`, timezone `America/New_York`, flexible window Off, target Lambda Invoke. Let it create its own role.

Pick the named timezone, not a fixed UTC offset, or the schedule drifts an hour at DST.

## Gym sessions

`handlers/gym_command_handler.py` sits behind API Gateway and takes four commands from an iPhone Shortcut. Every request carries an `x-gym-command-secret` header compared against `GYM_COMMAND_SECRET` with `hmac.compare_digest`.

| Command | Effect |
|---|---|
| `START` | Opens a session. Rejects it unless `workout` matches the current rotation entry and `location` is in `VALID_GYM_LOCATIONS`. |
| `STOP` | Closes it, computes the real duration, advances the rotation. |
| `SKIP` | Advances the rotation without a session. Requires a `reason` from `REASONS_TO_SKIP`. |
| `STATUS` | Current and next workout, active session, last completed. |

The rotation is `WORKOUTS` in `config.py` — an eight-entry cycle, so the same workout lands on different weekdays each pass.

`SKIP` takes two reasons, and they are not interchangeable:

| `reason` | When it is accepted | Record written |
|---|---|---|
| `rest` | Only when the current rotation entry is `Rest-days`. | `status: rest_completed` |
| `injured` | Any entry, including a training day. | `status: skipped_due_to_injury` |

`rest` stays strict on purpose — it is the normal path through a `Rest-days` entry, and letting it fire on a training day would quietly turn every missed workout into a logged rest day. `injured` is the deliberate escape hatch. Injuries land on days the rotation says you should train, and without it the only options are lying with `rest` or leaving the rotation stuck until you heal.

Both reasons advance the rotation. That is a knowing deviation from `planning.md` §11.3, which says injury should hold the index. Holding it means a week off leaves you staring at the same workout with no way past it; advancing means you drift through the cycle instead, which costs nothing here because `WORKOUTS` is an eight-entry rotation rather than a weekday schedule. The missed workout is not made up.

Every `SKIP` record carries `skip_reason` and `training_eligible: false`, so an injured day never reaches the model as a missed-attendance signal. `injured` is written on every session record, `START` included, so it is safe to filter on directly rather than treating a missing attribute as `false`.

Both writes are `transact_write_items` against a single session item and a `GYM_STATE` item, guarded by condition expressions. Two `START`s cannot open two sessions; a stale `STOP` cannot close a session someone else already closed. `TransactionCanceledException` comes back as a 409 telling the caller to re-run `STATUS`.

A session outside `MIN_PLAUSIBLE_SESSION_MINUTES`..`MAX_PLAUSIBLE_SESSION_MINUTES` (10–240) closes as `needs_review` with `training_eligible: false` and returns 202. That flag is what keeps a forgotten `STOP` out of the training data.

## Session validator

`handlers/validate_sesh_handler.py` runs at 02:00 America/New_York on an EventBridge **Scheduler** schedule. Nothing invokes it but the clock.

The command handler only writes when the phone sends something, so a day you simply did not go leaves no record at all — and the training set reads that as *no data* rather than *he skipped*. The validator closes that gap after the day is over.

| Outcome | When | Written |
|---|---|---|
| `already_logged` | newest session's local day ≥ yesterday | nothing |
| `unattended` | training day, nothing logged | `attended: false`, `injured: false`, `training_eligible: true`, rotation **frozen** |
| `rest_auto_completed` | rotation entry was `Rest-days`, no `SKIP` arrived | `rest_completed` record, rotation **advanced** |
| `no_history` | table has no sessions at all | nothing |

The rotation deliberately does not advance on `unattended` — the workout you missed is still the workout that is up.

Day attribution reads `checkin_at` first and `created_at` only as a fallback. An 11 PM check-in is 03:00 UTC the next morning, so going by `created_at` would file Tuesday's session under Wednesday and then mark Tuesday missed. `SKIP` records carry no `checkin_at` at all, which is why the fallback exists.

Writes are idempotent by construction: `session_id` is a `uuid5` of the validated date, so the SK is deterministic and a retried invocation collides on `attribute_not_exists(SK)` and cancels harmlessly. The SK timestamp is local end-of-day in UTC, which keeps the "newest session is the last key" ordering intact.

It also releases a stale `active_session_id`. A `START` with no `STOP` holds that lock forever and every later `START` answers 409; a lock older than today closes as `needs_review` with `anomaly_reason: session_never_stopped`, without inventing a duration or advancing the rotation.

Replay one specific day instead of waiting for 02:00:

```json
{"target_date": "2026-08-19"}
```

It refuses any date that is not over yet.

```bash
python deploy/deploy_validator_handler.py   # ~/Downloads/gym-session-validator.zip
```

That zip bundles `tzdata`, unlike the gym command zip. This handler builds `LOCAL_TZ = ZoneInfo(TIMEZONE)` at module scope, so a missing tzdata is not one failed request — it is an init failure that kills every invocation, and a cron that silently never runs is noticed weeks later.

| Setting | Value |
|---|---|
| Handler | `validate_sesh_handler.lambda_handler` |
| Timeout | 30s |
| Env | `TABLE_NAME`, `TIMEZONE` |
| IAM | `dynamodb:GetItem`, `Query`, `PutItem`, `UpdateItem` |

There is no `dynamodb:TransactWriteItems` action — transactions are authorized through the underlying `PutItem`/`UpdateItem`. `Query` is separate from `GetItem`, and it is the call that decides whether you attended.

## Gym ML pipeline

> **Status: in progress.** Phases 1 and 2 are built and tested — `repository.py`, `normalize.py`, `duration_profile.py`, `candidate_generator.py`, `backfill.py`, `features.py`. Still missing: `train.py`, `predict.py`, the scoring handler, and every Lambda in `GymMlStack` (the stack creates only the artifacts bucket today). Until this ships, gym keeps using the fixed values in `config.py`. Build order is in [ml/gym_ml_cdk_plan.md](ml/gym_ml_cdk_plan.md).

Today gym is scheduled with two hardcoded assumptions in `config.GYM`: every session is 90 minutes, and the best time is whichever free slot sits closest to 17:30. Neither reflects what actually happens. `Sharms` does not take as long as `Back-Biceps`, and the 5:30 PM slot you keep skipping is worse than the 8:00 PM one you keep attending.

This pipeline replaces both with values learned from your own logged sessions.

### What replaces what

| | Now | After |
|---|---|---|
| Duration | `config.GYM["duration"]` = 90 for everything | Mean actual duration for that workout type |
| Slot choice | `scheduler._score()`, minutes away from `preferred_start` | Logistic regression attendance probability, highest wins |

Meals stay static and deterministic. Only gym becomes dynamic — and meals must be placed first, so the gym candidates see a full picture of the day. `SCHEDULING_ORDER` in `scheduler.py` already encodes that: each placed activity becomes busy for the next.

The meal-gap penalty in `_score()` stays. It encodes "do not lift right after eating," which is a constraint, not a preference the model should be free to learn away.

### 1. Duration profile

Read completed sessions from DynamoDB, group by workout, take the mean duration. Only `training_eligible` records count.

```json
{
  "Chest-Triceps": 82,
  "Back-Biceps": 76,
  "Sharms": 69
}
```

Written to S3 as `duration_profiles.json`. A workout with no history falls back to `config.GYM["duration"]`.

Only attended sessions count toward the mean. Unattended days are in the training set on purpose, but they carry a duration of 0 — averaging them in would shrink every profile toward zero and hand the candidate generator a window too small to hold the real workout.

### 2. Candidate generation

Each morning: read the current rotation entry, load its mean duration, reuse `calculate_free_windows` and `candidate_starts` from `scheduler.py`, and return up to three slots that fit.

Candidate generation stays deterministic. The model ranks slots; it never invents one.

Candidates are *spread*, not merely valid. A 15-minute grid across an open evening yields twenty near-identical slots, and scoring those produces three probabilities that differ in the third decimal place. `MIN_SPACING_MINUTES` (60) thins the grid, then the survivors are sampled evenly across the day so the model sees a morning, an afternoon, and an evening option rather than three flavours of 5:30.

The rotation is state-driven, not date-driven, so `resolve_workout()` takes no date argument — `next_workout_index` moves only when a session completes or the validator closes out a rest day.

### 3. Training data

The model is a **ranker**, not a day-level classifier. At prediction time the question is "which of these three slots is best", so `backfill.py` builds training rows that are slots, not days:

| Day | Rows |
|---|---|
| attended | the slot you used is `chosen: true`; the other viable slots that day are `chosen: false` |
| unattended | every viable slot is `chosen: false` |

Day-level labelling gives one row per day and a handful of negatives per season — enough to support roughly one predictor. Slot-level gives one row per option, and the comparison the model must learn (5:30 beat 8:00 *on that day, given that calendar*) is exactly the one it will be asked to make.

Two rules that keep the data honest:

- **Duration always comes from the profile, never `actual_duration_minutes`.** The real duration is only knowable after the session. Using it for positives and the profile for negatives would let the model separate the classes on a field that does not exist at prediction time.
- **Rest and injury days never appear.** `fetch_sessions` filters them out via `training_eligible`, and they do not belong regardless — a rest day is not a scheduling decision, and no proposed time would have prevented an injury. Rest days are also perfectly predictable from `cycle_index`, so including them would buy free accuracy and zero information.

An attended day whose check-in matches no viable slot is dropped whole, positives and negatives together — you trained during something the calendar called busy, and keeping only the negatives would record a day you attended as a total refusal. `build_examples` returns diagnostics; `days_dropped_no_slot` measures how often the calendar disagrees with your life.

Rows carry `day` so a train/test split can group by it. Slots from one day share a calendar and are not independent observations — split them across the boundary and the test set holds near-duplicates of training rows.

### 4. Attendance model

`features.py` turns a row into a vector, and it is the *only* thing that does, in both directions: `TrainingExample.to_row()` at train time, `Candidate.to_dict()` at predict time. Those two dicts expose the same feature keys deliberately.

The `FeatureSpec` — ordered feature names and frozen category lists — ships inside `model_metadata.json` beside the coefficients and is passed in, never inferred at predict time. A spec derived from whatever data is at hand produces a different column order for the same row, and the coefficients then land on columns they were never fitted on. Nothing raises. The predictions just quietly stop meaning anything.

The default spec is three features — `start_hour`, `gap_after_minutes`, `is_weekend` — not the eight in the original plan. With roughly 30 attended days, each a choice set of "here were the options, I took this one", three or four parameters is the budget before the fit memorizes days instead of learning times. Widen it as the season fills in and retrain; nothing else changes, because the spec travels with the model.

```text
3:00 PM → 0.52
5:30 PM → 0.84
8:00 PM → 0.61
```

The highest-scoring candidate that still passes the validator wins. Ordering matters more than calibration: if all three slots score 0.3 the best one still wins, so the fallback to `scheduler._score()` triggers on *whether a usable model exists* — no artifact in S3, or too few negatives in `model_metadata.json` — not on the winning probability. `CONFIDENCE_THRESHOLD` does not gate a ranker.

### Artifacts and stack

S3 holds `duration_profiles.json`, `attendance_model.joblib`, and `model_metadata.json`.

**`model_metadata.json` is the model that gets deployed**, not the joblib. Logistic regression inference is a dot product and a sigmoid, so exporting coefficients plus the feature spec lets the scoring Lambda score in pure Python with no ML dependencies at all. scikit-learn pulls numpy and scipy — roughly 200 MB unzipped, against a 50 MB console upload limit and a 250 MB unzipped ceiling — and a joblib pickle is version-locked to the scikit-learn that wrote it. The joblib stays in S3 for retraining and inspection.

`GymMlStack` creates the artifacts bucket, a training Lambda, a candidate-scoring Lambda, an EventBridge training schedule, and their IAM permissions. It references the existing `AssistantData` table **by name** — the table, the gym command Lambda, API Gateway, the Shortcut, the calendar Lambda, and the Bedrock logic are all left alone.

### Deploying the stack

`cdk.json` is gitignored, so create it in `infra/` first:

```json
{
  "app": "python app.py",
  "context": {
    "gym_table_name": "AssistantData"
  }
}
```

```bash
cd infra
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
cdk synth
cdk diff
cdk deploy
```

`app.py` reads `CDK_DEFAULT_ACCOUNT` and `CDK_DEFAULT_REGION` from the environment — the AWS account is never written into the repo.

The pipeline gets proven end to end — read gym data, profile durations, generate candidates, score them, return a winner — before anything connects it to the calendar Lambda.

## Configuration

`config.py` — durations and time windows per activity, `DAY_START`/`DAY_END`, `CONFIDENCE_THRESHOLD` (0.80), `ALL_DAY_BLOCKS`, the `WORKOUTS` rotation, `VALID_GYM_LOCATIONS`, and the plausible-session bounds.

Environment variables:

| Variable | Default | |
|---|---|---|
| `CALENDAR_ID` | — | required |
| `TABLE_NAME` | — | required, read at import |
| `GOOGLE_SA_SECRET_ID` | — | Lambda; falls back to `service-account.json` |
| `GYM_COMMAND_SECRET` | — | gym command Lambda; unset rejects every request |
| `BEDROCK_MODEL_ID` | `deepseek.v3.2` | |
| `AWS_REGION` | `us-east-1` | |
| `TIMEZONE` | `America/New_York` | |

Gym means weightlifting. Cardio and sports don't satisfy it — edit `SYSTEM_PROMPT` in `llm_client.py` to change that.

## Layout

```
agent.py                        orchestration, CLI, lambda_handler
calendar_client.py              Google Calendar read/write
llm_client.py                   Bedrock classification, forced tool-call schema
scheduler.py                    interval merging, free windows, candidate starts
validator.py                    last gate before any write
models.py                       dataclasses
config.py                       preferences

handlers/
  gym_command_handler.py        START/STOP/SKIP/STATUS, DynamoDB writes
  validate_sesh_handler.py      02:00 cron, records days you did not log

ml/
  repository.py                 session + rotation-state reads
  normalize.py                  DynamoDB records -> flat training rows
  duration_profile.py           mean duration per workout -> S3
  candidate_generator.py        daily slots, spread and valid
  backfill.py                   slot-level training examples from history
  features.py                   FeatureSpec, row -> vector, both directions
  gym_ml_cdk_plan.md            pipeline build plan

ml_handlers/
  train_model_handler.py        training Lambda entrypoint

deploy/
  deploy.py                     calendar agent packaging
  deploy_gym_handler.py         gym command Lambda zip
  deploy_validator_handler.py   validator zip, bundles tzdata

infra/
  app.py                        CDK app entrypoint
  cdk.json                      Python app metadata plus AssistantData table name
  stack/gym_ml_stack.py         GymMlStack -- artifacts bucket only so far
  requirements.txt

test/                           103 tests, no network
```

Everything under `ml/` imports without credentials: no module builds a boto3 client or reads DynamoDB at import time, and the pure functions (`generate_candidates`, `examples_for_day`, `vectorize`) take their data as arguments. That is what keeps the test suite offline.

Agent-created events carry an `extendedProperties` marker and an `[AI Scheduler]` title prefix. Re-running the same day writes nothing.

## Not committed

`service-account.json`, `token.json`, `.aws`, `.env` are gitignored, along with `cdk.out/`, anddddd `cdk.context.json`. Model artifacts (`*.joblib`, `*.pkl`) and training data (`*.csv`, `*.parquet`) are ignored too they are build output, and the gym data is personal.

I actually pushed `cdk.json` since it does not contain any data that could compromise me lol.

`CALENDAR_ID` is an environment variable, not a file value.
