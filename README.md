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
python -m pytest test -q          # 31 tests, no network
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
python deploy.py            # build/calendar-agent.zip (~7.5 MB)
python deploy.py --upload   # needs lambda:UpdateFunctionCode
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
| `SKIP` | Completes a `Rest-days` entry and advances the rotation. |
| `STATUS` | Current and next workout, active session, last completed. |

The rotation is `WORKOUTS` in `config.py` — an eight-entry cycle, so the same workout lands on different weekdays each pass.

Both writes are `transact_write_items` against a single session item and a `GYM_STATE` item, guarded by condition expressions. Two `START`s cannot open two sessions; a stale `STOP` cannot close a session someone else already closed. `TransactionCanceledException` comes back as a 409 telling the caller to re-run `STATUS`.

A session outside `MIN_PLAUSIBLE_SESSION_MINUTES`..`MAX_PLAUSIBLE_SESSION_MINUTES` (10–240) closes as `needs_review` with `training_eligible: false` and returns 202. That flag is what keeps a forgotten `STOP` out of the training data.

## Gym ML pipeline

> **Status: in progress.** `ml/repository.py` and `infra/stack/gym_ml_stack.py` are stubs, and no ML Lambda is deployed. Until this ships, gym keeps using the fixed values in `config.py`. The build order is in [ml/gym_ml_cdk_plan.md](ml/gym_ml_cdk_plan.md).

Today gym is scheduled with two hardcoded assumptions in `config.GYM`: every session is 90 minutes, and the best time is whichever free slot sits closest to 17:30. Neither reflects what actually happens. `Sharms` does not take as long as `Back-Biceps`, and the 5:30 PM slot you keep skipping is worse than the 8:00 PM one you keep attending.

This pipeline replaces both with values learned from your own logged sessions.

### What replaces what

| | Now | After |
|---|---|---|
| Duration | `config.GYM["duration"]` = 90 for everything | Mean actual duration for that workout type |
| Slot choice | `scheduler._score()`, minutes away from `preferred_start` | Logistic regression attendance probability, highest wins |

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

### 2. Candidate generation

Each morning: read the current rotation entry, load its mean duration, reuse the existing free-window logic in `scheduler.py`, and return up to three slots that fit.

Candidate generation stays deterministic. The model ranks slots; it never invents one.

### 3. Attendance model

Eight features per candidate — start time, weekday, workout type, calendar busy minutes, gap before, gap after, location, previous attendance — into a scikit-learn logistic regression that returns a probability.

```text
3:00 PM → 0.52
5:30 PM → 0.84
8:00 PM → 0.61
```

The highest-scoring candidate that still passes the validator wins.

### Artifacts and stack

S3 holds `duration_profiles.json`, `attendance_model.joblib`, and `model_metadata.json`.

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
scheduler.py                    interval merging, free windows, candidate scoring
validator.py                    last gate before any write
models.py                       dataclasses
config.py                       preferences
deploy.py                       Lambda packaging

handlers/
  gym_command_handler.py        START/STOP/SKIP/STATUS, DynamoDB writes

ml/
  repository.py                 (stub) session reads
  gym_ml_cdk_plan.md            pipeline build plan

infra/
  app.py                        CDK app entrypoint
  cdk.json                      Python app metadata plus AssistantData table name
  stack/gym_ml_stack.py         (stub) GymMlStack
  requirements.txt

test/                           31 tests, no network
```

Agent-created events carry an `extendedProperties` marker and an `[AI Scheduler]` title prefix. Re-running the same day writes nothing.

## Not committed

`service-account.json`, `token.json`, `.aws`, `.env` are gitignored, along with `cdk.out/`, anddddd `cdk.context.json`. Model artifacts (`*.joblib`, `*.pkl`) and training data (`*.csv`, `*.parquet`) are ignored too they are build output, and the gym data is personal.

I actually pushed `cdk.json` since it does not contain any data that could compromise me lol.

`CALENDAR_ID` is an environment variable, not a file value.
