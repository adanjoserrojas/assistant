# Assistant

Reads your Google Calendar every morning at 6:00 AM, finds free time, and schedules breakfast, lunch, dinner, and gym around what's already there.

One LLM call classifies existing events (does "Lunch with Alex" already cover lunch?). Deterministic Python does all time arithmetic and calendar writes. The LLM never picks a timestamp.

```
EventBridge 6AM -> Lambda -> Google Calendar (read)
                          -> Bedrock (classify)
                          -> scheduler -> validator -> Google Calendar (write)
```

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

Verify the calendar connection:

```bash
python calendar_client.py         # prints today's events
python agent.py --dry-run         # prints the plan, writes nothing
python agent.py                   # writes to the calendar
python -m pytest test -q
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

## Configuration

`config.py` — durations and time windows per activity, `DAY_START`/`DAY_END`, `CONFIDENCE_THRESHOLD` (0.80), `ALL_DAY_BLOCKS`.

Environment variables:

| Variable | Default | |
|---|---|---|
| `CALENDAR_ID` | — | required |
| `GOOGLE_SA_SECRET_ID` | — | Lambda; falls back to `service-account.json` |
| `BEDROCK_MODEL_ID` | `deepseek.v3.2` | |
| `AWS_REGION` | `us-east-1` | |
| `TIMEZONE` | `America/New_York` | |

Gym means weightlifting. Cardio and sports don't satisfy it — edit `SYSTEM_PROMPT` in `llm_client.py` to change that.

## Layout

```
agent.py            orchestration, CLI, lambda_handler
calendar_client.py  Google Calendar read/write
llm_client.py       Bedrock classification, forced tool-call schema
scheduler.py        interval merging, free windows, candidate scoring
validator.py        last gate before any write
models.py           dataclasses
config.py           preferences
deploy.py           Lambda packaging
test/               31 tests, no network
```

Agent-created events carry an `extendedProperties` marker and an `[AI Scheduler]` title prefix. Re-running the same day writes nothing.

## Not committed

`service-account.json`, `token.json`, `.aws`, `.env` are gitignored. `CALENDAR_ID` is an environment variable, not a file value.
