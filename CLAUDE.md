# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup and commands

`calendar-agent/config.py` reads `TABLE_NAME` from the environment **at import with no default**,
so it must be set before anything imports `config` — including pytest. `CALENDAR_ID` is needed for anything that
touches Google Calendar.

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt

export TABLE_NAME=AssistantData        # required, or every import raises KeyError
export CALENDAR_ID=you@example.com     # required; the ONLY calendar written to
export CALENDAR_IDS=a@x.com,b@y.com    # optional; calendars read, defaults to [CALENDAR_ID]
export BUCKET_NAME=...                 # optional; unset means "no model", not an error
```

```bash
python -m pytest test -q                           # whole suite, offline
python -m pytest test/test_predict.py -q           # one file
python -m pytest test/test_predict.py::test_matches_sklearn -q            # one test
python -m pytest test -q -k "candidate and not spacing"

python calendar-agent/agent.py --dry-run             # print the plan, write nothing
python calendar-agent/agent.py --date 2026-09-19     # target a specific day
python calendar-agent/calendar_client.py             # smoke-test Google credentials
python calendar-agent/llm_client.py                  # smoke-test the Bedrock call in isolation
```

The root `conftest.py` puts both source roots — the repo root (for `ml`, `handlers`,
`ml_handlers`) and `calendar-agent/` (for the flat loop-1 modules) — on `sys.path`, and `pytest.ini`
anchors the rootdir so that happens no matter which directory pytest is invoked from. Test files
therefore just `import config` / `from scheduler import ...` with no path preamble of their own.

Packaging (each script builds a zip; only `deploy.py` can push):

```bash
python deploy/deploy.py [--upload]        # calendar agent -> build/calendar-agent.zip
python deploy/deploy_gym_handler.py       # -> ~/Downloads/gym-command-handler.zip
python deploy/deploy_validator_handler.py # -> ~/Downloads/gym-session-validator.zip

cd infra && cdk synth && cdk diff && cdk deploy   # reads CDK_DEFAULT_ACCOUNT / CDK_DEFAULT_REGION
```

## Architecture

Two loops that share one DynamoDB table and one config file.

**Loop 1 — daily scheduler.** EventBridge 06:00 → `agent.lambda_handler` →
`READ → ANALYZE → SCHEDULE → VALIDATE → WRITE`. `calendar_client` reads the day, `llm_client` asks
Bedrock which existing events already satisfy breakfast/lunch/dinner/gym, `scheduler` does interval
arithmetic, `validator` independently re-checks the result, and only then does `agent.run` write.

**Loop 2 — gym.** An iPhone Shortcut hits API Gateway → `handlers/gym_command_handler.py`
(`START`/`STOP`/`SKIP`/`STATUS`, HMAC-compared secret header, `transact_write_items` with condition
expressions). `handlers/validate_sesh_handler.py` runs at 02:00 and writes the record for a day you
never logged. Those accumulated records are the *only* training input;
`ml_handlers/train_model_handler.py` refits from scratch each run and publishes two S3 artifacts
that loop 1 reads the next morning.

### Invariants that span files

- **`calendar-agent/` is a source directory on `sys.path`, not a package.** It holds loop 1's eight
  modules and deliberately has no `__init__.py` — the hyphen makes it unimportable as one. Every
  import in the repo therefore stays flat and absolute (`import config`, `from scheduler import
  ...`), which is the same shape the Lambda zips need: `deploy/deploy.py` maps each
  `calendar-agent/<name>.py` to a bare `<name>.py` at the zip root, so the handler string stays
  `agent.lambda_handler`. Adding a package layer here would mean re-pointing every handler in the
  AWS console. `ml/`, `handlers/` and `ml_handlers/` stay at the repo root, and `ml/` imports
  `scheduler`, `models` and `config` across that boundary — both roots must be on the path.
- **The read set is plural; the write target is singular.** `calendar_client.read_calendar_ids()`
  returns every calendar scanned (`config.CALENDAR_IDS`, write target first); `calendar_id()` is the
  one calendar `create_event` writes to. Keeping writes singular is what keeps every `aiScheduler`
  marker in one place, so idempotency has a single source of truth. A calendar that fails to read
  is recorded in `failures` and the run continues — `get_today_events_detailed` returns
  `(events, failures)` and `agent.report` makes the degradation loud, because unread events are
  invisible to `validator.validate_schedule` as well as to the scheduler.
- **`get_today_events` must keep returning a bare list.** `ml/backfill.py` takes the function itself
  as its `events_for_day` callable, so its return type is part of the training-pipeline contract.
  Anything needing failure detail calls `get_today_events_detailed` instead.
- **The LLM never picks a timestamp.** It classifies existing events only, via a forced tool call
  with a flat schema (`llm_client.SYSTEM_PROMPT`). `agent.analyze` swallows every LLM failure and
  falls back to `default_analysis()` — a Bedrock outage must never stop the schedule.
- **Meals are placed before gym, and that ordering is load-bearing.** `agent.schedule_day` splits
  the two calls so meals become busy time for the gym search (`gym_allocator.meals_as_events`).
  The `already_scheduled` argument to `scheduler.schedule_activities` exists only so the meal-gap
  penalty in `_score()` still sees the meals after the split.
- **The gym model can only ever be an improvement, never a dependency.**
  `gym_allocator.allocate_safely` catches everything — missing artifact, S3 timeout, unreadable
  rotation state, stale spec — and returns a fallback verdict so `agent.schedule_day` places gym
  the deterministic way. The model/fallback decision is re-made every morning, never latched.
- **Nothing under `ml/` does I/O at import.** boto3 clients are built lazily
  (`ml.repository.client()`, `ml.candidate_generator.s3_client()`) and the pure functions take
  their data as arguments. That is what keeps the test suite offline and credential-free — preserve
  it when adding modules.
- **`ml/features.py` is the only thing that turns a row into numbers**, in both directions:
  `TrainingExample.to_row()` at train time and `Candidate.to_dict()` at predict time emit the same
  keys on purpose. The `FeatureSpec` (ordered names + frozen categories) and the standardization
  means/stds ship *inside* `model_metadata.json` and are passed in, never inferred — a spec
  re-derived at predict time silently lands coefficients on the wrong columns with no error.
- **The model is a ranker, not a classifier.** `predict.usable()` gates on having enough training
  days, enough negatives, and *beating the `scheduler._score()` heuristic on held-out days* —
  not on probability. `config.CONFIDENCE_THRESHOLD` (0.80) belongs to the LLM classification path
  and does not gate the ranker.
- **Training is deliberately pure Python** (`ml/train.py`: L2 logistic regression by gradient
  descent). scikit-learn/numpy would blow the Lambda size budget, and the exported coefficients let
  scoring be a dot product and a sigmoid. `test_predict.py` checks the fit against sklearn.
- **`training_eligible` is the data-quality gate.** `ml.repository.fetch_sessions` filters on it, so
  rest days, injuries, implausible durations and never-stopped sessions never reach the model.
  Unattended days *are* eligible — they are the negatives — and carry `duration: 0`, which is why
  `duration_profile.calculate_mean` averages only `attended` rows.

### Data model

Single DynamoDB table, `PK = USER#ADAN` for everything. `SK = GYM_SESSION#<ts>` for sessions,
`SK = GYM_STATE` for the rotation cursor (`next_workout_index`, `active_session_id`). The rotation
is `config.WORKOUTS` — eight entries, state-driven not date-driven, so `resolve_workout()` takes no
date. Session writes and the state update go in one transaction; the validator writes are
idempotent via a `uuid5` of the validated date.

S3 artifacts: `gym/duration_profiles.json` (mean minutes per workout) and `gym/model_metadata.json`
(spec + standardization + coefficients + evaluation). There is no deployed `.joblib`.

### Lambda packaging conventions

Handlers are **flattened to the zip root** (`gym_command_handler.lambda_handler`, not
`handlers.gym_command_handler...`) with `config.py` copied beside them from `calendar-agent/`,
because handler imports are absolute. All three deploy scripts use the same
`SOURCE_MODULES = {repo path: name at the zip root}` mapping to flatten. `deploy.py` installs `manylinux2014_x86_64` wheels (Windows wheels for `cryptography`
fail to import on Lambda) and deletes every Google discovery document but `calendar.v3.json`
(135 MB → 27 MB). `deploy_validator_handler.py` bundles `tzdata` and the gym one does not — the
validator builds `ZoneInfo` at module scope, so a missing tzdata is an init failure that kills a
silent cron.

## Current state — where the code and the docs disagree

- `infra/stack/gym_ml_stack.py:38` is `self.artifacts_function = lambd.function()` — not a real
  symbol (`Function`) and called with no arguments. **`cdk synth` fails today.** The stack builds
  only the artifacts bucket; the training Lambda, scoring Lambda and EventBridge schedule from
  `ml/gym_ml_cdk_plan.md` Phase 4 are unbuilt.
- `README.md` line 193 says the pipeline is not connected to the calendar Lambda. It is —
  `calendar-agent/agent.py:83` calls `gym_allocator.allocate_safely` and `deploy/deploy.py` ships `ml/` in the
  calendar zip. That status block is stale; the code is authoritative.
- `README.md` says 148 tests; `test/` has 178. 177 pass offline;
  `test_validator.py::test_start_after_latest_is_caught` has been failing since before the
  `calendar-agent/` move and is unrelated to it.
- `ml/normalize.py` still carries a `group_workouts()` stub and a `parse_time` helper written before
  unattended rows existed.

`README.md` is the long-form reference for setup, IAM policies, console settings and the reasoning
behind the ML design. `ml/gym_ml_cdk_plan.md` holds the remaining build order. `plan.md` and
`planning.md` are the original design docs — code comments cite them by section number, and where
they conflict with the code (e.g. `planning.md` §11.3 on injury freezing the rotation), the code
and its comment explaining the deviation win.
