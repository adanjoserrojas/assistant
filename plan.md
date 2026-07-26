# AI Calendar Agent — 24-Hour Build Plan

## 1. Goal

Build a Python agent that runs every morning at **6:00 AM**, reads the user's Google Calendar for the current day, identifies free time, and dynamically schedules:

- Breakfast
- Lunch
- Dinner
- Gym

The agent should use:

- **Google Calendar API** for reading and writing calendar events
- **One LLM call per morning** for semantic reasoning
- **Deterministic Python scheduling logic** for exact time placement
- **AWS Lambda** to run the agent
- **Amazon EventBridge Scheduler** to invoke it every morning at 6:00 AM
- **AWS Secrets Manager** for Google OAuth credentials

The MVP should remain intentionally small enough to build in approximately 24 hours.

---

# 2. Core Design Principle

The LLM should **not directly choose or create exact calendar timestamps**.

Instead:

```text
Google Calendar
      |
      v
Today's Events
      |
      v
One LLM Call
      |
      v
Structured Scheduling Context
      |
      v
Deterministic Python Scheduler
      |
      v
Validator
      |
      v
Google Calendar
```

The LLM handles:

- Semantic understanding
- Detecting existing meals
- Detecting exercise
- Interpreting ambiguous calendar titles
- Adjusting priorities
- Producing lightweight scheduling hints

Python handles:

- Free-time calculation
- Time arithmetic
- Conflict detection
- Candidate generation
- Exact start/end times
- Validation
- Calendar writes

This avoids giving an LLM uncontrolled authority over the calendar.

---

# 3. MVP Architecture

```text
                    6:00 AM
                       |
                       v
          Amazon EventBridge Scheduler
                       |
                       v
                  AWS Lambda
                   agent.py
                       |
          +------------+------------+
          |                         |
          v                         v
 AWS Secrets Manager        Google Calendar API
          |                         |
          +------------+------------+
                       |
                       v
               Fetch today's events
                       |
                       v
                 LLM reasoning
                       |
                       v
                Structured JSON
                       |
                       v
             Availability engine
                       |
                       v
              Scheduling engine
                       |
                       v
                  Validator
                       |
                       v
            Google Calendar writer
```

---

# 4. Repository Structure

Keep the repository small.

```text
calendar-agent/
|
├── agent.py
├── requirements.txt
├── README.md
├── .gitignore
|
├── config.py
|
├── calendar_client.py
├── llm_client.py
├── scheduler.py
├── validator.py
|
├── models.py
|
└── tests/
    ├── test_scheduler.py
    └── test_validator.py
```

Do not create additional folders until the MVP works.

---

# 5. File Responsibilities

## `agent.py`

Main orchestration entry point.

Responsibilities:

1. Load configuration.
2. Authenticate with Google Calendar.
3. Retrieve today's events.
4. Send events to the LLM.
5. Parse structured LLM output.
6. Determine which activities still need scheduling.
7. Calculate free windows.
8. Schedule remaining activities.
9. Validate all proposed events.
10. Write events to Google Calendar.

Conceptually:

```python
def run():
    events = get_today_events()

    context = analyze_with_llm(events)

    activities = determine_required_activities(context)

    free_windows = calculate_free_windows(events)

    schedule = create_schedule(
        activities,
        free_windows,
        context
    )

    validate_schedule(schedule, events)

    write_schedule(schedule)
```

AWS Lambda should call the same function:

```python
def lambda_handler(event, context):
    return run()
```

---

# 6. `config.py`

Store non-secret scheduling defaults.

Example:

```python
TIMEZONE = "America/New_York"

DAY_START = "06:30"
DAY_END = "23:00"

BREAKFAST = {
    "duration": 30,
    "earliest": "06:30",
    "preferred": "08:00",
    "latest": "10:30",
}

LUNCH = {
    "duration": 45,
    "earliest": "11:00",
    "preferred": "12:30",
    "latest": "15:00",
}

DINNER = {
    "duration": 45,
    "earliest": "17:00",
    "preferred": "19:00",
    "latest": "21:30",
}

GYM = {
    "duration": 90,
    "earliest": "07:00",
    "preferred": "17:30",
    "latest": "22:00",
}
```

For the first version, keep all preferences hard-coded.

Do not build a preferences database yet.

---

# 7. `models.py`

Use lightweight Python dataclasses.

Example models:

```python
@dataclass
class CalendarEvent:
    title: str
    start: datetime
    end: datetime
```

```python
@dataclass
class Activity:
    name: str
    duration_minutes: int
    earliest_start: time
    latest_start: time
    preferred_start: time
```

```python
@dataclass
class ScheduledActivity:
    name: str
    start: datetime
    end: datetime
```

Keep models small.

---

# 8. Google Calendar Integration

## `calendar_client.py`

Responsibilities:

```text
authenticate()
get_today_events()
create_event()
find_agent_events()
```

The module should hide Google API implementation details from the rest of the application.

Desired interface:

```python
events = get_today_events()
```

returns:

```python
[
    CalendarEvent(...),
    CalendarEvent(...),
]
```

and:

```python
create_event(
    title="Gym",
    start=...,
    end=...
)
```

creates the calendar event.

---

# 9. Local Authentication First

Do not start with AWS.

First make this work locally:

```bash
python agent.py
```

Use Google's OAuth development flow.

Target milestone:

```text
Successfully print every event from today's Google Calendar.
```

Only after this works should scheduling logic begin.

---

# 10. LLM Architecture

## Objective

Use exactly **one primary LLM call each morning**.

The LLM receives:

- Today's existing calendar events
- The activities the agent normally schedules
- Basic user scheduling preferences

The LLM returns structured JSON.

The LLM should answer questions such as:

- Does an existing event satisfy breakfast?
- Does an existing event satisfy lunch?
- Does an existing event satisfy dinner?
- Does an existing event satisfy exercise?
- Are any existing events likely physically demanding?
- Are there scheduling considerations the deterministic scheduler should know about?

---

# 11. LLM Input

Example input:

```json
{
  "date": "2026-07-25",
  "events": [
    {
      "title": "Research Meeting",
      "start": "10:00",
      "end": "11:00"
    },
    {
      "title": "Lunch with Alex",
      "start": "12:30",
      "end": "13:30"
    },
    {
      "title": "Basketball",
      "start": "18:00",
      "end": "20:00"
    }
  ],
  "required_activities": [
    "breakfast",
    "lunch",
    "dinner",
    "gym"
  ]
}
```

---

# 12. LLM Output Schema

The schema is enforced with a **forced tool call** on Bedrock, not by asking for
JSON in the prompt. The tool schema is flat (`breakfast_satisfied`,
`breakfast_confidence`, ...) because nested objects are unreliable with
open-weight models; `llm_client.py` reassembles the nested shape below.

Example (for the calendar in section 11):

```json
{
  "satisfied_activities": {
    "breakfast": false,
    "lunch": true,
    "dinner": false,
    "gym": false
  },
  "confidences": {
    "breakfast": 0.05,
    "lunch": 1.0,
    "dinner": 0.05,
    "gym": 1.0
  },
  "reasoning": "Lunch with Alex satisfies lunch; basketball is not weightlifting."
}
```

Note `gym` is **false** even though Basketball is strenuous — see section 14.

Confidence is reported for every activity, including the ones judged
unsatisfied, so section 15's threshold can be applied uniformly.

This means the deterministic scheduler needs to schedule:

```text
Breakfast
Dinner
Gym
```

---

# 13. `llm_client.py`

Responsibilities:

```text
build_prompt()
call_model()
parse_response()
validate_schema()
```

Expose one function:

```python
analysis = analyze_calendar(events)
```

Return a normal Python object or dictionary.

Do not expose model-specific logic throughout the repository.

---

# 14. LLM Prompt Strategy

Keep the system prompt narrow.

Example:

```text
You analyze a user's daily calendar.

Your job is to classify calendar events and decide whether existing
events already satisfy breakfast, lunch, dinner, or exercise.

Do not choose exact timestamps.
Do not create calendar events.
Do not modify events.

"gym" means weightlifting specifically -- resistance and strength
training. Lifting, squats, bench, deadlifts, "leg day", "upper body",
"strength session", or a plain "gym" entry all satisfy it.

Cardio and sports do NOT satisfy gym, no matter how strenuous.
Basketball, soccer, running, cycling, swimming, hiking, yoga, and
fitness classes must all be reported as gym unsatisfied.

Be conservative. If an event is ambiguous, report low confidence
rather than marking the activity satisfied.
```

This dramatically reduces hallucination risk.

The gym definition is load-bearing and not optional. Without it the model
treats any strenuous activity as exercise, which silently skips real
weightlifting sessions on days with a basketball game.

---

# 15. Confidence Threshold

Do not blindly trust semantic classification.

Example:

```python
CONFIDENCE_THRESHOLD = 0.80
```

If:

```text
confidence >= 0.80
```

accept classification.

Otherwise:

```text
treat activity as unsatisfied
```

Example:

```text
"Networking Dinner"

LLM:
satisfies dinner = true
confidence = 0.92
```

Accept.

But:

```text
"Networking Event"

LLM:
satisfies dinner = true
confidence = 0.55
```

Reject.

The scheduler will still create dinner.

---

# 16. Availability Engine

Inside `scheduler.py`:

```python
calculate_free_windows(events)
```

Assume day:

```text
06:30 -> 23:00
```

Existing calendar:

```text
09:00 -> 10:15
11:00 -> 12:00
14:00 -> 16:00
```

Return:

```text
06:30 -> 09:00
10:15 -> 11:00
12:00 -> 14:00
16:00 -> 23:00
```

This is pure interval arithmetic.

No LLM involvement.

---

# 17. Merge Overlapping Calendar Events

Before calculating availability:

```text
Event A
10:00 -> 11:30

Event B
11:00 -> 12:00
```

must become:

```text
10:00 -> 12:00
```

Otherwise the free-time calculation can break.

Implement:

```python
merge_busy_intervals(events)
```

before:

```python
calculate_free_windows()
```

---

# 18. Candidate Scheduling

For each unsatisfied activity:

1. Find free windows within its allowed range.
2. Verify the window is long enough.
3. Generate candidate start times.
4. Score candidates.
5. Select the lowest-score option.

For a 45-minute lunch:

```text
free:
11:15 -> 14:30

preferred:
12:30

candidates:
11:15
11:30
11:45
12:00
12:15
12:30
12:45
...
```

Prefer:

```text
12:30
```

when possible.

---

# 19. Simple Scoring Function

Do not build a sophisticated optimizer in the first 24 hours.

Use:

```python
score = abs(candidate_start - preferred_start)
```

Then add simple penalties:

```python
score += late_penalty
score += meal_gym_penalty
```

Example:

```text
Gym candidate: 5:30 PM
score = 0

Gym candidate: 7:30 PM
score = 120
```

Lower wins.

---

# 20. Scheduling Order

Use a fixed scheduling order for the MVP:

```text
1. Breakfast
2. Lunch
3. Dinner
4. Gym
```

After each activity is scheduled, mark its interval as busy.

This avoids overlapping generated events.

More sophisticated "least-flexible-first" scheduling can be added later.

For a 24-hour MVP, fixed order is easier to debug.

---

# 21. Meal/Gym Rule

Implement one simple relationship:

```text
Avoid starting gym less than 60 minutes after a meal.
```

During gym candidate scoring:

```python
if gym_starts_within_60_minutes_after_meal:
    score += LARGE_PENALTY
```

Do not build complex health reasoning.

---

# 22. Schedule Validation

## `validator.py`

Before writing anything:

Validate:

```text
No generated events overlap existing events.
No generated events overlap each other.
Each event fits inside its configured time range.
Each event has the correct duration.
Each activity appears at most once.
```

Expose:

```python
validate_schedule(
    existing_events,
    generated_events
)
```

If validation fails:

```text
DO NOT WRITE ANY EVENTS
```

---

# 23. Dry-Run Mode

Implement this before real calendar writes.

Example:

```bash
python agent.py --dry-run
```

Output:

```text
TODAY

Existing events:
09:00-10:15 Class
11:00-12:00 Meeting
18:00-20:00 Basketball

LLM classification:
Basketball satisfies gym.

Activities remaining:
Breakfast
Lunch
Dinner

Proposed schedule:

Breakfast  07:45-08:15
Lunch      12:30-13:15
Dinner     20:15-21:00

Validation: PASS

Dry run enabled.
Calendar unchanged.
```

This will save considerable debugging time.

---

# 24. Prevent Duplicate Agent Events

Every event created by the agent should have a recognizable marker.

Simplest MVP approach:

```text
[AI Scheduler] Breakfast
[AI Scheduler] Lunch
[AI Scheduler] Dinner
[AI Scheduler] Gym
```

Before scheduling:

```python
find_agent_events(today)
```

If an agent-generated activity already exists:

```text
do not create another
```

This gives basic idempotency without building a database.

---

# 25. No DynamoDB in the First 24 Hours

Do not build persistent state yet.

Google Calendar itself can temporarily be the source of truth.

Agent state can be inferred from:

```text
[AI Scheduler] ...
```

events.

Add DynamoDB later when you need:

- Preference learning
- Manual modification tracking
- Historical schedules
- Reconciliation
- Schedule versions

---

# 26. No Multi-Agent Architecture

Do not build:

```text
Planner Agent
Calendar Agent
Meal Agent
Gym Agent
Validator Agent
```

For this MVP, there should be **one agent process**.

Architecture:

```text
agent.py
    |
    +-- Google Calendar
    +-- LLM
    +-- Scheduler
    +-- Validator
```

This is enough.

---

# 27. No Autonomous Tool Loop

Do not implement:

```text
LLM
 -> tool call
 -> observe
 -> think
 -> tool call
 -> observe
 -> repeat
```

It is unnecessary for the first version.

Use a fixed execution pipeline:

```text
READ
  |
  v
ANALYZE
  |
  v
SCHEDULE
  |
  v
VALIDATE
  |
  v
WRITE
```

This is easier to reason about and much safer.

---

# 28. Local Development Order

## Milestone 1

Create repository:

```bash
mkdir calendar-agent
cd calendar-agent
python -m venv .venv
```

Create files.

Target:

```text
agent.py executes successfully.
```

---

# 29. Milestone 2 — Google Calendar Read

Implement OAuth.

Target:

```bash
python agent.py
```

prints:

```text
09:00 Class
11:00 Meeting
14:00 Study Session
```

Do not continue until this works.

---

# 30. Milestone 3 — Availability Calculation

Hard-code test events first.

Input:

```text
09:00-10:00
12:00-13:00
```

Output:

```text
06:30-09:00
10:00-12:00
13:00-23:00
```

Write unit tests.

---

# 31. Milestone 4 — Basic Activity Scheduler

Ignore the LLM initially.

Always schedule:

```text
Breakfast
Lunch
Dinner
Gym
```

Target:

```text
The deterministic scheduler produces a conflict-free daily plan.
```

---

# 32. Milestone 5 — Dry Run Against Real Calendar

Combine:

```text
Google Calendar events
+
availability engine
+
activity scheduler
```

Run:

```bash
python agent.py --dry-run
```

Confirm output manually.

---

# 33. Milestone 6 — LLM Classification

Now add the LLM.

Send the day's events.

Parse structured response.

Target:

```text
Calendar:
12:30 Lunch with Alex
18:00 Weightlifting

LLM:
Lunch already satisfied.
Gym already satisfied.

Scheduler:
Only schedules breakfast and dinner.
```

And the negative case, which matters just as much:

```text
Calendar:
18:00 Basketball

LLM:
Gym NOT satisfied (basketball is not weightlifting).

Scheduler:
Still schedules gym.
```

---

# 34. Milestone 7 — Google Calendar Writes

Once dry-run results look correct:

```python
create_event(...)
```

Enable real writes.

Target:

```text
Run agent once.
Google Calendar contains generated activities.
```

Then run it again.

Target:

```text
No duplicates appear.
```

---

# 35. Milestone 8 — AWS Secrets Manager

Move OAuth secrets out of local files.

Store required Google credentials in AWS Secrets Manager.

Lambda should retrieve them at runtime.

Do not commit:

```text
credentials.json
token.json
.env
OAuth refresh tokens
```

---

# 36. Milestone 9 — AWS Lambda

Package:

```text
agent.py
calendar_client.py
llm_client.py
scheduler.py
validator.py
models.py
config.py
dependencies
```

Create Lambda.

Handler:

```text
agent.lambda_handler
```

Test manually from AWS.

Target:

```text
Lambda reads calendar.
Lambda generates schedule.
Lambda writes events.
```

---

# 37. Milestone 10 — EventBridge Scheduler

Create one schedule:

```text
Name:
daily-calendar-agent

Time:
6:00 AM

Timezone:
America/New_York

Target:
Calendar Agent Lambda
```

That completes the core project.

---

# 38. 24-Hour Build Schedule

## Hour 0-1

Repository setup.

```text
Create files.
Create virtual environment.
Install dependencies.
Create Google Cloud project.
Enable Calendar API.
```

---

## Hour 1-4

Google Calendar integration.

Goal:

```text
Read today's calendar successfully.
```

Do not touch AWS yet.

---

## Hour 4-7

Build:

```text
Event models
Busy interval merging
Free-window calculation
```

Test locally.

---

## Hour 7-10

Build deterministic scheduler.

Implement:

```text
Breakfast
Lunch
Dinner
Gym
Preferred times
Candidate scoring
Conflict prevention
```

---

## Hour 10-12

Implement:

```text
validator.py
--dry-run
duplicate detection
```

Run against real calendar.

---

## Hour 12-14

Add the LLM.

Implement:

```text
one model call
structured JSON
event classification
activity satisfaction detection
confidence threshold
```

Do not let the LLM create timestamps.

---

## Hour 14-16

Integrate everything:

```text
Google Calendar
      |
      v
LLM
      |
      v
Scheduler
      |
      v
Validator
```

Run repeated local tests.

---

## Hour 16-18

Enable Google Calendar writes.

Test:

```text
first run -> events created
second run -> no duplicates
```

---

## Hour 18-21

AWS deployment.

Implement:

```text
Secrets Manager
Lambda
IAM permissions
CloudWatch logs
```

Invoke Lambda manually.

---

## Hour 21-22

Configure EventBridge Scheduler.

```text
6:00 AM
America/New_York
```

---

## Hour 22-24

Testing and cleanup.

Test:

```text
Empty calendar
Very busy calendar
Existing lunch
Existing dinner
Existing exercise
Overlapping calendar events
No valid gym slot
Agent invoked twice
LLM unavailable
```

Update README.

---

# 39. LLM Failure Strategy

The agent must still work if the LLM call fails.

Use:

```python
try:
    llm_analysis = analyze_calendar(events)
except Exception:
    llm_analysis = default_analysis()
```

Default:

```json
{
  "satisfied_activities": {
    "breakfast": false,
    "lunch": false,
    "dinner": false,
    "gym": false
  }
}
```

The deterministic scheduler can then continue.

The LLM should improve the scheduler, not become a single point of failure.

---

# 40. Google API Failure Strategy

If reading the calendar fails:

```text
ABORT
```

Never schedule without knowing existing events.

If writing fails:

```text
Log error.
Stop remaining writes if practical.
```

For the MVP, log failures clearly in CloudWatch.

---

# 41. LLM Provider Abstraction

Keep:

```python
def analyze_calendar(events):
    ...
```

independent from the rest of the application.

This allows switching providers later without touching scheduling code.

Do not prematurely build a large provider abstraction.

One file is enough.

---

# 42. Minimum Dependencies

Possible Python dependencies:

```text
google-api-python-client
google-auth
google-auth-oauthlib
boto3
python-dateutil
pydantic
```

Plus the SDK for whichever LLM provider you choose.

Avoid unnecessary frameworks.

You do not need:

```text
LangChain
LangGraph
FastAPI
Flask
Celery
Redis
Kubernetes
```

for this MVP.

---

# 43. Simplified Runtime Flow

The final morning run should look approximately like:

```text
START

|
v

Determine today's date

|
v

Read Google Calendar

|
v

Remove / identify existing AI Scheduler events

|
v

Send existing human events to LLM

|
v

LLM returns activity classifications

|
v

Determine required activities

|
v

Merge calendar busy intervals

|
v

Calculate free windows

|
v

Schedule breakfast

|
v

Schedule lunch

|
v

Schedule dinner

|
v

Schedule gym

|
v

Validate entire proposal

|
+--- invalid ---> log + abort
|
v

Create Google Calendar events

|
v

Log result

|
v

END
```

---

# 44. Example Morning

Existing calendar:

```text
09:00-10:30 Class
12:30-13:30 Lunch with Sarah
15:00-16:00 Research Meeting
18:30-20:00 Basketball
```

LLM returns:

```json
{
  "satisfied_activities": {
    "breakfast": false,
    "lunch": true,
    "dinner": false,
    "gym": false
  }
}
```

Scheduler calculates:

```text
Breakfast:
07:45-08:15

Dinner:
20:15-21:00

Gym:
16:15-17:45
```

Calendar after execution:

```text
07:45-08:15 [AI Scheduler] Breakfast

09:00-10:30 Class

12:30-13:30 Lunch with Sarah

15:00-16:00 Research Meeting

16:15-17:45 [AI Scheduler] Gym

18:30-20:00 Basketball

20:15-21:00 [AI Scheduler] Dinner
```

No lunch is created because an existing lunch event already exists.

Gym **is** created: basketball is exercise, but it is not weightlifting, and
gym means weightlifting (section 14). The scheduler places it in the 16:00-18:30
gap rather than at its 17:30 preferred start, because a 90-minute block at 17:30
would collide with basketball.

---

# 45. MVP Definition of Done

The project is complete enough for the 24-hour deadline when all of these work:

```text
[ ] Python project runs locally

[ ] Google OAuth works

[ ] Agent reads today's Google Calendar

[ ] Busy intervals are normalized

[ ] Free windows are calculated

[ ] Breakfast can be scheduled dynamically

[ ] Lunch can be scheduled dynamically

[ ] Dinner can be scheduled dynamically

[ ] Gym can be scheduled dynamically

[ ] Generated events never overlap

[ ] LLM classifies existing calendar events

[ ] LLM can identify an existing meal

[ ] LLM can identify existing exercise

[ ] LLM returns structured output

[ ] Scheduler works when the LLM fails

[ ] Dry-run mode works

[ ] Agent writes events to Google Calendar

[ ] Duplicate generated events are prevented

[ ] Google OAuth secrets are not committed

[ ] Lambda runs successfully

[ ] EventBridge invokes Lambda

[ ] Schedule is configured for 6:00 AM America/New_York

[ ] CloudWatch receives useful logs
```

---

# 46. Explicitly Out of Scope for the 24-Hour MVP

Do not build these unless everything above is already complete:

```text
DynamoDB preference history

Machine-learned preference optimization

Multi-agent workflows

Continuous calendar monitoring

Push notifications

Mobile application

Natural-language chat interface

Calendar event rescheduling throughout the day

Long-term memory

Vector databases

RAG

Complex constraint solvers

Google OR-Tools

Automatic preference learning

Multiple calendars

Travel-time estimation

Location-aware scheduling

Weather-aware scheduling

Nutrition tracking

Workout recommendation models
```

These are future iterations.

---

# 47. V2 Roadmap

Once the MVP is working:

## V2.1

Run reconciliation during the day.

```text
6 AM  -> initial plan
12 PM -> conflict check
4 PM  -> conflict check
```

---

## V2.2

Detect manually moved agent events.

Respect user overrides.

---

## V2.3

Add DynamoDB.

Track:

```text
Suggested time
Actual time
Manual modifications
Preference history
```

---

## V2.4

Use LLM to infer preferences.

Example:

```text
User repeatedly moves gym from morning to 5-7 PM.

LLM inference:
Preferred gym window = 5-7 PM.
```

---

## V2.5

Accept natural-language instructions.

Example:

```text
"Tomorrow is packed. Get my workout done early and don't let me eat dinner after 9."
```

LLM converts this into temporary constraints.

---

# 48. Final Design

The MVP should stay conceptually simple:

```text
                    AWS
                     |
                     v
                 agent.py
                     |
        +------------+-------------+
        |                          |
        v                          v
Google Calendar                 LLM Call
        |                          |
        +------------+-------------+
                     |
                     v
              Python Scheduler
                     |
                     v
                Validator
                     |
                     v
              Google Calendar
```

The most important architectural rule is:

> **Use the LLM to understand human context. Use deterministic code to control the calendar.**

That gives the project genuine AI reasoning while keeping the system predictable, testable, inexpensive, and realistic to build within 24 hours.
