import os

TIMEZONE = os.environ.get("TIMEZONE", "America/New_York")

DAY_START = "07:00"
DAY_END = "23:00"

TABLE_NAME = os.environ["TABLE_NAME"]
BUCKET_NAME = os.environ.get("BUCKET_NAME", "")
COMMAND_SECRET = os.environ.get("GYM_COMMAND_SECRET", "")
USER_ID = "ADAN"

# The calendar to read and write -- the address you shared with the service
# account. Must be a real address; "primary" would resolve to the service
# account's own (empty) calendar.
#
# Set it in the environment, never in this file:
#   local    setx CALENDAR_ID you@example.com   (then reopen the terminal)
#   Lambda   Configuration -> Environment variables
CALENDAR_ID = os.environ.get("CALENDAR_ID", "")


# Calendars to READ, comma-separated. Events get created from more than one
# Google account, and a calendar the agent cannot see is a calendar it will
# schedule straight over. Unset falls back to reading just the write target,
# which is exactly how this agent behaved before multi-calendar support.
#
# Writing stays singular: CALENDAR_ID above is the only calendar the agent ever
# creates events on, so every aiScheduler marker -- and therefore the whole
# idempotency story -- lives in one place.
#
# Each calendar must be shared with the service account separately. Reading
# needs "See all event details"; only CALENDAR_ID needs "Make changes to events".
#
#   local    setx CALENDAR_IDS "work@example.com,personal@gmail.com"
#   Lambda   Configuration -> Environment variables
def _read_calendar_ids():
    """Stripped, blank-free, de-duplicated, write target first.

    Order is load-bearing downstream: duplicate events across calendars are
    resolved by keeping the first one seen, so leading with the write target
    means the agent's own copy of an event wins.
    """
    found = [entry.strip() for entry in os.environ.get("CALENDAR_IDS", "").split(",") if entry.strip()]

    ordered = []
    for entry in ([CALENDAR_ID, *found] if CALENDAR_ID else found):
        if entry not in ordered:
            ordered.append(entry)
    return ordered


CALENDAR_IDS = _read_calendar_ids()

# Local dev credentials. In Lambda, set GOOGLE_SERVICE_ACCOUNT_JSON instead.
SERVICE_ACCOUNT_FILE = "service-account.json"

# Idempotency markers (plan.md section 24).
AGENT_PREFIX = "Assistant - "
AGENT_MARKER_KEY = "aiScheduler"

# All-day events ("PTO", birthdays) are returned with a date but no time.
# False means they do not block meals; the day is still schedulable.
ALL_DAY_BLOCKS = False

VALID_GYM_LOCATIONS = {
    "UCF",
    "CRUNCH",
}

CONFIDENCE_THRESHOLD = 0.80

# Training-eligible records required before the training Lambda will publish a
# model at all. Enforced in one place -- ml_handlers/train_model_handler.py --
# so the morning run never needs to know the number: no artifact means it uses
# the deterministic preferences path.
#
# Note this counts records, not attended sessions. Unattended days written by
# the validator are training-eligible too, and they are the negatives the model
# needs most.
MIN_TRAINING_RECORDS = 70

BREAKFAST = {
    "duration": 60,
    "earliest": "08:00",
    "preferred": "08:30",
    "latest": "12:00",
}

LUNCH = {
    "duration": 45,
    "earliest": "11:00",
    "preferred": "12:30",
    "latest": "16:00",
}

DINNER = {
    "duration": 45,
    "earliest": "17:00",
    "preferred": "19:00",
    "latest": "22:30",
}

GYM = {
    "duration": 90,
    "earliest": "07:00",
    "preferred": "17:30",
    "latest": "22:00",
}

# Workout types for the new feature of assistant, 
# the week turns into an 8 day week because I repeat this workout twice
WORKOUTS = [
    "Chest-Triceps", 
    "Back-Biceps", 
    "Sharms", 
    "Rest-days",
    "Chest-Triceps", 
    "Back-Biceps", 
    "Sharms", 
    "Rest-days"
]

MIN_PLAUSIBLE_SESSION_MINUTES = 10
MAX_PLAUSIBLE_SESSION_MINUTES = 240

REASONS_TO_SKIP = [
    "rest",
    "injured"
]
