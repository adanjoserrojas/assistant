import os

TIMEZONE = os.environ.get("TIMEZONE", "America/New_York")

DAY_START = "07:00"
DAY_END = "23:00"

TABLE_NAME = os.environ["TABLE_NAME"]
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

BREAKFAST = {
    "duration": 60,
    "earliest": "08:00",
    "preferred": "08:30",
    "latest": "09:00",
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
