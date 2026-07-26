import os

TIMEZONE = os.environ.get("TIMEZONE", "America/New_York")

DAY_START = "07:00"
DAY_END = "23:00"

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