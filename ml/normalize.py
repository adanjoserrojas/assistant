import json
from boto3.dynamodb.types import TypeDeserializer
from .repository import (
    fetch_sessions as logs,
    count_training_sessions as amount
)
from datetime import datetime

'''
Sole purpose of this thing is to clean and prepare daata for ML logistics regression model
And mean calculation
'''

# protos
def parse_time(time_date: str):
    days = time_date.split("T")
    day = days[0].split("-")
    time = days[1].split("-")

    string_day = datetime(int(day[0]), int(day[1]), int(day[2]))
    return [time[0], string_day.strftime('%A')]

def group_workouts() -> dict[dict]:
    pass

def weekday_from_date(day: str | None) -> str | None:
    """Weekday name from a plain YYYY-MM-DD, no time component required."""
    if not day:
        return None
    year, month, date_of_month = day.split("-")
    return datetime(int(year), int(month), int(date_of_month)).strftime('%A')

def build_training_data(old_data: list[dict]) -> list[dict]:

    new_data = []

    for item in old_data:

        # Unattended days are written by validate_sesh_handler and carry no
        # check-in, no duration and no location -- none of those happened. They
        # are the negative examples, so they must survive this loop rather than
        # KeyError it. session_date is what they carry instead.
        checkin_at = item.get("checkin_at")
        if checkin_at:
            started_at, weekday = parse_time(str(checkin_at))
        else:
            started_at = None
            weekday = weekday_from_date(item.get("session_date"))

        duration = item.get("actual_duration_minutes")

        new_data.append({
            "workout": item.get("workout"),
            "duration": float(duration) if duration is not None else 0.0,
            "location": item.get("location_code"),
            "started_at": started_at,
            "weekday": weekday,
            # Read the record instead of hardcoding False -- otherwise every row
            # is a negative and the regression has only one class to learn from.
            "attended": bool(item.get("attended", False)),
        })

    return new_data


def clean_data() -> list[dict]:
    """Fetch and normalize in one call.

    A function, not a module-level CLEAN_DATA constant: as a constant this ran a
    DynamoDB query on import, so merely importing anything under ml/ -- including
    candidate_generator, which needs no session history at all -- hit the table.
    That is a cold-start cost on every ML Lambda and makes the package impossible
    to import in a test without credentials.
    """
    return build_training_data(logs())