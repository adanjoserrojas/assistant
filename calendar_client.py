"""Google Calendar access via a service account.

Hides the Google API from the rest of the app (plan.md section 8).

Credentials are resolved in this order:
  1. GOOGLE_SERVICE_ACCOUNT_JSON -- raw JSON string (Lambda / Secrets Manager)
  2. config.SERVICE_ACCOUNT_FILE -- path on disk (local dev)

Run directly to smoke-test the connection:
    python calendar_client.py
"""

import json
import os
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from google.oauth2 import service_account
from googleapiclient.discovery import build

import config
from models import CalendarEvent

# Narrower than the full "calendar" scope: read/write events, nothing else.
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

_service = None


def _secret_json(secret_id):
    """Fetch the service account key from AWS Secrets Manager (Lambda path)."""
    import boto3

    region = os.environ.get("AWS_REGION", "us-east-1")
    secret = boto3.client("secretsmanager", region_name=region).get_secret_value(
        SecretId=secret_id
    )
    return json.loads(secret["SecretString"])


def _credentials():
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if raw:
        return service_account.Credentials.from_service_account_info(
            json.loads(raw), scopes=SCOPES
        )

    # Preferred in Lambda: the key never appears in an env var or the console.
    secret_id = os.environ.get("GOOGLE_SA_SECRET_ID")
    if secret_id:
        return service_account.Credentials.from_service_account_info(
            _secret_json(secret_id), scopes=SCOPES
        )

    path = os.environ.get("SERVICE_ACCOUNT_FILE", config.SERVICE_ACCOUNT_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No service account key at {path!r}. Download it from GCP Console -> "
            "IAM & Admin -> Service Accounts -> Keys -> Add Key -> JSON."
        )
    return service_account.Credentials.from_service_account_file(path, scopes=SCOPES)


def calendar_id():
    if not config.CALENDAR_ID:
        raise RuntimeError(
            "CALENDAR_ID is not set. Set it to the calendar address you shared "
            "with the service account -- locally `setx CALENDAR_ID you@example.com` "
            "(then reopen the terminal), or as a Lambda environment variable."
        )
    return config.CALENDAR_ID


def authenticate():
    """Build (and memoize) the Calendar service."""
    global _service
    if _service is None:
        _service = build(
            "calendar", "v3", credentials=_credentials(), cache_discovery=False
        )
    return _service


def timezone():
    return ZoneInfo(config.TIMEZONE)


def day_bounds(day=None):
    """Timezone-aware midnight-to-midnight bounds for a local day."""
    tz = timezone()
    day = day or datetime.now(tz).date()
    start = datetime.combine(day, time.min, tzinfo=tz)
    return start, start + timedelta(days=1)


def _parse_event(item, tz):
    start, end = item["start"], item["end"]
    title = item.get("summary", "(no title)")

    # Timed events carry dateTime; all-day events carry only date.
    if "dateTime" in start:
        return CalendarEvent(
            title=title,
            start=datetime.fromisoformat(start["dateTime"]).astimezone(tz),
            end=datetime.fromisoformat(end["dateTime"]).astimezone(tz),
            all_day=False,
        )

    # An all-day event's end date is exclusive.
    return CalendarEvent(
        title=title,
        start=datetime.combine(date.fromisoformat(start["date"]), time.min, tzinfo=tz),
        end=datetime.combine(date.fromisoformat(end["date"]), time.min, tzinfo=tz),
        all_day=True,
    )


def _raw_events(day=None, **params):
    """Page through events.list, returning raw API dicts."""
    start, end = day_bounds(day)
    service = authenticate()

    items, page_token = [], None
    while True:
        response = (
            service.events()
            .list(
                calendarId=calendar_id(),
                timeMin=start.isoformat(),  # RFC3339 with offset, required
                timeMax=end.isoformat(),
                singleEvents=True,  # expand recurring series into instances
                orderBy="startTime",  # only valid when singleEvents=True
                maxResults=250,
                pageToken=page_token,
                **params,
            )
            .execute()
        )
        items.extend(response.get("items", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            return items


def get_today_events(day=None):
    """Today's events as CalendarEvent objects, earliest first."""
    tz = timezone()
    return [
        _parse_event(item, tz)
        for item in _raw_events(day)
        if item.get("status") != "cancelled"
    ]


def find_agent_events(day=None):
    """Titles of events this agent already created today (plan.md section 24)."""
    seen = {item["id"]: item for item in _raw_events(
        day, privateExtendedProperty=f"{config.AGENT_MARKER_KEY}=1"
    )}

    # Fall back to the title prefix so events created before the marker existed
    # still count toward idempotency.
    for item in _raw_events(day):
        if item.get("summary", "").startswith(config.AGENT_PREFIX):
            seen.setdefault(item["id"], item)

    return [item.get("summary", "") for item in seen.values()]


def create_event(title, start, end):
    """Create a marked event. start/end must be timezone-aware datetimes."""
    body = {
        "summary": f"{config.AGENT_PREFIX} {title}",
        "description": "Created automatically by the calendar agent.",
        "start": {"dateTime": start.isoformat(), "timeZone": config.TIMEZONE},
        "end": {"dateTime": end.isoformat(), "timeZone": config.TIMEZONE},
        "extendedProperties": {"private": {config.AGENT_MARKER_KEY: "1"}},
    }
    return (
        authenticate()
        .events()
        .insert(calendarId=calendar_id(), body=body)
        .execute()
    )


if __name__ == "__main__":
    todays_events = get_today_events()
    print(f"{calendar_id()} -- {datetime.now(timezone()).date()}\n")
    if not todays_events:
        print("  (no events today)")
    for event in todays_events:
        when = "ALL DAY    " if event.all_day else f"{event.start:%H:%M}-{event.end:%H:%M}"
        print(f"  {when}  {event.title}")
