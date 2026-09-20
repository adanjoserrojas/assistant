"""Google Calendar access via a service account.

Hides the Google API from the rest of the app (plan.md section 8).

Credentials are resolved in this order:
  1. GOOGLE_SERVICE_ACCOUNT_JSON -- raw JSON string (Lambda / Secrets Manager)
  2. config.SERVICE_ACCOUNT_FILE -- path on disk (local dev)

Run directly to smoke-test the connection:
    python calendar-agent/calendar_client.py
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
    """The single calendar the agent WRITES to."""
    if not config.CALENDAR_ID:
        raise RuntimeError(
            "CALENDAR_ID is not set. Set it to the calendar address you shared "
            "with the service account -- locally `setx CALENDAR_ID you@example.com` "
            "(then reopen the terminal), or as a Lambda environment variable."
        )
    return config.CALENDAR_ID


def read_calendar_ids():
    """Every calendar the agent READS, write target first.

    Deliberately plural where calendar_id() is singular: the agent has to see
    the whole day across every account to avoid double-booking, but it only
    ever writes to one place.
    """
    if not config.CALENDAR_IDS:
        raise RuntimeError(
            "No calendars to read. Set CALENDAR_ID to the calendar you shared "
            "with the service account, and optionally CALENDAR_IDS to a "
            'comma-separated list to read several -- locally `setx CALENDAR_IDS '
            '"work@example.com,personal@gmail.com"` (then reopen the terminal), '
            "or as a Lambda environment variable."
        )
    return list(config.CALENDAR_IDS)


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
            calendar_id=item.get("_calendar_id", ""),
        )

    # An all-day event's end date is exclusive.
    return CalendarEvent(
        title=title,
        start=datetime.combine(date.fromisoformat(start["date"]), time.min, tzinfo=tz),
        end=datetime.combine(date.fromisoformat(end["date"]), time.min, tzinfo=tz),
        all_day=True,
        calendar_id=item.get("_calendar_id", ""),
    )


def _list_one(service, cal_id, start, end, params):
    """Page through one calendar's events.list."""
    items, page_token = [], None
    while True:
        response = (
            service.events()
            .list(
                calendarId=cal_id,
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


def _describe(error):
    """Short, readable failure text.

    googleapiclient's HttpError stringifies to the entire request URL, query
    string included, which buries the one thing that matters (404 vs 403 vs a
    timeout) in 300 characters of noise -- in the report and in every log line.
    """
    status = getattr(getattr(error, "resp", None), "status", None)
    if status:
        return f"HTTP {status}: {getattr(error, 'reason', None) or type(error).__name__}"

    return f"{type(error).__name__}: {' '.join(str(error).split())[:160]}"


def _raw_events(day=None, **params):
    """Raw API dicts from every read calendar, plus whatever failed.

    Returns (items, failures). A calendar that raises is recorded rather than
    propagated -- losing one calendar degrades the day, it does not cancel it.

    The caller owns making that degradation visible (see agent.report), and it
    matters more than it looks: the events we could not read are exactly the
    ones validator.validate_schedule can no longer check overlaps against, so a
    quiet failure here becomes a double-booking downstream.
    """
    start, end = day_bounds(day)
    service = authenticate()

    items, failures = [], []
    for cal_id in read_calendar_ids():
        try:
            found = _list_one(service, cal_id, start, end, params)
        except Exception as error:
            failures.append({"calendar_id": cal_id, "error": _describe(error)})
            continue

        # Tag provenance before the dicts lose their calendar context.
        for item in found:
            item["_calendar_id"] = cal_id
        items.extend(found)

    return items, failures


def _deduplicate(items):
    """One entry per real-world event, across calendars.

    An invite sent from one of your accounts to another lands on both calendars
    as separate resources sharing an iCalUID. merge_busy_intervals would collapse
    the duplicate intervals anyway, so this is about what a human and the LLM
    see -- the same meeting listed twice is noise in the prompt and the report.

    First occurrence wins, and read_calendar_ids() puts the write target first.
    """
    seen, unique = set(), []
    for item in items:
        key = item.get("iCalUID") or item.get("id")
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def get_today_events_detailed(day=None):
    """Today's events across every read calendar, plus per-calendar failures.

    Returns (events, failures), earliest first. The sort is not redundant:
    orderBy=startTime only orders within a single request, so merged calendars
    arrive interleaved and nothing else re-establishes the ordering that
    get_today_events promises.
    """
    tz = timezone()
    items, failures = _raw_events(day)

    events = [
        _parse_event(item, tz)
        for item in _deduplicate(items)
        if item.get("status") != "cancelled"
    ]
    events.sort(key=lambda event: (event.start, event.title))
    return events, failures


def get_today_events(day=None):
    """Today's events as CalendarEvent objects, earliest first.

    Deliberately still returns a bare list. ml/backfill.py takes this very
    function as its `events_for_day` callable, so the return type is part of the
    training pipeline's contract -- widening it here would quietly change how
    every historical day is reconstructed. Callers that need to know a calendar
    failed use get_today_events_detailed instead.
    """
    return get_today_events_detailed(day)[0]


def find_agent_events(day=None):
    """Titles of events this agent already created today (plan.md section 24).

    Scans every read calendar, not just the write target. Read failures are
    ignored here on purpose: this answers "have I already done this?", and a
    calendar that will not load cannot hold an event this agent wrote, because
    the agent only ever writes to CALENDAR_ID.
    """
    marked, _ = _raw_events(
        day, privateExtendedProperty=f"{config.AGENT_MARKER_KEY}=1"
    )
    seen = {item["id"]: item for item in marked}

    # Fall back to the title prefix so events created before the marker existed
    # still count toward idempotency.
    everything, _ = _raw_events(day)
    for item in everything:
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
    todays_events, read_failures = get_today_events_detailed()

    print(f"{datetime.now(timezone()).date()}\n")
    print("Calendars read:")
    broken = {failure["calendar_id"] for failure in read_failures}
    for source in read_calendar_ids():
        if source in broken:
            continue
        count = sum(1 for event in todays_events if event.calendar_id == source)
        marker = "  <- writes go here" if source == config.CALENDAR_ID else ""
        print(f"  {source}  ({count} event(s)){marker}")
    for failure in read_failures:
        print(f"  {failure['calendar_id']}  !! UNREADABLE -- {failure['error']}")

    print()
    if not todays_events:
        print("  (no events today)")
    for event in todays_events:
        when = "ALL DAY    " if event.all_day else f"{event.start:%H:%M}-{event.end:%H:%M}"
        print(f"  {when}  {event.title}")
