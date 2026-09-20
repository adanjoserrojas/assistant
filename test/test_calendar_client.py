"""Multi-calendar read tests. Run from the repo root:  python -m pytest test -q

Everything here is offline. calendar_client.authenticate is replaced with a fake
Google service, so no credentials are resolved and no request leaves the machine
-- the same property the rest of the suite relies on.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

import calendar_client
import config

TZ = ZoneInfo(config.TIMEZONE)
DAY = date(2026, 9, 19)

WORK = "work@example.com"
PERSONAL = "personal@gmail.com"


def timed(event_id, summary, hour, minute=0, ical_uid=None, status="confirmed"):
    start = datetime(2026, 9, 19, hour, minute, tzinfo=TZ)
    end = datetime(2026, 9, 19, hour + 1, minute, tzinfo=TZ)
    item = {
        "id": event_id,
        "summary": summary,
        "status": status,
        "start": {"dateTime": start.isoformat()},
        "end": {"dateTime": end.isoformat()},
    }
    if ical_uid:
        item["iCalUID"] = ical_uid
    return item


class FakeService:
    """Minimal stand-in for the Google Calendar service.

    `pages` maps a calendar id to either a list of items or an Exception to
    raise, which is how the partial-failure tests inject a broken calendar.
    """

    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def events(self):
        return self

    def list(self, calendarId, **params):
        self.calls.append((calendarId, params))
        self._pending = self.pages.get(calendarId, [])
        self._filter = params.get("privateExtendedProperty")
        return self

    def execute(self):
        if isinstance(self._pending, Exception):
            raise self._pending

        items = list(self._pending)
        # Google applies privateExtendedProperty server-side; without this the
        # fake would hand find_agent_events every event and the marker query
        # would look like it matched everything.
        if self._filter:
            key, _, value = self._filter.partition("=")
            items = [
                item
                for item in items
                if ((item.get("extendedProperties") or {}).get("private") or {}).get(key)
                == value
            ]
        return {"items": items}


@pytest.fixture
def calendars(monkeypatch):
    """Install a fake service and a read set, restoring config afterwards."""

    def install(pages, read_ids=None, write_id=WORK):
        monkeypatch.setattr(calendar_client, "authenticate", lambda: FakeService(pages))
        monkeypatch.setattr(config, "CALENDAR_IDS", read_ids or list(pages))
        monkeypatch.setattr(config, "CALENDAR_ID", write_id)

    return install


def test_merges_events_from_every_read_calendar(calendars):
    calendars({
        WORK: [timed("w1", "Standup", 9)],
        PERSONAL: [timed("p1", "Dentist", 14)],
    })

    events, failures = calendar_client.get_today_events_detailed(DAY)

    assert failures == []
    assert [event.title for event in events] == ["Standup", "Dentist"]


def test_merged_calendars_are_resorted_earliest_first(calendars):
    """orderBy=startTime only orders within one request, so interleaved
    calendars come back out of order unless the merge re-sorts."""
    calendars({
        WORK: [timed("w1", "Morning", 8), timed("w2", "Evening", 20)],
        PERSONAL: [timed("p1", "Noon", 12)],
    })

    events, _ = calendar_client.get_today_events_detailed(DAY)

    assert [event.title for event in events] == ["Morning", "Noon", "Evening"]
    assert events == sorted(events, key=lambda item: item.start)


def test_same_event_on_two_calendars_is_deduplicated_by_ical_uid(calendars):
    """A cross-account invite lands on both calendars as separate resources
    with different ids but the same iCalUID."""
    shared = "abc123@google.com"
    calendars({
        WORK: [timed("w1", "1:1 with Alex", 15, ical_uid=shared)],
        PERSONAL: [timed("p9", "1:1 with Alex", 15, ical_uid=shared)],
    })

    events, _ = calendar_client.get_today_events_detailed(DAY)

    assert len(events) == 1
    # First occurrence wins, and the write target is read first.
    assert events[0].calendar_id == WORK


def test_distinct_events_at_the_same_time_are_both_kept(calendars):
    """Dedup must key on identity, not on the time slot."""
    calendars({
        WORK: [timed("w1", "Work thing", 15, ical_uid="uid-a")],
        PERSONAL: [timed("p1", "Personal thing", 15, ical_uid="uid-b")],
    })

    events, _ = calendar_client.get_today_events_detailed(DAY)

    assert len(events) == 2


def test_events_are_tagged_with_their_source_calendar(calendars):
    calendars({
        WORK: [timed("w1", "Standup", 9)],
        PERSONAL: [timed("p1", "Dentist", 14)],
    })

    events, _ = calendar_client.get_today_events_detailed(DAY)

    assert {event.title: event.calendar_id for event in events} == {
        "Standup": WORK,
        "Dentist": PERSONAL,
    }


def test_cancelled_events_are_still_dropped(calendars):
    calendars({
        WORK: [timed("w1", "Standup", 9), timed("w2", "Called off", 11, status="cancelled")],
    })

    events, _ = calendar_client.get_today_events_detailed(DAY)

    assert [event.title for event in events] == ["Standup"]


def test_one_failing_calendar_does_not_lose_the_others(calendars):
    """The chosen policy is log-and-continue: a broken calendar degrades the
    day rather than cancelling it."""
    calendars({
        WORK: [timed("w1", "Standup", 9)],
        PERSONAL: RuntimeError("HttpError 404"),
    })

    events, failures = calendar_client.get_today_events_detailed(DAY)

    assert [event.title for event in events] == ["Standup"]
    assert len(failures) == 1
    assert failures[0]["calendar_id"] == PERSONAL
    assert "404" in failures[0]["error"]


def test_every_calendar_failing_yields_no_events_and_all_failures(calendars):
    calendars({
        WORK: RuntimeError("boom"),
        PERSONAL: RuntimeError("bang"),
    })

    events, failures = calendar_client.get_today_events_detailed(DAY)

    assert events == []
    assert {failure["calendar_id"] for failure in failures} == {WORK, PERSONAL}


def test_get_today_events_still_returns_a_bare_list(calendars):
    """ml/backfill.py passes this function itself as events_for_day, so the
    return type is part of the training pipeline's contract."""
    calendars({WORK: [timed("w1", "Standup", 9)]})

    events = calendar_client.get_today_events(DAY)

    assert isinstance(events, list)
    assert [event.title for event in events] == ["Standup"]


def test_read_set_defaults_to_the_write_target(monkeypatch):
    monkeypatch.setenv("CALENDAR_IDS", "")
    monkeypatch.setattr(config, "CALENDAR_ID", WORK)
    monkeypatch.setattr(config, "CALENDAR_IDS", config._read_calendar_ids())

    assert calendar_client.read_calendar_ids() == [WORK]


def test_read_set_puts_the_write_target_first_and_deduplicates(monkeypatch):
    monkeypatch.setenv("CALENDAR_IDS", f" {PERSONAL} , {WORK} ,, {PERSONAL} ")
    monkeypatch.setattr(config, "CALENDAR_ID", WORK)

    assert config._read_calendar_ids() == [WORK, PERSONAL]


def test_read_calendar_ids_raises_when_nothing_is_configured(monkeypatch):
    monkeypatch.setattr(config, "CALENDAR_IDS", [])

    with pytest.raises(RuntimeError, match="No calendars to read"):
        calendar_client.read_calendar_ids()


def test_find_agent_events_scans_every_read_calendar(calendars):
    marked = timed("w1", f"{config.AGENT_PREFIX} Breakfast", 8)
    marked["extendedProperties"] = {"private": {config.AGENT_MARKER_KEY: "1"}}
    calendars({
        WORK: [marked],
        PERSONAL: [timed("p1", "Dentist", 14)],
    })

    titles = calendar_client.find_agent_events(DAY)

    assert titles == [f"{config.AGENT_PREFIX} Breakfast"]
