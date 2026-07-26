"""Validator tests. Run from the repo root:  python -m pytest test -q"""

import os
import sys
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models import CalendarEvent, ScheduledActivity
from scheduler import activities_from_config, schedule_activities
from validator import validate_schedule

TZ = ZoneInfo(config.TIMEZONE)
DAY = date(2026, 7, 25)


def at(hhmm):
    hours, minutes = hhmm.split(":")
    return datetime.combine(DAY, time(int(hours), int(minutes)), tzinfo=TZ)


def event(title, start, end, all_day=False):
    return CalendarEvent(title=title, start=at(start), end=at(end), all_day=all_day)


def planned(name, start):
    duration = getattr(config, name.upper())["duration"]
    return ScheduledActivity(
        name=name, start=at(start), end=at(start) + timedelta(minutes=duration)
    )


def test_valid_schedule_has_no_problems():
    assert validate_schedule([], [planned("lunch", "12:30")], DAY, TZ) == []


def test_scheduler_output_always_validates():
    events = [event("Flagship Project Block", "10:00", "12:00")]
    scheduled, _ = schedule_activities(activities_from_config(), events, DAY, TZ)
    assert validate_schedule(events, scheduled, DAY, TZ) == []


def test_overlap_with_existing_event_is_caught():
    events = [event("Standup", "12:00", "13:00")]
    problems = validate_schedule(events, [planned("lunch", "12:30")], DAY, TZ)
    assert any("overlaps" in p for p in problems)


def test_overlap_between_generated_events_is_caught():
    problems = validate_schedule(
        [], [planned("lunch", "12:30"), planned("gym", "12:45")], DAY, TZ
    )
    assert any("overlaps" in p for p in problems)


def test_duplicate_activity_is_caught():
    problems = validate_schedule(
        [], [planned("lunch", "12:30"), planned("lunch", "14:00")], DAY, TZ
    )
    assert any("more than once" in p for p in problems)


def test_wrong_duration_is_caught():
    bad = ScheduledActivity(name="lunch", start=at("12:30"), end=at("14:30"))
    problems = validate_schedule([], [bad], DAY, TZ)
    assert any("duration" in p for p in problems)


def test_start_before_earliest_is_caught():
    problems = validate_schedule([], [planned("dinner", "12:00")], DAY, TZ)
    assert any("earlier than" in p for p in problems)


def test_start_after_latest_is_caught():
    problems = validate_schedule([], [planned("lunch", "16:00")], DAY, TZ)
    assert any("later than" in p for p in problems)


def test_end_before_start_is_caught():
    bad = ScheduledActivity(name="lunch", start=at("13:00"), end=at("12:00"))
    problems = validate_schedule([], [bad], DAY, TZ)
    assert problems


def test_all_day_event_does_not_trigger_overlap_by_default():
    assert config.ALL_DAY_BLOCKS is False
    events = [event("Research Sprint", "00:00", "00:00", all_day=True)]
    assert validate_schedule(events, [planned("lunch", "12:30")], DAY, TZ) == []


def test_empty_schedule_is_valid():
    assert validate_schedule([event("Busy", "07:00", "23:00")], [], DAY, TZ) == []
