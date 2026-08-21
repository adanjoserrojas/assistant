"""Backfill tests. Run from the repo root:  python -m pytest test -q

events_for_day is injected, so nothing here touches Google Calendar or AWS.
"""

import os
import sys
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models import CalendarEvent
from ml.backfill import (
    MAX_SLOTS_PER_DAY,
    SNAP_TOLERANCE_MINUTES,
    build_examples,
    busy_minutes,
    examples_for_day,
    session_day,
    session_start,
)

TZ = ZoneInfo(config.TIMEZONE)
DAY = date(2026, 7, 25)          # a Saturday
WORKOUT = "Chest-Triceps"
PROFILE = {"Chest-Triceps": 82.0, "Back-Biceps": 76.0}


def at(hhmm, day=DAY):
    hours, minutes = hhmm.split(":")
    return datetime.combine(day, time(int(hours), int(minutes)), tzinfo=TZ)


def event(title, start, end, day=DAY):
    return CalendarEvent(title=title, start=at(start, day), end=at(end, day))


def completed(day, workout, checkin):
    return {
        "workout": workout,
        "checkin_at": at(checkin, day).isoformat(),
        "actual_duration_minutes": 82,
        "attended": True,
        "status": "completed",
    }


def unattended(day, workout):
    return {
        "workout": workout,
        "session_date": day.isoformat(),
        "attended": False,
        "status": "unattended",
    }


# --- busy_minutes --------------------------------------------------------


def test_busy_minutes_sums_events():
    events = [event("A", "09:00", "10:00"), event("B", "13:00", "14:30")]
    assert busy_minutes(events, DAY, TZ) == 150


def test_busy_minutes_clamps_to_the_day_window():
    # DAY_START is 07:00; the 05:00-08:00 event only counts for one hour.
    assert busy_minutes([event("Early", "05:00", "08:00")], DAY, TZ) == 60


def test_busy_minutes_does_not_double_count_overlaps():
    events = [event("A", "09:00", "11:00"), event("B", "10:00", "12:00")]
    assert busy_minutes(events, DAY, TZ) == 180


def test_empty_calendar_is_zero_busy():
    assert busy_minutes([], DAY, TZ) == 0


# --- examples_for_day ----------------------------------------------------


def test_attended_day_has_exactly_one_positive():
    found = examples_for_day(WORKOUT, 82, [], DAY, at("17:30"), TZ)
    assert sum(1 for example in found if example.chosen) == 1
    assert len(found) > 1, "the slots you passed on are the negatives"


def test_the_positive_is_the_slot_you_used():
    found = examples_for_day(WORKOUT, 82, [], DAY, at("17:30"), TZ)
    chosen = [example for example in found if example.chosen]
    assert chosen[0].start_time == "17:30:00"


def test_unattended_day_is_all_negatives():
    found = examples_for_day(WORKOUT, 82, [], DAY, None, TZ)
    assert found, "an open day should offer slots you declined"
    assert not any(example.chosen for example in found)


def test_offgrid_checkin_snaps_to_nearest_slot():
    # 17:37 is not on the 15-minute grid; 17:30 is the honest match.
    found = examples_for_day(WORKOUT, 82, [], DAY, at("17:37"), TZ)
    chosen = [example for example in found if example.chosen]
    assert len(chosen) == 1
    assert chosen[0].start_time == "17:30:00"


def test_day_is_dropped_when_checkin_matches_no_slot():
    # Trained 09:00 on a day the calendar was fully booked until 17:00. There is
    # no slot to hang the positive on, so the day contributes nothing -- and
    # crucially not a pile of negatives for a day you actually attended.
    events = [event("Work", config.DAY_START, "17:00")]
    assert examples_for_day(WORKOUT, 82, events, DAY, at("09:00"), TZ) == []


def test_rest_day_yields_nothing():
    assert examples_for_day("Rest-days", 82, [], DAY, None, TZ) == []


def test_no_room_yields_nothing():
    events = [event("Booked", config.DAY_START, config.DAY_END)]
    assert examples_for_day(WORKOUT, 82, events, DAY, None, TZ) == []


def test_slot_count_is_capped():
    found = examples_for_day(WORKOUT, 82, [], DAY, None, TZ)
    assert len(found) <= MAX_SLOTS_PER_DAY


def test_every_example_carries_the_same_day_context():
    events = [event("Work", "09:00", "17:00")]
    found = examples_for_day(WORKOUT, 82, events, DAY, None, TZ)
    assert {example.day for example in found} == {DAY.isoformat()}
    assert {example.weekday for example in found} == {"Saturday"}
    # busy_minutes is a property of the day, identical across its slots.
    assert len({example.busy_minutes for example in found}) == 1


def test_start_hour_is_decimal():
    found = examples_for_day(WORKOUT, 82, [], DAY, at("17:30"), TZ)
    chosen = [example for example in found if example.chosen][0]
    assert chosen.start_hour == 17.5


def test_duration_never_comes_from_the_session():
    # Both classes must be sized by the profile: actual_duration_minutes does
    # not exist at prediction time, so letting it in would leak the label.
    attended = examples_for_day(WORKOUT, 82, [], DAY, at("17:30"), TZ)
    missed = examples_for_day(WORKOUT, 82, [], DAY, None, TZ)
    assert {e.duration_minutes for e in attended + missed} == {82}


# --- session_day / session_start -----------------------------------------


def test_session_day_from_checkin():
    assert session_day(completed(DAY, WORKOUT, "17:30")) == DAY


def test_session_day_from_session_date():
    assert session_day(unattended(DAY, WORKOUT)) == DAY


def test_session_day_of_junk_is_none():
    assert session_day({"workout": WORKOUT}) is None


def test_unattended_has_no_start():
    assert session_start(unattended(DAY, WORKOUT), TZ) is None


def test_completed_start_is_aware():
    start = session_start(completed(DAY, WORKOUT, "17:30"), TZ)
    assert start is not None and start.tzinfo is not None


# --- build_examples ------------------------------------------------------


def test_build_examples_mixes_both_classes():
    monday = date(2026, 7, 27)
    sessions = [
        completed(DAY, WORKOUT, "17:30"),
        unattended(monday, "Back-Biceps"),
    ]
    examples, diagnostics = build_examples(sessions, lambda day: [], PROFILE, TZ)

    assert diagnostics["days_used"] == 2
    assert diagnostics["positives"] == 1
    assert diagnostics["negatives"] == diagnostics["examples"] - 1
    assert diagnostics["negatives"] > 1, "slot-level negatives, not one per day"


def test_build_examples_reads_each_day_once():
    seen = []

    def events_for_day(day):
        seen.append(day)
        return []

    sessions = [completed(DAY, WORKOUT, "17:30"), unattended(date(2026, 7, 27), WORKOUT)]
    build_examples(sessions, events_for_day, PROFILE, TZ)
    assert seen == [DAY, date(2026, 7, 27)]


def test_build_examples_uses_that_days_calendar():
    busy = {DAY: [event("Work", config.DAY_START, "17:00")]}
    sessions = [completed(DAY, WORKOUT, "17:30")]
    examples, _ = build_examples(sessions, lambda day: busy.get(day, []), PROFILE, TZ)
    assert examples, "17:30 fits the evening gap"
    assert all(example.start_time >= "17:00:00" for example in examples)


def test_build_examples_counts_dropped_days():
    booked = [event("Booked", config.DAY_START, config.DAY_END)]
    sessions = [unattended(DAY, WORKOUT)]
    examples, diagnostics = build_examples(sessions, lambda day: booked, PROFILE, TZ)
    assert examples == []
    assert diagnostics["days_dropped_no_room"] == 1
    assert diagnostics["days_used"] == 0


def test_build_examples_flags_unreadable_records():
    examples, diagnostics = build_examples([{"workout": WORKOUT}], lambda d: [], PROFILE, TZ)
    assert examples == []
    assert diagnostics["days_dropped_unreadable"] == 1


def test_examples_are_groupable_by_day():
    sessions = [completed(DAY, WORKOUT, "17:30"), unattended(date(2026, 7, 27), WORKOUT)]
    examples, _ = build_examples(sessions, lambda day: [], PROFILE, TZ)
    days = {example.day for example in examples}
    assert days == {DAY.isoformat(), "2026-07-27"}


def test_to_row_is_flat_and_serializable():
    import json

    found = examples_for_day(WORKOUT, 82, [], DAY, at("17:30"), TZ)
    row = found[0].to_row()
    assert json.loads(json.dumps(row)) == row
    assert "chosen" in row
