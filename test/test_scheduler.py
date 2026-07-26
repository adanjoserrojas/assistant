"""Scheduler tests. Run from the repo root:  python -m pytest test -q"""

import os
import sys
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models import Activity, CalendarEvent
from scheduler import (
    activities_from_config,
    calculate_free_windows,
    merge_busy_intervals,
    schedule_activities,
)

TZ = ZoneInfo(config.TIMEZONE)
DAY = date(2026, 7, 25)


def at(hhmm):
    hours, minutes = hhmm.split(":")
    return datetime.combine(DAY, time(int(hours), int(minutes)), tzinfo=TZ)


def event(title, start, end, all_day=False):
    return CalendarEvent(title=title, start=at(start), end=at(end), all_day=all_day)


def as_hhmm(intervals):
    return [(f"{s:%H:%M}", f"{e:%H:%M}") for s, e in intervals]


# --- merge_busy_intervals ------------------------------------------------


def test_overlapping_events_merge():
    # plan.md section 17: 10:00-11:30 + 11:00-12:00 must become 10:00-12:00.
    merged = merge_busy_intervals(
        [event("A", "10:00", "11:30"), event("B", "11:00", "12:00")]
    )
    assert as_hhmm(merged) == [("10:00", "12:00")]


def test_disjoint_events_stay_separate():
    merged = merge_busy_intervals(
        [event("A", "09:00", "10:00"), event("B", "12:00", "13:00")]
    )
    assert as_hhmm(merged) == [("09:00", "10:00"), ("12:00", "13:00")]


def test_event_fully_contained_in_another():
    merged = merge_busy_intervals(
        [event("Outer", "09:00", "17:00"), event("Inner", "11:00", "12:00")]
    )
    assert as_hhmm(merged) == [("09:00", "17:00")]


def test_unsorted_input_still_merges():
    merged = merge_busy_intervals(
        [event("B", "11:00", "12:00"), event("A", "10:00", "11:30")]
    )
    assert as_hhmm(merged) == [("10:00", "12:00")]


def test_all_day_events_do_not_block_by_default():
    assert config.ALL_DAY_BLOCKS is False
    merged = merge_busy_intervals(
        [event("Research Sprint", "00:00", "00:00", all_day=True)]
    )
    assert merged == []


# --- calculate_free_windows ----------------------------------------------


def test_free_windows_match_plan_example():
    # plan.md section 30, adjusted to this repo's 07:00-23:00 day window.
    events = [event("Class", "09:00", "10:00"), event("Meeting", "12:00", "13:00")]
    assert as_hhmm(calculate_free_windows(events, DAY)) == [
        ("07:00", "09:00"),
        ("10:00", "12:00"),
        ("13:00", "23:00"),
    ]


def test_empty_calendar_is_one_window():
    assert as_hhmm(calculate_free_windows([], DAY)) == [("07:00", "23:00")]


def test_events_outside_day_window_are_ignored():
    events = [event("Early", "05:00", "06:00"), event("Late", "23:30", "23:59")]
    assert as_hhmm(calculate_free_windows(events, DAY)) == [("07:00", "23:00")]


def test_event_straddling_day_start_is_clamped():
    events = [event("Overnight", "06:00", "08:00")]
    assert as_hhmm(calculate_free_windows(events, DAY)) == [("08:00", "23:00")]


def test_fully_booked_day_has_no_windows():
    assert calculate_free_windows([event("All", "07:00", "23:00")], DAY) == []


# --- schedule_activities -------------------------------------------------


def test_empty_calendar_schedules_everything_at_preferred_times():
    scheduled, unplaced = schedule_activities(activities_from_config(), [], DAY)
    assert unplaced == []
    placed = {s.name: f"{s.start:%H:%M}" for s in scheduled}
    assert placed["breakfast"] == config.BREAKFAST["preferred"]
    assert placed["lunch"] == config.LUNCH["preferred"]
    assert placed["dinner"] == config.DINNER["preferred"]


def test_generated_events_never_overlap_existing_ones():
    events = [
        event("Flagship Project Block", "10:00", "12:00"),
        event("Research Meeting", "15:00", "16:00"),
    ]
    scheduled, _ = schedule_activities(activities_from_config(), events, DAY)
    for item in scheduled:
        for existing in events:
            assert item.start >= existing.end or item.end <= existing.start


def test_generated_events_never_overlap_each_other():
    scheduled, _ = schedule_activities(activities_from_config(), [], DAY)
    ordered = sorted(scheduled, key=lambda s: s.start)
    for earlier, later in zip(ordered, ordered[1:]):
        assert earlier.end <= later.start


def test_activity_is_pushed_when_preferred_slot_is_busy():
    # Lunch prefers 12:30; block it and confirm lunch moves rather than overlaps.
    events = [event("Standup", "12:00", "13:30")]
    scheduled, _ = schedule_activities(activities_from_config(), events, DAY)
    lunch = next(s for s in scheduled if s.name == "lunch")
    assert lunch.start >= at("13:30") or lunch.end <= at("12:00")


def test_durations_match_config():
    scheduled, _ = schedule_activities(activities_from_config(), [], DAY)
    for item in scheduled:
        expected = getattr(config, item.name.upper())["duration"]
        assert (item.end - item.start).total_seconds() / 60 == expected


def test_activity_stays_within_its_allowed_range():
    scheduled, _ = schedule_activities(activities_from_config(), [], DAY)
    for item in scheduled:
        settings = getattr(config, item.name.upper())
        assert f"{item.start:%H:%M}" >= settings["earliest"]
        assert f"{item.start:%H:%M}" <= settings["latest"]


def test_gym_avoids_starting_right_after_a_meal():
    scheduled, _ = schedule_activities(activities_from_config(), [], DAY)
    gym = next((s for s in scheduled if s.name == "gym"), None)
    assert gym is not None
    for meal in [s for s in scheduled if s.name in ("breakfast", "lunch", "dinner")]:
        gap = (gym.start - meal.end).total_seconds() / 60
        assert not (0 <= gap < 60), f"gym starts {gap} min after {meal.name}"


def test_unplaceable_activity_is_reported_not_forced():
    # Leave no room anywhere in the day.
    scheduled, unplaced = schedule_activities(
        activities_from_config(), [event("Booked", "07:00", "23:00")], DAY
    )
    assert scheduled == []
    assert set(unplaced) == {"breakfast", "lunch", "dinner", "gym"}


def test_partial_day_places_what_it_can():
    # Only a morning gap exists; breakfast fits, later activities do not.
    events = [event("Booked", "09:00", "23:00")]
    scheduled, unplaced = schedule_activities(activities_from_config(), events, DAY)
    assert [s.name for s in scheduled] == ["breakfast"]
    assert set(unplaced) == {"lunch", "dinner", "gym"}


def test_scheduling_is_deterministic():
    events = [event("Flagship Project Block", "10:00", "12:00")]
    first, _ = schedule_activities(activities_from_config(), events, DAY)
    second, _ = schedule_activities(activities_from_config(), events, DAY)
    assert [(s.name, s.start) for s in first] == [(s.name, s.start) for s in second]
