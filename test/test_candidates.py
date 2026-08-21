"""Candidate generation tests. Run from the repo root:  python -m pytest test -q

Everything here is pure -- no AWS, no credentials. generate_candidates takes the
day's events as an argument precisely so this file can stay that way.
"""

import os
import sys
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models import CalendarEvent
from ml.candidate_generator import (
    MIN_SPACING_MINUTES,
    REST_WORKOUT,
    Candidate,
    spread_starts,
    duration_for,
    generate_candidates,
)

TZ = ZoneInfo(config.TIMEZONE)
DAY = date(2026, 7, 25)          # a Saturday
WORKOUT = "Chest-Triceps"


def at(hhmm):
    hours, minutes = hhmm.split(":")
    return datetime.combine(DAY, time(int(hours), int(minutes)), tzinfo=TZ)


def event(title, start, end, all_day=False):
    return CalendarEvent(title=title, start=at(start), end=at(end), all_day=all_day)


def starts(candidates):
    return [f"{candidate.start:%H:%M}" for candidate in candidates]


# --- duration_for --------------------------------------------------------


def test_learned_duration_wins():
    assert duration_for(WORKOUT, {WORKOUT: 82.0}) == 82


def test_duration_rounds_up():
    # 76.2 rounded down would size a window half your sessions overrun.
    assert duration_for(WORKOUT, {WORKOUT: 76.2}) == 77


def test_unknown_workout_falls_back_to_config():
    assert duration_for("Brand-New-Split", {}) == config.GYM["duration"]


def test_zero_duration_falls_back_to_config():
    # A profile of 0 means no completed sessions, not a zero-minute workout.
    assert duration_for(WORKOUT, {WORKOUT: 0}) == config.GYM["duration"]


# --- generate_candidates -------------------------------------------------


def test_empty_calendar_returns_three_spread_candidates():
    found = generate_candidates(WORKOUT, 82, [], DAY, TZ)
    assert len(found) == 3
    # Earliest, middle, latest -- not three slots bunched at the open.
    assert found[0].start < found[1].start < found[2].start
    assert found[2].start - found[0].start >= timedelta(hours=6)


def test_candidates_respect_config_gym_bounds():
    found = generate_candidates(WORKOUT, 82, [], DAY, TZ)
    earliest = at(config.GYM["earliest"])
    latest = at(config.GYM["latest"])
    for candidate in found:
        assert earliest <= candidate.start <= latest


def test_candidates_never_overlap_a_busy_block():
    events = [event("Class", "09:00", "12:00"), event("Work", "13:00", "18:00")]
    found = generate_candidates(WORKOUT, 82, events, DAY, TZ)
    assert found, "a day with two gaps should still yield options"
    for candidate in found:
        for busy in events:
            assert not (candidate.start < busy.end and busy.start < candidate.end)


def test_candidates_are_spaced_apart():
    found = generate_candidates(WORKOUT, 82, [], DAY, TZ)
    gaps = [
        (b.start - a.start).total_seconds() / 60 for a, b in zip(found, found[1:])
    ]
    assert all(gap >= MIN_SPACING_MINUTES for gap in gaps), gaps


def test_duration_sizes_the_slot():
    found = generate_candidates(WORKOUT, 82, [], DAY, TZ)
    for candidate in found:
        assert candidate.end - candidate.start == timedelta(minutes=82)
        assert candidate.duration_minutes == 82


def test_longer_workout_fits_fewer_windows():
    # A 45-minute gap holds the short workout and not the long one.
    events = [event("Work", "08:00", "17:00"), event("Dinner", "17:45", "22:30")]
    short = generate_candidates(WORKOUT, 30, events, DAY, TZ)
    long = generate_candidates(WORKOUT, 90, events, DAY, TZ)
    assert short, "30 minutes fits the 17:00-17:45 gap"
    assert not long, "90 minutes does not fit anywhere on this day"


def test_fully_booked_day_returns_empty():
    events = [event("Booked", config.DAY_START, config.DAY_END)]
    assert generate_candidates(WORKOUT, 82, events, DAY, TZ) == []


def test_rest_day_returns_empty():
    assert generate_candidates(REST_WORKOUT, 82, [], DAY, TZ) == []


def test_limit_is_honoured():
    assert len(generate_candidates(WORKOUT, 82, [], DAY, TZ, limit=1)) == 1
    assert len(generate_candidates(WORKOUT, 82, [], DAY, TZ, limit=5)) == 5


def test_gaps_describe_the_surrounding_window():
    # One free window, 12:00-18:00. A 60-minute session somewhere inside it
    # should report the free time on each side, summing to the slack.
    events = [event("AM", config.DAY_START, "12:00"), event("PM", "18:00", config.DAY_END)]
    found = generate_candidates(WORKOUT, 60, events, DAY, TZ, limit=1)
    assert len(found) == 1
    candidate = found[0]
    slack = (6 * 60) - 60
    assert candidate.gap_before_minutes + candidate.gap_after_minutes == slack
    assert candidate.gap_before_minutes >= 0
    assert candidate.gap_after_minutes >= 0


def test_generation_is_deterministic():
    events = [event("Work", "09:00", "17:00")]
    first = generate_candidates(WORKOUT, 82, events, DAY, TZ)
    second = generate_candidates(WORKOUT, 82, events, DAY, TZ)
    assert starts(first) == starts(second)


def test_to_dict_carries_the_model_features():
    found = generate_candidates(WORKOUT, 82, [], DAY, TZ, limit=1)
    payload = found[0].to_dict()
    assert payload["workout"] == WORKOUT
    assert payload["weekday"] == "Saturday"
    assert payload["duration_minutes"] == 82
    assert set(payload) >= {
        "start", "end", "start_time", "weekday", "workout",
        "duration_minutes", "gap_before_minutes", "gap_after_minutes",
    }


# --- spread_starts -------------------------------------------------------------


def test_spread_of_empty_is_empty():
    assert spread_starts([], 3, timedelta(minutes=60)) == []


def test_spread_keeps_everything_when_under_limit():
    slots = [at("08:00"), at("12:00")]
    assert spread_starts(slots, 3, timedelta(minutes=60)) == slots


def test_spread_drops_clustered_slots():
    # 15-minute grid across one hour collapses to a single option.
    slots = [at("17:00"), at("17:15"), at("17:30"), at("17:45")]
    assert spread_starts(slots, 3, timedelta(minutes=60)) == [at("17:00")]


def test_spread_reaches_both_ends():
    slots = [at(f"{hour:02d}:00") for hour in range(7, 22)]
    picked = spread_starts(slots, 3, timedelta(minutes=60))
    assert picked[0] == slots[0]
    assert picked[-1] == slots[-1]
