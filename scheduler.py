"""Deterministic scheduling (plan.md sections 16-21).

Pure interval arithmetic -- no calendar calls, no LLM. Every function here
takes timezone-aware datetimes and returns timezone-aware datetimes.
"""

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import config
from models import Activity, CalendarEvent, ScheduledActivity

# Candidate start times are generated on this grid.
GRANULARITY_MINUTES = 15

# Applied when gym would start too soon after a meal (plan.md section 21).
MEAL_GYM_PENALTY = 10_000
MEAL_GYM_GAP_MINUTES = 60

MEALS = ("breakfast", "lunch", "dinner")

# Fixed order for the MVP; each scheduled activity becomes busy for the next.
SCHEDULING_ORDER = ("breakfast", "lunch", "dinner", "gym")


def parse_hhmm(value):
    hours, minutes = value.split(":")
    return time(int(hours), int(minutes))


def activities_from_config():
    """Build Activity objects from the config dicts, in scheduling order."""
    settings = {
        "breakfast": config.BREAKFAST,
        "lunch": config.LUNCH,
        "dinner": config.DINNER,
        "gym": config.GYM,
    }
    return [
        Activity(
            name=name,
            duration_minutes=settings[name]["duration"],
            earliest_start=parse_hhmm(settings[name]["earliest"]),
            latest_start=parse_hhmm(settings[name]["latest"]),
            preferred_start=parse_hhmm(settings[name]["preferred"]),
        )
        for name in SCHEDULING_ORDER
    ]


def day_window(day, tz=None):
    """The schedulable span of a day, from config.DAY_START to config.DAY_END."""
    tz = tz or ZoneInfo(config.TIMEZONE)
    return (
        datetime.combine(day, parse_hhmm(config.DAY_START), tzinfo=tz),
        datetime.combine(day, parse_hhmm(config.DAY_END), tzinfo=tz),
    )


def merge_busy_intervals(events):
    """Collapse overlapping/touching events into disjoint busy intervals.

    Without this, two overlapping events punch two holes in the free-time
    calculation and the scheduler can place an activity inside a busy block.
    """
    blocking = [
        (event.start, event.end)
        for event in events
        if not event.all_day or config.ALL_DAY_BLOCKS
    ]
    if not blocking:
        return []

    merged = []
    for start, end in sorted(blocking):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def calculate_free_windows(events, day, tz=None):
    """Gaps inside the day window that no event occupies."""
    window_start, window_end = day_window(day, tz)

    free, cursor = [], window_start
    for busy_start, busy_end in merge_busy_intervals(events):
        # Ignore anything entirely outside the day window.
        if busy_end <= window_start or busy_start >= window_end:
            continue
        busy_start = max(busy_start, window_start)
        busy_end = min(busy_end, window_end)

        if busy_start > cursor:
            free.append((cursor, busy_start))
        cursor = max(cursor, busy_end)

    if cursor < window_end:
        free.append((cursor, window_end))
    return free


def _candidates(activity, free_windows, day, tz):
    """Every grid-aligned start where the activity fits inside a free window."""
    duration = timedelta(minutes=activity.duration_minutes)
    earliest = datetime.combine(day, activity.earliest_start, tzinfo=tz)
    latest = datetime.combine(day, activity.latest_start, tzinfo=tz)
    step = timedelta(minutes=GRANULARITY_MINUTES)

    found = []
    for window_start, window_end in free_windows:
        start = max(window_start, earliest).replace(second=0, microsecond=0)

        # Snap forward onto the granularity grid.
        offset = start.minute % GRANULARITY_MINUTES
        if offset:
            start += timedelta(minutes=GRANULARITY_MINUTES - offset)

        while start <= latest and start + duration <= window_end:
            found.append(start)
            start += step
    return found


def _score(activity, start, already_scheduled, day, tz):
    preferred = datetime.combine(day, activity.preferred_start, tzinfo=tz)
    score = abs((start - preferred).total_seconds()) / 60

    # Avoid starting gym too soon after a meal (plan.md section 21).
    if activity.name == "gym":
        for scheduled in already_scheduled:
            if scheduled.name not in MEALS:
                continue
            gap = (start - scheduled.end).total_seconds() / 60
            if 0 <= gap < MEAL_GYM_GAP_MINUTES:
                score += MEAL_GYM_PENALTY

    return score


def schedule_activities(activities, events, day, tz=None):
    """Place each activity in its best free slot, in fixed order.

    Returns (scheduled, unplaced). An activity with no viable slot is reported
    rather than forced -- a missing gym is better than a double-booked one.
    """
    tz = tz or ZoneInfo(config.TIMEZONE)
    busy = list(events)
    scheduled, unplaced = [], []

    for activity in activities:
        free_windows = calculate_free_windows(busy, day, tz)
        candidates = _candidates(activity, free_windows, day, tz)
        if not candidates:
            unplaced.append(activity.name)
            continue

        # Ties break toward the earlier slot.
        best = min(
            candidates, key=lambda s: (_score(activity, s, scheduled, day, tz), s)
        )
        placed = ScheduledActivity(
            name=activity.name,
            start=best,
            end=best + timedelta(minutes=activity.duration_minutes),
        )
        scheduled.append(placed)

        # Occupy the slot so later activities cannot overlap it.
        busy.append(
            CalendarEvent(title=placed.name, start=placed.start, end=placed.end)
        )

    return scheduled, unplaced
