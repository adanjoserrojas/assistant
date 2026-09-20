"""Last gate before any calendar write (plan.md section 22).

The scheduler is supposed to produce a valid plan; this checks it independently.
If anything here fails, agent.py writes nothing at all -- a partial day is
recoverable, a corrupted calendar is not.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import config
from scheduler import parse_hhmm


def _overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def validate_schedule(existing_events, generated_events, day=None, tz=None):
    """Return a list of problems. Empty list means safe to write."""
    tz = tz or ZoneInfo(config.TIMEZONE)
    day = day or (generated_events[0].start.date() if generated_events else None)
    problems = []

    # Each activity appears at most once.
    seen = set()
    for item in generated_events:
        if item.name in seen:
            problems.append(f"{item.name}: scheduled more than once")
        seen.add(item.name)

    for item in generated_events:
        settings = getattr(config, item.name.upper(), None)
        if settings is None:
            problems.append(f"{item.name}: no config entry")
            continue

        # Correct duration.
        minutes = (item.end - item.start).total_seconds() / 60
        if minutes != settings["duration"]:
            problems.append(
                f"{item.name}: duration {minutes:.0f}min, expected {settings['duration']}min"
            )

        # Ends after it starts.
        if item.end <= item.start:
            problems.append(f"{item.name}: ends at or before it starts")

        # Inside its configured range.
        earliest = datetime.combine(item.start.date(), parse_hhmm(settings["earliest"]), tzinfo=tz)
        latest = datetime.combine(item.start.date(), parse_hhmm(settings["latest"]), tzinfo=tz)
        if item.start < earliest:
            problems.append(f"{item.name}: starts {item.start:%H:%M}, earlier than {settings['earliest']}")
        if item.start > latest:
            problems.append(f"{item.name}: starts {item.start:%H:%M}, later than {settings['latest']}")

        # Inside the schedulable day.
        day_start = datetime.combine(item.start.date(), parse_hhmm(config.DAY_START), tzinfo=tz)
        day_end = datetime.combine(item.start.date(), parse_hhmm(config.DAY_END), tzinfo=tz)
        if item.start < day_start or item.end > day_end:
            problems.append(f"{item.name}: falls outside {config.DAY_START}-{config.DAY_END}")

    # No generated event overlaps another generated event.
    ordered = sorted(generated_events, key=lambda s: s.start)
    for earlier, later in zip(ordered, ordered[1:]):
        if _overlaps(earlier.start, earlier.end, later.start, later.end):
            problems.append(f"{earlier.name} overlaps {later.name}")

    # No generated event overlaps an existing calendar event.
    for item in generated_events:
        for existing in existing_events:
            if existing.all_day and not config.ALL_DAY_BLOCKS:
                continue
            if _overlaps(item.start, item.end, existing.start, existing.end):
                problems.append(
                    f"{item.name} ({item.start:%H:%M}-{item.end:%H:%M}) "
                    f"overlaps {existing.title!r} ({existing.start:%H:%M}-{existing.end:%H:%M})"
                )

    return problems
