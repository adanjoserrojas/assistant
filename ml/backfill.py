"""Reconstruct slot-level training examples from session history and calendar.

The attendance model is a *ranker*, not a day-level classifier. At prediction
time the question is "which of these three slots is best", so the training rows
have to be slots, not days:

    attended day    the slot you actually used is chosen=True; the other viable
                    slots that day are chosen=False -- you had them and passed
    unattended day  every viable slot is chosen=False -- you had them all and
                    went to none

This is what makes the data workable. A day-level label gives one row per day and
~10 negatives across a whole season, which supports roughly one predictor. Slot
level gives one row per option, and the comparison the model has to learn --
5:30 beat 8:00 *on that day, given that calendar* -- is exactly the comparison it
will be asked to make.

Two things it deliberately does NOT do:

  Duration always comes from the profile, never from actual_duration_minutes.
  The real duration is only knowable after the session; using it for positives
  and the profile for negatives would let the model separate the classes on a
  field that does not exist at prediction time.

  Rest and injury days never appear. fetch_sessions filters them out via
  training_eligible, and they do not belong here regardless: a rest day is not a
  scheduling decision, and no proposed time would have prevented an injury.

Examples carry `day` so a train/test split can group by it. Slots from one day
share a calendar and are not independent observations -- splitting them across
the boundary leaks, and the reported accuracy comes out flattering and wrong.
"""

from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import config
from scheduler import calculate_free_windows, candidate_starts

from .candidate_generator import (
    MIN_SPACING_MINUTES,
    REST_WORKOUT,
    busy_minutes,
    duration_for,
    gym_activity,
    spread_starts,
    window_for,
)

# How far a logged check-in may sit from a grid slot and still be called the
# same decision. The grid is 15 minutes, so anything beyond this means you
# trained during a block the calendar called busy -- see build_examples.
SNAP_TOLERANCE_MINUTES = 45

# Enumerate generously for training; serving takes the top 3 of the same grid.
MAX_SLOTS_PER_DAY = 12


@dataclass(frozen=True)
class TrainingExample:
    day: str
    workout: str
    weekday: str
    start_time: str
    start_hour: float
    duration_minutes: int
    gap_before_minutes: int
    gap_after_minutes: int
    busy_minutes: int
    chosen: bool

    def to_row(self) -> dict:
        return asdict(self)


def examples_for_day(
    workout: str,
    duration_minutes: int,
    events: list,
    day: date,
    attended_start: datetime | None = None,
    tz=None,
) -> list[TrainingExample]:
    """Every viable slot on one day, labelled by whether you took it.

    Returns [] for a rest day, a day with no room, or an attended day whose
    check-in matches no viable slot -- see SNAP_TOLERANCE_MINUTES. An attended
    day that yields no positive must yield no negatives either, or the day is
    recorded as a total refusal when in fact you went.
    """
    if workout == REST_WORKOUT:
        return []

    tz = tz or ZoneInfo(config.TIMEZONE)
    free_windows = calculate_free_windows(events, day, tz)

    # The full 15-minute grid, not the spread one serving uses. Snapping a 17:30
    # check-in onto 60-minute spacing would record it as 17:00 and throw away
    # resolution in the feature the model exists to learn -- and the spread
    # grid's phase moves with the free window, so the same habit lands on-grid
    # one day and off-grid the next.
    grid = sorted(candidate_starts(gym_activity(duration_minutes), free_windows, day, tz))
    if not grid:
        return []

    chosen_start = None
    if attended_start is not None:
        nearest = min(grid, key=lambda slot: abs(slot - attended_start))
        if abs(nearest - attended_start) > timedelta(minutes=SNAP_TOLERANCE_MINUTES):
            # You trained at a time the calendar said was unavailable. There is
            # no honest slot to attach the positive to, so drop the whole day.
            return []
        chosen_start = nearest

    # Thin the negatives so one open Saturday does not contribute twenty rows
    # that are all the same decision, then put the chosen slot back if the
    # thinning happened to drop it -- the positive is never optional.
    starts = spread_starts(
        grid, MAX_SLOTS_PER_DAY, timedelta(minutes=MIN_SPACING_MINUTES)
    )
    if chosen_start is not None and chosen_start not in starts:
        starts = sorted(set(starts) | {chosen_start})

    occupied = busy_minutes(events, day, tz)
    duration = timedelta(minutes=duration_minutes)

    examples = []
    for start in starts:
        window = window_for(start, free_windows)
        if window is None:
            continue
        window_start, window_end = window
        examples.append(
            TrainingExample(
                day=day.isoformat(),
                workout=workout,
                weekday=start.strftime("%A"),
                start_time=start.strftime("%H:%M:%S"),
                start_hour=start.hour + start.minute / 60,
                duration_minutes=duration_minutes,
                gap_before_minutes=int((start - window_start).total_seconds() // 60),
                gap_after_minutes=int(
                    (window_end - (start + duration)).total_seconds() // 60
                ),
                busy_minutes=occupied,
                chosen=start == chosen_start,
            )
        )
    return examples


def session_day(item: dict) -> date | None:
    """Local calendar day of a session record, however it was written."""
    if item.get("session_date"):
        try:
            return date.fromisoformat(str(item["session_date"]))
        except ValueError:
            return None
    checkin = item.get("checkin_at")
    if checkin:
        try:
            return datetime.fromisoformat(str(checkin)).date()
        except ValueError:
            return None
    return None


def session_start(item: dict, tz) -> datetime | None:
    """The check-in as a timezone-aware datetime, or None if you never went."""
    checkin = item.get("checkin_at")
    if not checkin:
        return None
    try:
        moment = datetime.fromisoformat(str(checkin))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=tz)


def build_examples(
    sessions: list[dict],
    events_for_day,
    profile: dict[str, float],
    tz=None,
) -> tuple[list[TrainingExample], dict]:
    """Turn session records into slot-level examples, one calendar read per day.

    events_for_day is a callable taking a date and returning that day's events --
    calendar_client.get_today_events in production, a dict lookup in tests. The
    seam is what keeps this function testable without network.

    Returns (examples, diagnostics). Read the diagnostics before trusting a
    model fitted on the result: `days_dropped_no_room` and `days_dropped_no_slot`
    are days silently missing from the training set, and a large count means the
    calendar history disagrees with when you actually train.
    """
    tz = tz or ZoneInfo(config.TIMEZONE)
    diagnostics = {
        "sessions_seen": len(sessions),
        "days_used": 0,
        "days_dropped_unreadable": 0,
        "days_dropped_no_room": 0,
        "days_dropped_no_slot": 0,
        "examples": 0,
        "positives": 0,
        "negatives": 0,
    }

    examples: list[TrainingExample] = []
    for item in sessions:
        day = session_day(item)
        workout = item.get("workout")
        if day is None or not workout:
            diagnostics["days_dropped_unreadable"] += 1
            continue

        attended_start = session_start(item, tz)
        events = events_for_day(day)
        duration = duration_for(str(workout), profile)

        found = examples_for_day(
            str(workout), duration, events, day, attended_start, tz
        )
        if not found:
            key = (
                "days_dropped_no_slot"
                if attended_start is not None
                else "days_dropped_no_room"
            )
            diagnostics[key] += 1
            continue

        diagnostics["days_used"] += 1
        examples.extend(found)

    diagnostics["examples"] = len(examples)
    diagnostics["positives"] = sum(1 for example in examples if example.chosen)
    diagnostics["negatives"] = len(examples) - diagnostics["positives"]
    return examples, diagnostics
