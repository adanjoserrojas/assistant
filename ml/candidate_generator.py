"""Phase 2 -- candidate generation (gym_ml_cdk_plan.md, "Candidate Generation").

Every morning: read the current workout, load its mean duration, find the
calendar windows that fit it, and offer up to three start times worth scoring.

Split deliberately at the I/O boundary. load_duration_profile and resolve_workout
reach S3 and DynamoDB; generate_candidates is pure arithmetic over events handed
to it, so test_candidates.py needs no mocks and no credentials.

This module does not rank by attendance probability -- that is predict.py's job.
Its contract is a *diverse and valid* menu: options far enough apart to be real
alternatives, rather than three variations on 5:30. A 15-minute grid across an
open evening yields twenty near-identical slots, and handing those to a logistic
regression produces three scores that differ in the third decimal place.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import boto3

import config
from models import Activity
from scheduler import calculate_free_windows, candidate_starts, parse_hhmm

from .repository import fetch_state

DURATION_PROFILE_KEY = "gym/duration_profiles.json"
REST_WORKOUT = "Rest-days"

# Two candidates closer together than this are the same decision wearing
# different hats. Wide enough to force genuinely different options, narrow
# enough that a half-booked day still yields more than one.
MIN_SPACING_MINUTES = 60

_s3_client = None


def s3_client():
    """The S3 client, built on first use -- see ml.repository.client()."""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name="us-east-1")
    return _s3_client


@dataclass(frozen=True)
class Candidate:
    """One proposed gym slot, with the context features.py will need.

    gap_before/gap_after measure the free time on either side inside the window
    this candidate sits in -- "how rushed is this slot", which is exactly the
    plan's Gap before / Gap after features and is free to compute here while the
    windows are still in hand.
    """

    workout: str
    start: datetime
    end: datetime
    duration_minutes: int
    gap_before_minutes: int
    gap_after_minutes: int

    def to_dict(self) -> dict:
        return {
            "workout": self.workout,
            "start": self.start.isoformat(timespec="seconds"),
            "end": self.end.isoformat(timespec="seconds"),
            "start_time": self.start.strftime("%H:%M:%S"),
            "weekday": self.start.strftime("%A"),
            "duration_minutes": self.duration_minutes,
            "gap_before_minutes": self.gap_before_minutes,
            "gap_after_minutes": self.gap_after_minutes,
        }


def load_duration_profile() -> dict[str, float]:
    """Read gym/duration_profiles.json, the artifact duration_profile.py writes."""
    if not config.BUCKET_NAME:
        raise RuntimeError("BUCKET_NAME is not set; run cdk deploy and export it")

    body = s3_client().get_object(
        Bucket=config.BUCKET_NAME, Key=DURATION_PROFILE_KEY
    )["Body"].read()
    return json.loads(body)


def duration_for(workout: str, profile: dict[str, float]) -> int:
    """Minutes to reserve for a workout, learned if known and config if not.

    A workout you have never completed has no mean to fall back on, and refusing
    to schedule it would strand a new rotation entry forever. config.GYM sizes it
    until real sessions exist. Rounded up: a window that fits the mean exactly
    fits half your actual sessions.
    """
    learned = profile.get(workout)
    if learned is None or learned <= 0:
        return int(config.GYM["duration"])
    return int(-(-float(learned) // 1))  # ceil without importing math


def resolve_workout() -> tuple[str, int]:
    """Today's rotation entry as (workout, cycle_index).

    Returns the rest entry rather than None so the caller keeps the index -- a
    rest day still needs to be recorded against its position in the cycle. Check
    the result against REST_WORKOUT.

    The rotation is state-driven, not date-driven: next_workout_index only moves
    when a session completes or the validator closes out a rest day, so two
    missed days in a row leave the same workout up. There is deliberately no
    `day` argument.
    """
    state = fetch_state()
    cycle_index = int(state.get("next_workout_index", 0)) % len(config.WORKOUTS)
    return config.WORKOUTS[cycle_index], cycle_index


def _gym_activity(duration_minutes: int) -> Activity:
    """The gym as an Activity, sized by the learned profile instead of config.

    config.GYM still supplies earliest/latest/preferred -- those are your rules
    about when you are willing to train, which no amount of data should override.
    Only the duration is learned.
    """
    return Activity(
        name="gym",
        duration_minutes=duration_minutes,
        earliest_start=parse_hhmm(config.GYM["earliest"]),
        latest_start=parse_hhmm(config.GYM["latest"]),
        preferred_start=parse_hhmm(config.GYM["preferred"]),
    )


def _window_for(start: datetime, free_windows: list[tuple[datetime, datetime]]):
    for window_start, window_end in free_windows:
        if window_start <= start and start < window_end:
            return window_start, window_end
    return None


def _spread(starts: list[datetime], limit: int, min_gap: timedelta) -> list[datetime]:
    """Thin a dense grid down to `limit` options that are actually different.

    Two passes. First a greedy sweep that drops anything within min_gap of the
    last kept slot, which is what removes the 5:30 / 5:45 / 6:00 cluster. Then,
    if more survive than we want, take them evenly across the range rather than
    the first few -- otherwise every candidate lands in the morning and the model
    never sees an evening option to compare against.
    """
    if not starts:
        return []

    spaced = [starts[0]]
    for start in starts[1:]:
        if start - spaced[-1] >= min_gap:
            spaced.append(start)

    if len(spaced) <= limit:
        return spaced
    if limit == 1:
        return [spaced[0]]

    step = (len(spaced) - 1) / (limit - 1)
    return [spaced[round(index * step)] for index in range(limit)]


def generate_candidates(
    workout: str,
    duration_minutes: int,
    events: list,
    day,
    tz=None,
    limit: int = 3,
) -> list[Candidate]:
    """Up to `limit` valid, well-separated slots for today's workout.

    Pure: events come in, candidates go out. Returns [] when the day genuinely
    has no room -- an empty list is a real answer, and the caller should skip the
    day rather than force a session into a busy block.
    """
    if workout == REST_WORKOUT:
        return []

    tz = tz or ZoneInfo(config.TIMEZONE)
    duration = timedelta(minutes=duration_minutes)

    free_windows = calculate_free_windows(events, day, tz)
    starts = candidate_starts(_gym_activity(duration_minutes), free_windows, day, tz)

    candidates = []
    for start in _spread(sorted(starts), limit, timedelta(minutes=MIN_SPACING_MINUTES)):
        window = _window_for(start, free_windows)
        if window is None:
            continue
        window_start, window_end = window
        end = start + duration
        candidates.append(
            Candidate(
                workout=workout,
                start=start,
                end=end,
                duration_minutes=duration_minutes,
                gap_before_minutes=int((start - window_start).total_seconds() // 60),
                gap_after_minutes=int((window_end - end).total_seconds() // 60),
            )
        )
    return candidates
