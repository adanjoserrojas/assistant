"""Where the gym goes, once the meals are placed.

Two paths, decided fresh every morning:

  model     an artifact exists, was trained on enough history, and beat the
            "nearest to config.GYM['preferred']" heuristic on held-out days
  fallback  anything else -- no artifact, too little data, or a model that
            lost to the heuristic. The caller places gym with
            scheduler.schedule_activities, exactly as it always has.

The check runs on every invocation rather than latching once. Re-checking costs
nothing (the numbers are already in the artifact being read), and it means a
retrain that produces a worse model self-corrects the next morning instead of
degrading the schedule until someone notices.

Imports nothing from calendar_client or llm_client, so the whole decision is
testable without Google credentials or a Bedrock call.
"""

import logging

import config
from models import CalendarEvent, ScheduledActivity
from scheduler import MEAL_GYM_GAP_MINUTES, MEALS

log = logging.getLogger("gym_allocator")


def meals_as_events(scheduled: list[ScheduledActivity]) -> list[CalendarEvent]:
    """Placed meals, as busy time for the gym search.

    This is what makes the ordering matter: gym candidates are generated
    against a calendar that already contains today's breakfast, lunch and
    dinner, so a slot is only offered if it survives the full day.
    """
    return [
        CalendarEvent(title=item.name, start=item.start, end=item.end)
        for item in scheduled
    ]


def clear_of_meals(
    candidates: list,
    scheduled: list[ScheduledActivity],
    gap_minutes: int = MEAL_GYM_GAP_MINUTES,
) -> list:
    """Drop candidates starting too soon after a meal.

    scheduler._score() expresses this as a large penalty, which a high enough
    probability could out-vote. Here it is a filter instead: "do not lift right
    after eating" is a constraint, and the model does not get a vote on it. The
    candidate is removed before scoring rather than scored and discouraged.

    Only the window *after* a meal matters -- eating after the gym is fine.
    """
    kept = []
    for candidate in candidates:
        conflict = False
        for meal in scheduled:
            if meal.name not in MEALS:
                continue
            gap = (candidate.start - meal.end).total_seconds() / 60
            if 0 <= gap < gap_minutes:
                conflict = True
                break
        if not conflict:
            kept.append(candidate)
    return kept


def as_scheduled(candidate) -> ScheduledActivity:
    return ScheduledActivity(name="gym", start=candidate.start, end=candidate.end)


def allocate(
    workout: str,
    duration_minutes: int,
    events: list,
    scheduled_meals: list[ScheduledActivity],
    day,
    tz,
    model=None,
) -> dict:
    """Choose today's gym slot.

    Returns a dict with `winner` (a ScheduledActivity or None), `method`, and a
    human-readable `reason`. A None winner is not an error -- it means the
    caller should place gym the deterministic way.
    """
    from ml.candidate_generator import REST_WORKOUT, generate_candidates
    from ml.predict import choose

    if workout == REST_WORKOUT:
        return {"winner": None, "method": "rest", "reason": "rotation entry is a rest day"}

    busy = list(events) + meals_as_events(scheduled_meals)
    candidates = generate_candidates(workout, duration_minutes, busy, day, tz)
    if not candidates:
        return {
            "winner": None,
            "method": "none",
            "reason": "no slot fits the workout around today's calendar",
        }

    allowed = clear_of_meals(candidates, scheduled_meals)
    if not allowed:
        return {
            "winner": None,
            "method": "none",
            "reason": f"every slot starts within {MEAL_GYM_GAP_MINUTES} minutes of a meal",
        }

    decision = choose(allowed, model)
    if decision["winner"] is None:
        return {
            "winner": None,
            "method": decision["method"],
            "reason": decision["reason"],
        }

    return {
        "winner": as_scheduled(decision["winner"]),
        "method": "model",
        "reason": decision["reason"],
        "scores": decision.get("scores", []),
        "candidates_considered": len(allowed),
        "candidates_dropped_for_meals": len(candidates) - len(allowed),
    }


def allocate_safely(events, scheduled_meals, day, tz) -> dict:
    """allocate(), with every piece of I/O and every failure mode absorbed.

    The gym model shares a Lambda with meal scheduling now. A missing artifact,
    an S3 timeout, a rotation state that will not read, a spec that no longer
    matches the row shape -- none of those are reasons for breakfast to go
    unscheduled. Anything unexpected logs and returns a fallback verdict.
    """
    try:
        from ml.candidate_generator import (
            duration_for,
            load_duration_profile,
            resolve_workout,
        )
        from ml.predict import load_model

        workout, _ = resolve_workout()
        duration = duration_for(workout, load_duration_profile())
        return allocate(
            workout, duration, events, scheduled_meals, day, tz, load_model()
        )
    except Exception as error:
        log.warning(
            "gym model path unavailable (%s: %s); using preferences",
            type(error).__name__,
            error,
        )
        return {
            "winner": None,
            "method": "fallback",
            "reason": f"{type(error).__name__}: {error}",
        }
