"""Gym allocation tests. Run from the repo root:  python -m pytest test -q

gym_allocator imports no calendar or LLM client, so the whole meals-then-gym
decision is exercised here without credentials.
"""

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import config
import gym_allocator
from models import CalendarEvent, ScheduledActivity
from scheduler import MEAL_GYM_GAP_MINUTES
from ml.candidate_generator import generate_candidates
from ml.predict import Model
from ml.train import train

TZ = ZoneInfo(config.TIMEZONE)
DAY = date(2026, 7, 27)          # Monday
WORKOUT = "Chest-Triceps"


def at(hhmm, day=DAY):
    hours, minutes = hhmm.split(":")
    return datetime.combine(day, time(int(hours), int(minutes)), tzinfo=TZ)


def meal(name, start, end):
    return ScheduledActivity(name=name, start=at(start), end=at(end))


def event(title, start, end):
    return CalendarEvent(title=title, start=at(start), end=at(end))


def synthetic_rows(days=40):
    import random

    random.seed(3)
    rows = []
    for index in range(days):
        taken = random.choice([17.0, 18.0])
        for hour in (7.0, 10.0, 13.0, 17.0, 18.0, 21.0):
            rows.append({
                "day": f"2026-05-{index + 1:02d}", "workout": WORKOUT,
                "weekday": "Monday", "start_hour": hour,
                "gap_after_minutes": 120, "busy_minutes": 400,
                "chosen": hour == taken,
            })
    return rows


def good_model() -> Model:
    payload = train(synthetic_rows())
    payload["training"]["n_days"] = 40
    payload["training"]["n_negatives"] = 150
    payload["evaluation"]["top1_accuracy"] = 0.9
    payload["evaluation"]["baseline_top1_accuracy"] = 0.4
    return Model.from_dict(payload)


# --- meals become busy time ----------------------------------------------


def test_meals_convert_to_events():
    events = gym_allocator.meals_as_events([meal("lunch", "12:00", "12:45")])
    assert len(events) == 1
    assert events[0].start == at("12:00") and events[0].end == at("12:45")


def test_gym_never_overlaps_a_placed_meal():
    meals = [meal("dinner", "19:00", "19:45")]
    result = gym_allocator.allocate(
        WORKOUT, 82, [], meals, DAY, TZ, good_model()
    )
    winner = result["winner"]
    assert winner is not None
    assert not (winner.start < meals[0].end and meals[0].start < winner.end)


# --- the meal gap is a constraint, not a preference ----------------------


def test_slots_right_after_a_meal_are_removed():
    meals = [meal("dinner", "18:00", "18:45")]
    candidates = generate_candidates(WORKOUT, 82, [], DAY, TZ, limit=12)
    kept = gym_allocator.clear_of_meals(candidates, meals)
    for candidate in kept:
        gap = (candidate.start - meals[0].end).total_seconds() / 60
        assert not (0 <= gap < MEAL_GYM_GAP_MINUTES), candidate.start


def test_a_slot_before_a_meal_is_fine():
    # Eating after the gym is allowed; only the window after a meal is blocked.
    meals = [meal("dinner", "19:00", "19:45")]
    candidates = generate_candidates(WORKOUT, 82, [], DAY, TZ, limit=12)
    kept = gym_allocator.clear_of_meals(candidates, meals)
    assert any(candidate.end <= meals[0].start for candidate in kept)


def test_non_meal_activities_do_not_block():
    kept = gym_allocator.clear_of_meals(
        generate_candidates(WORKOUT, 82, [], DAY, TZ, limit=12),
        [ScheduledActivity(name="gym", start=at("12:00"), end=at("13:00"))],
    )
    assert len(kept) == len(generate_candidates(WORKOUT, 82, [], DAY, TZ, limit=12))


def test_the_model_cannot_outvote_the_meal_gap():
    """A high probability must not buy a post-dinner slot."""
    meals = [meal("dinner", "17:00", "17:45")]
    result = gym_allocator.allocate(WORKOUT, 82, [], meals, DAY, TZ, good_model())
    if result["winner"] is not None:
        gap = (result["winner"].start - meals[0].end).total_seconds() / 60
        assert not (0 <= gap < MEAL_GYM_GAP_MINUTES)


def test_a_day_of_nothing_but_post_meal_slots_reports_none():
    events = [event("Work", config.DAY_START, "18:00")]
    meals = [meal("dinner", "18:00", "18:45")]
    result = gym_allocator.allocate(WORKOUT, 300, events, meals, DAY, TZ, good_model())
    assert result["winner"] is None
    assert result["method"] == "none"


# --- the two paths -------------------------------------------------------


def test_a_good_model_wins():
    result = gym_allocator.allocate(WORKOUT, 82, [], [], DAY, TZ, good_model())
    assert result["method"] == "model"
    assert result["winner"] is not None
    assert result["winner"].name == "gym"


def test_no_model_means_fallback():
    result = gym_allocator.allocate(WORKOUT, 82, [], [], DAY, TZ, None)
    assert result["method"] == "fallback"
    assert result["winner"] is None


def test_a_weak_model_means_fallback():
    payload = train(synthetic_rows())
    payload["training"]["n_days"] = 3
    result = gym_allocator.allocate(
        WORKOUT, 82, [], [], DAY, TZ, Model.from_dict(payload)
    )
    assert result["method"] == "fallback"


def test_rest_day_is_its_own_answer():
    result = gym_allocator.allocate("Rest-days", 82, [], [], DAY, TZ, good_model())
    assert result["method"] == "rest"
    assert result["winner"] is None


def test_a_full_day_reports_no_room():
    events = [event("Booked", config.DAY_START, config.DAY_END)]
    result = gym_allocator.allocate(WORKOUT, 82, events, [], DAY, TZ, good_model())
    assert result["method"] == "none"


def test_dropped_candidates_are_reported():
    meals = [meal("lunch", "12:00", "12:45")]
    result = gym_allocator.allocate(WORKOUT, 82, [], meals, DAY, TZ, good_model())
    assert result["candidates_dropped_for_meals"] >= 0
    assert result["candidates_considered"] >= 1


# --- allocate_safely absorbs everything ----------------------------------


def test_allocate_safely_never_raises(monkeypatch):
    """Meal scheduling must not go down with the gym model."""
    import ml.candidate_generator as generator

    def explode():
        raise RuntimeError("DynamoDB is having a day")

    monkeypatch.setattr(generator, "resolve_workout", explode)
    result = gym_allocator.allocate_safely([], [], DAY, TZ)
    assert result["winner"] is None
    assert result["method"] == "fallback"
    assert "DynamoDB is having a day" in result["reason"]


def test_allocate_safely_survives_a_missing_bucket(monkeypatch):
    import ml.candidate_generator as generator

    monkeypatch.setattr(generator, "resolve_workout", lambda: (WORKOUT, 0))
    monkeypatch.setattr(config, "BUCKET_NAME", "")
    result = gym_allocator.allocate_safely([], [], DAY, TZ)
    # No bucket means no profile and no model: config duration, fallback path.
    assert result["winner"] is None
    assert result["method"] == "fallback"


# --- the fallback path keeps the meal gap too -----------------------------


def test_fallback_still_honours_the_meal_gap():
    """Splitting meals and gym into two calls must not lose MEAL_GYM_PENALTY.

    _score() reads the gap off already_scheduled. When gym goes through its own
    call that list starts empty, so without seeding it the penalty never fires
    and gym lands the minute dinner ends.
    """
    import scheduler

    dinner = meal("dinner", "17:00", "17:45")
    gym = [a for a in scheduler.activities_from_config() if a.name == "gym"]
    busy = [event("Work", config.DAY_START, "17:00")] + gym_allocator.meals_as_events([dinner])

    seeded, _ = scheduler.schedule_activities(
        gym, busy, DAY, TZ, already_scheduled=[dinner]
    )
    gap = (seeded[0].start - dinner.end).total_seconds() / 60
    assert not (0 <= gap < MEAL_GYM_GAP_MINUTES), f"gym started {gap:.0f} min after dinner"


def test_seeded_activities_are_not_returned_twice():
    import scheduler

    dinner = meal("dinner", "12:00", "12:45")
    gym = [a for a in scheduler.activities_from_config() if a.name == "gym"]
    placed, _ = scheduler.schedule_activities(gym, [], DAY, TZ, already_scheduled=[dinner])
    assert [item.name for item in placed] == ["gym"]
