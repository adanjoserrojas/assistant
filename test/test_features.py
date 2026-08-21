"""Feature vector tests. Run from the repo root:  python -m pytest test -q

The point of most of these is train/predict parity: a row from backfill and a
live candidate must produce the same vector under the same spec.
"""

import json
import os
import sys
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import config
from models import CalendarEvent
from ml.backfill import examples_for_day
from ml.candidate_generator import generate_candidates
from ml.features import (
    DEFAULT,
    WEEKDAYS,
    FeatureSpec,
    design_matrix,
    groups,
    infer_categories,
    labels,
    vectorize,
    vectorize_all,
)

TZ = ZoneInfo(config.TIMEZONE)
DAY = date(2026, 7, 25)          # Saturday
WEEKDAY_DAY = date(2026, 7, 27)  # Monday
WORKOUT = "Chest-Triceps"

ROW = {
    "day": "2026-07-25",
    "workout": WORKOUT,
    "weekday": "Saturday",
    "start_hour": 17.5,
    "gap_after_minutes": 120,
    "gap_before_minutes": 30,
    "busy_minutes": 480,
    "chosen": True,
}


def at(hhmm, day=DAY):
    hours, minutes = hhmm.split(":")
    return datetime.combine(day, time(int(hours), int(minutes)), tzinfo=TZ)


# --- FeatureSpec ---------------------------------------------------------


def test_default_spec_is_four_features():
    assert len(DEFAULT.feature_names) == 4


def test_default_spec_can_express_a_peak_time_of_day():
    # Linear start_hour alone can only say "later is better"; the squared term
    # is what lets the fit put an optimum in the middle of the day.
    assert "start_hour" in DEFAULT.feature_names
    assert "start_hour_sq" in DEFAULT.feature_names


def test_feature_names_are_ordered_numeric_derived_categorical():
    spec = FeatureSpec(
        numeric=("start_hour",),
        derived=("is_weekend",),
        categorical={"workout": ("A", "B")},
    )
    assert spec.feature_names == ["start_hour", "is_weekend", "workout=A", "workout=B"]


def test_categorical_columns_are_sorted_by_name():
    # Dict insertion order must not decide column order.
    spec = FeatureSpec(categorical={"workout": ("A",), "weekday": ("Monday",)})
    assert spec.feature_names == ["weekday=Monday", "workout=A"]


def test_unknown_derived_feature_is_rejected():
    with pytest.raises(ValueError, match="unknown derived"):
        FeatureSpec(derived=("bogus_feature",))


def test_spec_round_trips_through_json():
    spec = FeatureSpec(
        numeric=("start_hour", "gap_after_minutes"),
        derived=("is_weekend",),
        categorical={"weekday": WEEKDAYS},
    )
    restored = FeatureSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored == spec
    assert restored.feature_names == spec.feature_names


# --- vectorize -----------------------------------------------------------


def test_vector_length_always_matches_feature_names():
    spec = FeatureSpec(
        numeric=("start_hour",), categorical={"weekday": WEEKDAYS}
    )
    assert len(vectorize(ROW, spec)) == len(spec.feature_names)


def test_numeric_values_pass_through():
    spec = FeatureSpec(numeric=("start_hour", "gap_after_minutes"))
    assert vectorize(ROW, spec) == [17.5, 120.0]


def test_is_weekend_reads_the_weekday():
    spec = FeatureSpec(derived=("is_weekend",))
    assert vectorize(ROW, spec) == [1.0]
    assert vectorize({**ROW, "weekday": "Monday"}, spec) == [0.0]


def test_one_hot_marks_exactly_one_column():
    spec = FeatureSpec(categorical={"weekday": WEEKDAYS})
    vector = vectorize(ROW, spec)
    assert sum(vector) == 1.0
    assert vector[WEEKDAYS.index("Saturday")] == 1.0


def test_unseen_category_is_all_zeros_not_an_error():
    # A workout the model was never fitted on gets no opinion, rather than
    # crashing the morning run or silently stealing another column.
    spec = FeatureSpec(categorical={"workout": ("Back-Biceps", "Sharms")})
    assert vectorize(ROW, spec) == [0.0, 0.0]


def test_missing_numeric_feature_raises():
    spec = FeatureSpec(numeric=("nonexistent_feature",))
    with pytest.raises(KeyError, match="nonexistent_feature"):
        vectorize(ROW, spec)


def test_missing_weekday_does_not_crash_derived():
    spec = FeatureSpec(derived=("is_weekend",))
    assert vectorize({"start_hour": 1}, spec) == [0.0]


def test_vectorize_is_deterministic():
    spec = FeatureSpec(
        numeric=("start_hour",), derived=("is_weekend",), categorical={"weekday": WEEKDAYS}
    )
    assert vectorize(ROW, spec) == vectorize(ROW, spec)


# --- train/predict parity ------------------------------------------------


def test_training_row_and_live_candidate_vectorize_identically():
    """The whole point of the module: same slot, same numbers, both directions."""
    events = [CalendarEvent(title="Work", start=at("09:00"), end=at("17:00"))]

    training = examples_for_day(WORKOUT, 82, events, DAY, at("17:00"), TZ)
    chosen = [row.to_row() for row in training if row.chosen][0]

    live = generate_candidates(WORKOUT, 82, events, DAY, TZ, limit=12)
    match = [
        candidate.to_dict()
        for candidate in live
        if candidate.to_dict()["start_time"] == chosen["start_time"]
    ]
    assert match, "the 17:30 slot should exist in both directions"

    spec = FeatureSpec(
        numeric=("start_hour", "gap_after_minutes", "gap_before_minutes", "busy_minutes"),
        derived=("is_weekend",),
        categorical={"weekday": WEEKDAYS, "workout": (WORKOUT,)},
    )
    assert vectorize(chosen, spec) == vectorize(match[0], spec)


def test_training_keeps_offgrid_checkins_that_serving_would_not_offer():
    """A deliberate asymmetry, pinned so it does not read as a bug later.

    Serving offers the 60-minute spread grid; training keeps the clock time you
    actually walked in at. start_hour is continuous, so a model fitted on 17.5
    scores 17.0 perfectly well -- and rounding the real check-in onto the coarse
    grid would throw away resolution in the feature that matters most.
    """
    training = examples_for_day(WORKOUT, 82, [], DAY, at("17:30"), TZ)
    chosen = [row for row in training if row.chosen][0]
    assert chosen.start_time == "17:30:00"

    live = generate_candidates(WORKOUT, 82, [], DAY, TZ, limit=12)
    assert "17:30:00" not in {c.to_dict()["start_time"] for c in live}


def test_both_row_shapes_expose_the_same_feature_keys():
    events = [CalendarEvent(title="Work", start=at("09:00"), end=at("17:00"))]
    training = set(examples_for_day(WORKOUT, 82, events, DAY, None, TZ)[0].to_row())
    live = set(generate_candidates(WORKOUT, 82, events, DAY, TZ, limit=1)[0].to_dict())

    features = {
        "workout", "weekday", "start_time", "start_hour",
        "duration_minutes", "gap_before_minutes", "gap_after_minutes", "busy_minutes",
    }
    assert features <= training
    assert features <= live


def test_default_spec_works_on_both_row_shapes():
    events = []
    training = examples_for_day(WORKOUT, 82, events, DAY, at("17:30"), TZ)[0].to_row()
    live = generate_candidates(WORKOUT, 82, events, DAY, TZ, limit=1)[0].to_dict()
    assert len(vectorize(training, DEFAULT)) == len(vectorize(live, DEFAULT)) == 4


# --- labels / groups / design_matrix -------------------------------------


def test_labels_read_chosen():
    assert labels([{"chosen": True}, {"chosen": False}, {}]) == [1.0, 0.0, 0.0]


def test_groups_read_day():
    assert groups([{"day": "2026-07-25"}, {"day": "2026-07-26"}]) == [
        "2026-07-25",
        "2026-07-26",
    ]


def test_design_matrix_shapes_line_up():
    rows = [ROW, {**ROW, "chosen": False, "start_hour": 8.0}]
    X, y, g = design_matrix(rows, DEFAULT)
    assert len(X) == len(y) == len(g) == 2
    assert all(len(vector) == len(DEFAULT.feature_names) for vector in X)
    assert y == [1.0, 0.0]


def test_design_matrix_groups_slots_of_one_day_together():
    rows = [row.to_row() for row in examples_for_day(WORKOUT, 82, [], DAY, at("17:30"), TZ)]
    _, y, g = design_matrix(rows, DEFAULT)
    assert len(set(g)) == 1, "one day is one group"
    assert sum(y) == 1.0, "exactly one slot was taken"


# --- infer_categories ----------------------------------------------------


def test_infer_categories_is_sorted_and_distinct():
    rows = [{"workout": "Sharms"}, {"workout": "Back-Biceps"}, {"workout": "Sharms"}]
    assert infer_categories(rows, "workout") == ("Back-Biceps", "Sharms")


def test_infer_categories_ignores_missing():
    assert infer_categories([{"workout": None}, {}], "workout") == ()


def test_infer_categories_is_row_order_independent():
    rows = [{"workout": "B"}, {"workout": "A"}]
    assert infer_categories(rows, "workout") == infer_categories(rows[::-1], "workout")


def test_vectorize_all_maps_every_row():
    rows = [ROW, {**ROW, "start_hour": 8.0}]
    assert len(vectorize_all(rows, DEFAULT)) == 2
