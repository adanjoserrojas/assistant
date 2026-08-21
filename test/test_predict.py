"""Training and prediction tests. Run from the repo root:  python -m pytest test -q

The scikit-learn comparison is skipped when sklearn is absent, so the suite still
runs offline with no ML dependencies -- which is the point of the pure-Python
implementation in the first place.
"""

import json
import math
import os
import random
import sys
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import config
from ml.candidate_generator import generate_candidates
from ml.features import DEFAULT, FeatureSpec
from ml.predict import (
    MIN_NEGATIVE_DAYS,
    MIN_TRAINING_DAYS,
    Model,
    best,
    choose,
    rank,
    score,
    usable,
)
from ml.train import (
    safe_learning_rate,
    baseline_scores,
    fit_logistic,
    predict_one,
    sigmoid,
    split_by_group,
    standardize,
    top1_accuracy,
    train,
)

TZ = ZoneInfo(config.TIMEZONE)
DAY = date(2026, 7, 25)
WORKOUT = "Chest-Triceps"


def at(hhmm, day=DAY):
    hours, minutes = hhmm.split(":")
    return datetime.combine(day, time(int(hours), int(minutes)), tzinfo=TZ)


def synthetic_rows(days=40, seed=3):
    """Days where the evening is genuinely better, so a fit should find it."""
    random.seed(seed)
    rows = []
    for index in range(days):
        day = f"2026-05-{index + 1:02d}"
        slots = [7.0, 10.0, 13.0, 17.0, 19.0, 21.0]
        # Evening slots get taken; mornings do not.
        taken = random.choice([17.0, 19.0])
        for hour in slots:
            rows.append(
                {
                    "day": day,
                    "workout": WORKOUT,
                    "weekday": "Monday",
                    "start_hour": hour,
                    "gap_after_minutes": 120,
                    "busy_minutes": 400,
                    "chosen": hour == taken,
                }
            )
    return rows


def artifact_from(rows, **overrides):
    payload = train(rows)
    payload["training"].update(overrides.pop("training", {}))
    payload["evaluation"].update(overrides.pop("evaluation", {}))
    return payload


def good_model(rows=None) -> Model:
    payload = train(rows or synthetic_rows())
    payload["training"]["n_days"] = MIN_TRAINING_DAYS + 10
    payload["training"]["n_negatives"] = 100
    payload["evaluation"]["top1_accuracy"] = 0.8
    payload["evaluation"]["baseline_top1_accuracy"] = 0.4
    return Model.from_dict(payload)


# --- numerics ------------------------------------------------------------


def test_sigmoid_midpoint():
    assert sigmoid(0.0) == 0.5


def test_sigmoid_is_monotonic():
    assert sigmoid(-2) < sigmoid(-1) < sigmoid(0) < sigmoid(1) < sigmoid(2)


def test_sigmoid_survives_extremes():
    # exp(800) overflows; a scored candidate must never raise.
    assert sigmoid(800.0) == pytest.approx(1.0)
    assert sigmoid(-800.0) == pytest.approx(0.0)


def test_standardize_centers_and_scales():
    X = [[1.0], [2.0], [3.0]]
    scaled, means, stds = standardize(X)
    assert means == [2.0]
    assert sum(row[0] for row in scaled) == pytest.approx(0.0)
    assert stds[0] == pytest.approx(math.sqrt(2 / 3))


def test_constant_column_does_not_divide_by_zero():
    scaled, _, stds = standardize([[5.0], [5.0], [5.0]])
    assert stds == [1.0]
    assert all(math.isfinite(row[0]) for row in scaled)


# --- fitting -------------------------------------------------------------


def test_fit_recovers_a_positive_relationship():
    X = [[-2.0], [-1.0], [1.0], [2.0]]
    y = [0.0, 0.0, 1.0, 1.0]
    weights, _, _ = fit_logistic(X, y, l2=0.0)
    assert weights[0] > 0


def test_fit_recovers_a_negative_relationship():
    X = [[-2.0], [-1.0], [1.0], [2.0]]
    y = [1.0, 1.0, 0.0, 0.0]
    weights, _, _ = fit_logistic(X, y, l2=0.0)
    assert weights[0] < 0


def test_regularization_shrinks_weights():
    X = [[-2.0], [-1.0], [1.0], [2.0]]
    y = [0.0, 0.0, 1.0, 1.0]
    weak, _, _ = fit_logistic(X, y, l2=0.001)
    strong, _, _ = fit_logistic(X, y, l2=10.0)
    assert abs(strong[0]) < abs(weak[0])


def test_heavy_regularization_stays_finite():
    # lr * l2 above 2 diverges without the step cap; NaN coefficients would then
    # score every slot identically and rank by list order.
    X = [[-2.0], [-1.0], [1.0], [2.0]]
    y = [0.0, 0.0, 1.0, 1.0]
    for l2 in (10.0, 100.0, 1000.0):
        weights, intercept, _ = fit_logistic(X, y, l2=l2)
        assert all(math.isfinite(w) for w in weights), l2
        assert math.isfinite(intercept), l2


def test_safe_learning_rate_leaves_defaults_alone():
    X, _, _ = standardize([[7.0, 100.0], [17.0, 300.0], [21.0, 50.0]])
    assert safe_learning_rate(X, l2=1.0, requested=0.5) == 0.5


def test_safe_learning_rate_shrinks_for_heavy_l2():
    X, _, _ = standardize([[7.0], [17.0], [21.0]])
    assert safe_learning_rate(X, l2=100.0, requested=0.5) < 0.5


def test_fit_of_nothing_is_empty():
    assert fit_logistic([], []) == ([], 0.0, False)


def test_fit_is_deterministic():
    X, y = [[-1.0], [1.0]], [0.0, 1.0]
    assert fit_logistic(X, y) == fit_logistic(X, y)


def test_matches_sklearn():
    """The pure-Python fit is a shortcut, so verify it against the real thing."""
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    random.seed(11)
    X = [[random.gauss(0, 1), random.gauss(0, 1)] for _ in range(300)]
    y = [1.0 if 1.5 * a - 0.8 * b + random.gauss(0, 0.3) > 0 else 0.0 for a, b in X]

    scaled, _, _ = standardize(X)
    l2 = 0.01
    mine, my_intercept, _ = fit_logistic(scaled, y, l2=l2, iterations=8000)

    # sklearn minimizes 0.5||w||^2 + C*sum(loss); this file minimizes
    # mean(loss) + (l2/2)||w||^2. Dividing sklearn's objective by C*n makes them
    # the same problem when C = 1 / (l2 * n).
    reference = LogisticRegression(C=1.0 / (l2 * len(scaled)), max_iter=5000)
    reference.fit(scaled, y)

    for ours, theirs in zip(mine, reference.coef_[0]):
        assert ours == pytest.approx(theirs, abs=0.02)
    assert my_intercept == pytest.approx(reference.intercept_[0], abs=0.02)


# --- grouped splitting and ranking metric --------------------------------


def test_split_keeps_days_whole():
    rows = synthetic_rows(days=20)
    groups = [row["day"] for row in rows]
    train_index, test_index = split_by_group(rows, groups)
    train_days = {groups[i] for i in train_index}
    test_days = {groups[i] for i in test_index}
    assert train_days and test_days
    assert not (train_days & test_days), "a day must not straddle the split"


def test_split_declines_when_there_are_too_few_days():
    rows = synthetic_rows(days=3)
    groups = [row["day"] for row in rows]
    _, test_index = split_by_group(rows, groups)
    assert test_index == []


def test_top1_counts_days_not_rows():
    scores = [0.9, 0.1, 0.2, 0.8]
    y = [1.0, 0.0, 0.0, 1.0]
    groups = ["a", "a", "b", "b"]
    assert top1_accuracy(scores, y, groups) == 1.0


def test_top1_misses_when_the_wrong_slot_wins():
    scores = [0.1, 0.9]
    y = [1.0, 0.0]
    assert top1_accuracy(scores, y, ["a", "a"]) == 0.0


def test_top1_skips_days_with_no_positive():
    # An all-negative day has no correct answer to rank first.
    assert top1_accuracy([0.5, 0.4], [0.0, 0.0], ["a", "a"]) is None


def test_baseline_prefers_the_configured_time():
    rows = [{"start_hour": 17.5}, {"start_hour": 7.0}]
    scores = baseline_scores(rows)
    assert scores[0] > scores[1]


# --- train ---------------------------------------------------------------


def test_train_produces_a_complete_artifact():
    artifact = train(synthetic_rows())
    for key in (
        "version", "trained_at", "spec", "feature_names",
        "standardization", "coefficients", "intercept", "training", "evaluation",
    ):
        assert key in artifact, key
    assert len(artifact["coefficients"]) == len(artifact["feature_names"])


def test_artifact_is_json_serializable():
    artifact = train(synthetic_rows())
    assert json.loads(json.dumps(artifact))["version"] == artifact["version"]


def test_train_learns_that_evenings_win():
    artifact = train(synthetic_rows())
    start_hour = artifact["feature_names"].index("start_hour")
    assert artifact["coefficients"][start_hour] > 0, "later slots should score higher"


def test_train_counts_the_classes():
    rows = synthetic_rows(days=10)
    artifact = train(rows)
    assert artifact["training"]["n_days"] == 10
    assert artifact["training"]["n_positives"] == 10
    assert artifact["training"]["n_examples"] == len(rows)


def test_train_evaluates_against_the_baseline():
    artifact = train(synthetic_rows())
    assert artifact["evaluation"]["top1_accuracy"] is not None
    assert artifact["evaluation"]["baseline_top1_accuracy"] is not None


def test_train_refuses_empty_input():
    with pytest.raises(RuntimeError, match="no training rows"):
        train([])


# --- usable --------------------------------------------------------------


def test_no_model_is_not_usable():
    ok, reason = usable(None)
    assert not ok and "no model" in reason


def test_too_few_days_is_not_usable():
    payload = train(synthetic_rows())
    payload["training"]["n_days"] = MIN_TRAINING_DAYS - 1
    ok, reason = usable(Model.from_dict(payload))
    assert not ok and "training days" in reason


def test_too_few_negatives_is_not_usable():
    payload = train(synthetic_rows())
    payload["training"]["n_days"] = MIN_TRAINING_DAYS + 5
    payload["training"]["n_negatives"] = MIN_NEGATIVE_DAYS - 1
    ok, reason = usable(Model.from_dict(payload))
    assert not ok and "negative" in reason


def test_losing_to_the_baseline_is_not_usable():
    payload = train(synthetic_rows())
    payload["training"]["n_days"] = MIN_TRAINING_DAYS + 5
    payload["training"]["n_negatives"] = 100
    payload["evaluation"]["top1_accuracy"] = 0.3
    payload["evaluation"]["baseline_top1_accuracy"] = 0.6
    ok, reason = usable(Model.from_dict(payload))
    assert not ok and "baseline" in reason


def test_an_unevaluated_model_is_not_usable():
    payload = train(synthetic_rows())
    payload["training"]["n_days"] = MIN_TRAINING_DAYS + 5
    payload["training"]["n_negatives"] = 100
    payload["evaluation"]["top1_accuracy"] = None
    ok, reason = usable(Model.from_dict(payload))
    assert not ok and "not evaluated" in reason


def test_a_good_model_is_usable():
    ok, _ = usable(good_model())
    assert ok


# --- scoring live candidates ---------------------------------------------


def test_score_returns_a_probability():
    model = good_model()
    candidate = generate_candidates(WORKOUT, 82, [], DAY, TZ, limit=1)[0]
    value = score(candidate.to_dict(), model)
    assert 0.0 <= value <= 1.0


def test_rank_is_sorted_best_first():
    model = good_model()
    candidates = generate_candidates(WORKOUT, 82, [], DAY, TZ)
    ranked = rank(candidates, model)
    probabilities = [probability for _, probability in ranked]
    assert probabilities == sorted(probabilities, reverse=True)


def test_best_picks_the_evening_when_evenings_won():
    model = good_model()
    candidates = generate_candidates(WORKOUT, 82, [], DAY, TZ)
    winner = best(candidates, model)
    earliest = min(candidate.start for candidate in candidates)
    assert winner.start > earliest


def test_best_of_nothing_is_none():
    assert best([], good_model()) is None


def test_ranking_is_deterministic():
    model = good_model()
    candidates = generate_candidates(WORKOUT, 82, [], DAY, TZ)
    first = [c.start for c, _ in rank(candidates, model)]
    second = [c.start for c, _ in rank(candidates, model)]
    assert first == second


# --- choose --------------------------------------------------------------


def test_choose_uses_the_model_when_it_is_good():
    candidates = generate_candidates(WORKOUT, 82, [], DAY, TZ)
    result = choose(candidates, good_model())
    assert result["method"] == "model"
    assert result["winner"] is not None
    assert len(result["scores"]) == len(candidates)


def test_choose_falls_back_without_a_model():
    candidates = generate_candidates(WORKOUT, 82, [], DAY, TZ)
    result = choose(candidates, None)
    assert result["method"] == "fallback"
    assert result["winner"] is None


def test_choose_falls_back_on_a_weak_model():
    payload = train(synthetic_rows())
    payload["training"]["n_days"] = 2
    candidates = generate_candidates(WORKOUT, 82, [], DAY, TZ)
    result = choose(candidates, Model.from_dict(payload))
    assert result["method"] == "fallback"
    assert "training days" in result["reason"]


def test_choose_reports_no_slots_separately_from_fallback():
    # "nowhere to put it" is a different outcome from "no model to pick with".
    result = choose([], good_model())
    assert result["method"] == "none"
    assert result["winner"] is None


# --- artifact round trip -------------------------------------------------


def test_model_survives_the_json_round_trip():
    artifact = train(synthetic_rows())
    restored = Model.from_dict(json.loads(json.dumps(artifact)))
    assert restored.coefficients == artifact["coefficients"]
    assert restored.spec == DEFAULT


def test_scoring_is_identical_before_and_after_the_round_trip():
    artifact = train(synthetic_rows())
    direct = Model.from_dict(artifact)
    restored = Model.from_dict(json.loads(json.dumps(artifact)))
    candidate = generate_candidates(WORKOUT, 82, [], DAY, TZ, limit=1)[0].to_dict()
    assert score(candidate, direct) == score(candidate, restored)


def test_a_missing_spec_does_not_explode_the_loader():
    model = Model.from_dict({})
    assert model.spec == FeatureSpec()
    ok, _ = usable(model)
    assert not ok
