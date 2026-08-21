"""Fit the attendance ranker and write model_metadata.json.

No scikit-learn, and none needed. This is L2-regularized logistic regression by
full-batch gradient descent over a few hundred rows and three features -- the fit
takes under a second in pure Python, and it keeps numpy and scipy (~200 MB
unzipped, against a 50 MB console upload limit) out of the training Lambda
entirely. test_train.py checks the coefficients against scikit-learn's, so the
shortcut is verified rather than assumed.

Retraining is always a full refit from DynamoDB. The previous model is never an
input, which is why no .joblib needs to survive between runs -- the session
history is the thing that has to persist, and the command handler and validator
are already accumulating it.

Two things the artifact carries besides coefficients, both of which silently
corrupt predictions if they go missing:

  The FeatureSpec, so predict.py builds columns in the same order they were
  fitted in.

  The standardization means and standard deviations. Features here span very
  different scales -- start_hour runs 7-22, gap_after_minutes runs 0-900 -- and
  gradient descent on raw values crawls. Predict must apply the same shift and
  scale, or every coefficient is being multiplied by the wrong magnitude.
"""

import json
import math
from datetime import datetime, timezone

import config
from .features import DEFAULT, FeatureSpec, design_matrix

ARTIFACT_KEY = "gym/model_metadata.json"
ARTIFACT_VERSION = 1

LEARNING_RATE = 0.5
MAX_ITERATIONS = 3000
TOLERANCE = 1e-7
L2 = 1.0

# Held-out fraction, by day rather than by row.
HOLDOUT_FRACTION = 0.25


def sigmoid(z: float) -> float:
    # Split on sign so neither branch ever calls exp() on a large positive
    # number; exp(800) overflows, and a scored candidate must never raise.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def standardize(X: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    """Center and scale each column, returning the transform for predict.py.

    A column with no variance (one workout, one weekday) gets a std of 1 rather
    than 0 -- it contributes nothing either way, and dividing by zero would turn
    the whole fit into NaN.
    """
    if not X:
        return [], [], []

    width = len(X[0])
    means = [sum(row[i] for row in X) / len(X) for i in range(width)]
    stds = []
    for i in range(width):
        variance = sum((row[i] - means[i]) ** 2 for row in X) / len(X)
        stds.append(math.sqrt(variance) or 1.0)

    scaled = [
        [(row[i] - means[i]) / stds[i] for i in range(width)] for row in X
    ]
    return scaled, means, stds


def apply_standardization(
    row: list[float], means: list[float], stds: list[float]
) -> list[float]:
    return [(value - mean) / std for value, mean, std in zip(row, means, stds)]


def fit_logistic(
    X: list[list[float]],
    y: list[float],
    l2: float = L2,
    learning_rate: float = LEARNING_RATE,
    iterations: int = MAX_ITERATIONS,
    tolerance: float = TOLERANCE,
) -> tuple[list[float], float, bool]:
    """Returns (weights, intercept, converged). Expects standardized X.

    The intercept is deliberately not regularized. Penalizing it would drag the
    predicted base rate toward 0.5, which is wrong here -- most slots on most
    days are not chosen, and that imbalance is real signal about the prior, not
    something to shrink away.
    """
    if not X:
        return [], 0.0, False

    width = len(X[0])
    weights = [0.0] * width
    intercept = 0.0
    rows = len(X)

    converged = False
    for _ in range(iterations):
        gradient = [0.0] * width
        intercept_gradient = 0.0

        for features, label in zip(X, y):
            total = intercept + sum(w * f for w, f in zip(weights, features))
            error = sigmoid(total) - label
            intercept_gradient += error
            for i, value in enumerate(features):
                gradient[i] += error * value

        for i in range(width):
            gradient[i] = gradient[i] / rows + l2 * weights[i]
        intercept_gradient /= rows

        step = max(
            max((abs(g) for g in gradient), default=0.0), abs(intercept_gradient)
        )
        weights = [w - learning_rate * g for w, g in zip(weights, gradient)]
        intercept -= learning_rate * intercept_gradient

        if step < tolerance:
            converged = True
            break

    return weights, intercept, converged


def predict_one(
    features: list[float], weights: list[float], intercept: float
) -> float:
    return sigmoid(intercept + sum(w * f for w, f in zip(weights, features)))


def split_by_group(
    rows: list[dict], groups: list[str], holdout: float = HOLDOUT_FRACTION
) -> tuple[list[int], list[int]]:
    """Indices for train/test, split on whole days.

    Slots from one day share a calendar and are near-duplicates of each other.
    Splitting by row would put a Tuesday 17:00 in train and the same Tuesday's
    18:00 in test, and the reported accuracy would be measuring memorization.
    """
    unique = sorted(set(groups))
    if len(unique) < 4:
        return list(range(len(rows))), []

    cut = max(1, int(len(unique) * holdout))
    test_days = set(unique[-cut:])
    train_index = [i for i, group in enumerate(groups) if group not in test_days]
    test_index = [i for i, group in enumerate(groups) if group in test_days]
    return train_index, test_index


def top1_accuracy(scores: list[float], y: list[float], groups: list[str]) -> float | None:
    """Fraction of days where the highest-scoring slot is the one you took.

    This is the metric that matches what actually ships. Plain accuracy over
    rows is close to meaningless here -- most slots are negatives, so predicting
    "not chosen" for everything scores around 90% while picking nothing.

    Days with no positive (you went nowhere) are skipped: there is no correct
    answer to rank first. Returns None when no day has one.
    """
    by_day: dict[str, list[tuple[float, float]]] = {}
    for score, label, group in zip(scores, y, groups):
        by_day.setdefault(group, []).append((score, label))

    scored = [day for day in by_day.values() if any(label for _, label in day)]
    if not scored:
        return None

    hits = sum(1 for day in scored if max(day, key=lambda pair: pair[0])[1] == 1.0)
    return hits / len(scored)


def baseline_scores(rows: list[dict]) -> list[float]:
    """The incumbent: closeness to config.GYM["preferred"].

    scheduler._score() picks whichever free slot sits nearest the preferred
    time. If the model cannot beat that on held-out days, it has learned nothing
    worth deploying, and predict.py should keep using the deterministic path.
    """
    hours, minutes = config.GYM["preferred"].split(":")
    preferred = int(hours) + int(minutes) / 60
    return [-abs(float(row.get("start_hour", 0.0)) - preferred) for row in rows]


def train(rows: list[dict], spec: FeatureSpec = DEFAULT, l2: float = L2) -> dict:
    """Fit on rows from backfill.build_examples and return the artifact dict."""
    if not rows:
        raise RuntimeError("no training rows; run the backfill first")

    X, y, groups = design_matrix(rows, spec)
    train_index, test_index = split_by_group(rows, groups)

    scaled, means, stds = standardize([X[i] for i in train_index])
    weights, intercept, converged = fit_logistic(
        scaled, [y[i] for i in train_index], l2=l2
    )

    evaluation: dict = {"holdout_days": len({groups[i] for i in test_index})}
    if test_index:
        test_rows = [rows[i] for i in test_index]
        test_y = [y[i] for i in test_index]
        test_groups = [groups[i] for i in test_index]
        scores = [
            predict_one(apply_standardization(X[i], means, stds), weights, intercept)
            for i in test_index
        ]
        evaluation["top1_accuracy"] = top1_accuracy(scores, test_y, test_groups)
        evaluation["baseline_top1_accuracy"] = top1_accuracy(
            baseline_scores(test_rows), test_y, test_groups
        )
    else:
        evaluation["top1_accuracy"] = None
        evaluation["baseline_top1_accuracy"] = None

    return {
        "version": ARTIFACT_VERSION,
        "trained_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "spec": spec.to_dict(),
        "feature_names": spec.feature_names,
        "standardization": {"means": means, "stds": stds},
        "coefficients": weights,
        "intercept": intercept,
        "training": {
            "n_days": len(set(groups)),
            "n_examples": len(rows),
            "n_positives": int(sum(y)),
            "n_negatives": int(len(y) - sum(y)),
            "l2": l2,
            "converged": converged,
        },
        "evaluation": evaluation,
    }


def main() -> dict:
    """Read history, rebuild examples, fit, and upload the artifact."""
    import calendar_client

    from .backfill import build_examples
    from .candidate_generator import load_duration_profile, s3_client
    from .repository import fetch_sessions

    sessions = fetch_sessions()
    if not sessions:
        raise RuntimeError("no training-eligible sessions in DynamoDB")

    profile = load_duration_profile()
    examples, diagnostics = build_examples(
        sessions, calendar_client.get_today_events, profile
    )
    artifact = train([example.to_row() for example in examples])
    artifact["backfill"] = diagnostics

    if not config.BUCKET_NAME:
        raise RuntimeError("BUCKET_NAME is not set; run cdk deploy and export it")

    s3_client().put_object(
        Bucket=config.BUCKET_NAME,
        Key=ARTIFACT_KEY,
        Body=json.dumps(artifact, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    return artifact


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
