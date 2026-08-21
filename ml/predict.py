"""Score candidates against the trained artifact and pick the winner.

Pure Python, no ML dependencies: the artifact carries coefficients, so scoring
is a dot product and a sigmoid. That is what keeps the scoring Lambda a small
zip instead of a container image.

The model is a ranker. Absolute probabilities are not trustworthy at this sample
size and do not need to be -- if all three slots score 0.3, the best one still
wins. So `usable()` gates on whether the model has earned its place at all, not
on how confident it sounds about today. A model that cannot beat "pick the slot
nearest 17:30" on held-out days has learned nothing worth deploying, and the
caller should fall back to scheduler._score().
"""

import json
from dataclasses import dataclass

import config
from .features import FeatureSpec, vectorize
from .train import ARTIFACT_KEY, apply_standardization, predict_one

# A model fitted on fewer days than this is describing a handful of afternoons.
MIN_TRAINING_DAYS = 20

# Without days where you declined every slot, the fit has no idea what a bad
# slot looks like -- it has only ever seen you say yes.
MIN_NEGATIVE_DAYS = 3


@dataclass(frozen=True)
class Model:
    spec: FeatureSpec
    coefficients: list[float]
    intercept: float
    means: list[float]
    stds: list[float]
    training: dict
    evaluation: dict

    @classmethod
    def from_dict(cls, payload: dict) -> "Model":
        standardization = payload.get("standardization") or {}
        return cls(
            spec=FeatureSpec.from_dict(payload.get("spec") or {}),
            coefficients=list(payload.get("coefficients") or []),
            intercept=float(payload.get("intercept") or 0.0),
            means=list(standardization.get("means") or []),
            stds=list(standardization.get("stds") or []),
            training=payload.get("training") or {},
            evaluation=payload.get("evaluation") or {},
        )


def load_model() -> Model | None:
    """Read gym/model_metadata.json from S3, or None if nothing is published."""
    from .candidate_generator import s3_client

    if not config.BUCKET_NAME:
        return None
    try:
        body = s3_client().get_object(Bucket=config.BUCKET_NAME, Key=ARTIFACT_KEY)[
            "Body"
        ].read()
    except Exception:
        # No artifact yet is the normal state before the first training run,
        # and a morning schedule must never fail because of it.
        return None
    return Model.from_dict(json.loads(body))


def usable(model: Model | None) -> tuple[bool, str]:
    """Whether to trust this model today. Returns (ok, reason)."""
    if model is None:
        return False, "no model artifact published"
    if not model.coefficients:
        return False, "artifact carries no coefficients"

    days = int(model.training.get("n_days") or 0)
    if days < MIN_TRAINING_DAYS:
        return False, f"only {days} training days, need {MIN_TRAINING_DAYS}"

    negatives = int(model.training.get("n_negatives") or 0)
    if negatives < MIN_NEGATIVE_DAYS:
        return False, f"only {negatives} negative examples"

    top1 = model.evaluation.get("top1_accuracy")
    baseline = model.evaluation.get("baseline_top1_accuracy")
    if top1 is None or baseline is None:
        return False, "model was not evaluated against the baseline"
    if top1 < baseline:
        return False, (
            f"model top-1 {top1:.2f} does not beat "
            f"preferred-time baseline {baseline:.2f}"
        )

    return True, f"top-1 {top1:.2f} vs baseline {baseline:.2f} over {days} days"


def score(row: dict, model: Model) -> float:
    """Attendance probability for one candidate row."""
    vector = vectorize(row, model.spec)
    if model.means and model.stds:
        vector = apply_standardization(vector, model.means, model.stds)
    return predict_one(vector, model.coefficients, model.intercept)


def rank(candidates: list, model: Model) -> list[tuple]:
    """(candidate, probability) sorted best first.

    Ties break toward the earlier slot, matching scheduler.schedule_activities --
    two slots the model cannot separate should not shuffle between runs.
    """
    scored = [(candidate, score(candidate.to_dict(), model)) for candidate in candidates]
    return sorted(scored, key=lambda pair: (-pair[1], pair[0].start))


def best(candidates: list, model: Model):
    """Highest-scoring candidate, or None when there is nothing to choose from."""
    if not candidates:
        return None
    return rank(candidates, model)[0][0]


def choose(candidates: list, model: Model | None) -> dict:
    """Pick today's slot, saying which path was taken and why.

    Never raises on a missing or weak model -- it reports `fallback` and leaves
    the choice to the caller's deterministic path. A morning with no gym event
    is a worse outcome than a morning with a merely adequate one.
    """
    if not candidates:
        return {"winner": None, "method": "none", "reason": "no viable slots today"}

    ok, reason = usable(model)
    if not ok:
        return {"winner": None, "method": "fallback", "reason": reason}

    ranked = rank(candidates, model)
    return {
        "winner": ranked[0][0],
        "method": "model",
        "reason": reason,
        "scores": [
            {
                "start": candidate.start.isoformat(timespec="seconds"),
                "probability": round(probability, 4),
            }
            for candidate, probability in ranked
        ],
    }
