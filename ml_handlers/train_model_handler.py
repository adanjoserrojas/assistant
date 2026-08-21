"""Training Lambda. Refits the attendance ranker from DynamoDB.

This is the only place config.MIN_TRAINING_RECORDS is enforced. Below the
threshold it publishes nothing at all, which means the morning run never has to
know the number -- it asks "is there a usable artifact?", finds none, and uses
the deterministic preferences path. One threshold, defined once, with no way for
two Lambdas to disagree about it.

Each run rebuilds both artifacts from scratch:

    gym/duration_profiles.json   mean duration per workout, sizes the slots
    gym/model_metadata.json      the model itself -- spec, standardization,
                                 coefficients, and the evaluation predict.py
                                 gates on

Retraining never reads the previous model. The session history in DynamoDB is
the only thing that has to survive between runs.

Note this needs Google Calendar credentials, unlike the other gym Lambdas. The
negatives are reconstructed from what your calendar looked like on each past
day, so the backfill reads history the same way the morning agent reads today.
"""

import json
from typing import Any

import calendar_client
import config
from ml.backfill import build_examples
from ml.candidate_generator import DURATION_PROFILE_KEY, s3_client
from ml.duration_profile import calculate_mean
from ml.normalize import build_training_data
from ml.repository import fetch_sessions
from ml.train import ARTIFACT_KEY, train


def _upload(payload: dict, key: str) -> str:
    if not config.BUCKET_NAME:
        raise RuntimeError("BUCKET_NAME is not set; run cdk deploy and export it")

    s3_client().put_object(
        Bucket=config.BUCKET_NAME,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    return f"s3://{config.BUCKET_NAME}/{key}"


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    sessions = fetch_sessions()
    records = len(sessions)

    if records < config.MIN_TRAINING_RECORDS:
        # Not an error. This is the expected state for months, and the morning
        # agent keeps scheduling gym from preferences the whole time.
        result = {
            "trained": False,
            "reason": "below the training threshold",
            "records": records,
            "required": config.MIN_TRAINING_RECORDS,
        }
        print(json.dumps(result))
        return result

    # Durations first: they size the slots the examples are built on, so a stale
    # profile would enumerate candidates the wrong width.
    profile = calculate_mean(build_training_data(sessions))
    profile_uri = _upload(profile, DURATION_PROFILE_KEY)

    examples, diagnostics = build_examples(
        sessions, calendar_client.get_today_events, profile
    )
    if not examples:
        result = {
            "trained": False,
            "reason": "backfill produced no usable examples",
            "records": records,
            "backfill": diagnostics,
        }
        print(json.dumps(result))
        return result

    artifact = train([example.to_row() for example in examples])
    artifact["backfill"] = diagnostics
    artifact_uri = _upload(artifact, ARTIFACT_KEY)

    result = {
        "trained": True,
        "records": records,
        "duration_profile": profile_uri,
        "artifact": artifact_uri,
        "training": artifact["training"],
        "evaluation": artifact["evaluation"],
        "backfill": diagnostics,
    }
    print(json.dumps(result, default=str))
    return result
