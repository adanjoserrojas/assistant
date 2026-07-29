import hmac
import json
import uuid
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import boto3
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.exceptions import ClientError

from config import (
    COMMAND_SECRET,
    MAX_PLAUSIBLE_SESSION_MINUTES,
    MIN_PLAUSIBLE_SESSION_MINUTES,
    TABLE_NAME,
    TIMEZONE,
    USER_ID,
    VALID_GYM_LOCATIONS,
    WORKOUTS,
)

USER_PK = f"USER#{USER_ID}"
STATE_SK = "GYM_STATE"
REST_WORKOUT = "Rest-days"

DDB = boto3.client("dynamodb")
SERIALIZER = TypeSerializer()
DESERIALIZER = TypeDeserializer()


def response(status_code: int, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def serialize_item(item: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {key: SERIALIZER.serialize(value) for key, value in item.items()}


def deserialize_item(
    item: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not item:
        return None
    return {key: DESERIALIZER.deserialize(value) for key, value in item.items()}


def parse_event(event: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    headers = {
        str(key).lower(): str(value)
        for key, value in (event.get("headers") or {}).items()
        if value is not None
    }

    raw_body = event.get("body")
    if raw_body is None:
        body = {key: value for key, value in event.items() if key != "headers"}
    elif isinstance(raw_body, str):
        body = json.loads(raw_body)
    elif isinstance(raw_body, dict):
        body = raw_body
    else:
        raise ValueError("The request body must be a JSON object.")

    if not isinstance(body, dict):
        raise ValueError("The request body must be a JSON object.")

    return headers, body


def authenticated(headers: dict[str, str]) -> bool:
    if not COMMAND_SECRET:
        return False
    supplied = headers.get("x-gym-command-secret", "")
    return hmac.compare_digest(supplied, COMMAND_SECRET)


def current_timestamps() -> tuple[datetime, str, str]:
    utc_now = datetime.now(timezone.utc)
    local_now = utc_now.astimezone(ZoneInfo(TIMEZONE))
    utc_iso = utc_now.isoformat(timespec="seconds").replace("+00:00", "Z")
    local_iso = local_now.isoformat(timespec="seconds")
    return utc_now, utc_iso, local_iso


def get_item(sk: str) -> dict[str, Any] | None:
    result = DDB.get_item(
        TableName=TABLE_NAME,
        Key={"PK": {"S": USER_PK}, "SK": {"S": sk}},
        ConsistentRead=True,
    )
    return deserialize_item(result.get("Item"))


def get_state() -> dict[str, Any]:
    state = get_item(STATE_SK)
    if state is None:
        raise RuntimeError(f"Missing {USER_PK} / {STATE_SK} item.")
    return state


def canonical_workout(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    for workout in WORKOUTS:
        if workout.lower() == normalized:
            return workout
    return None


def start_session(body: dict[str, Any]) -> dict[str, Any]:
    state = get_state()
    if state.get("active_session_id"):
        return response(
            409,
            {
                "accepted": False,
                "message": "A gym session is already active.",
                "session_id": str(state["active_session_id"]),
            },
        )

    cycle_index = int(state.get("next_workout_index", 0)) % len(WORKOUTS)
    expected_workout = WORKOUTS[cycle_index]
    supplied_workout = canonical_workout(body.get("workout"))
    location = str(body.get("location") or "").strip().upper()

    if expected_workout == REST_WORKOUT:
        return response(
            409,
            {
                "accepted": False,
                "message": "The current rotation entry is Rest-days. Send SKIP with reason rest.",
                "current_workout": expected_workout,
            },
        )

    if supplied_workout != expected_workout:
        return response(
            409,
            {
                "accepted": False,
                "message": f"Expected workout is {expected_workout}.",
                "current_workout": expected_workout,
            },
        )

    if location not in VALID_GYM_LOCATIONS:
        allowed = ", ".join(sorted(VALID_GYM_LOCATIONS))
        return response(
            400,
            {
                "accepted": False,
                "message": f"location must be one of: {allowed}.",
            },
        )

    _, utc_iso, local_iso = current_timestamps()
    session_id = str(uuid.uuid4())
    session_sk = f"GYM_SESSION#{utc_iso}#{session_id}"

    session_item = {
        "PK": USER_PK,
        "SK": session_sk,
        "entity_type": "gym_session",
        "session_id": session_id,
        "status": "active",
        "training_eligible": False,
        "workout": supplied_workout,
        "cycle_index": cycle_index,
        "location_code": location,
        "timezone": TIMEZONE,
        "checkin_at": local_iso,
        "checkin_at_utc": utc_iso,
        "source": "phone_command",
        "created_at": utc_iso,
        "updated_at": utc_iso,
        "schema_version": 1,
    }

    DDB.transact_write_items(
        TransactItems=[
            {
                "Put": {
                    "TableName": TABLE_NAME,
                    "Item": serialize_item(session_item),
                    "ConditionExpression": "attribute_not_exists(PK) AND attribute_not_exists(SK)",
                }
            },
            {
                "Update": {
                    "TableName": TABLE_NAME,
                    "Key": {"PK": {"S": USER_PK}, "SK": {"S": STATE_SK}},
                    "UpdateExpression": (
                        "SET active_session_id = :sid, active_session_sk = :ssk, "
                        "updated_at = :now, #version = if_not_exists(#version, :zero) + :one"
                    ),
                    "ConditionExpression": (
                        "attribute_not_exists(active_session_id) OR "
                        "attribute_type(active_session_id, :null_type)"
                    ),
                    "ExpressionAttributeNames": {"#version": "version"},
                    "ExpressionAttributeValues": {
                        ":sid": {"S": session_id},
                        ":ssk": {"S": session_sk},
                        ":now": {"S": utc_iso},
                        ":zero": {"N": "0"},
                        ":one": {"N": "1"},
                        ":null_type": {"S": "NULL"},
                    },
                }
            },
        ]
    )

    return response(
        200,
        {
            "accepted": True,
            "command": "START",
            "message": f"Started {supplied_workout} at {location}.",
            "session_id": session_id,
            "current_workout": supplied_workout,
            "next_workout": WORKOUTS[(cycle_index + 1) % len(WORKOUTS)],
            "server_timestamp": local_iso,
        },
    )


def stop_session() -> dict[str, Any]:
    state = get_state()
    session_id = state.get("active_session_id")
    session_sk = state.get("active_session_sk")

    if not session_id or not session_sk:
        return response(
            409,
            {"accepted": False, "message": "There is no active gym session."},
        )

    session = get_item(str(session_sk))
    if session is None:
        return response(
            500,
            {
                "accepted": False,
                "message": "The active session record could not be found.",
            },
        )

    checkin_utc = datetime.fromisoformat(
        str(session["checkin_at_utc"]).replace("Z", "+00:00")
    )
    utc_now, utc_iso, local_iso = current_timestamps()
    duration_seconds = int((utc_now - checkin_utc).total_seconds())
    duration_minutes = int(round(duration_seconds / 60))

    duration_valid = (
        duration_seconds >= 0
        and MIN_PLAUSIBLE_SESSION_MINUTES
        <= duration_minutes
        <= MAX_PLAUSIBLE_SESSION_MINUTES
    )

    cycle_index = int(session["cycle_index"])
    next_index = (cycle_index + 1) % len(WORKOUTS)

    if duration_valid:
        session_status = "completed"
        state_update = (
            "SET last_completed_session_id = :sid, last_completed_at = :now, "
            "next_workout_index = :next_index, updated_at = :now, "
            "#version = if_not_exists(#version, :zero) + :one "
            "REMOVE active_session_id, active_session_sk"
        )
        state_values = {
            ":sid": {"S": str(session_id)},
            ":ssk": {"S": str(session_sk)},
            ":now": {"S": utc_iso},
            ":next_index": {"N": str(next_index)},
            ":zero": {"N": "0"},
            ":one": {"N": "1"},
        }
        message = f"Completed {session['workout']} in {duration_minutes} minutes."
        next_workout = WORKOUTS[next_index]
        status_code = 200
    else:
        session_status = "needs_review"
        state_update = (
            "SET last_flagged_session_id = :sid, last_flagged_at = :now, "
            "updated_at = :now, #version = if_not_exists(#version, :zero) + :one "
            "REMOVE active_session_id, active_session_sk"
        )
        state_values = {
            ":sid": {"S": str(session_id)},
            ":ssk": {"S": str(session_sk)},
            ":now": {"S": utc_iso},
            ":zero": {"N": "0"},
            ":one": {"N": "1"},
        }
        message = (
            f"Session closed but flagged for review: {duration_minutes} minutes is outside "
            f"the {MIN_PLAUSIBLE_SESSION_MINUTES}-{MAX_PLAUSIBLE_SESSION_MINUTES} minute range."
        )
        next_workout = WORKOUTS[cycle_index]
        status_code = 202

    session_expression = (
        "SET #status = :session_status, checkout_at = :checkout_local, "
        "checkout_at_utc = :checkout_utc, actual_duration_minutes = :minutes, "
        "actual_duration_seconds = :seconds, duration_valid = :duration_valid, "
        "training_eligible = :training_eligible, updated_at = :now"
    )
    session_values = {
        ":session_status": {"S": session_status},
        ":active": {"S": "active"},
        ":sid": {"S": str(session_id)},
        ":checkout_local": {"S": local_iso},
        ":checkout_utc": {"S": utc_iso},
        ":minutes": {"N": str(duration_minutes)},
        ":seconds": {"N": str(duration_seconds)},
        ":duration_valid": {"BOOL": duration_valid},
        ":training_eligible": {"BOOL": duration_valid},
        ":now": {"S": utc_iso},
    }

    if not duration_valid:
        session_expression += " SET anomaly_reason = :anomaly_reason"
        # DynamoDB allows only one SET keyword, so normalize the expression.
        session_expression = session_expression.replace(
            " SET anomaly_reason", ", anomaly_reason"
        )
        session_values[":anomaly_reason"] = {
            "S": "session_duration_outside_plausible_range"
        }

    DDB.transact_write_items(
        TransactItems=[
            {
                "Update": {
                    "TableName": TABLE_NAME,
                    "Key": {"PK": {"S": USER_PK}, "SK": {"S": str(session_sk)}},
                    "UpdateExpression": session_expression,
                    "ConditionExpression": "#status = :active AND session_id = :sid",
                    "ExpressionAttributeNames": {"#status": "status"},
                    "ExpressionAttributeValues": session_values,
                }
            },
            {
                "Update": {
                    "TableName": TABLE_NAME,
                    "Key": {"PK": {"S": USER_PK}, "SK": {"S": STATE_SK}},
                    "UpdateExpression": state_update,
                    "ConditionExpression": (
                        "active_session_id = :sid AND active_session_sk = :ssk"
                    ),
                    "ExpressionAttributeNames": {"#version": "version"},
                    "ExpressionAttributeValues": state_values,
                }
            },
        ]
    )

    return response(
        status_code,
        {
            "accepted": True,
            "command": "STOP",
            "message": message,
            "session_id": str(session_id),
            "status": session_status,
            "actual_duration_minutes": duration_minutes,
            "duration_valid": duration_valid,
            "next_workout": next_workout,
            "server_timestamp": local_iso,
        },
    )


def complete_rest(body: dict[str, Any]) -> dict[str, Any]:
    state = get_state()
    if state.get("active_session_id"):
        return response(
            409,
            {
                "accepted": False,
                "message": "Stop the active session before completing a rest day.",
            },
        )

    cycle_index = int(state.get("next_workout_index", 0)) % len(WORKOUTS)
    if (
        WORKOUTS[cycle_index] != REST_WORKOUT
        or str(body.get("reason", "")).lower() != "rest"
    ):
        return response(
            409,
            {
                "accepted": False,
                "message": "SKIP advances only when the current rotation entry is Rest-days.",
                "current_workout": WORKOUTS[cycle_index],
            },
        )

    _, utc_iso, local_iso = current_timestamps()
    next_index = (cycle_index + 1) % len(WORKOUTS)
    session_id = str(uuid.uuid4())
    session_sk = f"GYM_SESSION#{utc_iso}#{session_id}"

    rest_item = {
        "PK": USER_PK,
        "SK": session_sk,
        "entity_type": "gym_session",
        "session_id": session_id,
        "status": "rest_completed",
        "training_eligible": False,
        "workout": REST_WORKOUT,
        "cycle_index": cycle_index,
        "timezone": TIMEZONE,
        "source": "phone_command",
        "created_at": utc_iso,
        "updated_at": utc_iso,
        "schema_version": 1,
    }

    DDB.transact_write_items(
        TransactItems=[
            {
                "Put": {
                    "TableName": TABLE_NAME,
                    "Item": serialize_item(rest_item),
                    "ConditionExpression": "attribute_not_exists(PK) AND attribute_not_exists(SK)",
                }
            },
            {
                "Update": {
                    "TableName": TABLE_NAME,
                    "Key": {"PK": {"S": USER_PK}, "SK": {"S": STATE_SK}},
                    "UpdateExpression": (
                        "SET next_workout_index = :next_index, updated_at = :now, "
                        "#version = if_not_exists(#version, :zero) + :one"
                    ),
                    "ConditionExpression": (
                        "attribute_not_exists(active_session_id) OR "
                        "attribute_type(active_session_id, :null_type)"
                    ),
                    "ExpressionAttributeNames": {"#version": "version"},
                    "ExpressionAttributeValues": {
                        ":next_index": {"N": str(next_index)},
                        ":now": {"S": utc_iso},
                        ":zero": {"N": "0"},
                        ":one": {"N": "1"},
                        ":null_type": {"S": "NULL"},
                    },
                }
            },
        ]
    )

    return response(
        200,
        {
            "accepted": True,
            "command": "SKIP",
            "message": "Rest day completed; rotation advanced.",
            "next_workout": WORKOUTS[next_index],
            "server_timestamp": local_iso,
        },
    )


def status() -> dict[str, Any]:
    state = get_state()
    cycle_index = int(state.get("next_workout_index", 0)) % len(WORKOUTS)
    return response(
        200,
        {
            "accepted": True,
            "command": "STATUS",
            "active_session_id": state.get("active_session_id"),
            "current_workout_index": cycle_index,
            "current_workout": WORKOUTS[cycle_index],
            "next_workout": WORKOUTS[(cycle_index + 1) % len(WORKOUTS)],
            "last_completed_session_id": state.get("last_completed_session_id"),
            "last_completed_at": state.get("last_completed_at"),
            "last_flagged_session_id": state.get("last_flagged_session_id"),
        },
    )


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    try:
        headers, body = parse_event(event)
    except (ValueError, json.JSONDecodeError) as error:
        return response(400, {"accepted": False, "message": str(error)})

    if not COMMAND_SECRET:
        return response(
            500,
            {
                "accepted": False,
                "message": "GYM_COMMAND_SECRET is not configured for this Lambda.",
            },
        )

    if not authenticated(headers):
        return response(401, {"accepted": False, "message": "Unauthorized."})

    command = str(body.get("command") or "").strip().upper()

    try:
        if command == "START":
            return start_session(body)
        if command == "STOP":
            return stop_session()
        if command == "SKIP":
            return complete_rest(body)
        if command == "STATUS":
            return status()
        return response(
            400,
            {
                "accepted": False,
                "message": "command must be START, STOP, SKIP, or STATUS.",
            },
        )
    except DDB.exceptions.TransactionCanceledException:
        return response(
            409,
            {
                "accepted": False,
                "message": "The gym state changed during the request. Run STATUS and retry.",
            },
        )
    except ClientError as error:
        print(json.dumps(error.response.get("Error", {})))
        return response(
            500,
            {
                "accepted": False,
                "message": "A DynamoDB operation failed. Check CloudWatch logs.",
            },
        )
    except Exception as error:
        print(f"{type(error).__name__}: {error}")
        return response(
            500,
            {
                "accepted": False,
                "message": "Unexpected server error. Check CloudWatch logs.",
            },
        )
