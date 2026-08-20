"""Nightly gym-session validator. EventBridge, 02:00 America/New_York.

The gym command handler only writes when the phone sends something. A day you
simply did not go leaves no record at all, which the training set reads as "no
data" rather than "he skipped". This handler runs after the day is over, looks at
yesterday, and writes the missing record itself.

One outcome per run:

  already_logged        yesterday has a session, rest, or injury record -- nothing to do
  rest_auto_completed   yesterday's rotation entry was Rest-days and no SKIP arrived;
                        record the rest day and advance the rotation
  unattended            yesterday was a training day and nothing was logged; record
                        attended=False and freeze the rotation, so the same workout
                        is still up today
  no_history            the table has no sessions at all; nothing to judge yesterday against

Separately: a session that was STARTed and never STOPped leaves active_session_id
set in GYM_STATE, and that blocks every future START with a 409. A lock older than
today is closed here as needs_review, without inventing a duration and without
advancing the rotation.

Every write is idempotent. session_id is a uuid5 of the validated date, so the SK
is deterministic and a retried invocation collides on
attribute_not_exists(SK), cancelling the transaction harmlessly.
"""

import json
import uuid
from datetime import date, datetime, time as clock, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import boto3
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.exceptions import ClientError

from config import (
    TABLE_NAME,
    TIMEZONE,
    USER_ID,
    WORKOUTS,
)

# Key layout written by gym_command_handler:
#   PK = USER#<USER_ID>
#   SK = GYM_STATE                                  (rotation state, one item)
#   SK = GYM_SESSION#<checkin_at_utc>#<session_id>  (one item per session)
#
# The UTC check-in timestamp leads the session SK and is fixed width (seconds
# precision, "Z" suffix), so lexicographic SK order is chronological order.
# That is what makes "newest session" the last key in the range.
USER_PK = f"USER#{USER_ID}"
STATE_SK = "GYM_STATE"
SESSION_SK_PREFIX = "GYM_SESSION#"
REST_WORKOUT = "Rest-days"

# uuid5 namespace for auto-generated records. Any fixed uuid works; this one is
# arbitrary and must never change, or idempotency breaks for past dates.
VALIDATOR_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

SOURCE = "auto_validator"

DDB = boto3.client("dynamodb")
SERIALIZER = TypeSerializer()
DESERIALIZER = TypeDeserializer()
LOCAL_TZ = ZoneInfo(TIMEZONE)


def serialize_item(item: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {key: SERIALIZER.serialize(value) for key, value in item.items()}


def deserialize_item(
    item: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not item:
        return None
    return {key: DESERIALIZER.deserialize(value) for key, value in item.items()}


def current_timestamps() -> tuple[datetime, datetime, str, str]:
    utc_now = datetime.now(timezone.utc)
    local_now = utc_now.astimezone(LOCAL_TZ)
    utc_iso = utc_now.isoformat(timespec="seconds").replace("+00:00", "Z")
    local_iso = local_now.isoformat(timespec="seconds")
    return utc_now, local_now, utc_iso, local_iso


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


def read_latest_session() -> dict[str, Any] | None:
    """Read the most recent gym session record from AssistantData.

    Returns the deserialized item, or None when the user has no session yet.

    Query, never Scan: reading the SK range backwards with Limit=1 charges for
    one item no matter how large the table grows. begins_with keeps the range on
    session items -- descending order would otherwise land on GYM_STATE first,
    since "GYM_STATE" sorts above "GYM_SESSION#". Key values travel as
    ExpressionAttributeValues, never as concatenated expression text. Strongly
    consistent, because this handler runs right after the day's writes and must
    not judge the day from a stale replica.
    """
    page = DDB.query(
        TableName=TABLE_NAME,
        KeyConditionExpression="PK = :pk AND begins_with(SK, :prefix)",
        ExpressionAttributeValues={
            ":pk": {"S": USER_PK},
            ":prefix": {"S": SESSION_SK_PREFIX},
        },
        ScanIndexForward=False,
        Limit=1,
        ConsistentRead=True,
    )

    items = page.get("Items", [])
    if not items:
        return None
    return {key: DESERIALIZER.deserialize(value) for key, value in items[0].items()}


def to_local_day(value: Any) -> date | None:
    """Local calendar day of an ISO timestamp, UTC or offset-bearing."""
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(LOCAL_TZ).date()


def session_local_day(session: dict[str, Any] | None) -> date | None:
    """Which local day a record belongs to.

    checkin_at first, because that is the day you were actually at the gym: a
    23:00 local check-in is 03:00 UTC the next morning, and created_at would file
    it under the wrong day. Rest and injury records carry no check-in, so
    created_at is the fallback -- it is the only timestamp present on every
    record shape gym_command_handler writes.
    """
    if session is None:
        return None
    if session.get("session_date"):
        try:
            return date.fromisoformat(str(session["session_date"]))
        except ValueError:
            pass
    for field in ("checkin_at", "created_at", "checkin_at_utc"):
        day = to_local_day(session.get(field))
        if day is not None:
            return day
    return None


def resolve_target_day(event: dict[str, Any], local_now: datetime) -> date:
    """The day being judged: yesterday, or an explicit override for testing.

    Invoke with {"target_date": "2026-08-19"} to replay a specific day without
    waiting for 02:00. The day must be over -- judging today would record a miss
    for a gym session you have not had the chance to log yet.
    """
    override = (event or {}).get("target_date")
    if override:
        target = date.fromisoformat(str(override))
    else:
        target = (local_now - timedelta(days=1)).date()

    if target >= local_now.date():
        raise ValueError(
            f"target_date {target} is not over yet (local date is {local_now.date()})."
        )
    return target


def deterministic_keys(target: date) -> tuple[str, str]:
    """A session_id and SK fixed by the validated date, so retries collide.

    The SK timestamp is the UTC instant of local end-of-day, which sorts the
    generated record after every real session from that day and before the next
    day's -- keeping the "newest session is the last key" invariant intact.
    """
    session_id = str(
        uuid.uuid5(VALIDATOR_NAMESPACE, f"gym-validator:{USER_ID}:{target.isoformat()}")
    )
    end_of_day = datetime.combine(target, clock(23, 59, 59), tzinfo=LOCAL_TZ)
    sk_stamp = (
        end_of_day.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    return session_id, f"{SESSION_SK_PREFIX}{sk_stamp}#{session_id}"


def close_stale_session(
    state: dict[str, Any], local_today: date, utc_iso: str
) -> dict[str, Any] | None:
    """Release an active_session_id left over from a previous day.

    A forgotten STOP holds the lock forever and every future START answers 409.
    The session is closed as needs_review rather than completed: there is no
    checkout, so any duration would be fabricated. The rotation does not advance
    -- an unfinished session is not a finished workout.

    Returns the session that was closed, or None if there was nothing to close.
    """
    session_id = state.get("active_session_id")
    session_sk = state.get("active_session_sk")
    if not session_id or not session_sk:
        return None

    session = get_item(str(session_sk))
    if session is None:
        # The lock points at a record that does not exist. Clear it anyway,
        # otherwise START stays blocked on a phantom.
        DDB.update_item(
            TableName=TABLE_NAME,
            Key={"PK": {"S": USER_PK}, "SK": {"S": STATE_SK}},
            UpdateExpression=(
                "SET updated_at = :now, #version = if_not_exists(#version, :zero) + :one "
                "REMOVE active_session_id, active_session_sk"
            ),
            ConditionExpression="active_session_id = :sid",
            ExpressionAttributeNames={"#version": "version"},
            ExpressionAttributeValues={
                ":sid": {"S": str(session_id)},
                ":now": {"S": utc_iso},
                ":zero": {"N": "0"},
                ":one": {"N": "1"},
            },
        )
        return None

    if (session_local_day(session) or local_today) >= local_today:
        # Started today and still running. Not stale -- leave it alone.
        return None

    DDB.transact_write_items(
        TransactItems=[
            {
                "Update": {
                    "TableName": TABLE_NAME,
                    "Key": {"PK": {"S": USER_PK}, "SK": {"S": str(session_sk)}},
                    "UpdateExpression": (
                        "SET #status = :needs_review, training_eligible = :false, "
                        "duration_valid = :false, anomaly_reason = :reason, "
                        "closed_by = :source, updated_at = :now"
                    ),
                    "ConditionExpression": "#status = :active AND session_id = :sid",
                    "ExpressionAttributeNames": {"#status": "status"},
                    "ExpressionAttributeValues": {
                        ":needs_review": {"S": "needs_review"},
                        ":active": {"S": "active"},
                        ":sid": {"S": str(session_id)},
                        ":false": {"BOOL": False},
                        ":reason": {"S": "session_never_stopped"},
                        ":source": {"S": SOURCE},
                        ":now": {"S": utc_iso},
                    },
                }
            },
            {
                "Update": {
                    "TableName": TABLE_NAME,
                    "Key": {"PK": {"S": USER_PK}, "SK": {"S": STATE_SK}},
                    "UpdateExpression": (
                        "SET last_flagged_session_id = :sid, last_flagged_at = :now, "
                        "updated_at = :now, #version = if_not_exists(#version, :zero) + :one "
                        "REMOVE active_session_id, active_session_sk"
                    ),
                    "ConditionExpression": (
                        "active_session_id = :sid AND active_session_sk = :ssk"
                    ),
                    "ExpressionAttributeNames": {"#version": "version"},
                    "ExpressionAttributeValues": {
                        ":sid": {"S": str(session_id)},
                        ":ssk": {"S": str(session_sk)},
                        ":now": {"S": utc_iso},
                        ":zero": {"N": "0"},
                        ":one": {"N": "1"},
                    },
                }
            },
        ]
    )
    return session


def record_unattended(
    target: date, cycle_index: int, workout: str, utc_iso: str
) -> dict[str, Any]:
    """Write the negative training example for a training day you did not log.

    No checkin_at, no actual_duration_minutes, no location_code -- none of them
    happened, and inventing them would teach the model a start time you never
    chose. ml/normalize.py tolerates their absence. The rotation is deliberately
    NOT advanced: the workout you missed is still the workout that is up.
    """
    session_id, session_sk = deterministic_keys(target)

    item = {
        "PK": USER_PK,
        "SK": session_sk,
        "entity_type": "gym_session",
        "session_id": session_id,
        "status": "unattended",
        "training_eligible": True,
        "workout": workout,
        "cycle_index": cycle_index,
        "timezone": TIMEZONE,
        "session_date": target.isoformat(),
        "source": SOURCE,
        "auto_generated": True,
        "created_at": utc_iso,
        "updated_at": utc_iso,
        "schema_version": 1,
        "injured": False,
        "attended": False,
    }

    DDB.transact_write_items(
        TransactItems=[
            {
                "Put": {
                    "TableName": TABLE_NAME,
                    "Item": serialize_item(item),
                    "ConditionExpression": (
                        "attribute_not_exists(PK) AND attribute_not_exists(SK)"
                    ),
                }
            },
            {
                "Update": {
                    "TableName": TABLE_NAME,
                    "Key": {"PK": {"S": USER_PK}, "SK": {"S": STATE_SK}},
                    "UpdateExpression": (
                        "SET last_validated_date = :target, last_validation_at = :now, "
                        "last_unattended_date = :target, updated_at = :now, "
                        "#version = if_not_exists(#version, :zero) + :one"
                    ),
                    "ConditionExpression": (
                        "(attribute_not_exists(last_validated_date) "
                        "OR last_validated_date <> :target) AND "
                        "(attribute_not_exists(active_session_id) "
                        "OR attribute_type(active_session_id, :null_type))"
                    ),
                    "ExpressionAttributeNames": {"#version": "version"},
                    "ExpressionAttributeValues": {
                        ":target": {"S": target.isoformat()},
                        ":now": {"S": utc_iso},
                        ":zero": {"N": "0"},
                        ":one": {"N": "1"},
                        ":null_type": {"S": "NULL"},
                    },
                }
            },
        ]
    )

    return {
        "outcome": "unattended",
        "session_id": session_id,
        "workout": workout,
        "cycle_index": cycle_index,
        "rotation_advanced": False,
        "next_workout": workout,
    }


def record_rest_day(target: date, cycle_index: int, utc_iso: str) -> dict[str, Any]:
    """Close out a Rest-days entry that never got a SKIP, and advance the rotation.

    Mirrors complete_rest() in gym_command_handler, with source=auto_validator so
    the two are distinguishable later. training_eligible stays False: a rest day
    is not a missed gym and does not belong in the attendance model.
    """
    session_id, session_sk = deterministic_keys(target)
    next_index = (cycle_index + 1) % len(WORKOUTS)

    item = {
        "PK": USER_PK,
        "SK": session_sk,
        "entity_type": "gym_session",
        "session_id": session_id,
        "status": "rest_completed",
        "training_eligible": False,
        "workout": REST_WORKOUT,
        "cycle_index": cycle_index,
        "timezone": TIMEZONE,
        "session_date": target.isoformat(),
        "source": SOURCE,
        "auto_generated": True,
        "created_at": utc_iso,
        "updated_at": utc_iso,
        "schema_version": 1,
        "injured": False,
        "skip_reason": "rest",
    }

    DDB.transact_write_items(
        TransactItems=[
            {
                "Put": {
                    "TableName": TABLE_NAME,
                    "Item": serialize_item(item),
                    "ConditionExpression": (
                        "attribute_not_exists(PK) AND attribute_not_exists(SK)"
                    ),
                }
            },
            {
                "Update": {
                    "TableName": TABLE_NAME,
                    "Key": {"PK": {"S": USER_PK}, "SK": {"S": STATE_SK}},
                    "UpdateExpression": (
                        "SET next_workout_index = :next_index, "
                        "last_validated_date = :target, last_validation_at = :now, "
                        "updated_at = :now, "
                        "#version = if_not_exists(#version, :zero) + :one"
                    ),
                    # The index guard makes the advance safe against a SKIP that
                    # lands between the read and this write.
                    "ConditionExpression": (
                        "(attribute_not_exists(last_validated_date) "
                        "OR last_validated_date <> :target) AND "
                        "(attribute_not_exists(next_workout_index) "
                        "OR next_workout_index = :expected_index) AND "
                        "(attribute_not_exists(active_session_id) "
                        "OR attribute_type(active_session_id, :null_type))"
                    ),
                    "ExpressionAttributeNames": {"#version": "version"},
                    "ExpressionAttributeValues": {
                        ":next_index": {"N": str(next_index)},
                        ":expected_index": {"N": str(cycle_index)},
                        ":target": {"S": target.isoformat()},
                        ":now": {"S": utc_iso},
                        ":zero": {"N": "0"},
                        ":one": {"N": "1"},
                        ":null_type": {"S": "NULL"},
                    },
                }
            },
        ]
    )

    return {
        "outcome": "rest_auto_completed",
        "session_id": session_id,
        "workout": REST_WORKOUT,
        "cycle_index": cycle_index,
        "rotation_advanced": True,
        "next_workout": WORKOUTS[next_index],
    }


def validate_day(event: dict[str, Any]) -> dict[str, Any]:
    _, local_now, utc_iso, local_iso = current_timestamps()
    target = resolve_target_day(event, local_now)

    state = get_state()
    stale = close_stale_session(state, local_now.date(), utc_iso)
    if stale is not None:
        # The state we read is now out of date in exactly one respect, and
        # nothing below reads the lock again.
        state = {
            key: value
            for key, value in state.items()
            if key not in ("active_session_id", "active_session_sk")
        }

    latest = read_latest_session()
    latest_day = session_local_day(latest)

    result: dict[str, Any] = {
        "target_date": target.isoformat(),
        "last_logged_date": latest_day.isoformat() if latest_day else None,
        "stale_session_closed": stale["session_id"] if stale else None,
        "server_timestamp": local_iso,
    }

    if latest is None:
        # Nothing has ever been logged. There is no history to say a gym day was
        # expected, so do not fabricate a miss on an empty table.
        return result | {"outcome": "no_history", "recorded": False}

    if latest_day is None:
        # A record with no usable timestamp -- refuse to guess.
        return result | {
            "outcome": "unreadable_latest_session",
            "recorded": False,
            "session_sk": str(latest.get("SK", "")),
        }

    if latest_day >= target:
        # Something was logged for the target day (or later): you went, you
        # rested, you were injured, or the stale session above covered it.
        return result | {
            "outcome": "already_logged",
            "recorded": False,
            "status": latest.get("status"),
        }

    cycle_index = int(state.get("next_workout_index", 0)) % len(WORKOUTS)
    workout = WORKOUTS[cycle_index]

    if workout == REST_WORKOUT:
        return result | record_rest_day(target, cycle_index, utc_iso) | {"recorded": True}

    return result | record_unattended(target, cycle_index, workout, utc_iso) | {
        "recorded": True
    }


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    try:
        result = validate_day(event or {})
    except ValueError as error:
        result = {"outcome": "bad_request", "recorded": False, "message": str(error)}
    except DDB.exceptions.TransactionCanceledException:
        # Deterministic keys mean the usual cause is a second invocation for a
        # day already validated. Not an error worth retrying or alarming on.
        result = {
            "outcome": "already_validated",
            "recorded": False,
            "message": "This day was already validated, or the gym state changed mid-write.",
        }
    except ClientError as error:
        print(json.dumps(error.response.get("Error", {})))
        raise
    except RuntimeError as error:
        result = {"outcome": "error", "recorded": False, "message": str(error)}

    print(json.dumps(result, default=str))
    return result
