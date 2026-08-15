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
    REASONS_TO_SKIP,
)

'''
At 11:45 pm UTC -4:00 (America/New_York) run cron job to run this handler

 Pseudo type shit, will be working on this soon after playing some valorant ngl

1. Reads AssistantData table for a record that was posted that day
2.  def read_table() -> bool:
3.      for record in records:
4.          today = record
5.          today.split("T")
6.
7.          if record["checkin_at"] == today[0]:
8.              return True   
9.       return False
10. if no record posted:
11.      create_record() -> dict:
12.      write_record() -> dict:
13. 
'''

# Key layout written by gym_command_handler:
#   PK = USER#<USER_ID>
#   SK = GYM_STATE                                  (rotation state, one item)
#   SK = GYM_SESSION#<checkin_at_utc>#<session_id>  (one item per session)
#
# The UTC check-in timestamp leads the session SK and is fixed width (seconds
# precision, "Z" suffix), so lexicographic SK order is chronological order.
# That is what makes "newest session" the last key in the range.
# Will be working on ts soon --> 
USER_PK = f"USER#{USER_ID}"
STATE_SK = "GYM_STATE"
SESSION_SK_PREFIX = "GYM_SESSION#"

DDB = boto3.client("dynamodb")
SERIALIZER = TypeSerializer()
DESERIALIZER = TypeDeserializer()


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

def certify_sesh() -> bool:
    return False
def lambda_handler() -> dict[str, Any]:
    return