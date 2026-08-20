"""
-======================================
Tonight I am literally spending my entire night setting up my entire workstation to use the AWS CLI
and the AWS CDK, do not judge my commits today lol

It's honestly a lot of docs lol...

Ok I am back, so highkey this is gonna read the the DynamoDB table at 21 rows...
Deployed in lambda, how do I know when there is 21 records...
Cron job every morning and highkey query to check table records?
"""

from config import TABLE_NAME, USER_ID
from typing import Any
import boto3
from boto3.dynamodb.types import TypeDeserializer


USER_PK = f"USER#{USER_ID}"
SESSION_SK_PREFIX = "GYM_SESSION#"
STATE_SK = "GYM_STATE"

_deserializer = TypeDeserializer()
_client = None


def client():
    """The DynamoDB client, built on first use.

    Lazy because boto3.client() resolves credentials as it constructs, so
    building one at module scope makes `import ml` fail outright anywhere a
    credential chain is absent or half-configured -- a test runner, or a laptop
    between `aws login` sessions. Lambda caches this across warm invocations
    exactly as a module-level client would.
    """
    global _client
    if _client is None:
        _client = boto3.client("dynamodb", region_name='us-east-1')
    return _client


def _deserialize_item(item: dict[str, Any]) -> dict[str, Any]:
    return {key: _deserializer.deserialize(value) for key, value in item.items()}

def fetch_sessions() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    request: dict[str, Any] = {
        "TableName": TABLE_NAME,
        "KeyConditionExpression": "PK = :pk AND begins_with(SK, :prefix)",
        "FilterExpression": "training_eligible = :eligible",
        "ExpressionAttributeValues": {
            ":pk": {"S": USER_PK},
            ":prefix": {"S": SESSION_SK_PREFIX},
            ":eligible": {"BOOL": True},
        },
    }

    while True:
        page = client().query(**request)
        items.extend(_deserialize_item(item) for item in page.get("Items", []))

        last_key = page.get("LastEvaluatedKey")
        if not last_key:
            return items
        request["ExclusiveStartKey"] = last_key


def count_training_sessions() -> int:
    return len(fetch_sessions())


def fetch_state() -> dict[str, Any]:
    """The rotation state item written by gym_command_handler.

    next_workout_index is what says which workout is up today. It lives here
    rather than in candidate_generator so every DynamoDB read in the ml package
    goes through one module. Strongly consistent: the morning run must not size
    a workout against a rotation the validator advanced hours earlier.
    """
    result = client().get_item(
        TableName=TABLE_NAME,
        Key={"PK": {"S": USER_PK}, "SK": {"S": STATE_SK}},
        ConsistentRead=True,
    )
    item = result.get("Item")
    if not item:
        raise RuntimeError(f"Missing {USER_PK} / {STATE_SK} item.")
    return _deserialize_item(item)
