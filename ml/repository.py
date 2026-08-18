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

client = boto3.client("dynamodb", region_name='us-east-1')
_deserializer = TypeDeserializer()


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
        page = client.query(**request)
        items.extend(_deserialize_item(item) for item in page.get("Items", []))

        last_key = page.get("LastEvaluatedKey")
        if not last_key:
            return items
        request["ExclusiveStartKey"] = last_key


def count_training_sessions() -> int:
    return len(fetch_sessions())
