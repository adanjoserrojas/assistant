'''
1. Duration Profile
Read completed gym sessions from DynamoDB.
Group them by workout.
Calculate mean duration.
Save the result to S3.
'''
import json
import boto3
from .normalize import (
    CLEAN_DATA
)

from config import BUCKET_NAME

s3_client = boto3.client('s3', region_name='us-east-1')

# Return the a dict of workout_type => [workouts{}], pops workout from keys to dminish redundancy
def group_workout(data: list[dict]) -> dict[str, list[dict]]:

    res: dict[str, list[dict]] = {}

    for item in data:

        session = dict(item)
        workout_type = session.pop("workout", None)
        res.setdefault(workout_type, []).append(session)

    return res

# Return the average time spend per workout type
def calculate_mean(data: list[dict]) -> dict[str, float]:

    grouped_data = group_workout(data)
    res: dict[str, float] = {}

    for workout_type, sessions in grouped_data.items():
        total = sum(session["duration"] for session in sessions)
        res[workout_type] = round((total / len(sessions)), 2)

    return res

def _send_S3(payload: dict, key: str) -> str:

    if not BUCKET_NAME:
        raise RuntimeError("BUCKET_NAME is not set; run cdk deploy and export it")

    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=json.dumps(payload, indent=4).encode("utf-8"),
        ContentType="application/json",
    )
    return f"s3://{BUCKET_NAME}/{key}"

def main() -> None:

    profiles = calculate_mean(CLEAN_DATA)

    print(json.dumps(profiles, indent=4))
    print(_send_S3(profiles, "gym/duration_profiles.json"))

if __name__ == "__main__":
    main()