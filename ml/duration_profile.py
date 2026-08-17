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

s3_client = boto3.client('s3')

# Return the a dict of workout_type => [workouts{}], pops workout from keys to dminish redundancy
def group_workout(data: list[dict]) -> dict[str, list[dict]]:

    res: dict[str, list[dict]] = {}

    for item in data:

        session = dict(item)
        workout_type = session.pop("workout", None)
        res.setdefault(workout_type, []).append(session)

    return res

# Return the average time spend per workout type
def calculate_mean() -> dict[str, float]:

    grouped_data = group_workout(CLEAN_DATA)
    res: dict[str, float] = {}

    for workout_type, sessions in grouped_data.items():
        total = sum(session["duration"] for session in sessions)
        res[workout_type] = round((total / len(sessions)), 2)

    return res

def send_S3() -> None:
    pass