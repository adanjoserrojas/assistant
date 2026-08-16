'''
1. Duration Profile
Read completed gym sessions from DynamoDB.
Group them by workout.
Calculate mean duration.
Save the result to S3.
'''
import json
from .normalize import (
    CLEAN_DATA
)

# Return the a dict of workout_type => [workouts{}], pops workout from keys to dminish redundancy
def group_workout(data: list[dict]) -> dict[str, list[dict]]:

    res: dict[str, list[dict]] = {}

    for item in data:

        workout_type = item.pop("workout", None)
        res.setdefault(workout_type, []).append(item)

    return res


def calculate_mean() -> int:
    pass
def send_S3() -> None:
    pass

def main():
    print(json.dumps(group_workout(CLEAN_DATA), indent=4))

main()
