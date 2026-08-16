'''
1. Duration Profile
Read completed gym sessions from DynamoDB.
Group them by workout.
Calculate mean duration.
Save the result to S3.
'''
from .normalize import (
    build_training_data as eligible
)

def group_workout() -> list[dict]:
    pass
def calculate_mean() -> int:
    pass
def send_S3() -> None:
    pass

def main():
    pass
