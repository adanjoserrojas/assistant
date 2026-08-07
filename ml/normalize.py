import json
from boto3.dynamodb.types import TypeDeserializer
from .repository import (
    fetch_sessions as logs,
    count_training_sessions as amount
)
from datetime import datetime

'''
Sole purpose of this thing is to clean and prepare daata for ML logistics regression model
And mean calculation
'''

# protos
def parse_time(time_date: str):
    days = time_date.split("T")
    day = days[0].split("-")
    time = days[1].split("-")

    string_day = datetime(int(day[0]), int(day[1]), int(day[2]))
    return [time[0], string_day.strftime('%A')]

def build_training_data(old_data: list[dict]) -> list[dict]:

    new_data = []

    for item in old_data:
        
        started_at, weekday = parse_time(item["checkin_at"])
        new_data.append({
            "workout": item["workout"],
            "duration": float(item["actual_duration_minutes"]),
            "location": item["location_code"],
            "started_at": started_at,
            "weekday": weekday,
        })

    return new_data


if __name__ == "__main__":
    data = logs()
    print(json.dumps(build_training_data(data), indent=4))