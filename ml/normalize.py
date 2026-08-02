import json
from boto3.dynamodb.types import TypeDeserializer
from ml.repository import (
    fetch_sessions as logs,
    count_training_sessions as amount
)
from datetime import datetime

'''
Sole purpose of this thing is to clean and prepare daata for ML logistics regression model
And mean calculation
'''

# protos

def good_boy() -> list[dict]:
    data = logs()
    print(json.dumps(data, indent=2, default=str)) # I see what we are deling with, my son
    return data

def parse_time(old: dict):
    date: datetime = None
    weekday: str = ""

    return [date, weekday]

def decimal_convert() -> int:
    pass

def build_training_data(old_data: list[dict]) -> list[dict]:

    new_data = []

    for item in old_data:
        
        started_at, weekday = parse_time(item)
        new_data.append({
            "workout": item["workout"],
            "duration": decimal_convert(item["actual_duration_minutes"]),
            "location": item["location_code"],
            "started_at": started_at,
            "weekday": weekday
        })

    return new_data
        
good_boy()