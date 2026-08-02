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

def good_boy() -> int:
    # print(json.dumps(logs(), indent=2, default=str)) I see what we are deling with, my son
    return 1
def parse_time() -> datetime:
    pass
def decimal_convert() -> int:
    pass
def build_training_data():
    pass

good_boy()