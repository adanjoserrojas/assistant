from boto3.dynamodb.types import TypeDeserializer
from repository import (
    fetch_sessions as logs,
    count_training_sessions as amount
)

'''
Sole purpose of this thing is to clean and prepare daata for ML logistics regression model
And mean calculation
'''

