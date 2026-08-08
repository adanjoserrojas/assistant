import hmac
import json
import uuid
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import boto3
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.exceptions import ClientError

from config import (
    COMMAND_SECRET,
    MAX_PLAUSIBLE_SESSION_MINUTES,
    MIN_PLAUSIBLE_SESSION_MINUTES,
    TABLE_NAME,
    TIMEZONE,
    USER_ID,
    VALID_GYM_LOCATIONS,
    WORKOUTS,
    REASONS_TO_SKIP,
)

'''
At 11:45 pm UTC -4:00 (America/New_York) run cron job to run this handler

 Pseudo type shit

1. Reads AssistantData table for a record that was posted that day
2.  def read_table() -> bool:
3.      for record in records:
4.          today = record
5.          today.split("T")
6.
7.          if record["checkin_at"] == today[0]:
8.              return True   
9.       return False
10. if no record posted:
11.      create_record() -> dict:
12.      write_record() -> dict:
13. 
'''