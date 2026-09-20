from dataclasses import dataclass
from datetime import datetime, time


@dataclass
class CalendarEvent:
    title: str
    start: datetime
    end: datetime
    all_day: bool = False
    # Which calendar this came from. Defaulted because events synthesized from
    # placed activities (scheduler, gym_allocator) never came from one.
    calendar_id: str = ""

@dataclass
class Activity:
    name: str
    duration_minutes: int
    earliest_start: time
    latest_start: time
    preferred_start: time

@dataclass
class ScheduledActivity:
    name: str
    start: datetime
    end: datetime

