"""One LLM call per morning, via DeepSeek V3.2 on Amazon Bedrock (plan.md 10-15).

The model classifies existing events only. It never picks timestamps and never
touches the calendar -- that is scheduler.py's job.

Schema is enforced with a forced tool call rather than "please return JSON".
The tool schema is deliberately flat (breakfast_satisfied, breakfast_confidence,
...) because nested objects are unreliable with open-weight models; the nested
shape agent.py expects is reassembled here.

Run directly to test the call in isolation:
    python llm_client.py
"""

import json
import os

import boto3

import config

# ON_DEMAND, no inference profile, no model agreement to accept.
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "deepseek.v3.2")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

ACTIVITIES = ("breakfast", "lunch", "dinner", "gym")

SYSTEM_PROMPT = """You analyze a user's daily calendar.

Your job is to classify calendar events and decide whether existing events
already satisfy breakfast, lunch, dinner, or exercise.

Do not choose exact timestamps. Do not create or modify calendar events.

"gym" means weightlifting specifically -- resistance and strength training.
Lifting, squats, bench, deadlifts, "leg day", "upper body", "strength session",
or a plain "gym" entry all satisfy it.

Cardio and sports do NOT satisfy gym, no matter how strenuous. Basketball,
soccer, running, cycling, swimming, hiking, yoga, and fitness classes must all
be reported as gym unsatisfied.

Be conservative. If an event is ambiguous, report low confidence rather than
marking the activity satisfied. A work meeting that merely overlaps the lunch
hour does not satisfy lunch unless it is clearly a meal. Report a confidence
between 0.0 and 1.0 for every activity."""

TOOL_NAME = "record_classification"


def _tool_spec():
    properties, required = {}, []
    for name in ACTIVITIES:
        properties[f"{name}_satisfied"] = {
            "type": "boolean",
            "description": f"True if an existing event already satisfies {name}.",
        }
        properties[f"{name}_confidence"] = {
            "type": "number",
            "description": f"Confidence 0.0-1.0 in the {name} decision.",
        }
        required += [f"{name}_satisfied", f"{name}_confidence"]

    properties["reasoning"] = {
        "type": "string",
        "description": "One short sentence naming the events that drove the decisions.",
    }
    required.append("reasoning")

    return {
        "toolSpec": {
            "name": TOOL_NAME,
            "description": "Record which daily activities are already satisfied.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            },
        }
    }


def default_analysis():
    """Used when the model is unreachable (plan.md section 39)."""
    return {
        "satisfied_activities": {name: False for name in ACTIVITIES},
        "confidences": {name: 0.0 for name in ACTIVITIES},
        "reasoning": "LLM unavailable; scheduling all activities.",
    }


def build_prompt(events, day):
    payload = {
        "date": str(day),
        "timezone": config.TIMEZONE,
        "events": [
            {
                "title": event.title,
                "all_day": event.all_day,
                "start": "all day" if event.all_day else f"{event.start:%H:%M}",
                "end": "all day" if event.all_day else f"{event.end:%H:%M}",
            }
            for event in events
        ],
        "required_activities": list(ACTIVITIES),
    }
    return (
        "Here is today's calendar. Decide which of the required activities are "
        f"already satisfied by an existing event.\n\n{json.dumps(payload, indent=2)}\n\n"
        f"Call the {TOOL_NAME} tool with your answer."
    )


def _normalize(flat):
    """Flat tool input -> the nested shape agent.py reads. Tolerates bad values."""
    satisfied, confidences = {}, {}
    for name in ACTIVITIES:
        satisfied[name] = bool(flat.get(f"{name}_satisfied", False))
        try:
            score = float(flat.get(f"{name}_confidence", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        confidences[name] = min(max(score, 0.0), 1.0)

    return {
        "satisfied_activities": satisfied,
        "confidences": confidences,
        "reasoning": str(flat.get("reasoning", "")),
    }


def analyze_calendar(events, day=None):
    """Classify today's events. Raises on failure; agent.py handles the fallback."""
    day = day or (events[0].start.date() if events else "today")
    runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    response = runtime.converse(
        modelId=MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": build_prompt(events, day)}]}],
        inferenceConfig={"maxTokens": 2048, "temperature": 0},
        toolConfig={
            "tools": [_tool_spec()],
            "toolChoice": {"tool": {"name": TOOL_NAME}},
        },
    )

    for block in response["output"]["message"]["content"]:
        if "toolUse" in block and block["toolUse"]["name"] == TOOL_NAME:
            return _normalize(block["toolUse"]["input"])

    raise ValueError(f"model returned no {TOOL_NAME} tool call")


if __name__ == "__main__":
    from datetime import datetime

    import calendar_client

    todays_events = calendar_client.get_today_events()
    analysis = analyze_calendar(
        todays_events, datetime.now(calendar_client.timezone()).date()
    )
    print(json.dumps(analysis, indent=2, ensure_ascii=True))
