"""Orchestration entry point (plan.md section 5).

    python agent.py --dry-run     inspect the plan, write nothing
    python agent.py               write the plan to Google Calendar

Fixed pipeline: READ -> ANALYZE -> SCHEDULE -> VALIDATE -> WRITE.
"""

import argparse
import logging
import sys
from datetime import date, datetime

import calendar_client
import config
import gym_allocator
import scheduler
from validator import validate_schedule

log = logging.getLogger("agent")

ACTIVITIES = ("breakfast", "lunch", "dinner", "gym")
GYM = "gym"


def analyze(events, day):
    """LLM classification, with a safe fallback (plan.md section 39).

    The LLM improves the scheduler; it must never be able to stop it.
    """
    import llm_client

    try:
        return llm_client.analyze_calendar(events, day) or llm_client.default_analysis()
    except Exception as exc:
        log.warning("LLM analysis unavailable (%s); scheduling all activities", exc)
        return llm_client.default_analysis()


def determine_required_activities(analysis, already_created):
    """Drop activities the LLM says are satisfied, and ones we already made."""
    satisfied = analysis.get("satisfied_activities", {})
    confidences = analysis.get("confidences", {})

    required = []
    for name in ACTIVITIES:
        if any(name in title.lower() for title in already_created):
            log.info("%s: already created by a previous run, skipping", name)
            continue

        # Only trust a confident classification (plan.md section 15).
        if satisfied.get(name):
            confidence = confidences.get(name, 1.0)
            if confidence >= config.CONFIDENCE_THRESHOLD:
                log.info("%s: already satisfied by an existing event", name)
                continue
            log.info("%s: classified satisfied but confidence %.2f is too low", name, confidence)

        required.append(name)
    return required


def schedule_day(required, events, day, tz):
    """Meals deterministically, then gym on top of them.

    The order is the point. Meals are placed first and become busy time, so the
    gym search sees the whole day rather than competing with meals that have not
    been decided yet -- which is what scheduler.SCHEDULING_ORDER has always
    encoded, and what splitting the call preserves.

    Meals stay on preferences forever; they are not a prediction problem. Only
    gym is dynamic, and only when the model has earned it.
    """
    activities = [a for a in scheduler.activities_from_config() if a.name in required]
    meals = [a for a in activities if a.name != GYM]
    gym = [a for a in activities if a.name == GYM]

    scheduled, unplaced = scheduler.schedule_activities(meals, events, day, tz)
    if not gym:
        return scheduled, unplaced, {"method": "not_required", "reason": "gym not required today"}

    decision = gym_allocator.allocate_safely(events, scheduled, day, tz)
    if decision["winner"] is not None:
        scheduled.append(decision["winner"])
        return scheduled, unplaced, decision

    if decision["method"] == "rest":
        return scheduled, unplaced, decision

    # No model, a model that has not earned it, or an outright failure: place
    # gym exactly as it has always been placed. The already-scheduled meals
    # travel in as busy time so this path sees the same day the model did.
    busy = list(events) + gym_allocator.meals_as_events(scheduled)
    gym_scheduled, gym_unplaced = scheduler.schedule_activities(
        gym, busy, day, tz, already_scheduled=scheduled
    )
    scheduled.extend(gym_scheduled)
    unplaced.extend(gym_unplaced)
    return scheduled, unplaced, decision


def report(day, events, analysis, required, scheduled, unplaced, problems, dry_run, gym_decision=None):
    print(f"\n{calendar_client.calendar_id()} -- {day}\n")

    print("Existing events:")
    if not events:
        print("  (none)")
    for item in events:
        when = "ALL DAY    " if item.all_day else f"{item.start:%H:%M}-{item.end:%H:%M}"
        print(f"  {when}  {item.title}")

    satisfied = [k for k, v in analysis.get("satisfied_activities", {}).items() if v]
    print("\nLLM classification:")
    print(f"  satisfied: {', '.join(satisfied) if satisfied else 'nothing'}")

    print(f"\nActivities remaining:\n  {', '.join(required) if required else '(none)'}")

    print("\nProposed schedule:")
    if not scheduled:
        print("  (nothing to schedule)")
    for item in sorted(scheduled, key=lambda s: s.start):
        print(f"  {item.start:%H:%M}-{item.end:%H:%M}  {item.name}")
    if unplaced:
        print(f"  no viable slot for: {', '.join(unplaced)}")

    if gym_decision:
        print(f"\nGym allocation: {gym_decision['method']}")
        print(f"  {gym_decision['reason']}")
        for entry in gym_decision.get("scores", []):
            print(f"  {entry['start'][11:16]}  p={entry['probability']}")

    print(f"\nValidation: {'PASS' if not problems else 'FAIL'}")
    for problem in problems:
        print(f"  - {problem}")

    if dry_run:
        print("\nDry run enabled.\nCalendar unchanged.\n")


def run(dry_run=False, day=None):
    tz = calendar_client.timezone()
    day = day or datetime.now(tz).date()

    # Reading is mandatory; never schedule blind (plan.md section 40).
    events = calendar_client.get_today_events(day)
    already_created = calendar_client.find_agent_events(day)

    analysis = analyze(events, day)
    required = determine_required_activities(analysis, already_created)

    # Agent-created events are real events -- keep them as busy time.
    scheduled, unplaced, gym_decision = schedule_day(required, events, day, tz)

    problems = validate_schedule(events, scheduled, day, tz)
    report(
        day, events, analysis, required, scheduled, unplaced, problems, dry_run,
        gym_decision,
    )

    if problems:
        log.error("validation failed; writing nothing")
        return {"written": 0, "problems": problems}

    if dry_run:
        return {
            "written": 0,
            "problems": [],
            "gym_method": gym_decision["method"],
            "gym_reason": gym_decision["reason"],
        }

    written = 0
    for item in sorted(scheduled, key=lambda s: s.start):
        try:
            calendar_client.create_event(item.name.title(), item.start, item.end)
            written += 1
            log.info("created %s at %s", item.name, item.start.isoformat())
        except Exception:
            log.exception("failed to create %s; stopping further writes", item.name)
            break

    print(f"Wrote {written} event(s) to {calendar_client.calendar_id()}.\n")
    return {
        "written": written,
        "problems": [],
        "gym_method": gym_decision["method"],
        "gym_reason": gym_decision["reason"],
    }


def lambda_handler(event, context):
    """EventBridge sends its own payload, which is ignored.

    For console testing, accepts an optional {"dry_run": true} to plan without
    writing, and {"date": "YYYY-MM-DD"} to target a specific day.
    """
    logging.getLogger().setLevel(logging.INFO)
    event = event or {}
    requested_day = event.get("date")
    return run(
        dry_run=bool(event.get("dry_run", False)),
        day=date.fromisoformat(requested_day) if requested_day else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily calendar agent")
    parser.add_argument("--dry-run", action="store_true", help="plan only, write nothing")
    parser.add_argument("--date", help="YYYY-MM-DD, defaults to today")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = run(dry_run=args.dry_run, day=date.fromisoformat(args.date) if args.date else None)
    sys.exit(1 if result["problems"] else 0)
