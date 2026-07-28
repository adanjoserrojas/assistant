# FXX01_Learning_Gym

## Repo with current system

https://github.com/adanjoserrojas/assistant

## Rules

- With your answers, be concise and straightforward. Output in actionable steps, not paragraphs. Preferred format is bullet-points. You will help me robust this system by designing a new feature described in this planning.md.

- Ask questions to enhance your knowledge about the current system, to make the one we are designing more robust.

- Utilize the README.md provided to comprehend the current system.

## Idea + System + Context
I want to make my AI Agent assistant (The assistant already exists) turn my time at the gym into data that can be tensorized, and then fed into a small neural network that will predict what times and how much time is the best for me to go to the gym based on my previously logged gym sessions. 

For context, I do not go to the gym everyday, but my current assistant allocates time for the gym everyday. I have modified the config to have a rest-day, that way it could be taken into account. Currently, the AI assistant uses a hard-coded config that represents a time interval I would like to use to go to the gym. Example: 

```json
{
"GYM": {

    "duration": 90,

    "earliest": "07:00",

    "preferred": "17:30",

    "latest": "22:00",

}}
```

For the new feature, the assistant must track my location on trigger at the time I am supposed to go to the gym to check if I am there, and if I am, track how much time I spent at that location using Google Maps API, then save that data over time...

For now, before tensorizing the data, I need data... So I could deploy a system that if I follow my gym sessions hard-coded scheduled for the next 16 days, and accumulate that attendance data with the following format: 

```json
{
    "Logs": [
        {

            "timestamp": "2026-07-28_14:25:00",

            "day": "Tuesday",

            "duration": 65,

            "location": "Crunch East-Colonial",

            "attended": true,

            "workout": "Back-Biceps"

        },
        {

            "timestamp": "2026-07-29_16:45:00",

            "day": "Wednesday",

            "duration": 45,

            "location": "Crunch East-Colonial",

            "attended": true,

            "workout": "Sharms"

        },
        {

            "timestamp": "",

            "day": "Thursday",

            "duration": 0,

            "location": "",

            "attended": false,

            "workout": "Rest-days"

        }
    ]
}
```

- **timestamp**: Represents the day and time I went to the gym. Maybe I could strip it into 2 attributes: time and date. Date representing numerical representation of calendar day, not string day. \
- **day**: String representing the actual day of the week I went (Ex. "Tuesday"). \
- **duration**: A rough measurement of time I spent at that place. \
- **location**: Which gym I attended, this one could be hardcoded because there is just 2 locations I attend: The UCF gym, or the Crunch by the UCF gym, I could set up either tuples that have the coordinates floats, or full addresses in strings. \
- **attended**: Boolean representing if I logged at that location **during the proposed time or not**. Maybe I logged but it was not tracked because it was outside of the proposed time, but it is too expensive for me to run the cron job every 5 minutes to check my location during an entire day. So, we can resume during the explicitly given time. \
- **workout**: hard-coded string that is in the configs. There is just 4 types of workouts I hit, WORKOUTS = ["Chest-Triceps", "Back-Biceps", "Sharms", "Rest-days"]. 

The system must be able to follow the indexed order of the workouts so I do not repeat workouts. The workouts repeat after Rest-day, so the week turns into an 8-day week... The only thing the system must worry about is dynamically allocating the time to go through the day, and allocating the right amount of minutes.

## Machine Learning algo

I have very little knowledge about ML. I think I could train a neural network with an arbitrary number of attention layers and a dropout layer that basically based on my 16 day attendance data set predicts what the best times to allocate my gym sesh are...

But there is a thing, my days are different every week, they are annotated but different... So, let say the prediction given for day X has already an event allocated, then I need a fallback method that looks for an empty slot in my calendar with the closest

## Task

Grill my idea to further build this feature the best way possible and deploy it today. I will build the system and the code, I just want to learn while building something cool, so I just want you to help me understand if this is duable. 