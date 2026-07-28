# FXX01_Learning_Gym

## Rules

- With your answers, be concise and straightforward. Output in actionable steps, not paragraphs. Preferred format is bullet-points. You will help me robust this system by designing a new feature described in this planning.md.

- Ask questions to enhance your knowledge about the current system, to make the one we are designing more robust.

- Utilize the README.md provided to comprehend the current system.

## System
I want to make an AI Agent assistant (The assistant already exists) turn my time at the gym into data that can be tensorized, and then fed into a small neural network that will predict what days and times are the best for me to go to the gym based on my previously logged gym sessions. Currently, the AI assistant uses a hard-coded config that represents a time interval I would like to use to go to the gym (Ex. GYM = {

    "duration": 90,

    "earliest": "07:00",

    "preferred": "17:30",

    "latest": "22:00",

}) I would like the assistant to track my location on trigger at the time I am supposed to go to the gym to check if I am there, and if I am, track how much time I spent at that location using Google Maps, then save that data over time... For now, I just need to hardcode them for the next 15 days, and accumulate that attendance data with the following format: Ex. { 

    "Logs": [

        {

            "timestamp": "2026-07-28_14:25:00",

            "Day": "Tuesday",

            "duration": 65,

            "location": "Crunch East-Colonial",

            "Attended": true,

            "Workout": "Back"

        },

        {

            "timestamp": "2026-07-29_16:45:00",

            "Day": "Wednesday",

            "duration": 45,

            "location": "Crunch East-Colonial",

            "Attended": true,

            "Workout": "Sharms"

        },

        {

            "timestamp": "2026-07-20_10:30:00",

            "Day": "Thursday",

            "duration": 96,

            "location": "Recreation Welness Center",

            "Attended": true

        }

        

    ]

}