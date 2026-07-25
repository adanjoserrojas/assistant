'''

orchestration entry point...

Responsibilities:

Load configuration.
Authenticate with Google Calendar.
Retrieve today's events.
Send events to the LLM.
Parse structured LLM output.
Determine which activities still need scheduling.
Calculate free windows.
Schedule remaining activities.
Validate all proposed events.
Write events to Google Calendar.
'''

import config
