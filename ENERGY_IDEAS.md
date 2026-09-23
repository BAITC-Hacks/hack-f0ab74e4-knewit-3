# Energy track: preparation hypotheses

**These are brainstorming options, not the HackAlem AI Energy task.** The [updated regulations](https://edu.astanahub.com/hackathons/df4743f5-c492-415c-b45a-1f13adb78e06?tab=regulations) say there are ten tasks and that each task has its own technical brief and scoring criteria (sections 5.5 and 5.8). The Energy task-setter has not been publicly identified in the sources checked. Replace this note with the official brief when it arrives. Checked 23 September 2026.

## What seems plausible

Kazakhstan's [Ministry of Energy](https://www.gov.kz/memleket/entities/energo/press/news/details/1136583?lang=ru) names grid-line defect inspection, heat-pipe diagnosis, and meter reading as active AI pilots. [KEGOC](https://www.kegoc.kz/ru/press-center/press-releases/167360/) highlights demand and renewable forecasting, equipment defects, weather-related failures, and grid losses. These signals suggest three useful directions, but none establishes what the organizers will ask teams to build.

| Hypothesis | One focused agent workflow | Demo evidence to prepare | Main unknown |
| --- | --- | --- | --- |
| Grid reliability and maintenance | Inspect an equipment report or image, identify a possible fault, retrieve the relevant maintenance rule, and draft a prioritized work order for human approval. | A few labeled example incidents; show the evidence behind each recommendation and compare the priority with a simple baseline. | Whether images, asset data, or operating rules will be provided. |
| Demand and renewable balancing | Read a short load and generation forecast, detect a coming shortfall, and recommend a small set of operational actions with assumptions. | A time-series scenario showing the projected gap before and after the recommendation. | Whether the task permits simulated data and what operational actions are valid. |
| Heat-network leak triage | Combine a maintenance log, sensor anomaly, and weather context; rank likely leak locations and create an inspection plan. | One known-fault case with a traceable ranking and an operator review step. | Whether heat-network data is in scope or available. |

## Narrowing by task-setter

The company behind the task is still unknown. Once named, its current work is a better predictor than the broad Energy label:

| If the task-setter is… | Recent public work | More likely problem family |
| --- | --- | --- |
| [KEGOC](https://www.kegoc.kz/ru/press-center/press-releases/167360/) | Grid forecasting, weather-related disruptions, defect detection, and loss analysis | Operator decision support for outages, maintenance, or balancing |
| [Samruk-Energy](https://samruk-energy.kz/en/press-center/news/2773-predictive-ai-analytics-intelligent-equipment-monitoring-introduced-at-ekibastuz-sdpp-1) | Predictive equipment monitoring at Ekibastuz SDPP-1 using vibration and acoustic data | Plant equipment anomaly triage and maintenance planning |
| [QazaqGaz Aimaq](https://www.gov.kz/memleket/entities/energo/press/news/details/1041105?lang=ru) | AI-assisted gas meter reading in its customer app | Meter-data quality, customer service, or gas-network operations |
| [Ministry of Energy](https://www.gov.kz/memleket/entities/energo/press/news/details/1136583?lang=ru) | Power-line defect inspection, heat-pipe diagnostics, and gas-meter pilots | Cross-utility monitoring, defect triage, or digital data workflows |

These are **conditional inferences**, not sponsor announcements. A new task may be unrelated to any published pilot. If the company remains unknown, the strongest general starting bet is an operator workflow for grid or equipment reliability: gather evidence, explain the risk, and propose a reviewable action. Build its core only after 13:00 and only if it fits the actual task.

## Questions to settle when the brief is released

1. Is Energy an open theme or a fixed case with a supplied dataset?
2. Who is the intended user: grid operator, utility maintenance team, building manager, or consumer?
3. What exact result will judges score: accuracy, savings, reliability, agent autonomy, usability, or something else?
4. What pre-event work, external data, APIs, and existing code are allowed?
5. What must be submitted, in what format, and at what time?

Once those are answered, keep only the idea that directly addresses the prompt and can be demonstrated with available data.
