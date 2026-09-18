# Experiments

This study compares IQL with normal `TripInfoETA` against the same learner with its route ETA entries fixed to free-flow values. The fixed condition retains the same observation shape, route-specific free-flow information, and constant origin, destination, and start-time entries.

## Software

- **URB revision:** Base revision `57ebc3f` plus the supplied study-specific files
- **RouteRL version:** 2.0.0
- **Python environment:** Repository `venv`, Python 3.12.7
- **SUMO version:** 1.27.1
- **Relevant machine details:** Apple Silicon macOS; no GPU is required

## Runs

The six runs form three matched pairs. Each pair uses identical environment and model seeds in both conditions.

| Run or group | Purpose | Status | Supplied evidence |
| --- | --- | --- | --- |
| `eta_iql_ing2_dynamic_s40` | Dynamic ETA, seed pair 40 | complete | [`run_metrics.csv`](evidence/run_metrics.csv) |
| `eta_iql_ing2_static_s40` | Fixed free-flow ETA, seed pair 40 | complete | [`run_metrics.csv`](evidence/run_metrics.csv) |
| `eta_iql_ing2_dynamic_s41` | Dynamic ETA, seed pair 41 | complete | [`run_metrics.csv`](evidence/run_metrics.csv) |
| `eta_iql_ing2_static_s41` | Fixed free-flow ETA, seed pair 41 | complete | [`run_metrics.csv`](evidence/run_metrics.csv) |
| `eta_iql_ing2_dynamic_s42` | Dynamic ETA, seed pair 42 | complete | [`run_metrics.csv`](evidence/run_metrics.csv) |
| `eta_iql_ing2_static_s42` | Fixed free-flow ETA, seed pair 42 | complete | [`run_metrics.csv`](evidence/run_metrics.csv) |

## Implementation

`iql_eta_ablation.py` is a direct copy of `scripts/iql.py` with a small study-specific diff. `dynamic` passes `TripInfoETA` unchanged. `static` replaces only its four ETA entries with that agent's route free-flow times before acting and learning. Both inputs retain the same shape and the same origin, destination, and departure entries.

The exact algorithm, environment, and task settings are under `configs/`. The IQL settings are `training_eps=1000`, `eps_init=1.0`, `eps_decay=0.9952587776358824`, `eps_min=0.01`, `buffer_size=2048`, `batch_size=32`, `lr=0.001`, one training epoch, and two 64-unit hidden layers. Epsilon reaches 0.01 on the last training update and is zero during testing.

The environment and model use the matched seed pairs `(40, 40)`, `(41, 41)`, and `(42, 42)`. Model initialization, exploration, and replay are matched across each pair. Exploration and replay use private per-agent random streams derived from the model seed, so they do not consume RouteRL's random streams. Everything else follows `scripts/iql.py`, including its simulator settings.

The three dynamic runs formed the first batch. Each static run then copied `routes.csv` and `route.rou.xml` from its matched dynamic run, fixing route definitions and action meanings exactly. The analysis also checked that the recorded human-learning episodes and selected CAV IDs matched within each pair.

The runs recorded CAV, human, and system travel times after every training and test day. The main test metric therefore uses all 100 test days and the learning metric uses all 1,000 training days. Standard URB metrics were calculated as secondary evidence. The runs also saved their final policies and every test episode so `analyze_study.py` could measure how the dynamic policies responded when recorded ETA entries were replaced by free-flow values.

## Checks before analysis

- Confirm from file hashes that each matched pair has identical routes, CAV IDs, and recorded human-learning episodes.
- Confirm from the study diff that the static transform changes only the four ETA entries.
- Plot both all-day learning curves and inspect their first and final 250 days.
- Count how often dynamic ETA values differ from free-flow values, overall and by departure-time third.

## Analysis

The main result is the paired relative difference in test `t_CAV` defined in `PROPOSAL.md`. Report all three pairs and their mean.

Learning performance will be measured by mean CAV travel time over all 1,000 AV-training days, including exploration. Report all three paired differences and their mean, plus the first and final 250 days as descriptive checks. It is classified separately as better, worse, or unclear/practically similar using the rules in `PROPOSAL.md`; it does not change the main test-time verdict.

Also report the standard URB `t_HDV_test`, `t_test`, CAV advantage, and cost-of-learning metrics. These remain secondary and do not replace the main test metric.

For mechanism evidence, use all CAVs rather than choosing favourable examples. Report how often replacing recorded ETA values with free-flow values changes the dynamic network's greedy action and Q-values, overall and across early, middle, and late departure groups. This shows whether IQL uses ETA information; only the matched performance comparison determines whether that use is beneficial.

## Retained evidence

`evidence/` contains the per-run metrics, paired effects, pairing checks, sensitivity results, and final decision. The exact shared settings are in `configs/`; `iql_eta_ablation.py` and `analyze_study.py` preserve the study-specific implementation and analysis. Raw simulator output, episode dumps, models, losses, logs, plots, and run-management files were removed after the results and independent reviews were complete.
