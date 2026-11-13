# Travel-time plots

From the repository root, run `venv/bin/python visualize/<script>.py <experiment_id>`. The scripts read saved episode CSVs and write 300 DPI PNGs to `visualize/plots/`.

| Scripts | What they show |
| --- | --- |
| `travel_times.py`, `travel_times_kde.py` | Baseline human travel times and test-phase human and AV travel times. |
| `travel_time_cohorts.py`, `travel_time_cohorts_kde.py` | Before and after travel times for matched agents who stayed human or converted to AVs. |
| `compare_experiments.py`, `compare_experiments_kde.py` | Test-phase human and AV travel times and per-agent travel-time reduction for converted AVs. Pass space-separated experiment IDs. |
| `training_travel_times.py` | Human and AV mean travel times over saved episodes, with pointwise confidence bands. |

Distribution plots use one mean per agent per phase. Dashed KDE markers and single-run summaries use the values shown after trimming. Positive travel-time reduction means a shorter trip. Edit [`config.json`](config.json) for the baseline window, tail trimming, fonts, colors, and KDE bandwidth. Set `outlier_tail_fraction` to `0` to include all agents.

For a single run, the training curve uses rolling intervals over saved episode means within each phase. For multiple seeds of the same setup, run `venv/bin/python visualize/training_travel_times.py seed_42 seed_43 seed_44`. The curve averages the seed means at shared episode numbers and computes a Student’s t interval across seeds at each episode. The default confidence level is 95%.
