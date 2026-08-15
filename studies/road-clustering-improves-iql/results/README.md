# Results

This directory contains the 12 completed runs from the paired
`2 route sets × 2 controllers × 3 seeds` experiment.

Each `rciql_ing2_*` directory is one run. Its `exp_config.json` records the effective
configuration, `metrics/BenchmarkMetrics.csv` contains the primary metrics, and `plots/`
contains selected run-level figures. Large raw artifacts remain local and are ignored.

The paired values and conclusion are reported in the [study README](../README.md).
They can be reproduced with:

```bash
venv/bin/python studies/road-clustering-improves-iql/experiments/analyze_results.py
```
