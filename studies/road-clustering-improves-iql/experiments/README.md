# Experiments

This directory contains the predeclared protocol and the code used to run and summarize
it. Generated configurations, metrics, and plots are stored in `../results/`.

Run one paired suite with:

```bash
./studies/road-clustering-improves-iql/experiments/run_replication.sh <seed> <janux-first|clustered-first>
```

The local study launch uses three concurrent suites:

| Seed | Order | Arms run sequentially inside the suite |
| --- | --- | --- |
| 40 | JanuX first | AON JanuX, AON clustered, IQL JanuX, IQL clustered |
| 41 | Clustered first | AON clustered, AON JanuX, IQL clustered, IQL JanuX |
| 42 | JanuX first | AON JanuX, AON clustered, IQL JanuX, IQL clustered |

Generated data are stored under this study's `results/rciql_ing2_*` directories. Local
process logs are stored under ignored `.local/studies/road-clustering-improves-iql/`
and are not research evidence by themselves.

After the suites finish, run:

```bash
venv/bin/python studies/road-clustering-improves-iql/experiments/analyze_results.py
```

The analysis refuses to summarize the primary result until all 12 expected benchmark
metric files are present. The completed experiment is summarized in the study README.
