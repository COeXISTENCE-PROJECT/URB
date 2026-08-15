# Results

This directory contains evidence produced by the procedures in `../experiments/`.

Use one subdirectory per run when retaining standard URB outputs:

```text
results/
├── README.md
└── <run-id>/
    ├── exp_config.json
    ├── metrics/BenchmarkMetrics.csv
    └── plots/
```

Keep the compact configuration, primary metrics, and only plots needed to audit the
conclusion. Raw episodes, SUMO output, combined data, losses, and diagnostic plots may
remain locally for follow-up analysis but should stay ignored by Git.

Summarize the important results in the study's main `README.md`; this directory is the
supporting evidence, not the interpretation.
