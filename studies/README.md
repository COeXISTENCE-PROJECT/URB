# URB studies

This directory organizes focused research studies built with URB. Each study begins
with a falsifiable hypothesis and collects the evidence needed to support, reject, or
refine it.

A study is not a replacement for URB's algorithm structure. Implementations remain in
`scripts/` and `baseline_models/`, and configurations remain in `config/`. Focused
study outputs may live in that study's `results/` directory so its evidence stays
self-contained; shared benchmark outputs remain in the repository-level `results/`.

The direction is intentionally simple: **experiments produce results; the study README
interprets those results**.

## Study registry

| Study | Hypothesis | Status | Conclusion |
| --- | --- | --- | --- |
| [Road clustering improves IQL](road-clustering-improves-iql/) | Diverse route representatives improve IQL route choice | falsified | Current clustered routes worsened IQL AV time by 7.54%; use JanuX as the tabular-IQL base. |

Suggested statuses are `proposed`, `active`, `supported`, `falsified`, `inconclusive`,
and `archived`. Supported and falsified describe the evidence under the stated scope;
they are not claims of universal proof.

## Directory structure

Create a study by copying `_template/` to a short, descriptive, lowercase name. For
example:

```text
studies/
├── README.md
├── _template/
└── road-clustering-improves-iql/
    ├── README.md
    ├── experiments/
    ├── notes/
    ├── papers/
    └── results/
```

The study `README.md` is the authoritative research record. It should state:

- the hypothesis and plausible mechanism;
- the scope in which the claim is expected to hold;
- predictions that distinguish it from competing explanations;
- evidence that would falsify or weaken it;
- controlled experiment comparisons, seeds, and metrics;
- links to exact scripts, configurations, and result directories;
- the analysis and final conclusion, including negative results.

The supporting directories have distinct roles:

| Directory | Contains | Does not contain |
| --- | --- | --- |
| `experiments/` | Design, run matrix, commands, launchers, analysis code | Generated data or algorithm implementations |
| `results/` | Per-run configurations, primary metrics, selected plots, derived summaries | Experiment plans or shared algorithm code |
| `notes/` | Decisions, anomalies, interpretation changes, follow-up ideas | Private or machine-specific information |
| `papers/` | Bibliography and concise relevance notes | Unlicensed or unnecessary PDF copies |

Raw episodes, SUMO output, combined data, losses, and diagnostic plots may live locally
under a study's `results/`, but should remain ignored. Retain only compact evidence
needed to audit the conclusion in Git.

Additional directories such as `figures/` or `data/` should be added only when the
study actually needs them. Prefer small, derived, reproducible artifacts over copied
raw outputs.

## Research expectations

### Formulate before running

Write the hypothesis, primary metric, important controls, and falsification criterion
before examining the target results. Exploratory work is welcome, but it should be
identified as exploratory rather than presented as confirmatory.

### Change one scientific factor at a time

An algorithm comparison should keep the network, demand, route set, observation,
reward, AV population, training budget, and evaluation procedure fixed unless one of
those factors is the subject of the study. Factorial studies should list every factor
and interaction intentionally examined.

### Preserve pairing and uncertainty

Use the same scenario and seed schedule for compared methods where possible. Record
failed and interrupted runs. Report per-run values and uncertainty alongside aggregate
metrics; hyperparameter variants are not independent repetitions.

### Separate evidence from interpretation

Tables and plots should identify their source results. The conclusion should distinguish
observed outcomes from proposed explanations and document important alternative
explanations that remain unresolved.

### Finish with a decision

A completed study should say what URB should do next: adopt a method, reject it, narrow
the claim, change the benchmark, or run a specific follow-up study. Inconclusive studies
are valid outcomes when the reason is recorded.

## Recommended workflow

1. Copy `_template/` and register the study in the table above.
2. Complete the hypothesis, scope, predictions, and falsification sections.
3. Define the smallest experiment matrix under the study's `experiments/` directory.
4. Run it through existing URB scripts and write outputs under the study's `results/`.
5. Analyze all planned runs, including failures, and link the exact evidence.
6. Write the conclusion in the study README and update the registry status.
