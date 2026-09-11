# URB studies

This directory contains focused studies carried out with URB. Each study starts with a clear hypothesis, supplies the experiments used to test it, gives the authors' conclusion, and includes at least one independent review.

Each directory other than `_template/` represents one study.

## Study structure

Create a study by copying `_template/` to a short lowercase name.

```text
studies/<study-name>/
├── COVER.png             # Optional study cover image
├── PROPOSAL.md            # Hypothesis and experiment plan
├── FINAL_REPORT.md        # Authors' assessment and conclusion
├── REFERENCES.md          # Relevant papers and sources
├── experiments/
│   ├── README.md          # What was run and why
│   └── <supplied runs>/   # Compact experiment evidence
└── review/
    ├── PROMPT.md          # Prompt for independent reviewers
    ├── REVIEW.md          # Review template
    └── <completed reviews>
```

## Process

1. Copy `_template/` and complete `PROPOSAL.md` before running the study.
2. Optionally add a cover image as `COVER.png` in the study's top-level directory. Use an image that represents the study and record its source in `REFERENCES.md`.
3. Run the planned experiments. Keep the exact configurations, main metrics, and other evidence needed to check the result under `experiments/`.
4. Describe the supplied runs in `experiments/README.md`. Include failed or excluded runs when they affect the evidence.
5. Complete `FINAL_REPORT.md`. Apply the rules from the proposal and keep the conclusion within the tested setting.
6. Give `review/PROMPT.md` to a fresh LLM that did not help design, run, or interpret the study.
7. Save each completed review under `review/` using the required model and timestamp filename.

## Study rules

- Test one main hypothesis and explain why it may be true.
- Decide the main metric, meaningful effect, seeds, and possible conclusions before seeing the new results.
- Keep important settings fixed unless they are part of the question. Use matched seeds for compared conditions when possible.
- Supply the full range of relevant results rather than selecting only favourable runs.
- Show variation across seeds, not only averages. Hyperparameter candidates do not count as repeat runs.
- Clearly label analysis added after seeing the results.
- Keep reusable algorithms and configurations in their normal URB directories. The study should contain only study-specific material and compact experiment evidence.
- After a study is complete, remove run launchers, status files, logs, caches, and full simulator output. Keep the exact settings, study-specific code, per-seed results, and compact evidence needed to audit the conclusion.
- A study may conclude `supported`, `falsified`, or `inconclusive`. None of these claims should extend beyond the setting that was tested.
- Independent review judges whether a focused study is clear and trustworthy enough to display. It does not require publication-level breadth, novelty, or statistical power beyond the stated scope.

## Stable format

The studies will later be read automatically for the URB website. Keep the filenames, headings, field names, score table, and allowed conclusion and review values from `_template/` unchanged. Fill in placeholders rather than restructuring individual studies. The optional cover must be named `COVER.png`. New optional sections may be added after the standard sections.
