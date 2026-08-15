# Study title

## Metadata

- **Status:** proposed
- **Started:** YYYY-MM-DD
- **Last updated:** YYYY-MM-DD
- **Primary method:**
- **Comparison:**

## Hypothesis

State one falsifiable claim in one or two sentences.

## Motivation and mechanism

Explain why the effect should occur. Identify the mechanism rather than only predicting
that one method will have a better aggregate score.

## Scope

State the networks, tasks, AV behaviors, observations, route construction, and training
regimes to which the claim applies. Record important exclusions.

## Predictions

List observable predictions, including at least one result that distinguishes the
proposed mechanism from a plausible alternative explanation.

## Falsification criteria

State what evidence would reject or materially weaken the hypothesis. Define the primary
metric and meaningful effect before inspecting the target results.

## Experiment plan

Describe the controlled comparison, seed schedule, training budget, evaluation policy,
metrics, and planned analysis. Put detailed commands and study-specific analysis code
under `experiments/`. Keep algorithm implementations in their canonical URB locations.

| Experiment | Network/task | Method | Seeds | Purpose | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  | planned |  |

## Evidence

Summarize the observations and link to exact files under `results/`. Include failed,
interrupted, and contradictory runs where they affect interpretation. Keep conclusions
here; keep the supporting artifacts in `results/`.

## Interpretation

Explain what the evidence shows, what remains uncertain, and which alternative
explanations remain viable.

## Conclusion

Use one of: `supported`, `falsified`, or `inconclusive`, always qualified by the stated
scope. Summarize the practical consequence for URB.

## Follow-up work

List only concrete experiments or implementation changes motivated by the conclusion.
