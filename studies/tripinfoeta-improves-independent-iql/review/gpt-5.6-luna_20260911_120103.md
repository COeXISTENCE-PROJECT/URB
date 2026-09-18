# Independent study review

## Reviewer

- **Model, provider, and version:** GPT-5.6 Luna, OpenAI, version unavailable (medium effort)
- **Review date:** 2026-09-11
- **Repository revision reviewed:** `57ebc3f8399dfb5af7f6f0d6cace00c2675a581f`
- **Role in the study:** none
- **Independent:** yes

I confirm that I did not design the study, change its code, run its experiments, choose its results, or help interpret it: **yes**.

## Knowledge of the repository

- **Important files inspected:** `studies/tripinfoeta-improves-independent-iql/PROPOSAL.md`, `FINAL_REPORT.md`, `REFERENCES.md`, `experiments/README.md`, `experiments/iql_eta_ablation.py`, `experiments/analyze_study.py`, `experiments/run_study.sh`, all three experiment configs, all six saved `exp_config.json` files, `experiments/evidence/{pair_results.csv,pair_checks.csv,run_metrics.csv,sensitivity_summary.csv,decision.json}`, all six `DailyMetrics.csv` and `BenchmarkMetrics.csv` files, `scripts/iql.py`, `analysis/metrics.py`, `networks/ingolstadt_custom2/agents.csv`, and the installed RouteRL 2.0.0 implementation at `venv/lib/python3.12/site-packages/routerl/environment/{observations.py,environment.py,agent.py}`. I also inspected `studies/README.md` and the other completed study as repository references.
- **How this experiment works in URB:** RouteRL first runs 300 human-learning days, then converts 40% of 1,035 agents to selfish AVs. Each AV independently selects one of four fixed routes with a one-step neural DQN/IQL model from `scripts/iql.py`. The dynamic condition feeds RouteRL's `TripInfoWithETA` observation unchanged; the control replaces only its first four route-ETA entries with the corresponding free-flow values before both action selection and learning. Each condition trains for 1,000 days and then runs 100 deterministic test days. The three dynamic/static pairs use matched environment and model seeds, and the static pair reuses the dynamic pair's generated route files.
- **My repository knowledge is sufficient for this review:** yes. The study code, saved configurations, raw aggregate metrics, pair checks, saved policies, route files, and RouteRL observation/environment code were available. I did not rerun the expensive experiments.

## Understanding of the hypothesis

- **Hypothesis in my own words:** In this one Ingolstadt demand and behavior setting, dynamic same-day route ETA information will make neural independent IQL produce at least a practically meaningful improvement in AV travel time over the otherwise identical free-flow-ETA control after the stated training budget.
- **Tested setting:** `ingolstadt_custom2`, `selfish_40`, 40% AVs, four JanuX routes, 300 human-learning days, 1,000 AV-learning days, 100 deterministic test days, and seed pairs 40, 41, and 42. The result is explicitly limited to this IQL implementation, network, demand, behavior, and budget.
- **Main metric and meaningful-effect threshold:** The mean over the three paired relative reductions in mean AV travel time across all 100 test days; dynamic must be at least 1% better.
- **Rules for support, falsification, and an inconclusive result:** Support requires mean improvement of at least 1% and all three pairs favoring dynamic ETA. Falsification requires mean improvement at most 0%, at least two pairs not improving, and all six runs/pair checks passing. Any other case is inconclusive. Training-time effects are assessed separately with the same 1% and unanimity logic.
- **I understand the hypothesis and how it is tested:** yes. The question is narrow but useful for URB because it tests whether a specific RouteRL observation signal helps a specific independent learner, while clearly separating performance sensitivity from performance benefit.

## Review

### Experiment design

The comparison is appropriate for the stated ablation. The static transform changes exactly the four ETA entries identified by RouteRL's `TripInfoWithETA`; origin, destination, start time, observation dimension, routes, reward, model, learning parameters, human phase, and test procedure are otherwise held fixed. The paired seed design is fair and the scope does not require more baselines. Three pairs are weak for estimating small effects, but the predefined 1% threshold and unanimity rule make that limitation explicit and appropriate for a focused study.

### Runs and evidence

All six named runs are present and complete. Each has 1,000 training and 100 testing rows in `DailyMetrics.csv`; the independently recomputed means match `experiments/evidence/pair_results.csv`. Pair checks are true for all seeds, including route CSV/XML hashes, saved human-learning episode hashes, AV IDs, and non-ETA settings. The observed test effects are −0.0886%, −0.3047%, and +0.2720%, with mean −0.0404%. The sensitivity analysis is consistent with ETA being used sometimes: it differs from free flow in about 52.4–54.4% of recorded AV decision observations and changes greedy actions about 6.1–7.0% of the time. These are mechanism checks, not substitutes for the paired outcome, correctly.

### Analysis and conclusion

The relative-effect calculations in `analyze_study.py` use the proposal's direction and denominator. The saved decision correctly classifies the test result as `falsified`: the mean is below zero, only seed 42 favors dynamic ETA, and all pairs are fair. The learning mean of −0.0347% is correctly classified as unclear/practically similar rather than as a meaningful disadvantage. The final report does not generalize beyond the tested setting and distinguishes policy sensitivity from demonstrated benefit. No confidence interval is supplied, so uncertainty remains limited to the three displayed paired effects; this is a limitation, not a contradiction of the declared lightweight decision rule.

### Strongest challenge

The strongest challenge is low statistical resolution: only three seed pairs are available, the observed effects are all far below the 1% practical threshold, and seed 42 favors dynamic ETA by +0.272%. Thus the result cannot rule out a small or setting-dependent benefit, and the policy-sensitivity evidence shows that the signal is not ignored. It nevertheless does support the narrower conclusion that the predefined meaningful, consistent improvement was not observed in this setting.

### Serious problems

None.

## Score

| Area | Score | Reason |
| --- | ---: | --- |
| Relevant question with potential value for URB | 3/4 | A focused observation-ablation question with clear downstream relevance to RouteRL/IQL studies. |
| Clear hypothesis, scope, and decision rules | 3/4 | Thresholds, pairing, separate learning outcome, and limits are explicit. |
| Fair comparison and enough evidence for the stated scope | 3/4 | The four ETA entries are isolated and all three matched pairs pass checks; precision is intentionally limited. |
| Complete evidence and sufficiently reproducible setup | 3/4 | Configs, scripts, routes, policies, aggregate metrics, and compact checks are supplied; the exact external machine environment is not fully captured. |
| Correct analysis and honest treatment of uncertainty | 2/4 | Calculations and reporting are correct, but three pairs provide only limited uncertainty information and no interval estimate. |
| Conclusion matches the evidence and tested setting | 3/4 | The falsification follows the proposal's rule and is appropriately scoped. |
| **Total** | **17/24** |  |

Scores of 12–24 normally support `accept`, and this study has no serious problem that overrides that guidance.

## Verdict

**Accept — 17/24.**

The study is clear and trustworthy enough to display as a focused URB result. It provides a fair, reproducible paired ablation and a correctly scoped negative conclusion; its small seed count is openly reflected in the decision rules and limitations.

### Required changes

None.

### Optional follow-up

Test normalized ETA inputs, a longer training budget, or another independent learner as separate studies. More seed pairs would improve resolution around the small effects observed here.
