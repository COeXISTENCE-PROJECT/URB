# Independent study review

## Reviewer

- **Model, provider, and version:** Gemini 3.8 Flash (High), Google, version unavailable
- **Review date:** 2026-09-11
- **Repository revision reviewed:** `57ebc3f8399dfb5af7f6f0d6cace00c2675a581f` (HEAD), plus uncommitted study files
- **Role in the study:** none
- **Independent:** yes

I confirm that I did not design the study, change its code, run its experiments, choose its results, or help interpret it: **yes**.

## Knowledge of the repository

- **Important files inspected:**
  - `studies/tripinfoeta-improves-independent-iql/PROPOSAL.md`
  - `studies/tripinfoeta-improves-independent-iql/FINAL_REPORT.md`
  - `studies/tripinfoeta-improves-independent-iql/REFERENCES.md`
  - `studies/tripinfoeta-improves-independent-iql/experiments/README.md`
  - `studies/tripinfoeta-improves-independent-iql/experiments/run_study.sh`
  - `studies/tripinfoeta-improves-independent-iql/experiments/iql_eta_ablation.py`
  - `studies/tripinfoeta-improves-independent-iql/experiments/analyze_study.py`
  - `studies/tripinfoeta-improves-independent-iql/experiments/configs/{algorithm.json,environment.json,task.json}`
  - `studies/tripinfoeta-improves-independent-iql/experiments/evidence/{decision.json,pair_checks.csv,pair_results.csv,run_metrics.csv,sensitivity_summary.csv,sensitivity_by_agent.csv}`
  - Saved run directories in `experiments/results/` (`exp_config.json`, `DailyMetrics.csv`, `BenchmarkMetrics.csv`, `routes.csv`, `route.rou.xml`, `agents.csv`, `models/policies.pt`)
  - Execution logs in `experiments/logs/`
  - Repository baseline code and RouteRL environment: `scripts/iql.py`, `analysis/metrics.py`, `networks/ingolstadt_custom2/agents.csv`, and RouteRL's observation implementation in `venv/lib/python3.12/site-packages/routerl/environment/observations.py` (`TripInfoWithETA`)
  - Existing reviews and reference studies under `studies/independent-tabular-iql-cannot-beat-humans-provins/` and `studies/tripinfoeta-improves-independent-iql/review/gpt-5.6-luna_20260911_120103.md`
- **How this experiment works in URB:** In the `ingolstadt_custom2` network (1,035 agents), RouteRL first simulates 300 days of Gawron human learning with 4 candidate JanuX routes. At day 300, 40% of the population (414 agents) are converted to autonomous vehicles (CAVs) driven by independent neural DQN agents (two 64-unit hidden layers, replay buffer of 2048, Adam optimizer with lr=0.001), while human learning is frozen. The dynamic condition feeds RouteRL's `TripInfoWithETA` observation, whose first 4 dimensions contain the moving average of same-day travel times from earlier departures sharing the same OD pair. The static control isolates this signal by using `study_observation()` to overwrite exactly those 4 entries with static free-flow travel times, keeping origin, destination, start time, route definitions, and seeds identical. The static runs copy `routes.csv` and `route.rou.xml` from the dynamic runs to ensure identical action-to-path mappings. Both conditions train for 1,000 days (epsilon decaying from 1.0 to 0.01) and evaluate for 100 test days with epsilon fixed to 0.0.
- **My repository knowledge is sufficient for this review:** yes. The study proposal, execution scripts, run configurations, raw episode outputs, derived metrics, verification scripts, and underlying RouteRL observation classes were directly inspected and verified.

## Understanding of the hypothesis

- **Hypothesis in my own words:** For selfish 40% CAV penetration in the `ingolstadt_custom2` network, supplying neural independent IQL agents with dynamic same-day route ETA information in `TripInfoETA` will improve average CAV test travel times by at least 1% compared to supplying static free-flow route times, after 1,000 training days.
- **Tested setting:** `ingolstadt_custom2`, `selfish_40` demand (414 CAVs, 621 frozen HDVs), four JanuX routes, neural IQL, 300 human-learning days, 1,000 CAV training days, 100 deterministic test days, and three matched seed pairs (40, 41, 42).
- **Main metric and meaningful-effect threshold:** The mean paired relative reduction in CAV test travel time across all 100 test days: `100 * (t_CAV_static - t_CAV_dynamic) / t_CAV_static`. The meaningful-effect threshold is +1.0% (positive indicates dynamic ETA outperforms static free-flow).
- **Rules for support, falsification, and an inconclusive result:**
  - Support: Mean paired test improvement >= 1.0% and all 3 seed pairs favor dynamic ETA (> 0%).
  - Falsification: Mean paired test improvement <= 0.0%, at least 2 of 3 seed pairs do not improve (<= 0%), and all runs and pairing checks complete normally.
  - Inconclusive: Any intermediate outcome failing both criteria, or any broken pairing / corrupted run.
  - Learning effect: Assessed separately across all 1,000 training days using the same +1% unanimous rule for "better", -1% unanimous for "worse", and "unclear or practically similar" otherwise.
- **I understand the hypothesis and how it is tested:** yes. The question is focused and useful for URB: it tests whether a real-time observation feature in RouteRL actually provides actionable value to independent decentralized learners or merely adds non-stationary noise.

## Review

### Experiment design

The experimental design is well-controlled and directly isolates the target variable. The ablation in `iql_eta_ablation.py` alters only the 4 route ETA values in the agent observation vector via `study_observation()`, preserving the input dimensionality, static features (origin, destination, departure time), reward function, learning dynamics, and neural architecture. Copying `routes.csv` and `route.rou.xml` from dynamic runs to their static counterparts guarantees that the action index corresponds to the identical physical route in SUMO. The use of matched seed pairs (40, 41, 42) with independent RNG streams for agent exploration and replay prevents simulation divergence while maintaining identical initializations. A sample size of three seed pairs is modest, but the predefined 1% effect threshold, unanimity requirement, and explicit falsification criteria make this design well-suited for a lightweight, focused benchmark study.

### Runs and evidence

All six specified runs are present, complete, and uncorrupted in `experiments/results/`. Each run contains the complete 1,000 training and 100 testing records in `DailyMetrics.csv`, serialized PyTorch policies in `models/policies.pt`, and associated SUMO outputs. The automated pair checks in `experiments/evidence/pair_checks.csv` confirm that route files, route XMLs, human pre-training episode logs, CAV assignments, and non-ETA hyperparameters match identically across all three pairs (`fair_pair = True`).

The empirical results match the reported numbers exactly:
- Test relative effect (`100 * (static - dynamic) / static`): -0.089% (seed 40), -0.305% (seed 41), and +0.272% (seed 42), yielding a mean paired effect of -0.040%.
- Learning relative effect: -0.052% (seed 40), -0.064% (seed 41), and +0.012% (seed 42), with a mean of -0.035%.
- Descriptive early (first 250 days: mean -0.052%) and late (final 250 days: mean -0.058%) training windows confirm the absence of an emerging learning advantage.
- The sensitivity analysis (`sensitivity_summary.csv`) confirms that dynamic ETA was functionally active: test ETA values differed from free-flow in 53.2% of decision steps, shifting greedy policy actions 6.6% of the time (up to 9.7% for middle departures), which rules out the possibility that the feature was completely ignored due to numerical scale or dead weights.

### Analysis and conclusion

The calculations in `analyze_study.py` are correct and reproduce all figures in the final report. Because the mean paired test improvement is negative (-0.040%), two of the three seed pairs fail to improve (seeds 40 and 41), and all pairing checks passed, the study strictly fulfills its preregistered falsification criterion. The training assessment is correctly classified as "unclear or practically similar" rather than worse, given that the mean difference (-0.035%) falls well within the [-1%, +1%] practical indifference zone. The final report avoids unwarranted generalizations, appropriately restricting its claims to independent neural IQL under the tested 1,000-day budget and specific network setting.

### Strongest challenge

The primary limitation is statistical resolution: with only three seed pairs, seed 42 exhibited a slight positive effect (+0.272%), and the sensitivity check confirmed that ETA does alter greedy action selection in 6.6% of test instances. Consequently, the data cannot rule out a subtle (<0.5%) positive effect or a scenario where longer training or specialized feature normalization enables agents to exploit intra-day ETA. Nevertheless, this does not weaken the study's central conclusion, which was specifically framed around achieving a consistent >=1% practical improvement within a 1,000-day budget.

### Serious problems

None.

## Score

Score the study as a focused contribution to the URB website, not as a publication. Use 0 for missing or unusable, 1 for a substantial weakness, 2 for adequate within the stated scope, 3 for strong, and 4 for exceptional. A sound small study should normally receive 2s; 4s are not expected.

| Area | Score | Reason |
| --- | ---: | --- |
| Relevant question with potential value for URB | 3/4 | Investigates whether a standard RouteRL observation feature provides practical utility to decentralized learners, offering direct architectural guidance for future benchmark tasks. |
| Clear hypothesis, scope, and decision rules | 4/4 | Predefined effect thresholds (+1%), explicit three-pair consistency rules, separate learning-curve criteria, and well-bounded scope. |
| Fair comparison and enough evidence for the stated scope | 3/4 | Meticulous controls (exact route copying, identical seed schedules, isolated ETA replacement) with adequate paired runs for the declared decision rules. |
| Complete evidence and sufficiently reproducible setup | 3/4 | Fully intact run directories, configuration files, daily episode metrics, policy checkpoints, and pairing validation artifacts. |
| Correct analysis and honest treatment of uncertainty | 3/4 | Computations are fully verified, seed variation is transparently displayed, and policy sensitivity is audited without overstating the negative result. |
| Conclusion matches the evidence and tested setting | 4/4 | Falsification directly adheres to the preregistered decision rules and remains strictly within the tested environment and training horizon. |
| **Total** | **20/24** |  |

Scores of 12–24 normally support `accept`, 7–11 support `revise`, and 0–6 support `reject`. `Accept` means the study is clear and trustworthy enough to display, not that it is exhaustive or publication-ready. `Revise` means the existing work may become acceptable through specific corrections. `Reject` means the central evidence or conclusion is not trustworthy enough to display.

A serious problem prevents `accept` regardless of the total. Examples include an unfair comparison, missing central results, undisclosed result selection, or an analysis error that could change the conclusion. A clearly stated limitation that fits the study's narrow scope is not a serious problem by itself.

## Verdict

**Accept — 20/24.**

The study is clear, rigorously controlled, and trustworthy. The empirical evidence cleanly falsifies the hypothesis that dynamic `TripInfoETA` provides a >=1% performance improvement for independent neural IQL in the tested Ingolstadt setting, and the report remains disciplined and honest regarding its limitations.

### Required changes

None.

### Optional follow-up

- Evaluate feature normalization or scaling for ETA inputs, as raw travel times (order of 10^2 - 10^3) are mixed with categorical origin/destination IDs and start times without batch normalization or standard scaling.
- Test longer training budgets (>1,000 episodes) or coordinated multi-agent architectures (e.g., QMIX or MAPPO) to evaluate whether multi-agent non-stationarity was the primary bottleneck hindering effective use of dynamic ETA signals.
