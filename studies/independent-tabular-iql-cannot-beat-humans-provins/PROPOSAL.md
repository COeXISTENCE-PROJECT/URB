# Hypothesis proposal

This is a retrospective demonstration study. It uses completed results already in URB and does not present its decision rules as preregistered.

## Study details

- **Study:** Tabular IQL cannot beat learned human routing in Provins
- **Designed by:** Onur Akman and GPT-5.6 Sol
- **Date proposed:** 2026-09-09
- **URB revision:** `57ebc3f`, with later local documentation changes
- **Keywords:** provins, tabularql, 40%, selfish

## Hypothesis

**In Provins with 40% selfish CAVs, tabular IQL cannot consistently achieve meaningfully shorter test travel times than the learned pre-mutation human routing baseline after 800 CAV-learning days.**

### Reason

Independent learners act in a congestion game where every agent changes the travel times experienced by others. This non-stationarity may prevent independently learned policies from improving on experienced human routing.

### Main alternative explanation

Each agent has a fixed trip and only four route choices. This small decision problem may let independent tabular learners discover stable routes that are better than the human baseline.

## Scope

- **Network and demand:** `provins` with its recorded 523-agent demand
- **Task and CAV behaviour:** `selfish_40`; humans learn for 300 days and do not adapt during CAV learning
- **CAV share:** 40% (209 CAVs)
- **Methods and observations:** Tabular IQL using `trip_info_eta`, four routes, and `bin_width=20`
- **Training and testing:** 800 CAV-learning days and 100 test days
- **Not covered:** Other networks, methods, hyperparameters, CAV shares, behaviours, or training budgets

## How the result will be judged

- **Main metric:** `Effect_of_change = t_HDV_pre / t_CAV`, reported separately for each seed
- **Smallest meaningful effect:** `Effect_of_change >= 1.02`
- **Support the hypothesis if:** The median ratio is at most 1.00 and no more than one of four runs reaches 1.02
- **Falsify the hypothesis if:** The median ratio is at least 1.02 and at least three of four runs reach 1.02
- **Call the result inconclusive if:** Neither rule is met or the four runs are not comparable

Other metrics may explain the result but do not change this decision rule.

## Experiment design

- **Changed factor:** Independently learned CAV routing compared with the learned human baseline recorded within each run
- **Settings kept fixed:** Algorithm, configurations, demand, task, observation, routes, training length, and test length
- **Environment seeds:** 40, 41, 42, and 43
- **Model seeds:** 40, 41, 42, and 43, paired with the matching environment seed
- **Number of repeat runs and why it is enough:** Four runs are sufficient for this demonstration because falsification requires a meaningful effect in at least three and a meaningful median effect
- **Reasons a run may be excluded:** Missing configuration or metrics, failed execution, or a setting that differs from the stated family

## Run limit

No new runs. The study uses all four existing `tbq_prv_short_s40` through `tbq_prv_short_s43` results.

## Known limits

The hypothesis and rules were written after the results existed. The configuration came from earlier tuning, and environment and model seeds share the same numeric value. This study can reject the stated impossibility claim in this setting, but it cannot establish that tabular IQL usually wins in Provins.
