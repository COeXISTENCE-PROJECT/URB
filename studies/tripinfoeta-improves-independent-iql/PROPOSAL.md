# Hypothesis proposal

Fill in this file before running the study. This proposal, combined with the supplied evidence and final report, will determine the quality of the study and therefore its inclusion in URB.

## Study details

- **Study:** Dynamic TripInfoETA values improve IQL
- **Designed by:** Onur Akman and GPT-5.6 Sol
- **Date proposed:** 2026-09-09
- **URB revision:** `57ebc3f`, before adding the study-specific experiment script
- **Keywords:** tripinfoeta, observations, iql, dqn, ingolstadt

## Hypothesis

**In `ingolstadt_custom2` with selfish 40% CAVs, the dynamic route ETA values in `TripInfoETA` allow neural IQL to achieve at least 1% lower mean CAV test travel time than the same learner receiving fixed free-flow route values after 1,000 training days.**

### Learning-performance question

The study will also test whether dynamic ETA reduces CAV travel time during learning. This matters because useful information may lower the cost of exploration or help IQL learn faster even when the final policies are similar. Training performance includes all exploratory actions; it is not a deterministic-policy evaluation.

### Reason

Each IQL agent has a fixed origin, destination, and start time. The ETA values are therefore its only changing input. They summarize same-day travel times from earlier vehicles with the same OD and may help later agents avoid routes that are currently slow.

### Main alternative explanation

Each agent may only need to learn a good fixed route. The ETA values may be too sparse, noisy, or unstable in a learning population to improve decisions.

## Scope

- **Network and demand:** `ingolstadt_custom2` and its current `agents.csv`
- **Task and CAV behaviour:** `selfish_40`; humans learn for 300 days and are then frozen
- **CAV share:** 40%
- **Methods and observations:** Neural IQL based on `scripts/iql.py`; four JanuX routes; normal `TripInfoETA` versus the same observation with its ETA values fixed to route free-flow times
- **Training and testing:** 1,000 AV-learning days and 100 deterministic-policy test days
- **Not covered:** Other networks, CAV shares or behaviours, clustered routes, other independent learners, or longer training budgets

## How the result will be judged

- **Main metric:** Paired relative reduction in mean CAV travel time across all 100 test days: `100 × (t_CAV_static − t_CAV_dynamic) / t_CAV_static`, averaged over the three seed pairs
- **Smallest meaningful effect:** 1% lower mean CAV test travel time
- **Learning metric:** Paired relative reduction in mean CAV travel time across all 1,000 training days, equivalent to the normalized area under the CAV learning curve
- **Smallest meaningful learning effect:** 1% lower mean CAV training travel time
- **Support the hypothesis if:** The mean paired improvement is at least 1% and all three pairs favour dynamic ETA
- **Falsify the hypothesis if:** The mean paired improvement is at most 0%, at least two of three pairs do not improve, and all six runs and pairing checks completed normally
- **Call the result inconclusive if:** Neither rule is met or a fair paired comparison cannot be completed

The test metric determines whether the main hypothesis is supported. Learning performance is assessed separately: it is **better** if its mean paired improvement is at least 1% and all three pairs favour dynamic ETA, **worse** if the mean is at most −1% and all three pairs favour fixed ETA, and **unclear or practically similar** otherwise. This separation will show whether the final result comes with a learning-time benefit or trade-off without combining two different outcomes into one score.

The first and final 250 training days will be reported descriptively to distinguish early learning from later convergence. These descriptive results do not change the predefined assessments.

## Experiment design

- **Changed factor:** Whether the four route ETA entries are updated from current-day trips or remain at their free-flow values
- **Settings kept fixed:** Script structure, network, demand, route generation, observation size, constant observation entries, model architecture, reward, learning parameters, human learning, test policy, and metric calculation
- **Environment seeds:** 40, 41, and 42, matched between conditions and passed to RouteRL
- **Model seeds:** 40, 41, and 42 respectively, matched between conditions and used for network initialization, exploration, and replay sampling
- **Number of repeat runs and why it is enough:** Three matched pairs can support a lightweight conclusion when the effect is practically meaningful and consistent; mixed or small effects will be called inconclusive
- **Reasons a run may be excluded:** Failure to finish, corrupt or missing output, or a verified implementation error unrelated to the tested condition. Teleports and poor performance are not exclusion reasons

## Run limit

Six confirmatory training runs: two conditions for each of three seed pairs. A failed technical run may be repeated once with the same seeds. Network-sensitivity analysis uses these trained models and does not add training runs.

## Known limits

The study tests one IQL implementation at a deliberately short training budget. Its raw observation also mixes route travel times with much larger fixed origin, destination, and departure values. A negative result will therefore show that this implementation did not benefit from ETA within 1,000 days, not that ETA has no predictive value or cannot help another learner or a longer run. Three seed pairs can identify a clear effect but will leave a small or inconsistent effect unresolved.
