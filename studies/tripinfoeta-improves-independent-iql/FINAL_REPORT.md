# Final report

## Study details

- **Study:** Dynamic TripInfoETA values improve IQL
- **Authors:** Onur Akman and GPT-5.6 Sol
- **Report date:** 2026-09-11
- **Proposal:** [PROPOSAL.md](PROPOSAL.md)

## Short answer

**Falsified in the tested setting.** Dynamic route ETA produced a mean paired change of −0.040% in CAV test travel time, with only one of three seed pairs favouring it. It also provided no meaningful learning-time benefit.

## Hypothesis and scope

- **Hypothesis:** In `ingolstadt_custom2` with selfish 40% CAVs, the dynamic route ETA values in `TripInfoETA` allow neural IQL to achieve at least 1% lower mean CAV test travel time than the same learner receiving fixed free-flow route values after 1,000 training days.
- **Tested setting:** `ingolstadt_custom2`, `selfish_40`, 40% CAVs, neural IQL, four JanuX routes, three matched environment/model seed pairs, 1,000 CAV-training days, and 100 deterministic test days
- **Main metric:** Paired relative reduction in mean CAV travel time over all 100 test days
- **Smallest meaningful effect:** 1%

## Evidence

- **Supplied runs:** Six runs: dynamic and fixed-ETA conditions for seeds 40, 41, and 42
- **Main observed effect:** −0.040% mean paired improvement; the seed effects ranged from −0.305% to +0.272%

Positive effects favour dynamic ETA.

| Seed | Dynamic test `t_CAV` | Fixed test `t_CAV` | Test effect | Learning effect | First 250 days | Final 250 days |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 4.232 | 4.229 | −0.089% | −0.052% | −0.007% | −0.059% |
| 41 | 4.389 | 4.376 | −0.305% | −0.064% | −0.114% | −0.086% |
| 42 | 4.320 | 4.332 | +0.272% | +0.012% | −0.035% | −0.030% |
| **Mean** | **4.314** | **4.312** | **−0.040%** | **−0.035%** | **−0.052%** | **−0.058%** |

The learning assessment is **unclear or practically similar**: its mean effect was −0.035%, far below the 1% threshold, and the pairs were not unanimous. The first and final training windows likewise show no emerging advantage.

Standard URB metrics tell the same practical story:

| Condition | `t_HDV_test` | `t_test` | CAV advantage | Cost of learning |
| --- | ---: | ---: | ---: | ---: |
| Dynamic ETA | 4.478 | 4.412 | 1.038 | 1.890 |
| Fixed ETA | 4.483 | 4.415 | 1.040 | 1.897 |

The dynamic signal was present and affected the learned networks. Across all recorded CAV decisions, ETA differed from free-flow values 53.2% of the time. Replacing it with free-flow values changed the dynamic policies' greedy action 6.6% of the time, with a mean absolute Q-value change of 0.043.

| Departure group | ETA differs | Greedy action changes | Mean absolute Q change |
| --- | ---: | ---: | ---: |
| Early | 22.8% | 2.6% | 0.011 |
| Middle | 65.7% | 9.7% | 0.055 |
| Late | 71.2% | 7.5% | 0.064 |

All three pair checks passed: route CSV and XML files, recorded human-learning episodes, selected CAV IDs, and settings other than the tested ETA mode matched exactly. The compact evidence is under [`experiments/evidence/`](experiments/evidence/).

## Assessment

### Does the evidence support the hypothesis?

No. The proposal defines falsification as a mean effect at or below 0%, at least two pairs not improving, and all runs and pairing checks completing normally. The mean was −0.040%, seeds 40 and 41 did not improve, and every completion and pairing check passed. The hypothesis is therefore falsified in the tested setting.

### Evidence against this conclusion

Seed 42 favoured dynamic ETA by 0.272%, and the sensitivity analysis confirms that ETA sometimes changes the policy's decision. However, this one small positive result does not meet the predefined 1% effect or consistency requirements. It shows that the networks can respond to ETA, not that doing so improves performance.

### Other possible explanations

A good fixed route may be sufficient for these agents, while same-day ETA may be too noisy or non-stationary to improve routing. The raw observation also mixes ETA with much larger fixed values, which may make the signal harder to use. These explanations were not separated by this study. The result therefore does not show that ETA lacks predictive information or that another representation, learner, network, CAV share, or longer budget could not benefit from it.

### Changes and problems

An earlier launch was interrupted and its partial artifacts were discarded. The full six-run suite was then restarted with the original design and seeds; no partial result entered the analysis. No completed run was excluded or repeated selectively. The logs contain repeated benign PROJ database warnings but no traceback, failed run, or incomplete output.

## Conclusion

For neural IQL with selfish 40% CAVs in `ingolstadt_custom2`, dynamic `TripInfoETA` values did not improve deterministic test performance after 1,000 training days and did not reduce the cost of learning. The policies were sensitive to the values, especially for middle and late departures, but those changed decisions produced no measurable benefit over fixed free-flow route values.

## What URB should do with this result

Do not treat dynamic `TripInfoETA` as an established performance advantage for this IQL setting. Keep it available for compatibility and further observation studies, but do not prefer it over the fixed-information control on the strength of this experiment.

## Further work

- **Needed before acting:** None for the scoped conclusion.
- **Useful but optional:** Test normalized ETA inputs, longer training, another independent learner, or another network as separate hypotheses.

## Author check

- [x] The results were supplied faithfully without cherrypicking favorable ones.
- [x] Unplanned analyses are labelled as exploratory.
- [x] The conclusion does not go beyond the tested setting.
- [x] The report is ready for independent review.
