# Final report

## Study details

- **Study:** Tabular IQL cannot beat learned human routing in Provins
- **Authors:** Onur Akman and GPT-5.6 Sol
- **Report date:** 2026-09-09
- **Proposal:** [PROPOSAL.md](PROPOSAL.md)

## Short answer

**Falsified in the tested setting.** All four tabular-IQL runs produced meaningfully shorter CAV travel times than their learned human baselines.

## Hypothesis and scope

- **Hypothesis:** In Provins with 40% selfish CAVs, tabular IQL cannot consistently achieve meaningfully shorter test travel times than the learned pre-mutation human routing baseline after 800 CAV-learning days.
- **Tested setting:** `provins`, `selfish_40`, 40% CAVs, tabular IQL with `trip_info_eta`, 800 CAV-learning days, and 100 test days
- **Main metric:** `Effect_of_change = t_HDV_pre / t_CAV` for each seed
- **Smallest meaningful effect:** `Effect_of_change >= 1.02`

## Evidence

- **Supplied runs:** `tbq_prv_short_s40`, `tbq_prv_short_s41`, `tbq_prv_short_s42`, and `tbq_prv_short_s43`
- **Main observed effect:** All four ratios exceed 1.02; the median is 1.083 and the range is 1.041–1.124

| Run | `t_HDV_pre` | `t_CAV` | `Effect_of_change` |
| --- | ---: | ---: | ---: |
| `tbq_prv_short_s40` | 2.998 | 2.880 | 1.041 |
| `tbq_prv_short_s41` | 3.093 | 2.751 | 1.124 |
| `tbq_prv_short_s42` | 3.047 | 2.779 | 1.097 |
| `tbq_prv_short_s43` | 3.027 | 2.830 | 1.070 |

The direction is consistent across all four seed pairs and every run clears the meaningful-effect threshold.

## Assessment

### Does the evidence support the hypothesis?

No. The proposal requires at least three of four ratios and the median ratio to reach 1.02 for falsification. All four ratios do so, and the median is 1.083. The hypothesis is therefore falsified in the tested setting.

### Evidence against this conclusion

The weakest run, seed 40, has a smaller effect than the others. Its ratio is still 1.041, above the predefined threshold. Other independent-learning configurations may perform worse, but they do not restore the scoped claim that this method cannot outperform the human baseline.

### Other possible explanations

The tuned hyperparameters, coarse observation binning, frozen humans, or short training budget may help this particular method. These factors limit generalization but are part of the stated setting. Comparing each run with its own pre-mutation baseline reduces concern that the result is only caused by differences between environment seeds.

### Changes and problems

This was intentionally retrospective: no new runs were made, and the decision rules were written after the data existed. All four results in the selected run family were included; none failed or was excluded.

## Conclusion

Tabular IQL can outperform learned human routing in the tested Provins task. This rejects the narrow impossibility claim, but it does not show that independent learning will outperform humans across other configurations or tasks.

## What URB should do with this result

Keep tabular IQL as a credible Provins baseline and reject claims that independent learning is inherently unable to beat human routing there.

## Further work

- **Needed before acting:** None for rejecting the scoped impossibility claim.
- **Useful but optional:** Prospectively repeat the comparison with independently chosen environment and model seeds and other CAV shares.

## Author check

- [x] The results were supplied faithfully without cherrypicking favorable ones.
- [x] Unplanned analyses are labelled as exploratory.
- [x] The conclusion does not go beyond the tested setting.
- [x] The report is ready for independent review.
