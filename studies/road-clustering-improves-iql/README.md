# Road clustering improves IQL

**Status:** Falsified for the current route set.  
**Decision:** Use standard JanuX routes as the base for the upcoming tabular IQL implementation.

## Question

We tested whether the `default-pre-integration` clustered routes make IQL more
effective than four routes generated directly by JanuX on `ingolstadt_custom2`.
The proposed benefit was that less-overlapping routes would provide more distinct
actions and therefore make exploration and value estimation easier.

## Experiment design

We used a paired `2 route sets × 2 controllers × 3 seeds` design:

- **Route sets:** seeded JanuX routes versus the fixed
  `default-pre-integration` clustered route set.
- **Controllers:** IQL as the primary comparison and all-or-nothing (AON) as a
  non-learning route-menu control.
- **Seeds:** 40, 41, and 42, paired across all four conditions. Route order was
  counterbalanced across seeds.
- **Scenario:** `ingolstadt_custom2`, 40% selfish AVs, frozen human policies,
  and `previous_agents_plus_start_time` observations.
- **Schedule:** 300 human-stabilization days, 4,000 IQL training days, and 100
  greedy test days. AON used its standard 200-day phase and the same test setup.
- **Primary metric:** mean AV travel time during testing, in minutes.

Both IQL treatments used [`scripts/iql_clusters.py`](../../scripts/iql_clusters.py);
only route construction and its action masks changed. This isolates the practical
choice between the two route pipelines. AON indicates whether a difference already
exists in the supplied route menus without IQL.

The predeclared decision rule required all three seeds to favor one route method,
at least 1% mean AV improvement, and no greater than 1% system-travel-time penalty.

## Results

`Δ = clustered − JanuX`; positive values mean clustering was worse.

| Controller | Seed | JanuX AV time | Clustered AV time | Δ minutes | Δ percent |
| --- | ---: | ---: | ---: | ---: | ---: |
| AON | 40 | 4.3498 | 4.6215 | +0.2717 | +6.25% |
| AON | 41 | 4.4366 | 4.7533 | +0.3167 | +7.14% |
| AON | 42 | 4.4204 | 4.6624 | +0.2421 | +5.48% |
| **AON mean** | — | **4.4023** | **4.6791** | **+0.2768** | **+6.29%** |
| IQL | 40 | 4.2939 | 4.6648 | +0.3709 | +8.64% |
| IQL | 41 | 4.4566 | 4.8125 | +0.3558 | +7.98% |
| IQL | 42 | 4.4224 | 4.6877 | +0.2653 | +6.00% |
| **IQL mean** | — | **4.3910** | **4.7216** | **+0.3307** | **+7.54%** |

What we observed:

1. Clustered IQL had higher AV travel time in every seed, with a mean penalty of
   7.54%.
2. Its overall system test time was also worse in every seed, by 7.97% on average.
3. AON showed the same direction in every seed, with a 6.29% AV penalty. Much of
   the result therefore comes from the clustered route menu itself.
4. Clustering did not reveal an additional learning benefit: its mean IQL gain over
   AON was 0.0539 minutes smaller than with JanuX.

## Conclusion

The hypothesis is falsified for `default-pre-integration` on this scenario. The
clustered routes were consistently worse as a route menu, and IQL did not overcome
that disadvantage. The symmetric predeclared rule clearly selects standard JanuX
routes as the tabular-IQL base.

This does **not** show that road clustering is universally harmful. It shows that
this particular clustered route set should not be the base for the next method.
Any future clustering study should first verify route-menu quality independently of
learning.

## Artifacts and reproduction

- Completed outputs: [`results/`](results/)
- Run matrix and command: [`experiments/README.md`](experiments/README.md)
- Paired analysis: [`experiments/analyze_results.py`](experiments/analyze_results.py)
- Operational notes: [`notes/README.md`](notes/README.md)

Run the read-only summary with:

```bash
venv/bin/python studies/road-clustering-improves-iql/experiments/analyze_results.py
```
