# Calculating metrics

Experiment scripts produce raw records that `metrics.py` turns into benchmark metrics and plots.

## Usage

```bash
python analysis/metrics.py --id <exp_id> [--results-folder <results-folder>] [--verbose True] [--skip-collecting True]
```

By default this reads from `results/<exp_id>/` and writes to its `metrics/` directory. `--results-folder` selects another results root, `--verbose True` prints more detail, and `--skip-collecting True` reuses previously collected episode data.

## Reported indicators

The core metric is the travel time $t$, which is both the core term of the utility for human drivers (rational utility maximizers) and of the CAVs reward.
We report the average travel time for the system $\hat{t}$, human drivers $\hat{t}\_{HDV}$, and autonomous vehicles $\hat{t}\_{CAV}$. We record each during the training, testing phase and for 50 days before CAVs are introduced to the system ( $\hat{t}^{train}, \hat{t}^{test}$, $\hat{t}^{pre}$). Using these values, we introduce: 

-  CAV advantage as  $\hat{t}\_{HDV}^{post}$ / $\hat{t}\_{CAV}$, 
-  Effect of changing to CAV as ${\hat{t}\_{HDV}^{pre}}/{\hat{t}\_{CAV}}$, and
-  Effect of remaining HDV as ${\hat{t}\_{HDV}^{pre}}/{\hat{t}\_{HDV}^{test}}$, which reflect the relative performance of HDVs and the CAV fleet from the point of view of individual agents.

To better understand the causes of the changes in travel time, we track the _Average speed_ and _Average mileage_ (directly extracted from SUMO). 

We measure the _Cost of training_, expressed as the average of: $\sum_{\tau \in train}(t^\tau_a - \hat{t}^{pre}_a)$ over all agents $a$, i.e. the cumulated disturbance that CAV cause during the training period. We define $c\_{CAV}$ and $c\_{HDV}$ accordingly.
We call an experiment _won_ by CAVs if their policy was on average faster than human drivers' behaviour. A final _winrate_ is a percentage of runs that were won by CAVs.

## Units of measurement

All the metrics are expressed in the following units:
- Time: minutes
- Distance: kilometers
- Speed: kilometers per hour
