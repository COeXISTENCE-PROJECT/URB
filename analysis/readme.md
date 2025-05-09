 ## 📊 Measures and indicators  

Each experiment outputs set of raw records, which are then processed with the script in this folder for a set of performance indicators which we report and several additional metrics that track the quality of the solution and its impact to the system.



To use the analysis script, you have to provide in the command line the following command:

```bash
python metrics.py --id <exp_id> --verbose <verbose> 
```

that will collect the results from the experiment with identifier ```<exp_id>``` and save them in the folder ```results/<exp_id>/metric/```. The ```--verbose``` flag is optional and if set to ```True``` will print additional information about the analysis process.

#### Reported indicators
---

The core metric is the travel time $t$, which is both the core term of the utility for human drivers (rational utility maximizers) and of the CAVs reward.
We report the average travel time for the system $\hat{t}$, human drivers $\hat{t}_{HDV}$, and autonomous vehicles $\hat{t}_{CAV}$. We record each during the training, testing phase and for 50 days before CAVs are introduced to the system ( $\hat{t}^{train}, \hat{t}^{test}$, $\hat{t}^{pre}$). Using these values, we introduce: 

-  CAV advantage as ${\hat{t}_{HDV}^{post}}/\hat{t}_{CAV}$, 
-  Effect of changing to CAV as ${\hat{t}_{HDV}^{pre}}/{\hat{t}_{CAV}}$, and
-  Effect of remaining HDV as ${\hat{t}_{HDV}^{pre}}/{\hat{t}_{HDV}^{test}}$), which reflect the relative performance of HDVs and the CAV fleet from the point of view of individual agents.

To better understand the causes of the changes in travel time, we track the _Average speed_ and _Average mileage_ (directly extracted from SUMO). 

We measure the _Cost of training_, expressed as the average of: $\sum_{\tau \in train}(t^\tau_a - \hat{t}^{pre}_a)$ over all agents $a$, i.e. the cumulated disturbance that CAV cause during the training period.
We call an episode _won_ by CAVs if on average they were faster than human drivers. A final _winrate_ is percentage of such days during training, which additionally describes how quickly the fleet improvement was.