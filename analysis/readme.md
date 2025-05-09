 ## 📊 Measures and indicators  

Each experiment outputs set of raw records, which are then processed with the script in this folder for a set of performance indicators which we report and several additional metrics that track the quality of the solution and its impact to the system.

---

The core metric is the travel time $t$, which is both the core term of the utility for human drivers (rational utility maximizers) and of the CAVs reward.
We report the average travel time for the system $\hat{t}$, human drivers $\hat{t}_{\text{HDV}}$, and autonomous vehicles $\hat{t}_{\text{CAV}}$. We record each during the training, testing phase and for 50 days before CAVs are introduced to the system ( $\hat{t}^{train}, \hat{t}^{test}$, $\hat{t}^{pre}$). Using these values, we introduce: 

-  CAV advantage as ${\hat{t}_{\text{HDV}}^{post}}/\hat{t}_{\text{CAV}}$, 
-  Effect of changing to CAV as ${\hat{t}_{\text{HDV}}^{pre}}/{\hat{t}_{\text{CAV}}}$, and
-  Effect of remaining HDV as ${\hat{t}_{\text{HDV}}^{pre}}/{\hat{t}_{\text{HDV}}^{test}}$), which reflect the relative performance of HDVs and the CAV fleet from the point of view of individual agents.

To better understand the causes of the changes in travel time, we track the _Average speed_ and _Average mileage_ (directly extracted from SUMO). 

We measure the _Cost of training_, expressed as the average of: $\sum_{\tau \in train}(t^\tau_a - \hat{t}^{pre}_a)$ over all agents $a$, i.e. the cumulated disturbance that CAV cause during the training period.
We call an episode _won_ by CAVs if on average they were faster than human drivers. A final _winrate_ is percentage of such days during training, which additionally describes how quickly the fleet improvement was.