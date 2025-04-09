#### Measures and indicators  

Each experiment outputs set of raw records, which are then processed with the scripts in this folder for a set of indicators which we report and (some) identify as meaningful to track the quality of the solution and its impact to the system.

---

The metrics we use to assess the quality of a solution are generally divided into three groups:

1. Individual metrics  
2. Group metrics  
3. System-wide metrics  

### Individual Metrics

- **User travel time** (from RouteRL) – average over the last 100 days of the simulation  
- **Delays** (from SUMO)  

### Group Metrics

- **Group travel time** (average)  
- **CAV Advantage**, effect of changing, effect of not changing fleet (as in the *Scientific Reports* paper)  
- **Cost of #** – how CAV advantage grows/decreases with increasing fleet size  
- **COeXISTENCE metric** (*dominance*) – how bad CAVs are for humans (the name is good, that’s most important)  

### System-Wide Metrics

- **Total travel time** (`totalTravelTime`)  
- **# of stops, teleports, sum of delays** – `agent_<id>_waitingCount`, `teleports_total`, `totalDepartDelay`  
  - The question is: do we count this as a delay?  
  - Another possibility: `timeLoss * count`  
- **Speed of convergence** – day where the algorithm beats Baseline  
  - (Does it have to be smoothed heavily?)  
- **Cost of training** – area under the curve (integral)  
- **Convergence stability** – entropy of human actions  
- **Human preference shifts**  
- **Winrate** – all good RL papers have some form of this to encapsulate how good the solution is  
  - If our objective is to beat naive solutions, we can define **winrate** as the probability (percentage) of achieving a better result than Baseline  
  - Another possibility: count the number of days where the effect of change is positive (if we aim to maximize fleet size)  
