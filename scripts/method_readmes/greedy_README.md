# Greedy approach based on driving history – working notes

#### Algorithm
Each day, every driver selects, for a given origin–destination–departure time triple, the route with the minimal travel time ever recorded (after mutation) for this triple. Travel time records are updated daily.
If no time has been recorded yet for a given route, origin, destination, and departure time, then free-flow travel time is assumed until the route is first tried.

#### Limitations
If a high travel time was reported for the route early on, it may never be chosen again—even if it later becomes the fastest option in given timepoint.

#### Possible Extensions:
- The algorithm only considers exact timepoint histories; it may be extended by taking into consideration also nearby timepoints for the same OD pair for a given driver.
- AV simulation starts with an empty history; instead, learned human actions after mutation could be used as AV choices on the first day after mutation.
- All agents choose their actions simultaneously at the beginning of each episode. They base decisions only on historical data; later drivers on the same day have no information about earlier drivers’ choices or congestion. This can lead to suboptimal or cyclic solutions.
- Instead of selecting the route with the minimal travel time ever recorded (after mutation), the driver can select the route based on the weighted average of travel times, with higher weights for the newer routes 

#### Note:
A “first-to-last” greedy approach could be implemented and compared with the current one. However, for now, this is technically challenging (e.g., running only the first k agents in simulation and incrementing k in subsequent episodes).
In this variant, the first agent tests all n possible routes (e.g., four), selects the fastest, and commits to it; the second agent then chooses among his four routes, with the first agent fixed on their choice, the process repeats for all agents (total simulation days: n_route_choices × n_agents).
