# URB
#### Urban Routing Benchmark - benchmarking MARL algorithms on the fleet routing tasks. 


> Autonomous Vehicles promise better performance to our congested urban networks in the future. One of potential improvements lays in routing decisions which, unlike for humans, can be collective, data-driven and come from deep-learning algorithms. To foster this unexploited opportunity, we propose the `urb` - Urban Mobility Benchmark for RL-equipped Connected Autonomous Vehicles (CAVs). `urb`, for a selection of real-world urban networks with realistic demand patterns simulates the future system state, with CAVs. It explores state-of-the-art MARL algorithms from torchRL, for various reward formulations (from selfish, via social to malicious), collaborations, observations,  market shares and compositions (monopolies vs oligopolies) and human adaptations. It reports variety of performance indicators not only on the RL algorithms performance, but also for the system (like impact on human drivers or on total travel times). Via the broad experimental scheme, it aims to:
>
> i) understand which state-of-the-art algorithms outperform others in this class of tasks,
>
> ii) open competition for future algorithmic improvements,
>
> iii) better understand impact of collective CAV routing for future cities (on congestion, emissions and sustainability) to equip policymakers with solid arguments for CAV regulations.
---


## Tasks:

#### Core (benchmarked with the results):

Collaborative fleet minimizing group average travel time with 40\% market share with non-adaptive human routing behaviour.

>In a French small town of _Gurgerville_ , where human routing decision were stable and close to the well-known User Equilibrium, 40\% of drivers decide to switch on the autonomous driving mode, delegating their routing decisions. Then, each machine will apply some kind of algorithm to select the route maximising the reward function: group average travel time.

#### Illustrative (sample results only):

1. Fully autonomous fleet
2. Altruistic fleet (minimize time for humans)
3. Malicious fleet (maximize time for humans)
4. Modal shifts - humans join or leave the fleet according to its performance
5. Human adaptations - drivers react to actions of the fleet and change their behaviour


#### Possible (doable yet not implemented):

6. Two competing fleets with various strategies
7. Communication protocols (sharing information to collaborate).
8. Heterogenous human populations - varying willingness to switch to CAV
9. Different choice set than humans
10. Adaptive traffic signals
11. 

# To be improved

## Setup:
- Clone the repository:
```
git clone https://github.com/COeXISTENCE-PROJECT/URB.git
```
- (Recommended) Create a virtual environment.
- Run: 
```
cd URB
pip3 install -r requirements.txt
```

---

## Usage:
```
python3 scipts/script_name.py --id <exp_id> --conf <configuration_id> --net <net_name> --seed <seed>
```

- Replace `<exp_id>` with the experiment identifier. 
- Replace `<configuration_id>` with the configuration identifier. Must be a key from `experiment_metadata.json`.
- Replace `<net_name>` with the name of the network you wish to use. Must be one of the folder names in `networks/`.
- Replace `<seed>` with your reproducibility random seed. For consistency with others, set to 42. 

For example:
```
python3 scripts/ippo_torchrl.py --id onur_1 --conf 1_ippo --net gargenville --seed 42
```

Records will be saved to ```records/<exp_id>```. Plots will be saved to ```plots/<exp_id>```.
