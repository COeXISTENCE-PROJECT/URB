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

#### Workflow

`urb`:
* runs a `RouteRl` script
* with a RL algorithm (possibly `torchRL` implementation)
* opens a config file (stored in `experiment_metadata.json`)
* loads the network and demand from `networks`
* executes a typical `RouteRL` routine of
   * first learning of human drivers,
   * which then 'mutate` to CAVs,
   * are trained to optimize routing policies with the implemented algorithm.
* When the training is finished, it uses raw results to compute a wide-set of KPIs.


![image](https://github.com/user-attachments/assets/1a2858e7-c1a7-4e4f-bb4a-c7289e366ceb)



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

### Prerequisites 

Make sure you have SUMO installed in your system. This procedure should be carried out separately, by following the instructions provided [here](https://sumo.dlr.de/docs/Installing/index.html).

### Cloning repository

Clone the **URB** repository from github by

```
git clone https://github.com/COeXISTENCE-PROJECT/URB.git
```

### Creating enviroment for URB

- **Option 1** (Recommended): Create a virtual enviroment with `venv`:

```bash
$ python3.13.1 -m venv .venv
```

and then install dependencies by:

```
$ cd URB
$ pip install --force-reinstall --no-cache-dir -r requirements.txt
```

- **Option2** (Alternative): Use conda environment with `conda`:

```bash
$ conda create -n URB python=3.13.1
```

and then install dependencies by: 

```bash
$ cd URB
$ conda activate URB
$ pip install --force-reinstall --no-cache-dir -r requirements.txt
```

## Usage:
```
python scripts/<script_name> --id <exp_id> --conf <configuration_id> --net <net_name> --env-seed <env_seed> --torch-seed <torch_seed>
```

- Replace `<scipt_name>` with the script you wish to run.
- Replace `<exp_id>` with the experiment identifier. 
- Replace `<configuration_id>` with the configuration identifier. Must be compatible with the script and a key from `experiment_metadata.json`.
- Replace `<net_name>` with the name of the network you wish to use. Must be one of the folder names in `networks/`.
- Replace `<env_seed>` with your reproducibility random seed for the traffic environment. (default=42)
- Replace `<torch_seed>` with your reproducibility random seed for PyTorch. (default=42)

For example:
```
python scripts/ippo_torchrl.py --id gar_ippo_0 --conf 1_ippo --net gargenville --env-seed 42 --torch-seed 0
```

Records and plots will be saved in ```results/<exp_id>```.

> ⚠️ **Attention**: There is one additional flag `model` for `scripts/baselines.py`, instead of `torch-seed`:

```
python scripts/baselines.py --id <exp_id> --conf <configuration_id> --net <net_name> --env-seed <env_seed> --model <model_name>
```

And `<model_name>` should be from `routerl.Keychain.HUMAN_MODELS` (see [here](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/6af53cfb0174c72a75216c8fce256aac96b044ae/routerl/keychain.py#L124)). For example:

```
python scripts/baselines.py --id ing_aon --conf 1_baseline --net ingolstadt_custom --env-seed 42 --model aon
```
