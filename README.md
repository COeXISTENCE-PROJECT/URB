<p align="center">
  <img src="docs/urb.png" align="center" width="30%"/>
</p>

# Urban Routing Benchmark: Benchmarking MARL algorithms on the fleet routing tasks

<p align="center">
  <img src="docs/urb_overview.png" align="center" width="90%"/>
</p>

Autonomous Vehicles promise better performance to our congested urban networks in the future. One of potential improvements lays in routing decisions which, unlike for humans, can be collective, data-driven and come from deep-learning algorithms. To foster this unexploited opportunity, we propose the `URB` - Urban Mobility Benchmark for RL-equipped Connected Autonomous Vehicles (CAVs). `URB`, for a selection of real-world urban networks with realistic demand patterns simulates the future system state, with CAVs. It explores state-of-the-art MARL algorithms from TorchRL, for various reward formulations (from selfish, via social to malicious), collaborations, observations,  market shares and compositions (monopolies vs oligopolies) and human adaptations. It reports variety of performance indicators not only on the RL algorithms performance, but also for the system (like impact on human drivers or on total travel times). Via the broad experimental scheme, it aims to:

1) understand which state-of-the-art algorithms outperform others in this class of tasks,
2) open competition for future algorithmic improvements,
3) better understand impact of collective CAV routing for future cities (on congestion, emissions and sustainability) to equip policymakers with solid arguments for CAV regulations.

---

## 🔗 Workflow

`URB` (as depicted in the above figure):
* Runs an experiment script using the `TrafficEnvironment` from `RouteRL`,
* With a RL algorithm (possibly `TorchRL` implementation) or a baseline method (from `baseline_models/`),
* Opens algorithm, environment and task configuration files from `config/`,
* Loads the network and demand from `networks`
* Executes a typical `RouteRL` routine of
   * first learning of human drivers,
   * which then 'mutate` to CAVs,
   * are trained to optimize routing policies with the implemented algorithm.
* When the training is finished, it uses raw results to compute a wide-set of KPIs.

## 📝 Tasks

#### Core (benchmarked with the results):

Collaborative fleet minimizing group average travel time with 40\% market share with non-adaptive human routing behaviour.

>In a French small town of _Saint Arnoult_, where human routing decision were stable and close to the well-known User Equilibrium, 40\% of drivers decide to switch on the autonomous driving mode, delegating their routing decisions. Then, each machine will apply some kind of algorithm to select the route maximising the reward function: group average travel time.

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

## 📦 Setup

#### Prerequisites 

Make sure you have SUMO installed in your system. This procedure should be carried out separately, by following the instructions provided [here](https://sumo.dlr.de/docs/Installing/index.html).

#### Cloning repository

Clone the **URB** repository from github by

```bash
git clone https://github.com/COeXISTENCE-PROJECT/URB.git
```

#### Creating enviroment for URB

- **Option 1** (Recommended): Create a virtual enviroment with `venv`:

```bash
python3.13.1 -m venv .venv
```

and then install dependencies by:

```bash
cd URB
pip install --force-reinstall --no-cache-dir -r requirements.txt
```

- **Option 2** (Alternative): Use conda environment with `conda`:

```bash
conda create -n URB python=3.13.1
```

and then install dependencies by: 

```bash
cd URB
conda activate URB
pip install --force-reinstall --no-cache-dir -r requirements.txt
```

## 📌 Usage

#### Usage of **URB** for Reinforcement Learning algorithms

To use **URB** while using RL algorithm, you have to provide in the command line the following command:

```bash
python scripts/<script_name> --id <exp_id> --alg-conf <hyperparam_id> --env-conf <env_conf_id> --task-conf <task_id> --net <net_name> --env-seed <env_seed> --torch-seed <torch_seed>
```

where

- ```<scipt_name>``` is the script you wish to run, available scripts are ```ippo_torchrl```, ```iql_torchrl```, ```mappo_torchrl```, ```vdn_torchrl``` and ```qmix_torchrl```,
- ```<exp_id>``` is your own experiment identifier, for instance ```random_ing```, 
- ```<hyperparam_id>``` is the hyperparameterization identifier, it must correspond to a `.json` filename (without extension) in [`config/algo_config`](config/algo_config/). Provided scripts automatically select the algorithm-specific subfolder in this directory.
- ```<env_conf_id>``` is the environment configuration identifier. It must correspond to a `.json` filename (without extension) in [`config/env_config`](config/env_config/). It is used to parameterize environment-specific processes, such as path generation, disk operations, etc. It is **optional** and by default is set to `config1`.
- ```<task_id>``` is the task configuration identifier. It must correspond to a `.json` filename (without extension) in [`config/task_config`](config/task_config/). It is used to parameterize the simulated scenario, such as portion of AVs, duration of human learning, AV behavior, etc.
- ```<net_name>``` is the name of the network you wish to use. Must be one of the folder names in ```networks/``` i.e. ```ingolstadt_custom```, ```nangis```, ```nemours```, ```provins``` or ```saint_arnoult```,
- ```<env_seed>``` is reproducibility random seed for the traffic environment, default seed is set to be 42,
- ```<torch_seed>``` is reproducibility random seed for PyTorch, it is **optional** and by default is set to 42.

For example, the following command runs an experiment using:
- QMIX algorithm, hyperparameterized by `config/algo_config/qmix/config3.json`, 
- The task specified in `config/task_config/config4.json`,
- The environment parameterization specified in `config/env_config/config1.json` (by default),
- Experiment identifier `sai_qmix_0`, which will be used as the folder name in `results/` to save the experiment data,
- Saint Arnoult network and demand, from `networks/saint_arnoult`,
- Environment (also used for `random` and `numpy`) and PyTorch seeds 42 and 0, respectively.

```bash
python scripts/qmix_torchrl.py --id sai_qmix_0 --alg-conf config3 --task-conf config4 --net saint_arnoult --env-seed 42 --torch-seed 0
```

####  Usage **URB** for baselines

Similarly as for RL algorithms, you have to provide command, but there is one additional flag ```model``` for ```scripts/baselines.py```, instead of ```torch-seed```, then you have command of form:

```bash
python scripts/baselines.py --id <exp_id> --alg-conf <hyperparam_id> --env-conf <env_conf_id> --task-conf <task_id> --net <net_name> --env-seed <env_seed> --model <model_name>
```

And ```<model_name>``` should be one of ```random```, ```aon``` (included in [baseline_models](baseline_models/)) or ```gawron``` (from [RouteRL](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/993423d101f39ea67a1f7373e6856af95a0602d4/routerl/human_learning/learning_model.py#L42)). 

For example:

```bash
python scripts/baselines.py --id ing_aon --alg-conf config1 --task-conf config2 --net ingolstadt_custom --model aon
```

## 📊 Measures and indicators  

Each experiment outputs set of raw records, which are then processed with the script in this folder for a set of performance indicators which we report and several additional metrics that track the quality of the solution and its impact to the system.

#### Usage

To use the analysis script, you have to provide in the command line the following command:

```bash
python metrics.py --id <exp_id> --verbose <verbose> 
```

that will collect the results from the experiment with identifier ```<exp_id>``` and save them in the folder ```results/<exp_id>/metrics/```. The ```--verbose``` flag is optional and if set to ```True``` will print additional information about the analysis process.

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

## 💎 Extending URB

We provide templates for extending the possible experiments that can be conducted using `URB`.

### Adding new baselines
Users can define and use their own baseline methods by creating a new model by extending [`baseline_models/BaseLearningModel`](baseline_models/base.py).

### Adding new scripts
Users can add new experiment scripts for testing different algorithms, different implementations and different training pipelines. The recommended script structure is provided in [`scripts/base_script.py`](scripts/base_script.py).

### New scenarios and hyperparameterizations
Users can extend possible experiment configurations by adding:

* Algorithm hyperparameterization in [`config/algo_config`](config/algo_config/),
* Experiment setting in [`config/env_config`](config/env_config/), and
* New tasks in [`config/task_config`](config/task_config/).
