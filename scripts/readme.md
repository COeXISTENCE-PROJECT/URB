### (MA)RL algorithms and baselines.

We deliver here scripts for the experiment runs. Each associated algorithm with selected implementations from `TorchRL`:
* ```ippo_torchrl.py``` uses Independent Proximal Policy Optimization algorithm,
* ```mappo_torchrl.py``` uses Multi Agent Proximal Policy Optimization algorithm,
* ```iql_torchrl.py``` uses Implicit Q-Learning algorithm,
* ```qmix_torchrl.py``` uses QMIX algorithm,
* ```vdn_torchrl.py``` uses Value Decomposition Network algorithm.

We selected five most promising RL algorithms implemented in `TorchRL` applicable for the class of `URB` problems. You can tune them, adjust, hyperparameterize and modify, or create own scripts.

Apart from RL algorithms, we provide baseline algorithms to compare with, can be used with ```baselines.py``` and model options consist:
* **Baselines included in URB**
    * ```aon``` model which deterministically picks the shortest free-flow route regardless of the congestion,
    * ```random``` model which is fully undeterministic,
* **Additionally, available from `RouteRL`**
    * ```gawron``` model is based on: `Gawron (1998) <https://kups.ub.uni-koeln.de/9257/>`, the model iteratively shifts the cost expectations towards the received reward.
