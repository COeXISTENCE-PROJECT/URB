### (MA)RL algorithms and baselines.

We deliver here scripts for the experiment runs. Each associated with selected implementation of `torchRL` algorithm:
* ```ippo_torchrl.py``` uses Independent Proximal Policy Optimization algorithm,
* ```mappo_torchrl.py``` uses Multi Agent Proximal Policy Optimization algorithm,
* ```iql_torchrl.py``` uses Implicit Q-Learning algorithm,
* ```qmix_torchrl.py``` uses QMIX algorithm,
* ```vdn_torchrl.py``` uses Value Decomposition Network algorithm.

We selected five most promising RL algorithms implemented in `TorchRL` applicable for the class of `urb` problems. You can tune them, adjust, hyperparameterize and modify, or create own scripts.

Apart from RL immplementations we provide baseline algorithms to compare with, there are included in ```baselines.py``` and consist
* ```aon``` model which is All or Nothing algorithm,
* ```random``` model which is fully undeterministic approach for finding the routes,
* ```gawron``` model is based on: `Gawron (1998) <https://kups.ub.uni-koeln.de/9257/>`,
* ```weighted``` model is based on: `Cascetta (2009) <https://link.springer.com/book/10.1007/978-0-387-75857-2/>`, the model uses the reward and a weighted average of the past cost expectations to update the current cost expectation.
