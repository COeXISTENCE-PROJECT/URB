## (MA)RL algorithms and baselines.

### (MA)RL algorithms

We deliver here scripts for the experiment runs. Each associated algorithm with selected implementations from `TorchRL`:
* ```ippo_torchrl.py``` uses Independent Proximal Policy Optimization algorithm,
* ```mappo_torchrl.py``` uses Multi Agent Proximal Policy Optimization algorithm,
* ```iql_torchrl.py``` uses Implicit Q-Learning algorithm,
* ```qmix_torchrl.py``` uses QMIX algorithm,
* ```vdn_torchrl.py``` uses Value Decomposition Network algorithm.

Moreover, we have two independent algorithms with our custom implementations:
* ```iql.py``` uses Independent Q-Learning,
* ```ippo.py``` uses Independent Proximal Policy Optimization,
* ```hyp_ippo.py``` uses Independent Proximal Policy Optimization with hypernetworks.

You can tune, adjust, hyperparameterize and modify all the provided implementations, or create own scripts.

### Output conventions

- At the end of each experiment script, metrics are automatically computed by calling `analysis/metrics.py`. Pass `--skip-metrics` to retain the raw experiment outputs and calculate metrics later.
- For learning-based scripts, training losses are saved in a unified CSV file:
  `results/<exp_id>/losses/losses.csv`

### Baselines

In addition to RL algorithms, we provide baseline algorithms for comparison.
They can be executed with ```scripts/baselines.py``` or as standalone scripts ```scripts/<model_name>.py```, depending on the selected model.

The options consist of:

| Method   |  Description                                            | Location          | Execution     | Source          |
| -------- | ------------------------------------------------------- | ----------------- | ------------- | --------------- |
| `greedy` | Selects the route with the lowest recorded travel time based on past episodes. Uses a global structure to store per-agent past records. | `scripts/`     | Run as standalone script: `scripts/greedy.py` | Included in URB 
| `aon`    | Deterministically picks the shortest free-flow route regardless of congestion. | `baseline_models/` | Run via `scripts/baselines.py` with `--model aon` | Included in URB
| `random` | Fully undeterministic. | `baseline_models/` |     Run via `scripts/baselines.py` with  `--model random` | Included in URB |
| `gawron` | Human learning model based on [Gawron (1998)](https://kups.ub.uni-koeln.de/9257/); iteratively shifts cost expectations toward received rewards. | `baseline_models/` | Run via `scripts/baselines.py` with `--model gawron` | [RouteRL](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/993423d101f39ea67a1f7373e6856af95a0602d4/routerl/human_learning/learning_model.py#L42) |

