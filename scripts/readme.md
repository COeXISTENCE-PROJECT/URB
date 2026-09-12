# Experiment scripts

The scripts in this directory run URB methods and save their outputs under `results/<exp_id>/`.

## Learning methods

- `iql.py`, `ippo.py`, and `hyp_ippo.py`: custom independent-learning implementations.
- `iql_tabular.py` and `reinforce_tabular.py`: simple tabular methods.
- `centralized_controller.py`, `centralized_dqn.py`, and `feudal_hrl.py`: centralized or hierarchical controllers.
- `ippo_torchrl.py`, `mappo_torchrl.py`, `iql_torchrl.py`, `qmix_torchrl.py`, and `vdn_torchrl.py`: TorchRL implementations.
- Scripts ending in `_clusters.py` use pre-generated clustered route sets and action masks. `ippo_clusters_shared_params.py` shares one policy across AVs.

Algorithm configurations live under `config/algo_config/<algorithm>/`. Use `base_script.py` as the starting point for a new experiment script.

### Output conventions

- At the end of each experiment script, metrics are automatically computed by calling `analysis/metrics.py`. Pass `--skip-metrics` to retain the raw experiment outputs and calculate metrics later.
- Pass `--project <name>` to record an optional project label in `exp_config.json`.
- For learning-based scripts, training losses are saved in a unified CSV file:
  `results/<exp_id>/losses/losses.csv`

### Baselines

In addition to RL algorithms, we provide baseline algorithms for comparison.
They can be executed with ```scripts/baselines.py``` or as standalone scripts ```scripts/<model_name>.py```, depending on the selected model.

The options consist of:

| Method   |  Description                                            | Location          | Execution     | Source          |
| -------- | ------------------------------------------------------- | ----------------- | ------------- | --------------- |
| `greedy` | Selects the route with the lowest recorded travel time based on past episodes. Uses a global structure to store per-agent past records. | `scripts/` | Run as standalone script: `scripts/greedy.py` | Included in URB |
| `aon` | Deterministically picks the shortest free-flow route regardless of congestion. | `baseline_models/` | Run via `scripts/baselines.py` with `--model aon` | Included in URB |
| `random` | Selects routes uniformly at random. | `baseline_models/` | Run via `scripts/baselines.py` with `--model random` | Included in URB |
| `gawron` | Human learning model based on [Gawron (1998)](https://kups.ub.uni-koeln.de/9257/); iteratively shifts cost expectations toward received rewards. | `baseline_models/` | Run via `scripts/baselines.py` with `--model gawron` | [RouteRL](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/993423d101f39ea67a1f7373e6856af95a0602d4/routerl/human_learning/learning_model.py#L42) |
