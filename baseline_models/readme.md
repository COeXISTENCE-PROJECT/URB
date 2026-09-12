### Baseline models available in URB

Apart from RL models, we provide baseline algorithms for comparison.

Based on the implementation, baselines can be divided into *per-agent* methods, executed via `scripts/baselines.py`, where each agent stores its own model and collects agent-level observations, and *centralized* methods, executed via `scripts/<script_name>.py`, which use a shared data structure or model during execution.

The available options consist of:


| Method   |  Description                                            | Implementation                     | Execution     | Source          |
| -------- | ------------------------------------------------------- | ------------------------- | ------------- | --------------- |
| `aon` | Deterministically picks the shortest free-flow route regardless of congestion. | Per-agent | Run via `scripts/baselines.py` with `--model aon` | Included in URB |
| `random` | Selects routes uniformly at random. | Per-agent | Run via `scripts/baselines.py` with `--model random` | Included in URB |
| `gawron` | Human learning model based on [Gawron (1998)](https://kups.ub.uni-koeln.de/9257/); iteratively shifts cost expectations toward received rewards. | Per-agent | Run via `scripts/baselines.py` with `--model gawron` | [RouteRL](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/993423d101f39ea67a1f7373e6856af95a0602d4/routerl/human_learning/learning_model.py#L42) |
| `greedy` | Selects the route with the lowest recorded travel time based on past episodes. Uses a global structure to store per-agent past records. | Centralized | Run as standalone script: `scripts/greedy.py` | Included in URB |
