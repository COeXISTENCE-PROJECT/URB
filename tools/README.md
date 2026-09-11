# Experiment tools

These small tools work with experiment directories under `results/`.

## Rerun an experiment

`rerun.py` reads the source experiment's `exp_config.json`, uses its saved script and configuration, and starts a new experiment. Explicit options override saved values. The new ID is required and must not already exist.

```bash
python tools/rerun.py old_exp new_exp --env-seed 43
python tools/rerun.py old_exp new_exp --net provins --task-conf selfish_40
```

Common overrides include `--alg-conf`, `--env-conf`, `--task-conf`, `--net`, `--env-seed`, `--torch-seed`, `--model`, `--route-set`, `--shuffle`, `--skip-metrics`, and `--save-model-every`.

## Rename an experiment

`rename.py` renames one existing results directory and refuses to overwrite another. If `exp_config.json` contains an `exp_id` field, it is updated too.

```bash
python tools/rename.py old_exp new_exp
```
