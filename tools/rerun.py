"""Rerun an existing experiment, optionally overriding its CLI configuration."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def experiment_id(value):
    if not value or Path(value).name != value or value in {".", ".."}:
        raise argparse.ArgumentTypeError("experiment IDs must be directory names")
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Rerun an experiment using its saved configuration."
    )
    parser.add_argument("experiment_id", type=experiment_id)
    parser.add_argument("new_experiment_id", type=experiment_id)
    parser.add_argument("--alg-conf", default=None)
    parser.add_argument("--env-conf", default=None)
    parser.add_argument("--task-conf", default=None)
    parser.add_argument("--net", default=None)
    parser.add_argument("--env-seed", type=int, default=None)
    parser.add_argument("--torch-seed", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--route-set", default=None)
    parser.add_argument("--shuffle", action="store_true", default=None)
    parser.add_argument("--skip-metrics", action="store_true", default=False)
    parser.add_argument("--save-model-every", type=int, default=None, metavar="N")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    source_dir = RESULTS_DIR / args.experiment_id
    destination_dir = RESULTS_DIR / args.new_experiment_id
    if not source_dir.is_dir():
        parser.error(f"experiment does not exist: {source_dir}")
    if destination_dir.exists() or destination_dir.is_symlink():
        parser.error(f"new experiment ID already exists: {destination_dir}")
    if args.save_model_every is not None and args.save_model_every <= 0:
        parser.error("--save-model-every must be a positive integer")

    config_path = source_dir / "exp_config.json"
    if not config_path.is_file():
        parser.error(f"experiment has no exp_config.json: {source_dir}")
    try:
        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"could not read {config_path}: {error}")

    saved_script = config.get("script")
    if not saved_script:
        parser.error("exp_config.json does not record the experiment script")
    saved_path = Path(saved_script)
    candidates = [
        saved_path if saved_path.is_absolute() else REPO_ROOT / saved_path,
        REPO_ROOT / "scripts" / saved_path.name,
    ]
    script_path = next(
        (
            candidate
            for candidate in candidates
            if candidate.is_file()
            and candidate.resolve().is_relative_to(REPO_ROOT)
        ),
        None,
    )
    if script_path is None:
        parser.error(f"experiment script does not exist locally: {saved_script}")

    command = [sys.executable, str(script_path), "--id", args.new_experiment_id]
    options = (
        ("--alg-conf", "alg_config", args.alg_conf),
        ("--env-conf", "env_config", args.env_conf),
        ("--task-conf", "task_config", args.task_conf),
        ("--net", "network", args.net),
        ("--env-seed", "env_seed", args.env_seed),
        ("--torch-seed", "torch_seed", args.torch_seed),
        ("--model", "baseline_model", args.model),
        ("--route-set", "route_set", args.route_set),
        ("--save-model-every", "save_model_every", args.save_model_every),
        ("--project", "project", args.project),
    )
    for flag, config_key, override in options:
        value = override if override is not None else config.get(config_key)
        if value is not None:
            command.extend([flag, str(value)])

    shuffle = args.shuffle if args.shuffle is not None else config.get("shuffle", False)
    if shuffle:
        command.append("--shuffle")
    if args.skip_metrics:
        command.append("--skip-metrics")

    print(f"Running: {shlex.join(command)}")
    result = subprocess.run(command, cwd=REPO_ROOT)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
