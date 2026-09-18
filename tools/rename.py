"""Rename an experiment directory without overwriting another experiment."""

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def experiment_id(value):
    if not value or Path(value).name != value or value in {".", ".."}:
        raise argparse.ArgumentTypeError("experiment IDs must be directory names")
    return value


def main():
    parser = argparse.ArgumentParser(description="Rename an existing experiment.")
    parser.add_argument("old_experiment_id", type=experiment_id)
    parser.add_argument("new_experiment_id", type=experiment_id)
    args = parser.parse_args()

    source_dir = RESULTS_DIR / args.old_experiment_id
    destination_dir = RESULTS_DIR / args.new_experiment_id
    if not source_dir.is_dir():
        parser.error(f"experiment does not exist: {source_dir}")
    if destination_dir.exists() or destination_dir.is_symlink():
        parser.error(f"new experiment ID already exists: {destination_dir}")

    config_path = destination_dir / "exp_config.json"
    source_config_path = source_dir / "exp_config.json"
    config = None
    if source_config_path.is_file():
        try:
            with source_config_path.open(encoding="utf-8") as file:
                config = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"could not read {source_config_path}: {error}")

    source_dir.rename(destination_dir)

    if config is not None and "exp_id" in config:
        config["exp_id"] = args.new_experiment_id
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=4)
            file.write("\n")

    print(f"Renamed {args.old_experiment_id} to {args.new_experiment_id}.")


if __name__ == "__main__":
    main()
