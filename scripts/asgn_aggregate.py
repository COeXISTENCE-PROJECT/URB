"""
Sampling jobs:
- read their own per-job config file
- run joint-action generation and/or replay simulations
- write partial results under the shared experiment root
- produce partial simulations.csv and simulations_static.csv files

Aggregation job:
- reads all partial simulations.csv files under one experiment root
- writes the merged simulations.csv at the experiment root
- computes best_joint_action.json from the merged simulation rows

Current partial-results layout:
results/<experiment_id>/<task_conf>/<run_id>/<method>/

Current final-results layout:
results/<experiment_id>/

Generated files at the experiment root:
- simulations.csv
- simulations_static.csv
- best_joint_action.json
- simulation_stats.json
"""

import json
import argparse
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def _find_partial_results(experiment_root):
    simulation_csvs = []

    for child in experiment_root.iterdir():
        if not child.is_dir():
            continue

        simulation_csvs.extend(child.rglob("simulations.csv"))

    simulation_csvs.sort()
    return simulation_csvs

def _append_simulations_csv(simulation_csv_paths, output_path):
    # remaps ids which results in removing gaps caused by removed simulations rows
    # because of teleports (e.g. files with simulation ids 1,2,3,5 and 1,4,6 will result
    # in ids 0,1,2,3,4,5,6 in the aggregated version)
    total_travel_times = []
    total_travel_times_by_method = defaultdict(list)
    best_joint_actions = {
        "sum": {"score": float("inf"), "joint_action": None, "source": None},
        "minmax": {"score": float("inf"), "joint_action": None, "source": None},
    }
    global_simulation_id = 0

    with open(output_path, "w", encoding="utf-8", newline="") as output_file:
        with tqdm(total=len(simulation_csv_paths), desc="Aggregating simulations") as pbar:
            wrote_header = False
            for simulation_csv_path in simulation_csv_paths:
                partial_df = pd.read_csv(simulation_csv_path)

                if partial_df.empty:
                    pbar.update(1)
                    continue

                required_columns = {"simulation_id", "agent_id", "action", "travel_time"}
                missing_columns = required_columns.difference(partial_df.columns)
                if missing_columns:
                    missing_list = ", ".join(sorted(missing_columns))
                    raise ValueError(f"Missing required columns [{missing_list}] in {simulation_csv_path}")

                partial_df["travel_time"] = pd.to_numeric(partial_df["travel_time"], errors="coerce")
                partial_df = partial_df.dropna(subset=["simulation_id", "agent_id", "action", "travel_time"])
                if partial_df.empty:
                    pbar.update(1)
                    continue

                partial_simulation_ids = sorted(int(value) for value in partial_df["simulation_id"].dropna().unique())
                local_to_global_id = {
                    local_id: global_simulation_id + index
                    for index, local_id in enumerate(partial_simulation_ids)
                }

                partial_df = partial_df.loc[:, ["simulation_id", "agent_id", "action", "travel_time"]].copy()
                partial_df["simulation_id"] = partial_df["simulation_id"].map(local_to_global_id)
                partial_df.to_csv(output_file, index=False, header=not wrote_header)
                wrote_header = True

                method = simulation_csv_path.parent.name.split("_")[1] # extract the method name from the subdirectory name (e.g. 123123_grid_7 -> grid)
                grouped_totals = []

                for simulation_id, group in partial_df.groupby("simulation_id"):
                    current_sum = float(group["travel_time"].sum())
                    current_minmax = float(group["travel_time"].max())
                    joint_action = {
                        str(int(row.agent_id)): int(row.action)
                        for row in group.itertuples(index=False)
                    }
                    source = {
                        "file": str(simulation_csv_path),
                        "simulation_id": int(simulation_id),
                    }

                    if current_sum < best_joint_actions["sum"]["score"]:
                        best_joint_actions["sum"] = {
                            "score": current_sum,
                            "joint_action": joint_action,
                            "source": source,
                        }

                    if current_minmax < best_joint_actions["minmax"]["score"]:
                        best_joint_actions["minmax"] = {
                            "score": current_minmax,
                            "joint_action": joint_action,
                            "source": source,
                        }

                    grouped_totals.append(current_sum)

                total_travel_times.extend(grouped_totals)
                total_travel_times_by_method[method].extend(grouped_totals)

                global_simulation_id += len(partial_simulation_ids)
                pbar.update(1)

    return total_travel_times, total_travel_times_by_method, best_joint_actions

def _write_best_joint_action(best_joint_actions, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(best_joint_actions, f, indent=2)

def _summarize_total_travel_times(total_travel_times):
    if not total_travel_times:
        raise ValueError("No simulation totals found")

    values = np.asarray(total_travel_times, dtype=float)
    count = int(values.size)
    minimum = float(values.min())
    maximum = float(values.max())
    total = float(values.sum())
    mean = float(values.mean())
    median = float(np.median(values))
    p10, p25, p75, p90 = (float(value) for value in np.percentile(values, [10, 25, 75, 90]))
    squared_error_sum = float(np.sum((values - mean) ** 2))

    return {
        "count": count,
        "sum": total,
        "min": minimum,
        "p10": p10,
        "p25": p25,
        "median": median,
        "p75": p75,
        "p90": p90,
        "max": maximum,
        "range": maximum - minimum,
        "iqr": p75 - p25,
        "mean": mean,
        "std": math.sqrt(squared_error_sum / (count - 1)) if count > 1 else 0.0,
    }

def _write_simulation_stats(total_travel_times, total_travel_times_by_method, output_path):
    if not total_travel_times:
        raise ValueError(f"No simulation totals found under {output_path.parent}")

    overall = _summarize_total_travel_times(total_travel_times)
    by_method = {
        method: _summarize_total_travel_times(method_values)
        for method, method_values in sorted(total_travel_times_by_method.items())
    }

    summary = {
        **overall,
        "by_method": by_method,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

def create_histogram(total_travel_times, output_path):
    values = np.asarray(total_travel_times, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot create a histogram for an empty set of travel times")

    plt.figure(figsize=(8, 5))
    if np.all(values > 0) and values.max() / values.min() > 3:
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 30)
        plt.hist(values, bins=bins, edgecolor="black", alpha=0.8)
        plt.xscale("log")
        plt.xlabel("Total travel time (log scale)")
    else:
        plt.hist(values, bins="auto", edgecolor="black", alpha=0.8)
        plt.xlabel("Total travel time")

    plt.title("Per-simulation total travel time")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate simulation partial results into one final result set.")
    parser.add_argument("--experiment-id", required=True, help="Experiment id used under the results base directory")
    parser.add_argument("--results-base-dir", default="../results", help="Base results directory used by sampling jobs") #?
    args = parser.parse_args()

    experiment_root = Path(args.results_base_dir) / args.experiment_id
    if not experiment_root.exists():
        raise FileNotFoundError(f"Experiment root not found: {experiment_root}")

    simulation_csvs = _find_partial_results(experiment_root)
    if not simulation_csvs:
        raise FileNotFoundError(f"No partial simulations.csv files found under {experiment_root}")

    output_root = experiment_root
    output_root.mkdir(parents=True, exist_ok=True)

    total_travel_times, total_travel_times_by_method, best_joint_actions = _append_simulations_csv(simulation_csvs, output_root / "simulations.csv")
    _write_best_joint_action(best_joint_actions, output_root / "best_joint_action.json")
    _write_simulation_stats(total_travel_times, total_travel_times_by_method, output_root / "simulation_stats.json")
    create_histogram(total_travel_times, output_root / "total_travel_times_histogram.png")

    print(f"Aggregated {len(total_travel_times)} simulations into {output_root / 'simulations.csv'}")
    print(f"Saved best joint action summary to {output_root / 'best_joint_action.json'}")
