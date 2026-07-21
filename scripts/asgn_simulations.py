"""
Local runs, i.e. runs not launched through one of the server scripts, generate
samples for all methods with positive *_num_samples values and save them into
one experiment directory. num_tasks_grid is ignored.

With the current 4-action, grid_values=21 setup, the full grid has 1771 points.
To cover every grid point at least once, set grid_num_samples >= 1771. Extra
samples are assigned deterministically from the start of the grid, so use a
multiple of 1771 to ensure even coverage.
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import ast
import csv
import json
import logging
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from routerl import Keychain as kc
from routerl import TrafficEnvironment
from tqdm import tqdm

from clustered_routes import (
    ClusteredRoutesLoader,
    resolve_route_set,
    validate_clustered_route_set,
)
from utils import clear_SUMO_files


def _get_agents_valid_actions(all_agents, action_masks, num_actions):
    valid_by_agent = {}
    default_mask = np.ones(num_actions, dtype=np.int8)

    for agent in all_agents:
        mask = action_masks.get((agent.origin, agent.destination), default_mask)
        valid = np.flatnonzero(mask == 1)
        # valid = np.asarray(mask == 1).nonzero()[0] # indices of valid actions
        if len(valid) == 0:
            raise ValueError(f"No valid actions for agent {agent.id}")
        valid_by_agent[agent.id] = valid

    return valid_by_agent

def _open_joint_actions_memmap(output_path, num_rows, num_agents, dtype=np.uint8):
    return np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=dtype,
        shape=(num_rows, num_agents),
    )

def _joint_action_row_to_dict(all_agents, joint_action_row):
    """
    Converts one joint action row from the memmap file (1D numpy array) and converts it back to a dict.
    This is because the memmap stores joint actions as a compact array but the rest of the code uses
    more readable dicts.
    """
    return {
        agent.id: int(action)
        for agent, action in zip(all_agents, joint_action_row)
    }

def _append_rows_to_csv(csv_path, rows, fieldnames):
    """Appends rows with given field names to a csv file."""
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

def _filter_csv_by_experiment_ids(csv_path, discarded_experiment_ids):
    if not discarded_experiment_ids or not os.path.exists(csv_path):
        return

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return

    if "exp_id" not in df.columns:
        return

    filtered_df = df[~df["exp_id"].isin(discarded_experiment_ids)]
    if filtered_df.empty:
        os.remove(csv_path)
    else:
        filtered_df.to_csv(csv_path, index=False)

def _split_samples_evenly(total_samples, num_points):
    if num_points <= 0:
        return []

    base_count, remainder = divmod(total_samples, num_points)
    return [base_count + (1 if idx < remainder else 0) for idx in range(num_points)]

def _split_indices_evenly(total_items, num_chunks, chunk_index):
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")

    if chunk_index < 0 or chunk_index >= num_chunks:
        raise ValueError(f"chunk_index {chunk_index} is out of range for num_chunks={num_chunks}")

    base_count, remainder = divmod(total_items, num_chunks)
    start = chunk_index * base_count + min(chunk_index, remainder)
    stop = start + base_count + (1 if chunk_index < remainder else 0)
    return start, stop

def _write_single_method_joint_actions(all_agents, valid_by_agent, num_actions, rng, num_samples, mode, joint_actions_output, save_joint_actions_every):
    num_agents = len(all_agents)

    if mode == "greedy":
        # outside the for agent in all_agents loop because here we don't generate all actions for each
        # agent at once but instead create joint actions greedily, one by one
        def hamming_dists(past_actions, action):
            # array of Hamming distances between each of the past actions and the new action
            # Hamming distance is counted here as the sum of mismatched positions ([1,2,3] and [4,5,6] differ by 3)
            return np.sum(past_actions != action, axis=1)

        greedy_num_candidates = int(globals()["greedy_num_candidates"])

        # valid_by_agent is built in all_agents order, so its values align with
        # sim_matrix columns and with _joint_action_row_to_dict(...).

        # np matrix to store simulations [num_samples, num_agents]. logic: sim_matrix[simulation_index, agent_index]
        sim_matrix = _open_joint_actions_memmap(joint_actions_output, num_samples, num_agents)

        with tqdm(total=num_samples, desc=f"sampling:{mode}") as pbar:
            # random first joint action
            for j, v in enumerate(valid_by_agent.values()):
                sim_matrix[0, j] = rng.choice(v)

            pbar.update(1)

            # subsequent simulations choose actions greedily by maximizing
            # the Hamming distance to its nearest neighbor (max-min)
            for i in range(1, num_samples):
                best_cand = None
                min_hamming_dist = -1

                for _ in range(greedy_num_candidates):
                    candidate = np.array([rng.choice(v) for v in valid_by_agent.values()])
                    curr_min_hamming_dist = hamming_dists(sim_matrix[:i], candidate).min()
                    if curr_min_hamming_dist > min_hamming_dist:
                        min_hamming_dist = curr_min_hamming_dist
                        best_cand = candidate

                sim_matrix[i] = best_cand
                pbar.update(1)
                if save_joint_actions_every > 0 and (i + 1) % save_joint_actions_every == 0:
                    sim_matrix.flush()

        sim_matrix.flush()
        return num_samples

    elif mode == "dirichlet":
        # This is a simplex sample - for 4 actions, weights sum to 1 and are >= 0
        dirichlet_alpha = float(globals()["dirichlet_alpha"])
        joint_actions_mm = _open_joint_actions_memmap(joint_actions_output, num_samples, num_agents)

        with tqdm(total=num_samples, desc=f"sampling:{mode}") as pbar:
            for simulation_idx in range(num_samples):
                # alpha = 1.0 gives uniform over the simplex (e.g. 0.9, 0.05, 0.03, 0.02)
                # alpha > 1.0 pushes weights toward being more balanced (e.g. 4 x 0.25)
                alpha = np.ones(num_actions) * dirichlet_alpha
                weights = rng.dirichlet(alpha)

                for agent_idx, agent in enumerate(all_agents):
                    valid = valid_by_agent[agent.id]
                    probs = weights[valid]
                    if probs.sum() <= 0:
                        probs = np.ones(len(valid), dtype=float) / len(valid)
                    else:
                        probs = probs / probs.sum()
                    joint_actions_mm[simulation_idx, agent_idx] = int(rng.choice(valid, p=probs))

                pbar.update(1)
                if save_joint_actions_every > 0 and (simulation_idx + 1) % save_joint_actions_every == 0:
                    joint_actions_mm.flush()

        joint_actions_mm.flush()
        return num_samples

    elif mode == "grid":
        grid_values = int(globals()["grid_values"])
        grid_num_samples = int(globals()["grid_num_samples"])
        task_id_env = (
            os.environ.get("ASGN_METHOD_TASK_INDEX")
            if "ASGN_METHOD_TASK_INDEX" in os.environ
            else os.environ.get("TASK_ID")
        )
        if task_id_env is None:
            # Local runs do not have a launcher-provided task id, so cover the
            # full grid even if the server config splits grid over many tasks.
            num_tasks_grid = 1
            task_id = 0
        else:
            num_tasks_grid = int(globals().get("num_tasks_grid", 1))
            task_id = int(task_id_env)

        if grid_values < 2:
            raise ValueError("grid_values must be at least 2")

        total_steps = grid_values - 1
        grid_scale = float(total_steps)
        agent_valid_items = list(valid_by_agent.items())

        if num_actions == 1:
            grid_count_iter = [(total_steps,)]
        else:
            separator_positions = total_steps + num_actions - 1
            grid_count_iter = []

            for cuts in combinations(range(separator_positions), num_actions - 1):
                counts = []
                previous_cut = -1

                for cut in cuts:
                    counts.append(cut - previous_cut - 1)
                    previous_cut = cut

                counts.append(separator_positions - previous_cut - 1)
                grid_count_iter.append(tuple(counts))

        total_grid_points = len(grid_count_iter)
        grid_start, grid_stop = _split_indices_evenly(total_grid_points, num_tasks_grid, task_id)
        grid_count_iter = grid_count_iter[grid_start:grid_stop]

        if not grid_count_iter:
            return 0

        print(
            f"Grid task {task_id + 1}/{num_tasks_grid} covers simplex points {grid_start}-{grid_stop - 1} "
            f"out of {total_grid_points} total points"
        )

        samples_per_grid_point = _split_samples_evenly(grid_num_samples, len(grid_count_iter))
        total_grid_samples = sum(samples_per_grid_point)
        joint_actions_mm = _open_joint_actions_memmap(joint_actions_output, total_grid_samples, num_agents)

        with tqdm(total=total_grid_samples, desc=f"sampling:{mode}") as pbar:
            row_idx = 0
            for counts, num_point_samples in zip(grid_count_iter, samples_per_grid_point):
                if num_point_samples <= 0:
                    continue

                weights = np.asarray(counts, dtype=float) / grid_scale

                for _ in range(num_point_samples):
                    for agent_idx, (agent_id, valid) in enumerate(agent_valid_items):
                        probs = weights[valid]
                        if probs.sum() <= 0:
                            probs = np.ones(len(valid), dtype=float) / len(valid)
                        else:
                            probs = probs / probs.sum()
                        joint_actions_mm[row_idx, agent_idx] = int(rng.choice(valid, p=probs))

                    row_idx += 1
                    pbar.update(1)
                    if save_joint_actions_every > 0 and row_idx % save_joint_actions_every == 0:
                        joint_actions_mm.flush()

        joint_actions_mm.flush()
        return total_grid_samples

    elif mode == "random":
        joint_actions_mm = _open_joint_actions_memmap(joint_actions_output, num_samples, num_agents)
        with tqdm(total=num_samples * len(all_agents), desc=f"sampling:{mode} (pbar: num_samples*num_agents)") as pbar:
            for agent_idx, agent in enumerate(all_agents):
                joint_actions_mm[:, agent_idx] = rng.choice(
                    valid_by_agent[agent.id],
                    size=num_samples
                ) # no vectorization of the other dimension because valid actions vary by agent

                pbar.update(num_samples)
                if save_joint_actions_every > 0 and (agent_idx + 1) % save_joint_actions_every == 0:
                    joint_actions_mm.flush()

        joint_actions_mm.flush()
        return num_samples

    elif mode == "balanced":
        joint_actions_mm = _open_joint_actions_memmap(joint_actions_output, num_samples, num_agents)
        with tqdm(total=num_samples * len(all_agents), desc=f"sampling:{mode} (pbar: num_samples*num_agents)") as pbar:
            for agent_idx, agent in enumerate(all_agents):
                valid = valid_by_agent[agent.id]
                full_repeats = num_samples // len(valid)
                agent_actions = np.tile(valid, full_repeats)

                remainder = num_samples % len(valid)
                if remainder > 0:
                    agent_actions_remainder = rng.choice(valid, remainder, replace=False)
                    agent_actions = np.concatenate([agent_actions, agent_actions_remainder])

                rng.shuffle(agent_actions)
                joint_actions_mm[:, agent_idx] = agent_actions

                pbar.update(num_samples)
                if save_joint_actions_every > 0 and (agent_idx + 1) % save_joint_actions_every == 0:
                    joint_actions_mm.flush()

        joint_actions_mm.flush()
        return num_samples

    elif mode == "hex":
        def entropy(counts):
            # ln 195 ~= 5.27 - max entropy

            counts = np.array(counts, dtype=float)
            total = counts.sum()

            if total == 0:
                return 0.0

            p = counts / total
            p = p[p > 0]
            # 0 log 0 = 0 (entropy of a zero probability contribution)
            # this doesn't change anything so remove the 0 probs
            # to avoid numerical errors

            H = - np.sum(p * np.log(p))

            return H

        num_hexes = len(all_hexes)
        if num_hexes == 0:
            raise ValueError("No hexes found in hex_lookup; cannot run hex sampling.")
        hex_to_idx = {h: i for i, h in enumerate(all_hexes)}

        # Precompute a count vector for each path/action
        # This avoids repeatedly doing new_hexes.count(hex), which is more expensive
        hex_delta_lookup = {}

        for key, sequence in hex_lookup.items():
            delta = np.zeros(num_hexes, dtype=np.int32)

            for h in sequence:
                if h in hex_to_idx:
                    delta[hex_to_idx[h]] += 1

            hex_delta_lookup[key] = delta

        # Validate that every feasible agent action has a hex sequence
        missing_keys = []
        for agent in all_agents:
            for action in valid_by_agent[agent.id]:
                key = (agent.origin, agent.destination, int(action))
                if key not in hex_delta_lookup:
                    missing_keys.append(key)

        if missing_keys:
            raise KeyError(
                f"Missing {len(missing_keys)} hex_lookup entries. "
                f"First few missing keys: {missing_keys[:10]}"
            )

        joint_actions_mm = _open_joint_actions_memmap(joint_actions_output, num_samples, num_agents)

        with tqdm(total=num_samples, desc=f"sampling:{mode}") as pbar:
            for simulation_idx in range(num_samples):
                hex_counts_sim = np.zeros(num_hexes, dtype=np.int32)

                # Randomize sequential joint action construction order between simulations
                # So that we don't end up with len([0,1,2,3]) distinct results
                agent_order = rng.permutation(num_agents)

                # First agent chooses randomly
                first_agent_idx = int(agent_order[0])
                first_agent = all_agents[first_agent_idx]
                first_action = int(rng.choice(valid_by_agent[first_agent.id]))
                joint_actions_mm[simulation_idx, first_agent_idx] = first_action

                first_key = (
                    first_agent.origin,
                    first_agent.destination,
                    first_action,
                )
                hex_counts_sim += hex_delta_lookup[first_key]

                # Subsequent agents choose actions greedily by maximizing the hex coverage (entropy)
                for agent_idx in agent_order[1:]:
                    agent_idx = int(agent_idx)
                    agent = all_agents[agent_idx]

                    max_H = -np.inf
                    eps = 1e-12
                    best_candidates = [] # list to randomly choose when there's a tie

                    for action in valid_by_agent[agent.id]:
                        action = int(action)
                        key = (agent.origin, agent.destination, action)

                        delta = hex_delta_lookup[key] # difference in hex counts
                        candidate_counts = hex_counts_sim + delta
                        H = entropy(candidate_counts)

                        if H > max_H + eps:
                            max_H = H
                            best_candidates = [(action, delta)]
                        elif abs(H - max_H) <= eps:
                            best_candidates.append((action, delta))

                    if not best_candidates:
                        raise RuntimeError(f"No hex action selected for agent {agent.id}")

                    choice_idx = int(rng.integers(len(best_candidates))) # rand int from 0 to len(best_candidates)-1
                    best_action, best_delta = best_candidates[choice_idx]

                    joint_actions_mm[simulation_idx, agent_idx] = best_action
                    hex_counts_sim += best_delta

                pbar.update(1)
                if save_joint_actions_every > 0 and (simulation_idx + 1) % save_joint_actions_every == 0:
                    joint_actions_mm.flush()

        joint_actions_mm.flush()
        return num_samples

def run_episode(env, joint_action):
    """
    Set default_action for all agents, run one episode, return travel times.

    Args:
        env: TrafficEnvironment (already started)
        joint_action: dict {agent_id (int): action (int)}

    Returns:
        travel_times: dict {agent_id (int): travel_time (float)}
        had_teleports: bool
    """

    # Set fixed actions
    for agent in env.all_agents:
        agent.default_action = joint_action.get(agent.id, None)

    env.reset()
    env.step() # humans-only: one call runs the full simulation

    # Ensure the default action was executed properly
    for agent in env.all_agents:
        intended = joint_action.get(agent.id, None)
        actual = agent.last_action
        if intended is not None and actual != intended:
            raise RuntimeError(
                f"ASGN action mismatch for agent {agent.id}: intended {intended}, actual {actual}"
            )

    # Read travel times snapshotted just before _reset_episode cleared them
    travel_times = {}
    for entry in env.last_episode_travel_times: # list that contains travel_times dicts with agent_id as key and travel_time as value
        agent_id = entry[kc.AGENT_ID]
        travel_times[agent_id] = entry[kc.TRAVEL_TIME]

    return travel_times, bool(getattr(env, "last_episode_had_teleports", False))

def _record_simulation_results(per_simulation_rows, all_path_stats, path_lookup, all_agents, joint_action, travel_times, simulation_id):
    for agent in all_agents:
        tt = travel_times.get(agent.id)
        if tt is None:
            continue # ?

        cluster = joint_action[agent.id]
        key = (agent.origin, agent.destination, cluster)

        if key not in all_path_stats:
            path_info = path_lookup[key]
            all_path_stats[key] = {
                "origin": agent.origin,
                "destination": agent.destination,
                "cluster": cluster,
                "path": path_info["path"],
                "free_flow_time": path_info["free_flow_time"],
                "count": 0,
                "mean": 0.0,
                "M2": 0.0,
                "std": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
            }

        per_simulation_rows.append({
            "simulation_id": simulation_id,
            "agent_id": agent.id,
            "action": cluster,
            "travel_time": f"{tt:.3f}"
        })

        # _update_path_stats(all_path_stats[key], tt)

    return len(travel_times)

def _run_joint_action_batch(env, joint_actions, all_path_stats, path_lookup, simulations_output, save_data_every, start_simulation_id, desc, discard_teleporting_simulations, discarded_experiment_ids, pbar=None):
    simulations_since_flush = 0
    fieldnames = ["simulation_id", "agent_id", "action", "travel_time"]
    per_simulation_rows = []

    for simulation_id, joint_action in enumerate(joint_actions, start=start_simulation_id):
        joint_action_dict = _joint_action_row_to_dict(env.all_agents, joint_action)

        env.unwrapped.simulator.experiment_id = simulation_id

        travel_times, had_teleports = run_episode(env, joint_action_dict)
        if discard_teleporting_simulations and had_teleports:
            discarded_experiment_ids.add(simulation_id)
            if pbar is not None:
                pbar.update()
                pbar.set_postfix({
                    "run": simulation_id,
                    "agents_recorded": 0,
                    "discarded_total": len(discarded_experiment_ids),
                })
            continue

        n_recorded = _record_simulation_results(
            per_simulation_rows,
            all_path_stats,
            path_lookup,
            env.all_agents,
            joint_action_dict,
            travel_times,
            simulation_id,
        )

        simulations_since_flush += 1
        if simulations_since_flush >= save_data_every:
            _append_rows_to_csv(simulations_output, per_simulation_rows, fieldnames)
            per_simulation_rows.clear()
            simulations_since_flush = 0

        if pbar is not None:
            pbar.update()
            pbar.set_postfix({
                "run": simulation_id,
                "agents_recorded": n_recorded,
                "discarded_total": len(discarded_experiment_ids),
            })

    if per_simulation_rows:
        _append_rows_to_csv(simulations_output, per_simulation_rows, fieldnames)
        per_simulation_rows.clear()

def _cleanup_simulation_outputs(env, records_folder):
    env.stop_simulation()
    clear_SUMO_files(
        os.path.join(records_folder, "SUMO_output"),
        os.path.join(records_folder, "episodes"),
        remove_additional_files=True,
    )

def _discard_teleporting_output_rows(records_folder, discarded_experiment_ids):
    _filter_csv_by_experiment_ids(os.path.join(records_folder, "SUMO_output", "all_snapshots.csv"), discarded_experiment_ids)
    _filter_csv_by_experiment_ids(os.path.join(records_folder, "SUMO_output", "all_departures.csv"), discarded_experiment_ids)

def _write_teleport_summary(records_folder, exp_id, mode, total_simulations, discard_teleporting_simulations, discarded_experiment_ids):
    num_discarded = len(discarded_experiment_ids)
    num_total = int(total_simulations)
    summary = {
        "experiment_id": exp_id,
        "mode": mode,
        "discard_teleporting_simulations": bool(discard_teleporting_simulations),
        "num_total_simulations": num_total,
        "num_discarded": num_discarded,
        "num_kept": num_total - num_discarded,
        "discard_rate": (num_discarded / num_total) if num_total > 0 else 0.0,
    }

    output_path = os.path.join(records_folder, "teleport_summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved teleport summary to {output_path}")

def save_agent_static_rows(all_agents, valid_by_agent, path_lookup, num_actions, output_path):
    task_id = int(os.environ.get("TASK_ID", "0")) # only save once - TASK_ID 0 always exists (as long as an array is used)
    if task_id != 0 or os.path.exists(output_path):
        return

    agent_static_rows = []

    for agent in all_agents:
        row = {
            "agent_id": agent.id,
            "origin": agent.origin,
            "destination": agent.destination,
            "start_time": agent.start_time,
        }

        valid_actions = {int(action) for action in valid_by_agent[agent.id]}

        for action in range(num_actions):
            if action in valid_actions:
                key = (agent.origin, agent.destination, action)
                row[f"action_{action}"] = path_lookup[key]["path"]
            else:
                row[f"action_{action}"] = None

        agent_static_rows.append(row)

    agent_static_df = pd.DataFrame(agent_static_rows)
    agent_static_df.to_csv(output_path, index=False)
    print(f"Saved static simulation data to {output_path}")

# ----------------------------
# ----------- MAIN -----------
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id',                 type=str, required=True)
    parser.add_argument('--task-conf',          type=str, default="asgn1")
    parser.add_argument('--net',                type=str, required=True)
    parser.add_argument('--env-seed',           type=int, default=42)
    parser.add_argument('--mode',               type=str, choices=("generate", "simulations", "sample"), default="sample")
    parser.add_argument(
        '--route-set',
        type=str,
        default=None,
        help="Named route-set subdirectory. Uses the network default when omitted.",
    )
    parser.add_argument('--sumo-output', action='store_true', default=False)
    args = parser.parse_args()

    exp_id          = args.id
    task_config     = args.task_conf
    network         = args.net
    env_seed        = args.env_seed
    mode            = args.mode
    route_set       = resolve_route_set(network, args.route_set)
    sumo_output     = args.sumo_output

    print("### ASSIGNMENT SAMPLER ###")
    # No external baseline script - actions get overwritten in this file
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Task config: {task_config}")
    print(f"Mode: {mode}")
    print(f"Generate SUMO departures and snapshots: {sumo_output}")
    print(f"Route set: {route_set}")

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    random.seed(env_seed)
    np.random.seed(env_seed)
    rng = np.random.default_rng(env_seed)

    # Parameter setting - except for alg_params
    params = dict()
    task_params = json.load(open(f"../config/task_config/{task_config}.json"))
    params.update(task_params)
    del task_params

    # set params as variables in this script
    for key, value in params.items():
        globals()[key] = value

    selected_method = os.environ.get("ASGN_METHOD") or None
    selected_num_samples = os.environ.get("ASGN_NUM_SAMPLES")
    if selected_method is not None:
        sample_methods = ("greedy", "random", "balanced", "dirichlet", "grid", "hex")
        if selected_method not in sample_methods:
            raise ValueError(f"Unknown ASGN_METHOD: {selected_method}")

        if selected_num_samples is None:
            raise ValueError("ASGN_NUM_SAMPLES must be set when ASGN_METHOD is set")

        selected_num_samples = int(selected_num_samples)
        for method in sample_methods:
            globals()[f"{method}_num_samples"] = selected_num_samples if method == selected_method else 0

    greedy_num_samples = int(globals().get("greedy_num_samples", 0))
    random_num_samples = int(globals().get("random_num_samples", 0))
    balanced_num_samples = int(globals().get("balanced_num_samples", 0))
    dirichlet_num_samples = int(globals().get("dirichlet_num_samples", 0))
    hex_num_samples = int(globals().get("hex_num_samples", 0))
    grid_num_samples = int(globals().get("grid_num_samples", 0))
    greedy_num_candidates = globals().get("greedy_num_candidates", 10)
    dirichlet_alpha = globals().get("dirichlet_alpha", 2.0)
    grid_values = globals().get("grid_values", 21)
    save_data_every = int(globals().get("save_data_every", 1))
    save_joint_actions_every = int(globals().get("save_joint_actions_every", save_data_every))
    discard_teleporting_simulations = os.environ.get("ASGN_DISCARD_TELEPORTING_SIMULATIONS", "1") != "0"

    print(f"Variable samples: greedy={greedy_num_samples}, random={random_num_samples}, balanced={balanced_num_samples}, dirichlet={dirichlet_num_samples}, grid={grid_num_samples}, hex={hex_num_samples}")
    print(f"Discard teleporting simulations: {discard_teleporting_simulations}")

    custom_network_folder = f"../networks/{network}"

    # Use /scratch on bonk
    base_results_dir = os.environ.get("RESULTS_BASE_DIR", "../results")
    experiment_name = os.environ.get("EXPERIMENT_NAME", exp_id)
    experiment_root = os.path.join(base_results_dir, experiment_name)
    records_folder = os.path.join(experiment_root, exp_id)
    plots_folder = os.path.join(records_folder, "plots")
    os.makedirs(records_folder, exist_ok=True)

    # Read origin-destinations
    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    data = ast.literal_eval(content)
    origins = data['origins']
    destinations = data['destinations']

    # Copy agents.csv from custom_network_folder to records_folder
    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    num_agents = len(pd.read_csv(agents_csv_path))
    if os.path.exists(agents_csv_path):
        os.makedirs(records_folder, exist_ok=True)
        new_agents_csv_path = os.path.join(records_folder, "agents.csv")
        with open(agents_csv_path, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(new_agents_csv_path, 'w', encoding='utf-8') as f:
            f.write(content)

    # Load clustered routes + action masks. ASGN sampling always requires clustered routes.
    route_set_dir = validate_clustered_route_set(
        network_name=network,
        route_set_dir=Path(custom_network_folder) / "clustered_routes" / route_set,
    )

    clustered_loader = ClusteredRoutesLoader(
        network,
        custom_network_folder,
        shuffle=False,
        seed=env_seed,
        route_set_dir=route_set_dir,
    )
    num_actions = clustered_loader.get_number_of_paths() # K
    clustered_loader.export_paths_routes(records_folder, origins, destinations)
    action_masks = clustered_loader.create_masks(origins, destinations)

    print(f"Number of clusters K = {num_actions}")

    # Create environment
    env = TrafficEnvironment(
        seed = env_seed,
        create_agents = False,
        create_paths = False, # use clustered routes
        action_masks = action_masks, # use clustered routes
        generate_asgn_data = sumo_output, # save SUMO_output files (use carefully; takes a lot of storage)
        save_detectors_info = False,
        agent_parameters = {
            "new_machines_after_mutation": 0, # no AVs at all; human actions are overwritten
            "human_parameters": {
                "model": "random" # model doesn't matter - default_action overwrites
            },
            "machine_parameters": {
                "behavior": "selfish", # doesn't matter no AVs anyway
            }
        },
        environment_parameters = {
            "save_every": 1_000_000_000, # don't save anything
        },
        simulator_parameters = {
            "network_name": network,
            "custom_network_folder": custom_network_folder,
            "sumo_type": "sumo",
            "use_libsumo": True,
        },
        plotter_parameters = {
            "smooth_by": 1,
            "plot_choices": "none",
            "records_folder": records_folder,
            "plots_folder": plots_folder,
        }, # no plots
        path_generation_parameters = {
            "origins": origins,
            "destinations": destinations,
            "number_of_paths": num_actions,
        } # placeholder
    )

    print(f"""
    Agents in the traffic:
    • Total agents  : {len(env.all_agents)}
    • Human agents  : {len(env.human_agents)}
    • AV agents     : {len(env.machine_agents)}
    """)

    env.start()
    env.reset()

    # Disable human learning - we just want travel times, no model updates
    env.human_learning = False

    # -----------------------------
    # -------- SIMULATIONS --------
    # -----------------------------

    # Init variables for online calculations
    all_path_stats = {} # path: stats
    valid_by_agent = _get_agents_valid_actions(env.all_agents, action_masks, num_actions)
    joint_action_sources = []
    total_joint_actions = 0
    simulations_output = os.path.join(records_folder, "simulations.csv")

    paths_df = pd.read_csv(os.path.join(records_folder, "paths.csv"))
    path_lookup = {}
    for row in paths_df.itertuples(index=False):
        key = (int(row.origins), int(row.destinations), int(row.cluster))
        if key in path_lookup:
            raise ValueError(f"Duplicate path mapping found for {key}")
        path_lookup[key] = {
            "path": row.path,
            "free_flow_time": row.free_flow_time,
        }

    # For the hex-greedy method:
    origin_to_idx = {str(edge): idx for idx, edge in enumerate(origins)}
    destination_to_idx = {str(edge): idx for idx, edge in enumerate(destinations)}

    hex_lookup = {}
    for row in clustered_loader.df.itertuples(index=False):
        key = (origin_to_idx[row.origins], destination_to_idx[row.destinations], int(row.cluster))
        if pd.isna(row.h3_sequence):
            sequence = tuple()
        else:
            sequence = tuple(
                h.strip() # remove surrounding whitespaces
                for h in str(row.h3_sequence).split(",")
                if h.strip() # filter out empty entries
            )

        hex_lookup[key] = sequence
    all_hexes = sorted({hex for sequence in hex_lookup.values() for hex in sequence})

    # to look up agent details using simulations.csv keys
    save_agent_static_rows(
        env.all_agents,
        valid_by_agent,
        path_lookup,
        num_actions,
        os.path.join(experiment_root, "simulations_static.csv"),
    )

    all_methods = [
        # name, num_samples
        ("greedy", greedy_num_samples),
        ("random", random_num_samples),
        ("balanced", balanced_num_samples),
        ("dirichlet", dirichlet_num_samples),
        ("grid", grid_num_samples),
        ("hex", hex_num_samples)
    ]

    discarded_experiment_ids = set()

    if mode in ("generate", "sample"):
        # writes joint actions to multiple npy (binary format for single numpy arrays) files
        # one per method, e.g. joint_actions_greedy.npy, joint_actions_random.npy, ...
        for method, count in all_methods:
            if count <= 0:
                continue

            joint_actions_output = os.path.join(records_folder, f"joint_actions_{method}.npy")
            rows_written = _write_single_method_joint_actions(
                env.all_agents,
                valid_by_agent,
                num_actions,
                rng,
                count, # doesn't matter if fixed
                method,
                joint_actions_output,
                save_joint_actions_every,
            )
            if rows_written <= 0:
                continue

            joint_action_sources.append({
                "method": method,
                "path": joint_actions_output,
                "rows": rows_written,
            })
            total_joint_actions += rows_written
            print(f"Saved joint actions for {method} to {joint_actions_output}")

        if total_joint_actions <= 0:
            raise ValueError("No joint actions were generated. Check the counts and fixed flags.")

        print(f"Saved {total_joint_actions} joint actions to {records_folder}")

    if mode == "generate":
        _cleanup_simulation_outputs(env, records_folder)
        sys.exit(0)

    if mode == "simulations":
        for method, count in all_methods:
            if count <= 0:
                continue

            joint_actions_output = os.path.join(records_folder, f"joint_actions_{method}.npy")
            if not os.path.exists(joint_actions_output):
                raise FileNotFoundError(f"Missing joint action file for {method}: {joint_actions_output}")

            joint_actions_mm = np.load(joint_actions_output, mmap_mode="r")
            rows_written = int(joint_actions_mm.shape[0])
            if rows_written <= 0:
                continue

            joint_action_sources.append({
                "method": method,
                "path": joint_actions_output,
                "rows": rows_written,
            })
            total_joint_actions += rows_written
            print(f"Loaded joint actions for {method} from {joint_actions_output}")

        if total_joint_actions <= 0:
            raise ValueError("No joint actions were requested by this job; exiting without simulations.")

    if mode in ("sample", "simulations"):
        print(f"Running {total_joint_actions} SUMO simulations ({total_joint_actions} sampling)\n")

        batch_size = 100
        start_simulation_id = 0
        with tqdm(total=total_joint_actions, desc="Sampling travel times") as pbar:
            for source in joint_action_sources:
                # open a joint action memmap file for a particular method
                joint_actions_mm = np.load(source["path"], mmap_mode="r")

                # access by loading small batches so as not to overload the memory
                for start in range(0, source["rows"], batch_size):
                    joint_actions_batch = joint_actions_mm[start:start + batch_size]

                    _run_joint_action_batch(
                        env,
                        joint_actions_batch,
                        all_path_stats,
                        path_lookup,
                        simulations_output,
                        save_data_every,
                        start_simulation_id + start,
                        f"Sampling travel times ({source['method']})",
                        discard_teleporting_simulations,
                        discarded_experiment_ids,
                        pbar=pbar,
                    )
                    # Keep SUMO output folders small by removing unwanted files
                    clear_SUMO_files(
                        os.path.join(records_folder, "SUMO_output"),
                        os.path.join(records_folder, "episodes"),
                        remove_additional_files=True,
                    )

                start_simulation_id += source["rows"]

        _discard_teleporting_output_rows(records_folder, discarded_experiment_ids)
        _write_teleport_summary(
            records_folder,
            exp_id,
            mode,
            total_joint_actions,
            discard_teleporting_simulations,
            discarded_experiment_ids,
        )

    # only deletes .xml files - all_departures.csv and all_snapshots.csv are safe
    _cleanup_simulation_outputs(env, records_folder)
