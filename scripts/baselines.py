"""
This script is used to train AV agents using the baseline methods in a traffic simulation environment.
Baseline methods can be found in the baseline_models/ directory.
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import ast
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
from routerl import Keychain as kc
from routerl import TrafficEnvironment
from tqdm import tqdm

from baseline_models import get_baseline
from utils import add_model_snapshot_argument
from utils import clear_SUMO_files
from utils import run_metrics_analysis
from utils import script_path_for_config
from utils import print_agent_counts
from utils import warn_model_snapshots_unsupported


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--env-conf', type=str, default="config1")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--skip-metrics', action='store_true', default=False)
    add_model_snapshot_argument(parser)
    args = parser.parse_args()
    warn_model_snapshots_unsupported(args.save_model_every)
    
    ALGORITHM = "baseline"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    baseline_model = args.model
    
    print("### STARTING EXPERIMENT ###")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")
    print(f"Baseline model: {baseline_model}")
    print(f"Metrics will {'NOT ' if args.skip_metrics else ''}be computed after the experiment.\n")

    # Check if baseline exists
    baseline_dir = Path(repo_root) / "baseline_models"
    available_models = {file.stem for file in baseline_dir.glob("*.py")}
    assert baseline_model in available_models, \
        f"Baseline model '{baseline_model}' not found in {baseline_dir}/. Available: {sorted(available_models)}"

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    random.seed(env_seed)
    np.random.seed(env_seed)
        
    # Parameter setting
    params = dict()
    alg_params = json.load(open(f"../config/algo_config/{ALGORITHM}/{alg_config}.json"))
    env_params = json.load(open(f"../config/env_config/{env_config}.json"))
    task_params = json.load(open(f"../config/task_config/{task_config}.json"))
    params.update(alg_params)
    params.update(env_params)
    params.update(task_params)
    del params["desc"], alg_params, env_params, task_params

    # set params as variables in this script
    for key, value in params.items():
        globals()[key] = value

    
    custom_network_folder = f"../networks/{network}"
    phases = [1, human_learning_episodes, int(training_eps) + human_learning_episodes]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]
    records_folder = f"../results/{exp_id}"
    plots_folder = f"../results/{exp_id}/plots"

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
        max_start_time = pd.read_csv(new_agents_csv_path)['start_time'].max()
    else:
        raise FileNotFoundError(f"Agents CSV file not found at {agents_csv_path}. Please check the network folder.")
            
    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + test_eps
            
    # Dump exp config to records
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    if args.project is not None:
        dump_config["project"] = args.project
    dump_config["network"] = network
    dump_config["env_seed"] = env_seed
    dump_config["env_config"] = env_config
    dump_config["task_config"] = task_config
    dump_config["alg_config"] = alg_config
    dump_config["baseline_model"] = baseline_model
    dump_config["num_agents"] = num_agents
    dump_config["num_machines"] = num_machines
    dump_config["script"] = script_path_for_config(__file__)
    dump_config["algorithm"] = ALGORITHM
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    # Initiate the traffic environment
    env = TrafficEnvironment(
        seed = env_seed,
        create_agents = False,
        create_paths = True,
        save_detectors_info = False,
        agent_parameters = {
            "new_machines_after_mutation": num_machines, 
            "human_parameters": {
                "model": human_model,
                "alpha": human_alpha,
                "beta": human_beta,
                "beta_randomness": human_beta_randomness,
                "deterministic": human_deterministic,
            },
            "machine_parameters" :{
                "behavior" : av_behavior,
                "observation_type" : observations
            }
        },
        environment_parameters = {
            "save_every" : save_every,
        },
        simulator_parameters = {
            "network_name" : network,
            "custom_network_folder" : custom_network_folder,
            "sumo_type" : "sumo",
            "simulation_timesteps" : max_start_time
        }, 
        plotter_parameters = {
            "phases" : phases,
            "phase_names" : phase_names,
            "smooth_by" : smooth_by,
            "plot_choices" : plot_choices,
            "records_folder" : records_folder,
            "plots_folder" : plots_folder
        },
        path_generation_parameters = {
            "origins" : origins,
            "destinations" : destinations,
            "number_of_paths" : number_of_paths,
            "beta" : path_gen_beta,
            "num_samples" : num_samples,
            "path_gen_workers" : path_gen_workers,
            "visualize_paths" : False
        } 
    )

    print_agent_counts(env)
    env.start()
    res = env.reset()

     
    # Human learning
    pbar = tqdm(total=total_episodes, desc="Human learning")
    for episode in range(human_learning_episodes):
        env.step()
        pbar.update()

    #  Mutation
    pre_mutation_agents = env.all_agents.copy()
    env.mutation(disable_human_learning = not should_humans_adapt, mutation_start_percentile = -1)
    print_agent_counts(env)

    # Replace AV models with baseline models
    machines = env.machine_agents.copy()
    mutated_humans = dict()

    for machine in machines:
        for human in pre_mutation_agents:
            if human.id == machine.id:
                mutated_humans[str(machine.id)] = human
                break
            
    human_learning_params = env.agent_params[kc.HUMAN_PARAMETERS]
    human_learning_params["model"] = baseline_model
    free_flows = env.get_free_flow_times()
    for h_id, human in mutated_humans.items():
        initial_knowledge = free_flows[(human.origin, human.destination)]
        initial_knowledge = [-1 * item for item in initial_knowledge]
        mutated_humans[h_id].model = get_baseline(human_learning_params, initial_knowledge)
       
    # Training
    pbar.set_description("AV learning")
    for episode in range(training_eps):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                obs = [{kc.AGENT_ID : int(agent), kc.TRAVEL_TIME : -reward}]
                last_action = mutated_humans[agent].last_action
                mutated_humans[agent].learn(last_action, obs)
                action = None
            else:
                action = mutated_humans[agent].act(0)
                mutated_humans[agent].last_action = action

            env.step(action)
        pbar.update()
    
    # Testing
    pbar.set_description("Testing")
    for episode in range(test_eps):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                action = mutated_humans[agent].act(0)
            env.step(action)
        pbar.update()

    pbar.close()
    os.makedirs(plots_folder, exist_ok=True)
    env.plot_results()

    env.stop_simulation()

    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)
    if not args.skip_metrics:
        run_metrics_analysis(exp_id, results_folder="../results")
