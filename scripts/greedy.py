"""
This script is used to train AV agents with baseline greedy algorithm (version with traffic history lookup).
"""

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
import argparse
import ast
import json
import logging
import random

import numpy as np
import pandas as pd


from routerl import Keychain as kc
from routerl import TrafficEnvironment
from utils import clear_SUMO_files
from tqdm import tqdm

import greedy_utils



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--env-conf', type=str, default="config1")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    # Any additional arguments can be added here
    
    
    args = parser.parse_args()
    ALGORITHM = 'greedy'
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    # ... and should be passed to the script
    
    # Initial print
    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {ALGORITHM.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}\n")

    

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

    # Set params as variables in this script
    for key, value in params.items():
        globals()[key] = value


    # Define input / output paths and plotting options 
    custom_network_folder = f"../networks/{network}"
    phases = [1, human_learning_episodes, int(training_eps) + human_learning_episodes] ## Define the phases as per your requirement
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

    
    num_machines = int(num_agents * ratio_machines) # Define the number of machines as per your requirement
    total_episodes =  human_learning_episodes + training_eps + test_eps ## Define the total number of episodes as per your requirement
            
    # Dump exp config to records
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    dump_config["network"] = network
    dump_config["env_seed"] = env_seed
    dump_config["env_config"] = env_config
    dump_config["task_config"] = task_config
    dump_config["alg_config"] = alg_config
    dump_config["num_agents"] = num_agents
    dump_config["num_machines"] = num_machines
    dump_config["algorithm"] = ALGORITHM
    # ...any other parameters you want to save in `exp_config.json` can be added here
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)


    # Create environment
    env = TrafficEnvironment(
        seed = env_seed,
        create_agents = False,
        create_paths = True,
        save_detectors_info = False,
        agent_parameters = {
            "new_machines_after_mutation": num_machines, 
            "human_parameters" : {
                "model" : human_model, ## Select the human model as per your requirement
            },
            "machine_parameters" :{
                "behavior" : av_behavior, ## Select the machine behavior as per your requirement
            }
        },
        environment_parameters = {
            "save_every" : save_every, ## Define the disk save frequency as per your requirement
        },
        simulator_parameters = {
            "network_name" : network,
            "custom_network_folder" : custom_network_folder,
            "sumo_type" : "sumo",
        }, 
        plotter_parameters = {
            "phases" : phases,
            "phase_names" : phase_names,
            "smooth_by" : smooth_by, ## Define the smoothing factor as per your requirement
            "plot_choices" : plot_choices, ## Define the plot choices as per your requirement,
            "records_folder" : records_folder,
            "plots_folder" : plots_folder
        },
        path_generation_parameters = {
            "origins" : origins,
            "destinations" : destinations,
            "number_of_paths" : number_of_paths, # Define the number of paths per OD as per your requirement
            "beta" : path_gen_beta,
            "num_samples" : num_samples,
            "visualize_paths" : False
        } 
    )

    print(f"""
    Agents in the traffic:
    • Total agents           : {len(env.all_agents)}
    • Human agents           : {len(env.human_agents)}
    • AV agents              : {len(env.machine_agents)}
    """)

    env.start()
    res = env.reset()

    
    # #### Human learning
    print(f"env.agents={env.agents}")
    pbar = tqdm(total=total_episodes, desc="Human learning")
    for episode in range(human_learning_episodes):
        env.step()
        pbar.update()

    # #### Mutation
    env.mutation(disable_human_learning = not should_humans_adapt, mutation_start_percentile = -1)

    print(f"""
    Agents in the traffic:
    • Total agents           : {len(env.all_agents)}
    • Human agents           : {len(env.human_agents)}
    • AV agents              : {len(env.machine_agents)}
    """)


    """
    ^
    |
    User defined AV learning pipeline!
    """
    
    pbar.set_description("AV learning\n")

    # Auxiliary structures
    experiment_records = greedy_utils.initialize_experiment_records(traffic_environment=env)
    agent_mapping = {agent.id : i for i,agent in enumerate(env.all_agents)} # mapping: agent id to agent position in env.all_agents (list[Agent]) 
    

    for episode in range(training_eps + test_eps):
        env.reset()
        episode_actions = dict()

        # Iterate over machine agents (only machine agents ids are added to env.possible_agents during mutation)
        for agentid in env.agent_iter():

            # Pick Agent object corresponding to agentid
            agentid_int = int(agentid)
            agent = env.all_agents[agent_mapping[agentid_int]]
            assert agent.id==agentid_int # ensure that agent object id matches agentid
            

            observation, reward, termination, truncation, info = env.last()
   
            if termination or truncation:

                # Update experiment records with agent's episode info
                travel_time = -reward
                greedy_utils.update_records(
                    od=(agent.origin, agent.destination),
                    timestamp=agent.start_time,
                    route=episode_actions[agentid_int],
                    duration=travel_time,
                    episode=episode,
                    records=experiment_records
                )
                action = None

            
            else:

                # Select action for the current agent
                action = greedy_utils.select_agent_action(agent=agent, records=experiment_records)
                episode_actions[agentid_int] = action

            env.step(action)
        pbar.update()

    """
    |
    v
    """
    
    # Save results
    os.makedirs(plots_folder, exist_ok=True)
    env.plot_results()

    env.stop_simulation()

    # Clean SUMO-generated redundant files
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)