import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
import argparse
import ast
import json
import logging
import random

import numpy as np
import pandas as pd

from routerl import TrafficEnvironment
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    # Any additional arguments can be added here
    
    PLACEHOLDER = None # Delete this line and add your own arguments in the following
    
    args = parser.parse_args()
    exp_id = args.id
    exp_config = args.conf
    network = args.net
    env_seed = args.env_seed
    # ... and should be passed to the script
    
    print("### STARTING EXPERIMENT ###")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Experiment config: {exp_config}")


    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    random.seed(env_seed)
    np.random.seed(env_seed)
        
    # #### Hyperparameters setting
    params = json.load(open("../experiment_metadata.json"))
    params = params[exp_config]["config"]
    
    # set params as variables in this script
    for key, value in params.items():
        globals()[key] = value

    custom_network_folder = f"../networks/{network}"
    phases = [1, PLACEHOLDER, PLACEHOLDER] # Define the phases as per your requirement
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
            
    should_humans_adapt = PLACEHOLDER # Define whether humans should adapt or not while AVs are learning
    human_learning_episodes = PLACEHOLDER # Define the number of human learning episodes as per your requirement
    num_machines = PLACEHOLDER # Define the number of machines as per your requirement
    total_episodes = PLACEHOLDER # Define the total number of episodes as per your requirement
            
    # Dump exp config to records
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    dump_config["network"] = network
    dump_config["env_seed"] = env_seed
    dump_config["config"] = exp_config
    dump_config["num_agents"] = num_agents
    dump_config["num_machines"] = num_machines
    # Any other parameters you want to save in `exp_config.json` can be added here
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    
    env = TrafficEnvironment(
        seed = env_seed,
        create_agents = False,
        create_paths = True,
        save_detectors_info = False,
        agent_parameters = {
            "new_machines_after_mutation": num_machines, 
            "human_parameters" : {
                "model" : PLACEHOLDER, # Select the human model as per your requirement
            },
            "machine_parameters" :{
                "behavior" : PLACEHOLDER, # Select the machine behavior as per your requirement
            }
        },
        environment_parameters = {
            "save_every" : PLACEHOLDER, # Define the disk save frequency as per your requirement
        },
        simulator_parameters = {
            "network_name" : network,
            "custom_network_folder" : custom_network_folder,
            "sumo_type" : "sumo",
        }, 
        plotter_parameters = {
            "phases" : phases,
            "phase_names" : phase_names,
            "smooth_by" : PLACEHOLDER, # Define the smoothing factor as per your requirement
            "plot_choices" : PLACEHOLDER, # Define the plot choices as per your requirement,
            "records_folder" : records_folder,
            "plots_folder" : plots_folder
        },
        path_generation_parameters = {
            "origins" : origins,
            "destinations" : destinations,
            "number_of_paths" : PLACEHOLDER, # Define the number of paths per OD as per your requirement
            "beta" : PLACEHOLDER,
            "num_samples" : PLACEHOLDER,
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
    |
    v
    """
    
    # Save results
    os.makedirs(plots_folder, exist_ok=True)
    env.plot_results()

    env.stop_simulation()