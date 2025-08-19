"""
This script is used to train AV agents with baseline greedy algorithm.
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

"""
Currently looks only at av travel times for route assesments (develop to enable also human times access).
Uses only the last day (episode) for the current episode action choice.
    =>Does not use past agent times to assess the route.

- [] how to retrieve human travel times for route times estimation

"""



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--env-conf', type=str, default="config1")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    # Any additional arguments can be added here
    
    
    ## PLACEHOLDER = None # Delete this line and add your own arguments in the following
    
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
    
    ################
    # Training
    ################
    pbar.set_description("AV learning\n")


    free_flows = env.get_free_flow_times() # free flow times for (origin, destination) pairs
    experiment_records = dict()

    # Auxiliary structures for working with env agents
    agent_mapping = {agent.id : i for i,agent in enumerate(env.all_agents)} # auxiliary mapping: agent id to agent position in env.all_agents (list[Agent]) 
    



    ##################################################################
    ## control print: show env.possible_agents and current env.agents
    print(f"env.agents={env.agents}")
    print(f"env.possible_agents={env.possible_agents}\n")
    ###################################################################

    for episode in range(training_eps + test_eps):
        env.reset()
        print(f"Dummy episode: {env.day}")
        print(f"Experiment records: {experiment_records}\n")


        episode_records = {int(aid): dict() for aid in env.possible_agents} # per-episode data
        

        for agentid in env.agent_iter(): # Iterate over machine agents (only machine agents ids are added to env.possible_agents during mutation)

            # Get Agent object
            agentid_int = int(agentid)
            agent = env.all_agents[agent_mapping[agentid_int]]
            assert agent.id==agentid_int # ensure that agent id matches
            
            ############################################################################
            # Sanity check (remove later)
            print(f"env.agents={env.agents}")
            print(f"Agent: id={agent.id}, kind={agent.kind}, start_time={agent.start_time}")
            #############################################################################


            observation, reward, termination, truncation, info = env.last()
            print(f"observation, reward, termination, truncation, info = ({observation}, {reward}, {termination}, {truncation}, {info})")

            
            if termination or truncation: # Case: agent finished their drive - save agent episode data
                print(f"in termination / truncation branch for agent {agentid}")

                travel_time = -reward
                episode_records[agentid_int].update(
                                            {
                                                'kind': agent.kind,
                                                'origin': agent.origin,
                                                'destination': agent.destination,
                                                #'route': action,
                                                'travel_time': travel_time,
                                                'time_start': agent.start_time,
                                                'time_end': agent.start_time + travel_time,
                                            })
                action = None

            
            else: # Case: agent is starting from their desination - select the action (route)
                print(f"in action branch for agent {agentid}")

                if env.day == 0:
                    od_free_flows = free_flows[(agent.origin, agent.destination)] # per-route free flow times
                    action = random.choice([route for route, time in enumerate(od_free_flows) if time == (min_time := min(od_free_flows))])
                else:
                    action = greedy_utils.choose_agent_action(agent=agent, day=env.day, experiment_records=experiment_records, free_flow_times=free_flows)

                episode_records[agentid_int]['route'] = action


            print(f"action: {action}\n")
            env.step(action)

        # Log episode data
        experiment_records[env.day-1] = episode_records # day incremented after last step (when no agents to act/terminate)
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

    # print("Cleaning up manually")
    # del experiment_data
    # del env
    # del agent_mapping
    #print("\nEND OF SCRIPT\n")