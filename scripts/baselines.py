
import argparse
import ast
import json
import logging
import os
import pandas as pd

import routerl

from routerl import Keychain as kc
from routerl import TrafficEnvironment

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    exp_id = args.id
    exp_config = args.conf
    network = args.net
    seed = args.seed
    baseline_model = args.model
    assert baseline_model in kc.HUMAN_MODELS, f"Model {baseline_model} not in {kc.HUMAN_MODELS}"
    print("### STARTING EXPERIMENT ###")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Seed: {seed}")
    print(f"Baseline model: {baseline_model}")

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

     
    # #### Hyperparameters setting

    
    params = json.load(open("../experiment_metadata.json"))
    params = params[exp_config]["config"]

    
    # set params as variables in this notebook
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
            
    num_machines = int(num_agents * ratio_machines)
            
    # Dump exp config to records
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    dump_config["network"] = network
    dump_config["seed"] = seed
    dump_config["config"] = exp_config
    dump_config["baseline_model"] = baseline_model
    dump_config["num_agents"] = num_agents
    dump_config["num_machines"] = num_machines
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    
    env = TrafficEnvironment(
        seed = seed,
        create_agents = False,
        create_paths = True,
        save_detectors_info = False,
        agent_parameters = {
            "new_machines_after_mutation": num_machines, 
            "human_parameters" : {
                "model" : human_model
            },
            "machine_parameters" :{
                "behavior" : av_behavior,
            }
        },
        environment_parameters = {
            "save_every" : save_every,
        },
        simulator_parameters = {
            "network_name" : network,
            "custom_network_folder" : custom_network_folder,
            "sumo_type" : "sumo",
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
            "visualize_paths" : False
        } 
    )

    
    print("Number of total agents is: ", len(env.all_agents), "\n")
    print("Number of human agents is: ", len(env.human_agents), "\n")
    print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

    
    env.start()
    res = env.reset()

     
    # #### Human learning
    
    for episode in range(human_learning_episodes):
        env.step()

    # #### Mutation

    
    pre_mutation_agents = env.all_agents.copy()
    env.mutation(mutation_start_percentile = -1)

    
    print("Number of total agents is: ", len(env.all_agents), "\n")
    print("Number of human agents is: ", len(env.human_agents), "\n")
    print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

    
    machines = env.machine_agents.copy()
    mutated_humans = dict()

    for machine in machines:
        for human in pre_mutation_agents:
            if human.id == machine.id:
                mutated_humans[str(machine.id)] = human
                break
            
    human_learning_params = env.agent_params[kc.HUMAN_PARAMETERS]
    human_learning_params[kc.MODEL] = baseline_model
    free_flows = env.get_free_flow_times()
    for h_id, human in mutated_humans.items():
        initial_knowledge = free_flows[(human.origin, human.destination)]
        mutated_humans[h_id].model = routerl.get_learning_model(human_learning_params, initial_knowledge)
       
    
    for episode in range(training_eps):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                obs = [{kc.AGENT_ID : int(agent), kc.TRAVEL_TIME : reward}]
                last_action = mutated_humans[agent].last_action
                mutated_humans[agent].learn(last_action, obs)
                action = None
            else:
                action = mutated_humans[agent].act(0)
                mutated_humans[agent].last_action = action

            env.step(action)
    
    
    for episode in range(test_eps):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                action = mutated_humans[agent].act(0)
            env.step(action)

    
    os.makedirs(plots_folder, exist_ok=True)
    env.plot_results()

    
    env.stop_simulation()