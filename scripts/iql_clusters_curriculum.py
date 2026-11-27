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
import re

import numpy as np
import pandas as pd
import torch

from routerl         import TrafficEnvironment, MachineAgent
from tqdm            import tqdm

from utils           import add_model_snapshot_argument
from utils           import clear_SUMO_files
from utils           import model_snapshot_path
from utils           import print_agent_counts
from utils           import run_metrics_analysis
from utils           import script_path_for_config
from utils           import should_save_model_snapshot

from clustered_routes import AVMaskWrapper, ClusteredRoutesLoader, resolve_route_set
from iql_clusters     import DQN


def format_json(data, min_indent=12):
    return re.sub(
        r"(?m)^( +)-?\d+,?(?:\n\1-?\d+,?){1,9}$",
        lambda m: m[0].replace("\n" + m[1], " ") if len(m[1]) >= min_indent else m[0],
        json.dumps(data, indent=4)
    )

def mutate_exact_agent_ids(traffic_env, selected_ids, disable_human_learning=True):
    humans_by_id = {int(agent.id): agent for agent in traffic_env.human_agents}
    machine_params = traffic_env.agent_params["machine_parameters"]

    for agent_id in sorted(set(selected_ids)):
        human = humans_by_id[int(agent_id)]
        traffic_env.human_agents.remove(human)
        traffic_env.machine_agents.append(
            MachineAgent(human.id, human.start_time, human.origin,
                         human.destination, machine_params, traffic_env.action_space_size)
        )
        traffic_env.possible_agents.append(str(human.id))

    traffic_env.n_agents = len(traffic_env.possible_agents)
    traffic_env.all_agents = traffic_env.machine_agents + traffic_env.human_agents
    if disable_human_learning:
        traffic_env.human_learning = False
    traffic_env._initialize_machine_agents()


# Main script to run the IQL experiment
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--env-conf', type=str, default="clusters")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--torch-seed', type=int, default=42)
    parser.add_argument(
        '--route-set',
        type=str,
        default=None,
        help="Named route-set subdirectory. Uses the network default when omitted.",
    )
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument('--skip-metrics', action='store_true', default=False)
    add_model_snapshot_argument(parser)
    args = parser.parse_args()
    ALGORITHM = "iql"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    torch_seed = args.torch_seed
    requested_route_set = args.route_set
    shuffle = args.shuffle
    save_model_every = args.save_model_every
    
    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {ALGORITHM.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")
    print(f"Requested route set: {requested_route_set or 'network default'}")
    print(f"Shuffle: {shuffle}")
    print(f"Metrics will {'NOT ' if args.skip_metrics else ''}be computed after the experiment.\n")

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(env_seed)
    np.random.seed(env_seed)

    device = (
        torch.device(0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Device is: ", device)
        
    # Parameter setting
    params = dict()
    alg_params = json.load(open(f"../config/algo_config/{ALGORITHM}/{alg_config}.json"))
    env_params = json.load(open(f"../config/env_config/{env_config}.json"))
    task_params = json.load(open(f"../config/task_config/{task_config}.json"))

    params.update(alg_params)
    params.update(env_params)
    params.update(task_params)
    del params["desc"], env_params, task_params

    # All AVs share one network and optimizer
    parameter_sharing = params.get("parameter_sharing", False)

    # JanuX fallback for non-clustered configs
    use_clustered_routes = params.get("use_clustered_routes", False)
    route_set = (
        resolve_route_set(network, requested_route_set)
        if use_clustered_routes
        else None
    )
    print(f"Route set: {route_set or 'none (unclustered)'}")

    # set params as variables in this script
    for key, value in params.items():
        globals()[key] = value

    custom_network_folder = f"../networks/{network}"
    curriculum_schedule = params["curriculum_schedule"]
    phases = [1, human_learning_episodes]
    phases += [human_learning_episodes + stage["episode"] for stage in curriculum_schedule[1:]]
    phases.append(human_learning_episodes + int(training_eps))
    phase_names = ["Human stabilization"]
    phase_names += [" + ".join(stage["groups"]) for stage in curriculum_schedule]
    phase_names.append("Testing phase")

    base_results_dir = os.environ.get("RESULTS_BASE_DIR", "../results")
    records_folder = os.path.join(base_results_dir, exp_id)
    plots_folder = os.path.join(records_folder, "plots")
    # records_folder = f"../results/{exp_id}"
    # plots_folder = f"../results/{exp_id}/plots"

    # Read origin-destinations
    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    data = ast.literal_eval(content)
    origins = data['origins']
    destinations = data['destinations']

    
    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    
    # Preselected groups from the previous Q-stability analysis
    with open("../config/agent_groups.json", 'r', encoding='utf-8') as f:
        groups = json.load(f)

    curriculum = {}
    selected_ids = []

    for stage in curriculum_schedule:
        agent_ids = [
            agent_id
            for name in stage["groups"]
            for agent_id in groups[name]
        ]

        curriculum[stage["episode"]] = agent_ids
        selected_ids.extend(agent_ids)

    num_agents = len(pd.read_csv(agents_csv_path))
    
    # Copy agents.csv
    if os.path.exists(agents_csv_path):
        os.makedirs(records_folder, exist_ok=True)
        new_agents_csv_path = os.path.join(records_folder, "agents.csv")
        with open(agents_csv_path, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(new_agents_csv_path, 'w', encoding='utf-8') as f:
            f.write(content)

    total_episodes = human_learning_episodes + training_eps + test_eps

    # Curriculum stages use exact agent IDs
    new_machines_after_mutation = len(set(curriculum[0]))
    num_machines = len(set(selected_ids))
            
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    unused_params = (
        "ratio_machines",
        "human_alpha",
        "human_beta",
        "human_beta_randomness",
        "human_deterministic",
        "path_gen_workers",
        "curriculum_schedule",
    )

    for key in unused_params:
        dump_config.pop(key, None)
    if args.project is not None:
        dump_config["project"] = args.project

    # Clustered routes: load action masks and generating paths.csv and route.rou.xml from the pregenerated routes
    create_paths_flag = True
    action_masks = None

    if use_clustered_routes:
        route_set_dir = os.path.join(custom_network_folder, "clustered_routes", route_set)
        clustered_loader = ClusteredRoutesLoader(
            network, custom_network_folder, shuffle, env_seed,
            route_set_dir=route_set_dir,
        )
        number_of_paths = clustered_loader.get_number_of_paths()
        clustered_loader.export_paths_routes(records_folder, origins, destinations)
        action_masks = clustered_loader.create_masks(origins, destinations)
        if action_masks is not None:
            create_paths_flag = False
            dump_config["number_of_paths"] = number_of_paths
    if action_masks is None:
        raise RuntimeError("Curriculum requires clustered action masks")

    # Dump exp config to records
    dump_config["network"] = network
    dump_config["env_seed"] = env_seed
    dump_config["torch_seed"] = torch_seed
    dump_config["route_set"] = route_set
    dump_config["env_config"] = env_config
    dump_config["task_config"] = task_config
    dump_config["alg_config"] = alg_config
    dump_config["script"] = script_path_for_config(__file__)
    dump_config["algorithm"] = ALGORITHM
    dump_config["num_agents"] = num_agents
    dump_config["num_machines"] = num_machines
    dump_config["initial_num_machines"] = new_machines_after_mutation
    dump_config["use_clustered_routes"] = use_clustered_routes
    dump_config["use_action_masks"] = action_masks is not None
    dump_config["shuffle"] = shuffle
    dump_config["parameter_sharing"] = parameter_sharing
    if save_model_every is not None:
        dump_config["save_model_every"] = save_model_every
    dump_config["curriculum"] = {
        "schedule": curriculum_schedule,
        "groups": groups,
    }

    # Save exact agent IDs used in this experiment
    with open(os.path.join(records_folder, 'agent_groups.json'), 'w', encoding='utf-8') as f:
        f.write(format_json(groups))

    with open(exp_config_path, 'w', encoding='utf-8') as f:
        f.write(format_json(dump_config))

    # Initialize the environment
    env = TrafficEnvironment(
        seed = env_seed,
        create_agents = False,
        create_paths = create_paths_flag, # Clustered routes: don't create paths if using own, clustered paths
        action_masks = action_masks, # Clustered routes: use action masks if available
        save_detectors_info = False,
        agent_parameters = {
            "new_machines_after_mutation": new_machines_after_mutation,
            "human_parameters" : {
                "model" : human_model
            },
            "machine_parameters" : {
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
            "sumo_type" : "sumo"
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

    env.start()
    env.reset()
    print_agent_counts(env)


    ### Human learning phase ###
    pbar = tqdm(total=total_episodes, desc="Human learning")
    for episode in range(human_learning_episodes):
        env.step()
        pbar.update()


    # First curriculum stage (episode 0)
    traffic_env = env
    mutate_exact_agent_ids(traffic_env, curriculum[0],
                           disable_human_learning=not should_humans_adapt)
    print_agent_counts(env)

    obs_size = env.observation_space(env.possible_agents[0]).shape[0]

    # Wrap env to support clustered routes
    if action_masks is not None:
        env = AVMaskWrapper(env, action_masks)

    # Assign policies to machine agents
    agent_lookup = {}
    def assign_policies():
        for agent in env.machine_agents:
            if str(agent.id) in agent_lookup:
                continue
            model = DQN(obs_size, agent.action_space_size,
                        device=device, eps_init=eps_init, eps_decay=eps_decay,
                        eps_min=eps_min, buffer_size=buffer_size, batch_size=batch_size, lr=lr,
                        num_epochs=num_epochs, num_hidden=num_hidden, widths=widths)
            if parameter_sharing and agent_lookup:
                shared_model = next(iter(agent_lookup.values())).model
                model.q_network = shared_model.q_network
                model.optimizer = shared_model.optimizer
            agent.model = model
            agent_lookup[str(agent.id)] = agent
    assign_policies()

    ### Learning phase ###
    pbar.set_description("AV learning")
    os.makedirs(plots_folder, exist_ok=True)
    for episode in range(training_eps):
        if episode in curriculum and episode != 0:
            mutate_exact_agent_ids(traffic_env, curriculum[episode],
                                   disable_human_learning=not should_humans_adapt)
            if action_masks is not None:
                env = AVMaskWrapper(traffic_env, action_masks)
            assign_policies()
            print_agent_counts(env)

        env.reset()
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                agent_lookup[agent_id].model.push(reward)
                if episode % update_every == 0:
                    agent_lookup[agent_id].model.learn()
                action = None
            else:
                action = agent_lookup[agent_id].model.act(observation)
                
            env.step(action)

        completed_episode = episode + 1
        if should_save_model_snapshot(completed_episode, training_eps, save_model_every):
            torch.save(
                {
                    "training_episode": completed_episode,
                    "models": {
                        str(agent.id): {
                            "q_network": agent.model.q_network.state_dict(),
                            "epsilon": agent.model.epsilon,
                        }
                        for agent in env.machine_agents
                    },
                },
                model_snapshot_path(records_folder, completed_episode, "pt"),
            )
            
        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()
    
    
    ### Testing phase ###
    for agent in env.machine_agents:
        agent.model.epsilon = 0.0
        agent.model.q_network.eval()
        
    pbar.set_description("Testing")
    for episode in range(test_eps):
        env.reset()
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                action = agent_lookup[agent_id].model.act(observation)
            env.step(action)
        pbar.update()
    
    # Finalize the experiment
    pbar.close()
    env.plot_results()
    losses_pd = pd.DataFrame([{"id": agent.id, "losses": agent.model.loss} for agent in env.machine_agents])
    losses_pd.to_csv(os.path.join(records_folder, "losses.csv"), index=False)
    env.stop_simulation()
    
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)
    if not args.skip_metrics:
        run_metrics_analysis(exp_id, results_folder="../results")
        