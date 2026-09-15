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
import xml.etree.ElementTree as ET
from types import MethodType

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from collections     import deque
from routerl         import TrafficEnvironment
from routerl.keychain import Keychain as kc
from tqdm            import tqdm

from baseline_models import BaseLearningModel
from utils           import clear_SUMO_files
from utils           import add_model_snapshot_argument
from utils           import model_snapshot_path
from utils           import print_agent_counts
from utils           import run_metrics_analysis
from utils           import save_loss_records
from utils           import script_path_for_config
from utils           import should_save_model_snapshot

from clustered_routes import (
    AVMaskWrapper,
    ClusteredRoutesLoader,
    resolve_route_set,
    validate_clustered_route_set,
)


def load_human_route_catalogs(path, agent_ids):
    """Load and validate the per-agent route distributions produced by the extractor."""
    routes_df = pd.read_csv(path)
    required = {"agent_id", "route_id", "probability", "edges"}
    missing_columns = required.difference(routes_df.columns)
    if missing_columns:
        raise ValueError(
            f"{path} is missing columns: {', '.join(sorted(missing_columns))}"
        )

    catalogs = {}
    for agent_id, rows in routes_df.groupby("agent_id", sort=False):
        agent_id = int(agent_id)
        rows = rows.sort_values("route_id")
        route_ids = [int(route_id) for route_id in rows["route_id"]]
        if route_ids != list(range(len(route_ids))):
            raise ValueError(f"Route IDs are not contiguous for agent {agent_id}")

        probabilities = rows["probability"].to_numpy(dtype=float, copy=True)
        if not np.isfinite(probabilities).all() or (probabilities <= 0).any():
            raise ValueError(f"Invalid route probabilities for agent {agent_id}")
        probabilities /= probabilities.sum()

        catalogs[agent_id] = []
        for route_id, probability, edges in zip(
            route_ids, probabilities, rows["edges"]
        ):
            edge_list = tuple(str(edges).split())
            if not edge_list:
                raise ValueError(f"Empty route for agent {agent_id}")
            catalogs[agent_id].append(
                {
                    "route_id": route_id,
                    "probability": probability,
                    "edges": edge_list,
                }
            )

    expected_ids = {int(agent_id) for agent_id in agent_ids}
    actual_ids = set(catalogs)
    if expected_ids != actual_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            "demand_routes.csv does not match agents.csv "
            f"(missing IDs: {missing[:10]}, extra IDs: {extra[:10]})"
        )
    return catalogs


def install_auto_routed_humans(env, catalogs, background_ids):
    """Use imported routes while preserving RouteRL's normal episode columns.

    Human ``action`` is the imported route ID and ``cost_table`` is that
    agent's imported route-probability vector, ordered by route ID.
    """
    regular_add_vehicle = env.simulator.add_vehicle
    regular_step = env.simulator.step
    regular_record = env._record
    env.simulator.auto_routed_human_routes = {}
    probabilities = {
        agent_id: [float(route["probability"]) for route in routes]
        for agent_id, routes in catalogs.items()
    }

    def add_vehicle(simulator, action):
        if action[kc.AGENT_KIND] != kc.TYPE_HUMAN:
            return regular_add_vehicle(action)

        agent_id = int(action[kc.AGENT_ID])
        selected = simulator.auto_routed_human_routes.get(agent_id)
        if selected is None:
            raise RuntimeError(f"No imported route was selected for human {agent_id}")

        sumo_route_id = f"arh_{agent_id}_{selected['route_id']}"
        simulator.sumo_connection.route.add(sumo_route_id, list(selected["edges"]))
        simulator.sumo_connection.vehicle.add(
            vehID=str(agent_id),
            routeID=sumo_route_id,
            depart=str(action[kc.AGENT_START_TIME]),
            typeID=kc.TYPE_HUMAN,
        )
        simulator.waiting_vehicles[str(agent_id)] = 0

        # This private field travels with the episode snapshot and is removed
        # before RouteRL writes the standard CSV schema.
        action["_arh_route_id"] = selected["route_id"]

    def record(environment, episode, episode_data, agents, detectors):
        # RouteRL uses the clustered action internally so its observations stay
        # valid. Only the saved copy receives the imported human route ID.
        for entry in episode_data:
            if entry.get(kc.AGENT_KIND) == kc.TYPE_HUMAN:
                entry[kc.ACTION] = entry.pop("_arh_route_id")
        for agent in agents:
            if agent.kind == kc.TYPE_HUMAN:
                agent.model.cost = probabilities[int(agent.id)]
        return regular_record(episode, episode_data, agents, detectors)

    def step(simulator):
        timestep, stopped, arrivals, teleported = regular_step()
        arrivals = [vehicle_id for vehicle_id in arrivals if vehicle_id not in background_ids]
        teleported = [vehicle_id for vehicle_id in teleported if vehicle_id not in background_ids]
        return timestep, stopped, arrivals, teleported

    env.simulator.add_vehicle = MethodType(add_vehicle, env.simulator)
    env.simulator.step = MethodType(step, env.simulator)
    env._record = MethodType(record, env)


def sample_human_routes(env, catalogs, action_masks, env_seed, episode):
    """Sample every current human route reproducibly for one episode."""
    selected_routes = {}
    for human in env.human_agents:
        options = catalogs[int(human.id)]
        probabilities = [option["probability"] for option in options]
        rng = np.random.default_rng(
            np.random.SeedSequence([int(env_seed), int(episode), int(human.id)])
        )
        selected_routes[int(human.id)] = options[
            int(rng.choice(len(options), p=probabilities))
        ]
        mask = action_masks[(int(human.origin), int(human.destination))]
        human.default_action = int(np.flatnonzero(mask)[0])

    env.simulator.auto_routed_human_routes = selected_routes


### Simplified single-DQN implementation for single-step decision-making
class DQN(BaseLearningModel):
    def __init__(self, state_size, action_space_size,
                 device="cpu", eps_init=0.99, eps_decay=0.998,
                 eps_min=0.0,buffer_size=256, batch_size=16, lr=0.003, 
                 num_epochs=1, num_hidden=2, widths=[32, 64, 32]):
        super().__init__()
        self.device = device
        self.action_space_size = action_space_size
        self.epsilon = eps_init
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.q_network = Network(state_size, action_space_size, num_hidden, widths).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.loss = list()

    def act(self, state):
        if isinstance(state, dict):
            observation = np.asarray(state["observation"])
            action_mask = np.asarray(state["action_mask"], dtype=bool)
        else:
            observation = np.asarray(state)
            action_mask = None

        if np.random.rand() < self.epsilon:
            valid_actions = (
                np.flatnonzero(action_mask)
                if action_mask is not None
                else np.arange(self.action_space_size)
            )
            action = int(np.random.choice(valid_actions))
        else:
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)

            if action_mask is not None:
                mask = torch.as_tensor(action_mask, device=self.device).unsqueeze(0)
                q_values = q_values.masked_fill(~mask, float("-inf"))
            action = int(torch.argmax(q_values).item())

        self.last_state = observation
        self.last_action = action
        return action
    
    def push(self, reward):
        # All interactions are single-step, so we only store the last state, action, and reward
        self.memory.append((self.last_state, self.last_action, reward))
        del self.last_state, self.last_action

    def learn(self):
        if len(self.memory) < self.batch_size: return
        step_loss = list()
        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards = zip(*batch)
            states_tensor = torch.FloatTensor(states).to(self.device)
            actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)

            current_q_values = self.q_network(states_tensor).gather(1, actions_tensor)
            target_q_values = rewards_tensor

            loss = self.loss_fn(current_q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            step_loss.append(loss.item())
        self.loss.append(sum(step_loss)/len(step_loss))
        self.decay_epsilon()

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

class Network(nn.Module):
    def __init__(self, in_size, out_size, num_hidden, widths):
        super(Network, self).__init__()
        assert len(widths) == (num_hidden + 1), "DQN widths and number of layers mismatch!"
        
        self.input_layer = nn.Linear(in_size, widths[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(widths[x], widths[x+1]) for x in range(num_hidden)])
        self.out_layer = nn.Linear(widths[-1], out_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.out_layer(x)
        return x
    
    
# Main script to run the IQL experiment with auto-routed humans
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--env-conf', type=str, default="config1")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--torch-seed', type=int, default=42)
    parser.add_argument('--route-set', type=str, default=None, help="Named clustered route set; uses the network default when omitted.")
    parser.add_argument('--skip-metrics', action='store_true', default=False)
    add_model_snapshot_argument(parser)
    args = parser.parse_args()
    
    ALGORITHM = "iql_arh"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    torch_seed = args.torch_seed
    route_set = resolve_route_set(network, args.route_set)
    save_model_every = args.save_model_every
    
    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {ALGORITHM.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")
    print(f"Route set: {route_set}")
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

    # set params as variables in this script
    for key, value in params.items():
        globals()[key] = value

    custom_network_folder = f"../networks/{network}"
    phases = [1, human_learning_episodes, int(training_eps) + human_learning_episodes]
    phase_names = ["Auto-routed human baseline", "Mutation and AV learning", "Testing phase"]
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
    demand_routes_path = os.path.join(custom_network_folder, "demand_routes.csv")
    if not os.path.exists(agents_csv_path):
        raise FileNotFoundError(agents_csv_path)
    if not os.path.exists(demand_routes_path):
        raise FileNotFoundError(demand_routes_path)

    agents_df = pd.read_csv(agents_csv_path)
    num_agents = len(agents_df)
    route_catalogs = load_human_route_catalogs(demand_routes_path, agents_df["id"])
    simulation_timesteps = max(180, int(agents_df["start_time"].max()) + 1)

    os.makedirs(records_folder, exist_ok=True)
    new_agents_csv_path = os.path.join(records_folder, "agents.csv")
    with open(agents_csv_path, 'r', encoding='utf-8') as f:
        content = f.read()
    with open(new_agents_csv_path, 'w', encoding='utf-8') as f:
        f.write(content)

    route_set_dir = os.path.join(custom_network_folder, "clustered_routes", route_set)
    validate_clustered_route_set(network, route_set_dir)
    clustered_loader = ClusteredRoutesLoader(
        network,
        custom_network_folder,
        route_set_dir=route_set_dir,
    )
    number_of_paths = clustered_loader.get_number_of_paths()
    clustered_loader.export_to_paths_csv(
        os.path.join(records_folder, "routes.csv"),
        origins,
        destinations,
    )
    clustered_loader.export_to_sumo_rou_xml(
        os.path.join(records_folder, "route.rou.xml"),
        origins,
        destinations,
    )
    background_path = os.path.join(custom_network_folder, "background_buses.rou.xml")
    if not os.path.exists(background_path):
        raise FileNotFoundError(background_path)
    background_root = ET.parse(background_path).getroot()
    background_vehicles = background_root.findall("vehicle")
    background_ids = {vehicle.get("id") for vehicle in background_vehicles}
    if len(background_ids) != len(background_vehicles) or None in background_ids:
        raise ValueError("Background bus IDs are missing or duplicated")
    if background_ids.intersection(str(agent_id) for agent_id in agents_df["id"]):
        raise ValueError("Background bus IDs overlap with RouteRL agent IDs")
    route_path = os.path.join(records_folder, "route.rou.xml")
    route_tree = ET.parse(route_path)
    route_tree.getroot().extend(list(background_root))
    ET.indent(route_tree, space="  ")
    route_tree.write(route_path, encoding="utf-8", xml_declaration=True)
    if background_vehicles:
        simulation_timesteps = max(
            simulation_timesteps,
            int(max(float(vehicle.get("depart")) for vehicle in background_vehicles)) + 1,
        )
    action_masks = clustered_loader.create_masks(origins, destinations)

    demand_ods = set(
        agents_df[["origin", "destination"]].itertuples(index=False, name=None)
    )
    missing_masks = sorted(demand_ods.difference(action_masks))
    empty_masks = sorted(od for od in demand_ods if not action_masks[od].any())
    if missing_masks or empty_masks:
        raise ValueError(
            "Clustered route set does not cover the demand "
            f"(missing ODs: {missing_masks[:10]}, empty masks: {empty_masks[:10]})"
        )
            
    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + test_eps
            
    # Dump exp config to records
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    if args.project is not None:
        dump_config["project"] = args.project
    dump_config["network"] = network
    dump_config["env_seed"] = env_seed
    dump_config["torch_seed"] = torch_seed
    dump_config["env_config"] = env_config
    dump_config["task_config"] = task_config
    dump_config["alg_config"] = alg_config
    dump_config["script"] = script_path_for_config(__file__)
    dump_config["algorithm"] = ALGORITHM
    dump_config["num_agents"] = num_agents
    dump_config["num_machines"] = num_machines
    dump_config["human_learning"] = False
    dump_config["human_routing"] = "per_agent_imported_distribution"
    dump_config["demand_routes_file"] = f"networks/{network}/demand_routes.csv"
    dump_config["background_traffic_file"] = f"networks/{network}/background_buses.rou.xml"
    dump_config["background_bus_count"] = len(background_ids)
    dump_config["should_humans_adapt"] = False
    dump_config["route_set"] = route_set
    dump_config["number_of_paths"] = number_of_paths
    dump_config["use_clustered_routes"] = True
    dump_config["use_action_masks"] = True
    if save_model_every is not None:
        dump_config["save_model_every"] = save_model_every
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    
    # Initialize the environment
    env = TrafficEnvironment(
        seed = env_seed,
        create_agents = False,
        create_paths = False,
        action_masks = action_masks,
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
            "sumo_type" : "sumo",
            "simulation_timesteps" : simulation_timesteps
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

    install_auto_routed_humans(env, route_catalogs, background_ids)
    env.human_learning = False
    env.start()
    env.reset()
    print_agent_counts(env)

    if should_humans_adapt:
        print("Warning: should_humans_adapt is ignored by iql_arh; humans never learn.")
    print(
        "Note: imported human routes do not share JanuX action indices. "
        "RouteRL observations see the first valid masked action. In saved episode "
        "files, human action is the imported route ID and human cost_table is the "
        "imported probability vector. Background buses affect traffic but are not agents."
    )


    ### Auto-routed human baseline ###
    routing_episode = 0
    pbar = tqdm(total=total_episodes, desc="Auto-routed human baseline")
    for episode in range(human_learning_episodes):
        sample_human_routes(env, route_catalogs, action_masks, env_seed, routing_episode)
        routing_episode += 1
        env.step()
        pbar.update()


    # Mutation
    env.mutation(disable_human_learning = True, mutation_start_percentile = -1)
    print_agent_counts(env)
    obs_size = env.observation_space(env.possible_agents[0]).shape[0]
    env = AVMaskWrapper(env, action_masks)
    
    # Set policies for machine agents
    for idx in range(len(env.machine_agents)):
        env.machine_agents[idx].model = DQN(obs_size, env.machine_agents[idx].action_space_size, 
                                            device=device, eps_init=eps_init, eps_decay=eps_decay,
                                            eps_min=eps_min, buffer_size=buffer_size, batch_size=batch_size, 
                                            lr=lr, num_epochs=num_epochs, num_hidden=num_hidden, widths=widths)
    agent_lookup = {str(agent.id): agent for agent in env.machine_agents}
    
    
    ### Learning phase ###
    pbar.set_description("AV learning")
    os.makedirs(plots_folder, exist_ok=True)
    for episode in range(training_eps):
        env.reset()
        sample_human_routes(env, route_catalogs, action_masks, env_seed, routing_episode)
        routing_episode += 1
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
        sample_human_routes(env, route_catalogs, action_masks, env_seed, routing_episode)
        routing_episode += 1
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
    loss_records = []
    for agent in env.machine_agents:
        for iteration, loss_value in enumerate(agent.model.loss, start=1):
            loss_records.append(
                {
                    "iteration": iteration,
                    "agent_id": agent.id,
                    "loss": loss_value,
                }
            )
    save_loss_records(
        records_folder,
        loss_records,
        columns=["iteration", "agent_id", "loss"],
    )

    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)
    if not args.skip_metrics:
        run_metrics_analysis(exp_id, results_folder="../results")
