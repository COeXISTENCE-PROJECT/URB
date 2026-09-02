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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


from routerl         import TrafficEnvironment
from tqdm            import tqdm

from baseline_models import BaseLearningModel
from iql             import Network
from utils           import clear_SUMO_files
from utils           import print_agent_counts
from utils           import run_metrics_analysis
from utils           import save_loss_records
from utils           import script_path_for_config

from clustered_routes import ClusteredRoutesLoader, resolve_route_set

### A simplified single-step actor-only PPO implementation with shared parameters.
class SharedPPO(BaseLearningModel):
    """One PPO policy/optimizer shared by all AV agents.

    Agent-specific pending transitions and action masks are kept separate so
    sequential AEC interactions do not overwrite another agent's experience.
    """

    def __init__(self, state_size, action_space_size,
                 device="cpu", batch_size=16, lr=0.003, num_epochs=4,
                 num_hidden=2, widths=[32, 64, 32], clip_eps=0.2,
                 normalize_advantage=True, entropy_coef=0.3):
        super().__init__()
        self.device = device
        self.action_space_size = action_space_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.clip_eps = clip_eps
        self.normalize_advantage = normalize_advantage
        self.entropy_coef = entropy_coef

        self.policy_net = Network(
            state_size, action_space_size, num_hidden, widths
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.softmax = nn.Softmax(dim=-1)

        self.loss = []
        self.memory = []
        self.pending = {}
        self.deterministic = False

    def _prepare_action_mask(self, action_mask, batch_size):
        if action_mask is None:
            return None

        mask = torch.as_tensor(
            action_mask,
            dtype=torch.bool,
            device=self.device,
        )

        if mask.ndim == 1:
            if mask.numel() != self.action_space_size:
                raise ValueError(
                    "Action mask size must match the shared action space size."
                )
            mask = mask.unsqueeze(0)
        elif mask.ndim == 2:
            if mask.shape[1] != self.action_space_size:
                raise ValueError(
                    "Action mask width must match the shared action space size."
                )
        else:
            raise ValueError("Action mask must be one- or two-dimensional.")

        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1)

        if mask.shape[0] != batch_size:
            raise ValueError(
                f"Action mask batch has {mask.shape[0]} rows; expected {batch_size}."
            )

        if not torch.all(mask.any(dim=1)).item():
            raise ValueError("Every action mask must contain at least one valid action.")

        return mask

    def _distribution(self, state_tensor, action_mask=None):
        logits = self.policy_net(state_tensor)
        mask = self._prepare_action_mask(action_mask, logits.shape[0])
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return torch.distributions.Categorical(probs=self.softmax(logits))

    def act(self, state, agent_id, action_mask=None, record=True):
        state_array = np.asarray(state, dtype=np.float32)
        state_tensor = torch.as_tensor(
            state_array, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            dist = self._distribution(state_tensor, action_mask)

        if self.deterministic:
            action = torch.argmax(dist.probs, dim=-1).item()
        else:
            action = dist.sample().item()

        if record:
            mask_array = (
                np.ones(self.action_space_size, dtype=np.bool_)
                if action_mask is None
                else np.asarray(action_mask, dtype=np.bool_).copy()
            )
            self.pending[str(agent_id)] = (
                state_array.copy(),
                int(action),
                float(
                    dist.log_prob(
                        torch.tensor(action, device=self.device)
                    ).item()
                ),
                mask_array,
            )

        return int(action)

    def push(self, agent_id, reward):
        key = str(agent_id)
        if key not in self.pending:
            return

        state, action, old_log_prob, action_mask = self.pending.pop(key)
        self.memory.append(
            (
                state,
                action,
                old_log_prob,
                float(reward),
                action_mask,
            )
        )

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        step_loss = []

        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            states, actions, old_log_probs, rewards, action_masks = zip(*batch)

            states_tensor = torch.as_tensor(
                np.stack(states), dtype=torch.float32, device=self.device
            )
            actions_tensor = torch.as_tensor(
                actions, dtype=torch.long, device=self.device
            )
            old_log_probs_tensor = torch.as_tensor(
                old_log_probs, dtype=torch.float32, device=self.device
            )
            rewards_tensor = torch.as_tensor(
                rewards, dtype=torch.float32, device=self.device
            )
            action_masks_tensor = torch.as_tensor(
                np.stack(action_masks), dtype=torch.bool, device=self.device
            )

            dist = self._distribution(states_tensor, action_masks_tensor)
            new_log_probs = dist.log_prob(actions_tensor)

            ratio = torch.exp(new_log_probs - old_log_probs_tensor)

            if self.normalize_advantage:
                advantage = (
                    rewards_tensor - rewards_tensor.mean()
                ) / (rewards_tensor.std() + 1e-8)
            else:
                advantage = rewards_tensor

            surr1 = ratio * advantage
            surr2 = torch.clamp(
                ratio, 1 - self.clip_eps, 1 + self.clip_eps
            ) * advantage

            entropy = dist.entropy().mean()
            loss = (
                -torch.min(surr1, surr2).mean()
                - self.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), max_norm=1.0
            )
            self.optimizer.step()
            step_loss.append(loss.item())

        self.loss.append(sum(step_loss) / len(step_loss))
        self.memory.clear()


# Main script to run the IPPO experiment
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
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
    args = parser.parse_args()
    ALGORITHM = "ippo"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    torch_seed = args.torch_seed
    requested_route_set = args.route_set
    shuffle = args.shuffle
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

    observation_type = params.get(
        "observation_type",
        params.get("observations", "previous_agents_plus_start_time"),
    )
    path_gen_workers_value = params.get("path_gen_workers", 4)

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

    # Load pre-generated clustered routes and their per-OD action masks.
    configured_number_of_paths = number_of_paths
    create_paths_flag = True
    action_masks = None

    if use_clustered_routes:
        try:
            route_set_dir = os.path.join(custom_network_folder, "clustered_routes", route_set)
            clustered_loader = ClusteredRoutesLoader(
                network,
                custom_network_folder,
                shuffle,
                env_seed,
                route_set_dir=route_set_dir,
            )
            number_of_paths = clustered_loader.get_number_of_paths()
            clustered_loader.export_paths_routes(records_folder, origins, destinations)
            action_masks = clustered_loader.create_masks(origins, destinations)
            if not action_masks:
                raise ValueError("The clustered route set contains no action masks.")

            for od_pair, mask in action_masks.items():
                mask_array = np.asarray(mask)
                if mask_array.shape != (number_of_paths,):
                    raise ValueError(
                        f"Action mask for OD pair {od_pair} has shape "
                        f"{mask_array.shape}; expected ({number_of_paths},)."
                    )
                if not np.isin(mask_array, (0, 1)).all():
                    raise ValueError(
                        f"Action mask for OD pair {od_pair} must be binary."
                    )
                if not mask_array.any():
                    raise ValueError(
                        f"Action mask for OD pair {od_pair} has no valid actions."
                    )

            agent_ods = {
                (int(row.origin), int(row.destination))
                for row in pd.read_csv(
                    agents_csv_path,
                    usecols=["origin", "destination"],
                ).itertuples()
            }
            missing_ods = sorted(agent_ods.difference(action_masks))
            if missing_ods:
                raise ValueError(
                    "Missing action masks for agent OD pairs: "
                    + ", ".join(map(str, missing_ods))
                )

            create_paths_flag = False
            dump_config["number_of_paths"] = number_of_paths
        except FileNotFoundError as e:
            use_clustered_routes = False
            number_of_paths = configured_number_of_paths
            print(f"[CLUSTERED ROUTES] Warning: {e}")
            print("[CLUSTERED ROUTES] Falling back to JanuX generation\n")

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
    dump_config["use_clustered_routes"] = use_clustered_routes
    dump_config["use_action_masks"] = action_masks is not None
    dump_config["shared_parameters"] = True
    dump_config["shuffle"] = shuffle
    dump_config["observation_type"] = observation_type
    dump_config["path_gen_workers"] = path_gen_workers_value
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    
    # Initialize the environment
    env = TrafficEnvironment(
        seed = env_seed,
        create_agents = False,
        create_paths = create_paths_flag,
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
                "observation_type" : observation_type
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
            "path_gen_workers" : path_gen_workers_value,
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


    # Mutation
    env.mutation(disable_human_learning = not should_humans_adapt, mutation_start_percentile = -1)
    print_agent_counts(env)
    obs_size = env.observation_space(env.possible_agents[0]).shape[0]
    
    # Set one shared policy for all machine agents.
    if not env.machine_agents:
        raise ValueError("No machine agents were created after mutation.")

    # A shared network needs one common output dimension. We use the largest
    # agent action space and mask invalid actions for agents with fewer routes.
    shared_action_space_size = max(
        agent.action_space_size for agent in env.machine_agents
    )

    shared_model = SharedPPO(
        obs_size,
        shared_action_space_size,
        device=device,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        num_hidden=num_hidden,
        widths=widths,
        clip_eps=clip_eps,
        normalize_advantage=normalize_advantage,
        entropy_coef=entropy_coef,
    )

    # Build one action mask per AV. This preserves the clustered-route masks
    # and also supports heterogeneous action-space sizes without clustered routes.
    agent_action_masks = {}
    for agent in env.machine_agents:
        if action_masks is not None:
            key = (agent.origin, agent.destination)
            if key not in action_masks:
                raise ValueError(
                    f"Missing action mask for agent {agent.id} "
                    f"({agent.origin} -> {agent.destination})."
                )
            mask = np.asarray(action_masks[key], dtype=np.bool_)

            if mask.shape != (shared_action_space_size,):
                raise ValueError(
                    f"Action mask for agent {agent.id} has shape {mask.shape}; "
                    f"expected ({shared_action_space_size},)."
                )
        else:
            mask = np.zeros(shared_action_space_size, dtype=np.bool_)
            mask[:agent.action_space_size] = True

        if not mask.any():
            raise ValueError(
                f"Agent {agent.id} has no valid actions in its action mask."
            )

        agent_action_masks[str(agent.id)] = mask
        agent.model = shared_model

    agent_lookup = {str(agent.id): agent for agent in env.machine_agents}

    print(
        f"Using shared-parameter IPPO: {len(env.machine_agents)} AVs "
        f"share one policy with {shared_action_space_size} output actions."
    )


    ### Learning phase ###
    pbar.set_description("AV learning")
    os.makedirs(plots_folder, exist_ok=True)

    for episode in range(training_eps):
        env.reset()

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            key = str(agent_id)

            if termination or truncation:
                shared_model.push(key, reward)
                action = None
            else:
                action = shared_model.act(
                    observation,
                    agent_id=key,
                    action_mask=agent_action_masks[key],
                    record=True,
                )

            env.step(action)

        # Update the shared policy once, using experience pooled from all AVs.
        if episode % update_every == 0:
            shared_model.learn()

        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()


    ### Testing phase ###
    shared_model.policy_net.eval()
    shared_model.deterministic = True

    pbar.set_description("Testing")
    for episode in range(test_eps):
        env.reset()

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            key = str(agent_id)

            if termination or truncation:
                action = None
            else:
                action = shared_model.act(
                    observation,
                    agent_id=key,
                    action_mask=agent_action_masks[key],
                    record=False,
                )

            env.step(action)

        pbar.update()


    # Finalize the experiment
    pbar.close()
    env.plot_results()

    loss_records = [
        {
            "iteration": iteration,
            "agent_id": "shared",
            "loss": loss_value,
        }
        for iteration, loss_value in enumerate(shared_model.loss, start=1)
    ]

    save_loss_records(
        records_folder,
        loss_records,
        columns=["iteration", "agent_id", "loss"],
    )

    env.stop_simulation()
    clear_SUMO_files(
        os.path.join(records_folder, "SUMO_output"),
        os.path.join(records_folder, "episodes"),
        remove_additional_files=True,
    )
    run_metrics_analysis(exp_id, results_folder="../results")