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
from collections     import deque, defaultdict

from baseline_models import BaseLearningModel
from iql             import Network
from utils           import clear_SUMO_files
from utils           import print_agent_counts
from utils           import run_metrics_analysis
from utils           import save_loss_records
from utils           import script_path_for_config
from clustered_routes import ClusteredRoutesLoader, resolve_route_set


class MAPPO(BaseLearningModel):
    def __init__(
        self,
        state_size: int,
        action_space_size: int,
        num_agents: int = 20,
        # --- policy settings ---
        shared_policy: bool = False,
        policy_nets: list[nn.Module] | None = None,
        policy_arch_kwargs: dict | None = None,
        # --- critic settings ---
        share_critic: bool = True,
        critic_nets: list[nn.Module] | None = None,
        critic_arch_kwargs: dict | None = None,
        # --- default architecture parameters ---
        default_widths: list[int] = [64,64],
        # --- hyperparameters ---
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        lr_actor: float = 0.0003,
        lr_critic: float = 0.0003,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        batch_size: int = 64,
        memory_size: int = 5000,
        max_norm: float = 0.5,
        device: torch.device | None = None,
        action_mask: dict | None = None,
        central_state_size: int | None = None,
        **kwargs
    ):
        super().__init__()
        # device setup 
        self.device = device or torch.device("cpu")
        self.state_size = state_size
        self.action_space_size = action_space_size
        self.num_agents = num_agents
        self.state_size = state_size
        self.max_norm = max_norm
        self.central_state_size = (
        central_state_size
            if central_state_size is not None
            else state_size
        )

        self.last_central_states = {}
        self.action_masks = action_mask if action_mask is not None else {}
        self.num_epochs = kwargs.get("num_epochs", 3)
        ws = kwargs.get("widths", default_widths)
        self.clip_ratio = kwargs.get("clip_eps", clip_ratio)
        lr_actor = kwargs.get("lr", lr_actor)
        lr_critic = kwargs.get("lr", lr_critic)

        # training phase flag
        self.training = True
        
        # hyperparameters
        self.gamma = gamma
        self.entropy_coef = kwargs.get("entropy_coef", entropy_coef)
        self.value_coef = kwargs.get("value_coef", value_coef)
        self.batch_size = kwargs.get("batch_size", batch_size)
        self.memory = []
        self.last_states = {}
        self.last_actions = {}
        self.last_log_probs = {}

        # --- Policy networks ---
        if policy_nets is not None:
            assert len(policy_nets) == num_agents or shared_policy, \
                "If policy_nets is provided, it must match the number of agents or be shared."
            self.policies = [net.to(self.device) for net in (policy_nets if not shared_policy else [policy_nets[0]] * num_agents)]
        else:
            self.policies = []
            for _ in range(num_agents):
                net = Network(state_size, action_space_size, len(ws) - 1, ws).to(self.device)
                self.policies.append(net)
            if shared_policy:
                self.policies = [self.policies[0]] * num_agents

        # create actor optimizers
        if shared_policy:
            self.actor_optimizer = optim.Adam(self.policies[0].parameters(), lr=lr_actor)
        else:
            self.actor_optimizer = [optim.Adam(policy.parameters(), lr=lr_actor) for policy in self.policies]
        
        # --- Critic networks ---
        if critic_nets is not None:
            assert len(critic_nets) == num_agents or share_critic, \
                "If critic_nets is provided, it must match the number of agents or be shared."
            self.critics = [net.to(self.device) for net in (critic_nets if not share_critic else [critic_nets[0]] * num_agents)]
        else:
            # build critics using generic Network class
            ch_ws = critic_arch_kwargs.get('widths', default_widths) if critic_arch_kwargs else default_widths
            self.critics = []
            for _ in range(num_agents):
                #net = Network(state_size, 1, len(ch_ws) - 1, ch_ws).to(self.device)
                net = Network(
                    self.central_state_size,
                    1,
                    len(ch_ws) - 1,
                    ch_ws
                ).to(self.device)

                self.critics.append(net)
            if share_critic:
                shared_critic = self.critics[0]
                self.critics = [shared_critic] * num_agents

        # create critic optimizers
        if share_critic:
            self.critic_optim = optim.Adam(self.critics[0].parameters(), lr=lr_critic)
        else:
            self.critic_optim = [optim.Adam(net.parameters(), lr=lr_critic) for net in self.critics]

        # loss tracking
        self.loss_actor = []
        self.loss_critic = []

    def train(self):
        """Set all models to training mode"""
        for policy in self.policies:
            policy.train()
        for critic in self.critics:
            critic.train()
        self.training = True
        return self
    
    def eval(self):
        """Set all models to evaluation mode"""
        for policy in self.policies:
            policy.eval()
        for critic in self.critics:
            critic.eval()
        self.training = False
        return self

    def push(self, agent_id: int | float, reward: float | None = None, next_state=None, done: bool = True):
        if reward is None:
            reward = float(agent_id)
            agent_id = 0
        if agent_id not in self.last_states:
            return
        if next_state is None:
            next_state = self.last_states[agent_id]
        
        self.memory.append(
        (
            self.last_states.pop(agent_id),
            self.last_central_states.pop(agent_id),
            self.last_actions.pop(agent_id),
            float(reward),
            self.last_log_probs.pop(agent_id),
            next_state,
            bool(done),
            agent_id,
            )
        )

    def act(self, state: any, agent_id: int, central_state=None):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policies[agent_id](state_tensor)
            
            if agent_id in self.action_masks:
                mask = self.action_masks[agent_id]
                logits = logits.masked_fill(~mask, float('-inf'))
                
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item() if self.training else torch.argmax(logits).item()
            
        log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()

        if central_state is None:
            central_state = state

        self.last_central_states[agent_id] = np.asarray(
        central_state,
        dtype=np.float32
        )

        self.last_states[agent_id] = state
        self.last_actions[agent_id] = action
        self.last_log_probs[agent_id] = log_prob

        return action

    def learn(self):
        if len(self.memory) < self.batch_size: return
        
        (
            s_batch,
            cs_batch,
            a_batch,
            r_batch,
            lp_batch,
            ns_batch,
            d_batch,
            id_batch,
        ) = zip(*self.memory)

        states_tensor = torch.FloatTensor(np.array(s_batch)).to(self.device)
        actions_tensor = torch.LongTensor(a_batch).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(r_batch).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(ns_batch)).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(lp_batch).unsqueeze(1).to(self.device)
        dones_tensor = torch.FloatTensor(d_batch).unsqueeze(1).to(self.device)
        id_tensor = torch.LongTensor(id_batch)

        central_states_tensor = torch.FloatTensor(
            np.array(cs_batch)
        ).to(self.device)

        unique_ids = id_tensor.unique().tolist()

        advantages = torch.zeros_like(rewards_tensor)
        targets = torch.zeros_like(rewards_tensor)

        with torch.no_grad():
            for aid in unique_ids:
                mask = (id_tensor == aid)
                if not mask.any(): continue
                
                v = self.critics[aid](central_states_tensor[mask])
                
                td_target = rewards_tensor[mask]
                adv = td_target - v
                
                advantages[mask] = adv
                targets[mask] = td_target

        for _ in range(self.num_epochs):
            indices = np.arange(len(self.memory))
            np.random.shuffle(indices)

            for start in range(0, len(self.memory), self.batch_size):
                end = start + self.batch_size
                mb_idx = indices[start:end]

                mb_states = states_tensor[mb_idx]
                mb_actions = actions_tensor[mb_idx]
                mb_old_log_probs = old_log_probs_tensor[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_targets = targets[mb_idx]
                mb_central_states = central_states_tensor[mb_idx]
                mb_ids = id_tensor[mb_idx]
                mb_unique_ids = mb_ids.unique().tolist()

                avg_policy_loss, avg_critic_loss, avg_entropy = 0.0, 0.0, 0.0
                total_count = len(mb_idx)

                for aid in mb_unique_ids:
                    agent_mask = (mb_ids == aid)
                    agent_count = agent_mask.sum().item()

                    #critic
                    v = self.critics[aid](mb_central_states[agent_mask])
                    critic_loss = nn.MSELoss()(v, mb_targets[agent_mask])

                    #actor
                    logits = self.policies[aid](mb_states[agent_mask])
                    
                    if aid in self.action_masks:
                        action_mask = self.action_masks[aid]
                        logits = logits.masked_fill(~action_mask, float('-inf'))
                        
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(mb_actions[agent_mask].squeeze(1)).unsqueeze(1)
                    entropy = dist.entropy().mean()

                    ratios = torch.exp(new_log_probs - mb_old_log_probs[agent_mask])
                    clipped_ratios = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)

                    surr1 = ratios * mb_advantages[agent_mask]
                    surr2 = clipped_ratios * mb_advantages[agent_mask]
                    policy_loss = -torch.min(surr1, surr2).mean()

                    avg_critic_loss += critic_loss * (agent_count / total_count)
                    avg_policy_loss += policy_loss * (agent_count / total_count)
                    avg_entropy += entropy * (agent_count / total_count)

                if isinstance(self.critic_optim, list):
                    for aid in mb_unique_ids: self.critic_optim[aid].zero_grad()
                    avg_critic_loss.backward()
                    for aid in mb_unique_ids: self.critic_optim[aid].step()
                else:
                    self.critic_optim.zero_grad()
                    avg_critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.critics[0].parameters(),
                        max_norm=self.max_norm
                    )
                    self.critic_optim.step()

                total_loss = avg_policy_loss - self.entropy_coef * avg_entropy
                if isinstance(self.actor_optimizer, list):
                    for aid in mb_unique_ids: self.actor_optimizer[aid].zero_grad()
                    total_loss.backward()
                    for aid in mb_unique_ids: self.actor_optimizer[aid].step()
                else:
                    self.actor_optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policies[0].parameters(),
                        max_norm=self.max_norm
                    )
                    self.actor_optimizer.step()

                self.loss_critic.append(avg_critic_loss.item())
                self.loss_actor.append(avg_policy_loss.item())

        self.memory.clear()

    def get_last_observation(self, agent_id: int):
        return self.last_states.get(agent_id, None)
    
    def get_last_action(self, agent_id: int):
        return self.last_actions.get(agent_id, None)
    
    def get_last_log_prob(self, agent_id: int):
        return self.last_log_probs.get(agent_id, None)
    
    def get_policy(self, agent_id: int):
        return self.policies[agent_id] if agent_id < len(self.policies) else None


def main():
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
    
    ALGORITHM = "mappo"
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

    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    data = ast.literal_eval(content)
    origins = data['origins']
    destinations = data['destinations']

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
        raise FileNotFoundError(f"Agents CSV file not found at {agents_csv_path}.")
            
    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + test_eps
    
    # Dump exp config to records
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    
    # Load pre-generated clustered routes and their per-OD action masks.
    configured_number_of_paths = params.get("number_of_paths", 4)
    number_of_paths = configured_number_of_paths
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

    
    # Set policies for machine agents
    shared_action_space_size = max(agent.action_space_size for agent in env.machine_agents)
    central_extra_size = shared_action_space_size + 1

    central_state_size = (
    obs_size + central_extra_size
    )
    agent_to_idx = {str(agent.id): idx for idx, agent in enumerate(env.machine_agents)}
    
    internal_action_masks = {}
    for idx, agent in enumerate(env.machine_agents):
        if action_masks is not None:
            key = (agent.origin, agent.destination)
            if key not in action_masks:
                key = (int(agent.origin), int(agent.destination))
                
            if key not in action_masks:
                raise ValueError(f"Missing action mask for agent {agent.id}")
                
            valid_mask = action_masks[key]
            mask_tensor = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
            
            padded_mask = torch.zeros(shared_action_space_size, dtype=torch.bool, device=device)
            limit = min(mask_tensor.size(0), shared_action_space_size)
            padded_mask[:limit] = mask_tensor[:limit]
            
            internal_action_masks[idx] = padded_mask
        else:
            mask_tensor = torch.zeros(shared_action_space_size, dtype=torch.bool, device=device)
            mask_tensor[:agent.action_space_size] = True
            internal_action_masks[idx] = mask_tensor
            
    model_params = params.copy()
    model_params.update({
        "state_size": obs_size,
        "central_state_size": central_state_size,
        "action_space_size": shared_action_space_size,
        "num_agents": len(env.machine_agents),
        "shared_policy": True,
        "share_critic": True,
        "device": device,
        "action_mask": internal_action_masks
    })

    shared_mappo = MAPPO(**model_params)

    for agent in env.machine_agents:
        agent.model = shared_mappo
        
    agent_lookup = {str(agent.id): agent for agent in env.machine_agents}
    
    ### Learning phase ###
    pbar.set_description("AV learning")
    os.makedirs(plots_folder, exist_ok=True)
    for episode in range(training_eps):
        od_route_counts = defaultdict(
            lambda: np.zeros(
            shared_action_space_size,
            dtype=np.float32
            )
        )

        num_avs_acted = 0
        env.reset()
        episode_rewards = []
        episode_travel_times = []
        
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            acted_od = None
            if termination or truncation:
                if agent_id in agent_lookup:
                    internal_id = agent_to_idx[str(agent_id)]
                    shared_mappo.push(agent_id=internal_id, reward=reward)
                
                action = None
                episode_rewards.append(reward)
                if "travel_time" in info:
                    episode_travel_times.append(info["travel_time"])
            else:
                if agent_id in agent_lookup:
                    internal_id = agent_to_idx[str(agent_id)]
                    machine = agent_lookup[agent_id]

                    # OD pair of the current AV
                    acted_od = (
                    int(machine.origin),
                    int(machine.destination)
                    )

                    # Previous route selections for this OD
                    counts = od_route_counts[acted_od]

                    if counts.sum() > 0:
                        od_route_fractions = (
                        counts / counts.sum()
                         )
                    else:
                        od_route_fractions = np.zeros_like(
                        counts,
                        dtype=np.float32
                        )

                    # How far through the AV decisions we are
                    progress = np.array(
                        [
                            num_avs_acted
                            / max(len(env.machine_agents), 1)
                        ],
                        dtype=np.float32
                    )

                    # Centralized critic observation
                    central_state = np.concatenate(
                        [
                            np.asarray(
                                observation,
                                dtype=np.float32
                            ),
                            od_route_fractions,
                            progress,
                        ]
                    ).astype(np.float32)

                    # Actor still receives ONLY local observation.
                    # Critic information is passed separately.
                    action = shared_mappo.act(
                        observation,
                        agent_id=internal_id,
                        central_state=central_state
                    )

                    #action = shared_mappo.act(observation, agent_id=internal_id)
                else:
                    action = None
                
                if acted_od is not None and action is not None:
                    od_route_counts[acted_od][int(action)] += 1
                    num_avs_acted += 1
                
            env.step(action)
            
        if episode % update_every == 0:
            shared_mappo.learn()
            
        episode_losses = [shared_mappo.loss_actor[-1]] if len(shared_mappo.loss_actor) > 0 else []
        metrics = {"episode": episode + human_learning_episodes}
        if episode_losses:
            metrics["train/avg_loss"] = sum(episode_losses) / len(episode_losses)
            
        metrics["train/reward_sum"] = float(np.sum(episode_rewards)) if episode_rewards else 0.0
        metrics["train/reward_mean"] = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        metrics["train/travel_time_mean"] = float(np.mean(episode_travel_times)) if episode_travel_times else 0.0
            
        
        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()
    
    ### Testing phase ###
    for agent in env.machine_agents:
        agent.model.eval()
        
    pbar.set_description("Testing")
    for episode in range(test_eps):
        env.reset()
        episode_rewards = []
        episode_travel_times = []
        
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if agent_id not in agent_lookup or termination or truncation:
                action = None
                if agent_id in agent_lookup:
                    episode_rewards.append(reward)
                    if "travel_time" in info:
                        episode_travel_times.append(info["travel_time"])
            else:
                if agent_id in agent_lookup:
                    internal_id = agent_to_idx[str(agent_id)]
                    action = shared_mappo.act(observation, agent_id=internal_id)
                else:
                    action = None
                
            env.step(action)
            
       
        pbar.update()
    
    # Finalize the experiment
    pbar.close()
    env.plot_results()
    
    plot_files = ["rewards.png", "travel_times.png"] 
    images_to_log = {}
    for plot_file in plot_files:
        plot_path = os.path.join(plots_folder, plot_file)
        if os.path.exists(plot_path):
            plot_name = f"plots/{plot_file.replace('.png', '')}"
            

    loss_records = [
        {"iteration": iteration, "agent_id": "shared", "loss": loss_value}
        for iteration, loss_value in enumerate(shared_mappo.loss_actor, start=1)
    ]
            
    save_loss_records(
        records_folder,
        loss_records,
        columns=["iteration", "agent_id", "loss"],
    )

    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)
    run_metrics_analysis(exp_id, results_folder="../results")

if __name__ == "__main__":
    main()
