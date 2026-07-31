"""Hyper-IPPO with pre-generated clustered routes and per-OD action masks."""

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

from routerl import TrafficEnvironment
from tqdm import tqdm

from baseline_models import BaseLearningModel
from clustered_routes import ClusteredRoutesLoader, resolve_route_set
from utils           import clear_SUMO_files
from utils           import print_agent_counts
from utils           import run_metrics_analysis
from utils           import script_path_for_config

class AgentFeatureEmbedder(nn.Module):
    def __init__(self, agents_df, embed_dim, device="cpu"):
        super().__init__()
        self.device = device
        
        num_locs = max(agents_df['origin'].max(), agents_df['destination'].max()) + 1
        self.loc_embedding = nn.Embedding(num_locs, 8) 
        self.max_time = agents_df['start_time'].max()        
        self.raw_locs = torch.LongTensor(agents_df[['origin', 'destination']].values).to(device)
        self.times = torch.FloatTensor(agents_df['start_time'].values / self.max_time).unsqueeze(1).to(device)

        self.fc = nn.Sequential(
            nn.Linear(17, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )

    def forward(self, agent_idx):
        locs = self.raw_locs[agent_idx] 
        o_emb = self.loc_embedding(locs[0])
        d_emb = self.loc_embedding(locs[1])
        t_val = self.times[agent_idx]
        
        combined = torch.cat([o_emb, d_emb, t_val], dim=-1)
        return self.fc(combined)

class HyperNetwork(nn.Module):
    def __init__(self, agent_embed_dim, state_dim, action_dim, hidden_sizes):
        super().__init__()

        layer_sizes = [state_dim] + hidden_sizes + [action_dim]
        self.layer_sizes = layer_sizes

        self.param_sizes = [
            layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1]
            for i in range(len(layer_sizes) - 1)
        ]
        self.total_params = sum(self.param_sizes)

        self.net = nn.Sequential(
            nn.Linear(agent_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.total_params)
        )

    def forward(self, agent_embed):
        return self.net(agent_embed)

def functional_mlp(x, weights, biases):
    for W, b in zip(weights[:-1], biases[:-1]):
        x = torch.relu(x @ W + b)
    return x @ weights[-1] + biases[-1]

class PPO(BaseLearningModel):
    def __init__(
        self,
        state_size,
        action_space_size,
        agent_id,
        hypernet,
        agent_embeddings,
        device="cpu",
        batch_size=16,
        lr=0.003,
        num_epochs=4,
        hidden_sizes=[32, 64, 32],
        clip_eps=0.2,
        normalize_advantage=True,
        entropy_coef=0.3,
        total_training_eps=10000,
        action_mask=None,
    ):
        super().__init__()
        self.device = device
        self.state_size = state_size
        self.action_space_size = action_space_size
        self.agent_id = agent_id
        self.hypernet = hypernet
        self.agent_embeddings = agent_embeddings

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.clip_eps = clip_eps
        self.normalize_advantage = normalize_advantage
        self.entropy_coef = entropy_coef
        self.hidden_sizes = hidden_sizes

        self.initial_lr = lr
        self.initial_entropy_coef = entropy_coef
        self.total_training_eps = total_training_eps

        self.optimizer = optim.Adam(
            list(self.hypernet.parameters()) +
            list(self.agent_embeddings.parameters()),
            lr=self.initial_lr
        )
        
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, 
            start_factor=1.0, 
            end_factor=0.01,
            total_iters=total_training_eps 
        )

        self.softmax = nn.Softmax(dim=-1)
        self.memory = []
        self.loss = []
        self.deterministic = False

        if action_mask is None:
            self.action_mask = None
        else:
            self.action_mask = torch.as_tensor(
                action_mask,
                dtype=torch.bool,
                device=self.device,
            )
            if self.action_mask.ndim != 1:
                raise ValueError("Action mask must be one-dimensional.")
            if self.action_mask.numel() != self.action_space_size:
                raise ValueError(
                    "Action mask size must match the action space size."
                )
            if not torch.any(self.action_mask).item():
                raise ValueError("Action mask must contain at least one valid action.")

    def save(self, path):
        torch.save({
            'hypernet_state_dict': self.hypernet.state_dict(),
            'agent_embeddings_state_dict': self.agent_embeddings.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.hypernet.load_state_dict(checkpoint['hypernet_state_dict'])
        self.agent_embeddings.load_state_dict(checkpoint['agent_embeddings_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def update_params(self, episode):
        self.scheduler.step()

        new_entropy_coef = self.initial_entropy_coef * (1 - episode / self.total_training_eps) + 0.01 * self.initial_entropy_coef
        for param_group in self.optimizer.param_groups:
            param_group['entropy_coef'] = new_entropy_coef 

    def _get_weights(self):
        agent_embed = self.agent_embeddings(self.agent_id) 

        params = self.hypernet(agent_embed)

        weights, biases = [], []
        idx = 0
        layer_sizes = [self.state_size] + self.hidden_sizes + [self.action_space_size]

        for i in range(len(layer_sizes) - 1):
            w_size = layer_sizes[i] * layer_sizes[i+1]
            b_size = layer_sizes[i+1]

            W = params[idx:idx + w_size].view(layer_sizes[i], layer_sizes[i+1])
            idx += w_size
            b = params[idx:idx + b_size]
            idx += b_size

            weights.append(W)
            biases.append(b)

        return weights, biases

    def _distribution(self, logits):
        if self.action_mask is not None:
            logits = logits.masked_fill(
                ~self.action_mask.unsqueeze(0),
                float("-inf"),
            )
        return torch.distributions.Categorical(probs=self.softmax(logits))

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        weights, biases = self._get_weights()

        logits = functional_mlp(state, weights, biases)
        dist = self._distribution(logits)
        action = torch.argmax(dist.probs) if self.deterministic else dist.sample()

        self.last_state = state.detach().cpu().numpy()
        self.last_action = action.item()
        self.last_log_prob = dist.log_prob(action).item()


        return action.item()

    def push(self, reward):
        self.memory.append((
            self.last_state,     
            self.last_action,
            self.last_log_prob,  
            reward                
        ))
        del self.last_state, self.last_action, self.last_log_prob


    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        step_losses = []

        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            states, actions, old_log_probs, rewards = zip(*batch)

            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)


            weights, biases = self._get_weights()
            logits = functional_mlp(states, weights, biases)
            dist = self._distribution(logits)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)

            if self.normalize_advantage:
                adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            else:
                adv = rewards

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv

            entropy = dist.entropy().mean()
            loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.hypernet.parameters()) +
                list(self.agent_embeddings.parameters()),
                0.5
            )
            self.optimizer.step()

            step_losses.append(loss.item())

        self.loss.append(np.mean(step_losses))
        self.memory.clear()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--env-conf', type=str, default="config1")
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

    ALGORITHM = "hyp_ippo"
    exp_id = args.id
    env_config = args.env_conf
    task_config = args.task_conf
    alg_config = args.alg_conf
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
    print(f"Torch seed: {torch_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")
    print(f"Requested route set: {requested_route_set or 'network default'}")
    print(f"Shuffle: {shuffle}")
    
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(env_seed)
    np.random.seed(env_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is:", device)
    params = {}
    params.update(json.load(open(f"../config/algo_config/{ALGORITHM}/{alg_config}.json")))
    params.update(json.load(open(f"../config/env_config/{env_config}.json")))
    params.update(json.load(open(f"../config/task_config/{task_config}.json")))
    del params["desc"]

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

    for k, v in params.items():
        globals()[k] = v

    custom_network_folder = f"../networks/{network}"
    records_folder = f"../results/{exp_id}"
    plots_folder = f"{records_folder}/plots"
    os.makedirs(records_folder, exist_ok=True)

    phases = [1, human_learning_episodes, int(training_eps) + human_learning_episodes]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]

    with open(os.path.join(custom_network_folder, f"od_{network}.txt")) as f:
        od_data = ast.literal_eval(f.read())

    origins = od_data["origins"]
    destinations = od_data["destinations"]

    agents_csv = os.path.join(custom_network_folder, "agents.csv")
    agents_df = pd.read_csv(agents_csv)
    agents_df.to_csv(os.path.join(records_folder, "agents.csv"), index=False)

    num_agents = len(agents_df)
    max_start_time = agents_df["start_time"].max()
    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + test_eps

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
                for row in agents_df.itertuples()
            }
            missing_ods = sorted(agent_ods.difference(action_masks))
            if missing_ods:
                raise ValueError(
                    "Missing action masks for agent OD pairs: "
                    + ", ".join(map(str, missing_ods))
                )

            create_paths_flag = False
            dump_config["number_of_paths"] = number_of_paths
        except FileNotFoundError as error:
            use_clustered_routes = False
            number_of_paths = configured_number_of_paths
            print(f"[CLUSTERED ROUTES] Warning: {error}")
            print("[CLUSTERED ROUTES] Falling back to JanuX generation\n")

    dump_config.update({
        "network": network,
        "env_seed": env_seed,
        "torch_seed": torch_seed,
        "route_set": route_set,
        "env_config": env_config,
        "task_config": task_config,
        "alg_config": alg_config,
        "algorithm": ALGORITHM,
        "num_agents": num_agents,
        "num_machines": num_machines,
        "script": script_path_for_config(__file__),
        "use_clustered_routes": use_clustered_routes,
        "use_action_masks": action_masks is not None,
        "shuffle": shuffle,
        "observation_type": observation_type,
        "path_gen_workers": path_gen_workers_value,
    })

    with open(os.path.join(records_folder, "exp_config.json"), "w") as f:
        json.dump(dump_config, f, indent=4)

    env = TrafficEnvironment(
        seed=env_seed,
        create_agents=False,
        create_paths=create_paths_flag,
        action_masks=action_masks,
        save_detectors_info=False,
        agent_parameters={
            "new_machines_after_mutation": num_machines,
            "human_parameters": {
                "model": human_model,
                "alpha": human_alpha,
                "beta": human_beta,
                "beta_randomness": human_beta_randomness,
                "deterministic": human_deterministic,
            },
            "machine_parameters": {
                "behavior": av_behavior,
                "observation_type": observation_type,
            }
        },
        environment_parameters={"save_every": save_every},
        simulator_parameters={
            "network_name": network,
            "custom_network_folder": custom_network_folder,
            "sumo_type": "sumo",
            "simulation_timesteps": max_start_time
        },
        plotter_parameters={
            "phases": phases,
            "phase_names": phase_names,
            "smooth_by": smooth_by,
            "plot_choices": plot_choices,
            "records_folder": records_folder,
            "plots_folder": plots_folder
        },
        path_generation_parameters={
            "origins": origins,
            "destinations": destinations,
            "number_of_paths": number_of_paths,
            "beta": path_gen_beta,
            "num_samples": num_samples,
            "path_gen_workers": path_gen_workers_value,
            "visualize_paths": False
        }
    )

    env.start()
    env.reset()
    print_agent_counts(env)

    pbar = tqdm(total=total_episodes, desc="Human learning")
    for _ in range(human_learning_episodes):
        env.step()
        pbar.update()

    env.mutation(
        disable_human_learning=not should_humans_adapt,
        mutation_start_percentile=-1
    )
    print_agent_counts(env)
    
    models_folder = os.path.join(records_folder, "saved_models")
    os.makedirs(models_folder, exist_ok=True)

    post_mutation_path = os.path.join(models_folder, "model_post_mutation.pth")
    print(f"Model zapisany po mutacji: {post_mutation_path}")
    
    obs_size = env.observation_space(env.possible_agents[0]).shape[0]
    action_size = env.machine_agents[0].action_space_size

    machine_ids = [int(a.id) for a in env.machine_agents]
    machine_features_df = agents_df[agents_df['id'].isin(machine_ids)].reset_index(drop=True)
    
    agent_embed_dim = 64

    agent_embeddings = AgentFeatureEmbedder(
            machine_features_df, 
            agent_embed_dim, 
            device=device
        ).to(device)
    
    hypernet = HyperNetwork(
        agent_embed_dim,
        obs_size,
        action_size,
        hidden_sizes=widths
    ).to(device)

    for idx, agent in enumerate(env.machine_agents):
        mask = None
        if action_masks is not None:
            key = (agent.origin, agent.destination)
            if key not in action_masks:
                raise ValueError(
                    f"Missing action mask for agent {agent.id} "
                    f"({agent.origin} -> {agent.destination})."
                )
            mask = action_masks[key]

        agent.model = PPO(
            state_size=obs_size,
            action_space_size=action_size,
            agent_id=idx,
            hypernet=hypernet,
            agent_embeddings=agent_embeddings,
            device=device,
            batch_size=batch_size,
            lr=lr,
            num_epochs=num_epochs,
            hidden_sizes=widths,
            clip_eps=clip_eps,
            normalize_advantage=normalize_advantage,
            entropy_coef=entropy_coef,
            total_training_eps=training_eps,
            action_mask=mask,
        )

    agent_lookup = {str(agent.id): agent for agent in env.machine_agents}

    pbar.set_description("AV learning")
    os.makedirs(plots_folder, exist_ok=True)

    for episode in range(training_eps):
        env.reset()
        env.machine_agents[0].model.update_params(episode) 
        
        for agent_id in env.agent_iter():
            obs, reward, term, trunc, _ = env.last()

            if term or trunc:
                agent_lookup[agent_id].model.push(reward)
                if episode % update_every == 0:
                    agent_lookup[agent_id].model.learn()
                action = None
            else:
                action = agent_lookup[agent_id].model.act(obs)

            env.step(action)

        if episode % plot_every == 0:
            env.plot_results()

        pbar.update()

    for agent in env.machine_agents:
        agent.model.deterministic = True

    pbar.set_description("Testing")
    for _ in range(test_eps):
        env.reset()
        for agent_id in env.agent_iter():
            obs, _, term, trunc, _ = env.last()
            action = None if (term or trunc) else agent_lookup[agent_id].model.act(obs)
            env.step(action)
        pbar.update()
    pbar.close()
    env.plot_results()

    losses = pd.DataFrame([
        {"id": agent.id, "losses": agent.model.loss}
        for agent in env.machine_agents
    ])
    losses.to_csv(os.path.join(records_folder, "losses.csv"), index=False)

    env.stop_simulation()
    clear_SUMO_files(
        os.path.join(records_folder, "SUMO_output"),
        os.path.join(records_folder, "episodes"),
        remove_additional_files=True
    )
    run_metrics_analysis(exp_id, results_folder="../results")
