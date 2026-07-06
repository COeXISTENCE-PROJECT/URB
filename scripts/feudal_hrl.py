"""Initial Feudal HRL experiment script for URB-style single-step route decisions.

This is a conservative first pass aligned with the repository's existing script pattern:
- one standalone script under scripts/
- one JSON config folder under config/algo_config/feudal_hrl/
- lightweight PyTorch models in models/

Current assumptions / limitations:
- manager picks a discrete subgoal every `manager_period` decision opportunities
- controller chooses the actual route conditioned on that subgoal
- intrinsic reward is simple consistency/progress shaping, not a domain-specific latent-goal loss
- action masking currently uses coarse uniform bins over action indices when enabled
- cluster IDs are scaffolded in config/model interfaces, but this script does not yet load cluster CSVs

This is meant to get the project started quickly with a code path that mirrors the existing IPPO/IQL scripts.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from baseline_models import BaseLearningModel
from scripts.controller import FeudalController
from scripts.manager import FeudalManager
from routerl import TrafficEnvironment
from utils import (  # type: ignore
    clear_SUMO_files,
    print_agent_counts,
    run_metrics_analysis,
    save_loss_records,
    script_path_for_config,
)

def load_cluster_lookup(cluster_csv_path, key_columns):
    df = pd.read_csv(cluster_csv_path)
    if "cluster" not in df.columns:
        raise ValueError(f"No 'cluster' column in {cluster_csv_path}")
    unique_clusters = sorted(df["cluster"].unique())
    cluster_to_idx = {c: i + 1 for i, c in enumerate(unique_clusters)}
    lookup = {}
    for _, row in df.iterrows():
        key = tuple(row[col] for col in key_columns)
        lookup[key] = cluster_to_idx[row["cluster"]]
    num_clusters = len(unique_clusters) + 1
    return lookup, num_clusters

def build_agent_cluster_map(agents_csv_path, cluster_lookup, key_columns):
    agents_df = pd.read_csv(agents_csv_path)
    cluster_map = {}
    missing = []
    for idx, row in agents_df.iterrows():
        key = tuple(row[col] for col in key_columns)
        if key in cluster_lookup:
            cluster_map[idx] = int(cluster_lookup[key])
        else:
            cluster_map[idx] = 0
            missing.append(idx)
    return cluster_map, missing

def build_mlp_optimizer(module: nn.Module, lr: float) -> optim.Optimizer:
    return optim.Adam(module.parameters(), lr=lr)


@dataclass
class Transition:
    state: np.ndarray
    subgoal: int
    action: int
    manager_log_prob: float
    controller_log_prob: float
    extrinsic_reward: float
    intrinsic_reward: float
    manager_step: bool


class FeudalAgent(BaseLearningModel):
    def __init__(
        self,
        state_size: int,
        action_space_size: int,
        config: Dict,
        device: torch.device,
        cluster_id: int = 0,
    ):
        super().__init__()
        self.device = device
        self.cluster_id = int(cluster_id)
        self.use_cluster_embedding = bool(config.get("use_cluster_embedding", False))
        self.num_clusters = int(config.get("num_clusters", 0))
        self.action_space_size = int(action_space_size)
        self.manager_period = int(config["manager_period"])
        self.num_subgoals = int(config["num_subgoals"])
        self.batch_size = int(config["batch_size"])
        self.manager_epochs = int(config["manager_epochs"])
        self.controller_epochs = int(config["controller_epochs"])
        self.update_every = int(config["update_every"])
        self.manager_clip_eps = float(config["manager_clip_eps"])
        self.controller_clip_eps = float(config["controller_clip_eps"])
        self.manager_entropy_coef = float(config["manager_entropy_coef"])
        self.controller_entropy_coef = float(config["controller_entropy_coef"])
        self.normalize_advantage = bool(config["normalize_advantage"])
        self.intrinsic_reward_weight = float(config["intrinsic_reward_weight"])
        self.manager_reward_weight = float(config["manager_reward_weight"])
        self.goal_switch_penalty = float(config.get("goal_switch_penalty", 0.0))
        self.action_mask_strategy = str(config.get("action_mask_strategy", "uniform_bins"))
        self.deterministic = False
        self.decision_count = 0
        self.current_subgoal: Optional[int] = None
        self.previous_subgoal: Optional[int] = None
        self.memory: List[Transition] = []
        self.loss: List[Dict[str, float]] = []
        
        self.last_transition_stub = None

        self.manager = FeudalManager(
            obs_dim=state_size,
            num_subgoals=self.num_subgoals,
            hidden_dims=config["manager_hidden_dims"],
            use_cluster_embedding=self.use_cluster_embedding,
            num_clusters=self.num_clusters,
            cluster_embed_dim=int(config.get("cluster_embed_dim", 8)),
        ).to(self.device)
        self.controller = FeudalController(
            obs_dim=state_size,
            action_dim=self.action_space_size,
            num_subgoals=self.num_subgoals,
            hidden_dims=config["controller_hidden_dims"],
            subgoal_embed_dim=int(config["subgoal_embed_dim"]),
        ).to(self.device)

        self.manager_optimizer = build_mlp_optimizer(self.manager, float(config["manager_lr"]))
        self.controller_optimizer = build_mlp_optimizer(self.controller, float(config["controller_lr"]))

    def _to_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _build_uniform_subgoal_mask(self, subgoal: int) -> torch.Tensor:
        if self.action_mask_strategy != "uniform_bins":
            return torch.ones((1, self.action_space_size), dtype=torch.float32, device=self.device)

        bins = np.array_split(np.arange(self.action_space_size), self.num_subgoals)
        rotated = (subgoal + self.cluster_id) % self.num_subgoals
        chosen = bins[rotated]
        mask = torch.zeros((1, self.action_space_size), dtype=torch.float32, device=self.device)
        mask[0, chosen] = 1.0
        return mask

    def _select_subgoal(self, state: np.ndarray) -> tuple[int, float]:
        state_tensor = self._to_tensor(state)
        if getattr(self, "use_cluster_embedding", False):
            cluster_tensor = torch.tensor([self.cluster_id], dtype=torch.long, device=self.device)
            output = self.manager.act(state_tensor, cluster_ids=cluster_tensor, deterministic=self.deterministic)
        else:
            output = self.manager.act(state_tensor, deterministic=self.deterministic)
        return output.subgoal, output.log_prob

    def act(self, state):
        state = np.asarray(state, dtype=np.float32)
        manager_step = self.current_subgoal is None or (self.decision_count % self.manager_period == 0)
        if manager_step:
            new_subgoal, manager_log_prob = self._select_subgoal(state)
            if self.current_subgoal is not None and new_subgoal != self.current_subgoal:
                self.previous_subgoal = self.current_subgoal
            self.current_subgoal = new_subgoal
        else:
            manager_log_prob = 0.0

        state_tensor = self._to_tensor(state)
        subgoal_tensor = torch.tensor([self.current_subgoal], dtype=torch.long, device=self.device)
        action_mask = self._build_uniform_subgoal_mask(self.current_subgoal)
        controller_output = self.controller.act(
            state_tensor,
            subgoal_tensor,
            action_mask=action_mask,
            deterministic=self.deterministic,
        )

        self.last_transition_stub = {
            "state": state.copy(),
            "subgoal": int(self.current_subgoal),
            "action": int(controller_output.action),
            "manager_log_prob": float(manager_log_prob),
            "controller_log_prob": float(controller_output.log_prob),
            "manager_step": manager_step,
        }
        self.decision_count += 1
        return int(controller_output.action)

    def push(self, reward):
        if self.last_transition_stub is None:
            return
            
        reward = float(reward)
        intrinsic_reward = self._intrinsic_reward()
        record = Transition(
            state=self.last_transition_stub["state"],
            subgoal=self.last_transition_stub["subgoal"],
            action=self.last_transition_stub["action"],
            manager_log_prob=self.last_transition_stub["manager_log_prob"],
            controller_log_prob=self.last_transition_stub["controller_log_prob"],
            extrinsic_reward=reward,
            intrinsic_reward=intrinsic_reward,
            manager_step=bool(self.last_transition_stub["manager_step"]),
        )
        self.memory.append(record)
        self.last_transition_stub = None

    def _intrinsic_reward(self) -> float:
        reward = 0.0
        if self.previous_subgoal is not None and self.current_subgoal != self.previous_subgoal:
            reward -= self.goal_switch_penalty
            self.previous_subgoal = self.current_subgoal
        reward += 1.0 / max(self.num_subgoals, 1)
        return reward

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_advantage or x.numel() <= 1:
            return x
        return (x - x.mean()) / (x.std() + 1e-8)

    def _controller_update(self, batch: List[Transition]) -> float:
        states = torch.as_tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self.device)
        subgoals = torch.as_tensor([b.subgoal for b in batch], dtype=torch.long, device=self.device)
        actions = torch.as_tensor([b.action for b in batch], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor([b.controller_log_prob for b in batch], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(
            [b.extrinsic_reward + self.intrinsic_reward_weight * b.intrinsic_reward for b in batch],
            dtype=torch.float32,
            device=self.device,
        )
        advantages = self._normalize(rewards)

        losses = []
        for _ in range(self.controller_epochs):
            action_masks = torch.cat([self._build_uniform_subgoal_mask(int(sg)) for sg in subgoals.tolist()], dim=0)
            dist = self.controller.dist(states, subgoals, action_mask=action_masks)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.controller_clip_eps, 1 + self.controller_clip_eps) * advantages
            entropy = dist.entropy().mean()
            loss = -torch.min(surr1, surr2).mean() - self.controller_entropy_coef * entropy
            self.controller_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.controller.parameters(), max_norm=1.0)
            self.controller_optimizer.step()
            losses.append(float(loss.item()))
        return float(np.mean(losses))

    def _manager_update(self, batch: List[Transition]) -> float:
        manager_batch = [b for b in batch if b.manager_step]
        if not manager_batch:
            return 0.0
        states = torch.as_tensor(np.stack([b.state for b in manager_batch]), dtype=torch.float32, device=self.device)
        subgoals = torch.as_tensor([b.subgoal for b in manager_batch], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor([b.manager_log_prob for b in manager_batch], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(
            [self.manager_reward_weight * b.extrinsic_reward for b in manager_batch],
            dtype=torch.float32,
            device=self.device,
        )
        advantages = self._normalize(rewards)

        losses = []

        if getattr(self, "use_cluster_embedding", False):
            cluster_ids = torch.full((len(manager_batch),), self.cluster_id, dtype=torch.long, device=self.device)

        for _ in range(self.manager_epochs):
            if getattr(self, "use_cluster_embedding", False):
                dist = self.manager.dist(states, cluster_ids=cluster_ids)
            else:
                dist = self.manager.dist(states)            
            new_log_probs = dist.log_prob(subgoals)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.manager_clip_eps, 1 + self.manager_clip_eps) * advantages
            entropy = dist.entropy().mean()
            loss = -torch.min(surr1, surr2).mean() - self.manager_entropy_coef * entropy
            self.manager_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.manager.parameters(), max_norm=1.0)
            self.manager_optimizer.step()
            losses.append(float(loss.item()))
        return float(np.mean(losses))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory[:]
        manager_loss = self._manager_update(batch)
        controller_loss = self._controller_update(batch)
        self.loss.append({
            "manager_loss": manager_loss,
            "controller_loss": controller_loss,
            "combined_loss": manager_loss + controller_loss,
        })
        self.memory.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--env-conf", type=str, default="config1")
    parser.add_argument("--task-conf", type=str, required=True)
    parser.add_argument("--alg-conf", type=str, required=True)
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--torch-seed", type=int, default=42)
    args = parser.parse_args()

    ALGORITHM = "feudal_hrl"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    torch_seed = args.torch_seed

    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {ALGORITHM.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Torch seed: {torch_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(env_seed)
    np.random.seed(env_seed)

    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
    print("Device is:", device)

    params = {}
    alg_params = json.load(open(f"../config/algo_config/{ALGORITHM}/{alg_config}.json"))
    env_params = json.load(open(f"../config/env_config/{env_config}.json"))
    task_params = json.load(open(f"../config/task_config/{task_config}.json"))
    params.update(alg_params)
    params.update(env_params)
    params.update(task_params)
    del params["desc"]

    for key, value in params.items():
        globals()[key] = value

    custom_network_folder = f"../networks/{network}"
    phases = [1, human_learning_episodes, int(training_eps) + human_learning_episodes]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]
    records_folder = f"../results/{exp_id}"
    plots_folder = f"../results/{exp_id}/plots"

    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, "r", encoding="utf-8") as f:
        data = ast.literal_eval(f.read())
    origins = data["origins"]
    destinations = data["destinations"]

    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    num_agents = len(pd.read_csv(agents_csv_path))
    if os.path.exists(agents_csv_path):
        os.makedirs(records_folder, exist_ok=True)
        new_agents_csv_path = os.path.join(records_folder, "agents.csv")
        with open(agents_csv_path, "r", encoding="utf-8") as src, open(new_agents_csv_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        max_start_time = pd.read_csv(new_agents_csv_path)["start_time"].max()
    else:
        raise FileNotFoundError(f"Agents CSV file not found at {agents_csv_path}.")

    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + test_eps

    cluster_csv_path = None
    if alg_params.get("use_cluster_embedding", False) and alg_params.get("cluster_csv_path"):
        cluster_csv_path = os.path.join(repo_root, alg_params["cluster_csv_path"])
    
    key_columns = alg_params.get("cluster_key_columns", ["start_time", "origin", "destination"])
    agent_cluster_map = {}

    if cluster_csv_path and os.path.exists(cluster_csv_path):
        cluster_lookup, num_clusters = load_cluster_lookup(cluster_csv_path, key_columns)
        agent_cluster_map, missing_indices = build_agent_cluster_map(agents_csv_path, cluster_lookup, key_columns)
        params["num_clusters"] = num_clusters
    else:
        params["num_clusters"] = 1

    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    dump_config.update(
        {
            "network": network,
            "env_seed": env_seed,
            "torch_seed": torch_seed,
            "env_config": env_config,
            "task_config": task_config,
            "alg_config": alg_config,
            "script": script_path_for_config(__file__),
            "algorithm": ALGORITHM,
            "num_agents": num_agents,
            "num_machines": num_machines,
        }
    )
    with open(exp_config_path, "w", encoding="utf-8") as f:
        json.dump(dump_config, f, indent=4)


    env = TrafficEnvironment(
        seed=env_seed,
        create_agents=False,
        create_paths=True,
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
                "observation_type": "previous_agents_plus_start_time",
            },
        },
        environment_parameters={
            "save_every": save_every,
        },
        simulator_parameters={
            "network_name": network,
            "custom_network_folder": custom_network_folder,
            "sumo_type": "sumo",
            "simulation_timesteps": max_start_time,
        },
        plotter_parameters={
            "phases": phases,
            "phase_names": phase_names,
            "smooth_by": smooth_by,
            "plot_choices": plot_choices,
            "records_folder": records_folder,
            "plots_folder": plots_folder,
        },
        path_generation_parameters={
            "origins": origins,
            "destinations": destinations,
            "number_of_paths": number_of_paths,
            "beta": path_gen_beta,
            "num_samples": num_samples,
            "path_gen_workers": path_gen_workers,
            "visualize_paths": False,
        },
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
        mutation_start_percentile=-1,
    )
    print_agent_counts(env)

    obs_size = env.observation_space(env.possible_agents[0]).shape[0]

    for idx in range(len(env.machine_agents)):
        agent_obj = env.machine_agents[idx]
        try:
            agent_int_id = int(str(agent_obj.id).split('_')[-1])
        except:
            agent_int_id = idx
            
        c_id = agent_cluster_map.get(agent_int_id, 0)
        
        agent_obj.model = FeudalAgent(
            state_size=obs_size,
            action_space_size=agent_obj.action_space_size,
            config=params,
            device=device,
            cluster_id=c_id, 
        )
        
    # Słownik MUSI być zainicjowany po stworzeniu modeli, inaczej pętla wyżej wybuchnie!
    agent_lookup = {str(agent.id): agent for agent in env.machine_agents}

    os.makedirs(plots_folder, exist_ok=True)
    pbar.set_description("AV learning")
    for episode in range(training_eps):
        env.reset()
        episode_rewards = []
        episode_travel_times = []
        episode_manager_losses = []
        episode_controller_losses = []
        episode_losses = []

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            # 1. Zawsze najpierw ładujemy poprzedni krok do pamięci
            agent_lookup[agent_id].model.push(reward)
            
            # 2. Sprawdzamy czy to koniec epizodu dla tego agenta
            if termination or truncation:
                reward = float(reward)
                episode_rewards.append(reward)
                if isinstance(info, dict) and "travel_time" in info:
                    episode_travel_times.append(float(info["travel_time"]))
                else:
                    episode_travel_times.append(-reward)

                if episode % update_every == 0:
                    losses_before = len(agent_lookup[agent_id].model.loss)
                    agent_lookup[agent_id].model.learn()
                    if len(agent_lookup[agent_id].model.loss) > losses_before:
                        loss_value = agent_lookup[agent_id].model.loss[-1]
                        episode_manager_losses.append(loss_value["manager_loss"])
                        episode_controller_losses.append(loss_value["controller_loss"])
                        episode_losses.append(loss_value["combined_loss"])
                action = None
            else:
                # 3. Jeśli to nie koniec, odpalamy nową akcję
                action = agent_lookup[agent_id].model.act(observation)
                
            env.step(action)

        log_data = {
            "episode": human_learning_episodes + episode,
            "training/reward_sum": float(np.sum(episode_rewards)),
            "training/reward_mean": float(np.mean(episode_rewards)),
            "training/travel_time_mean": float(np.mean(episode_travel_times)),
            "training/travel_time_sum": float(np.sum(episode_travel_times)),
        }
        if episode_losses:
            log_data.update(
                {
                    "training/manager_loss": float(np.mean(episode_manager_losses)),
                    "training/controller_loss": float(np.mean(episode_controller_losses)),
                    "training/loss": float(np.mean(episode_losses)),
                }
            )
        
        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()

    for agent in env.machine_agents:
        agent.model.deterministic = True
        agent.model.manager.eval()
        agent.model.controller.eval()

    pbar.set_description("Testing")
    for episode in range(test_eps):
        env.reset()
        episode_rewards = []
        episode_travel_times = []

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                reward = float(reward)
                episode_rewards.append(reward)
                if isinstance(info, dict) and "travel_time" in info:
                    episode_travel_times.append(float(info["travel_time"]))
                else:
                    episode_travel_times.append(-reward)
                action = None
            else:
                action = agent_lookup[agent_id].model.act(observation)
            env.step(action)


        pbar.update()

    pbar.close()
    env.plot_results()

    loss_records = []
    for agent in env.machine_agents:
        for iteration, loss_value in enumerate(agent.model.loss, start=1):
            loss_records.append(
                {
                    "iteration": iteration,
                    "agent_id": agent.id,
                    "manager_loss": loss_value["manager_loss"],
                    "controller_loss": loss_value["controller_loss"],
                    "loss": loss_value["combined_loss"],
                }
            )
    save_loss_records(
        records_folder,
        loss_records,
        columns=["iteration", "agent_id", "manager_loss", "controller_loss", "loss"],
    )

    env.stop_simulation()
    clear_SUMO_files(
        os.path.join(records_folder, "SUMO_output"),
        os.path.join(records_folder, "episodes"),
        remove_additional_files=True,
    )
    run_metrics_analysis(exp_id, results_folder="../results")
    
    rewards_path = os.path.join(plots_folder, "rewards.png")
    travel_times_path = os.path.join(plots_folder, "travel_times.png")
    plots_to_log = {}