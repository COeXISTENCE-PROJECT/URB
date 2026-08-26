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

from routerl         import TrafficEnvironment
from tqdm            import tqdm

from baseline_models import BaseLearningModel
from utils           import clear_SUMO_files
from utils           import print_agent_counts
from utils           import run_metrics_analysis
from utils           import save_loss_records
from utils           import script_path_for_config


class TabularREINFORCE(BaseLearningModel):
    """Independent tabular REINFORCE for URB's single-step decisions."""

    def __init__(self, action_space_size, rng, bin_width=1.0, lr=0.1,
                 baseline_lr=0.1, entropy_coef=0.0):
        super().__init__()
        if lr <= 0.0:
            raise ValueError("lr must be greater than zero")
        if not 0.0 < baseline_lr <= 1.0:
            raise ValueError("baseline_lr must be in (0, 1]")
        if entropy_coef < 0.0:
            raise ValueError("entropy_coef must be non-negative")

        self.action_space_size = action_space_size
        self.rng = rng
        self.bin_width = bin_width
        self.lr = lr
        self.baseline_lr = baseline_lr
        self.entropy_coef = entropy_coef

        self.preferences = dict()
        self.baselines = dict()
        self.loss = list()
        self.deterministic = False

    def _discretize(self, state):
        if isinstance(state, dict):
            if "observation" not in state:
                raise ValueError("Dictionary observations must contain an 'observation' entry")
            state = state["observation"]

        observation = np.asarray(state, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(observation)):
            raise ValueError("Observations must contain only finite values")

        widths = np.asarray(self.bin_width, dtype=np.float64).reshape(-1)
        if widths.size == 1:
            widths = np.full(observation.shape, widths.item())
        elif widths.size != observation.size:
            raise ValueError(
                "bin_width must be one number or contain one value per observation entry"
            )
        if np.any(widths <= 0.0) or not np.all(np.isfinite(widths)):
            raise ValueError("All bin_width values must be finite and greater than zero")

        return tuple(np.floor(observation / widths).astype(np.int64))

    def act(self, state):
        action_mask = state.get("action_mask") if isinstance(state, dict) else None
        if action_mask is None:
            valid_actions = np.arange(self.action_space_size)
        else:
            action_mask = np.asarray(action_mask, dtype=bool).reshape(-1)
            if action_mask.size != self.action_space_size:
                raise ValueError("Action mask size does not match the action space")
            valid_actions = np.flatnonzero(action_mask)
            if valid_actions.size == 0:
                raise ValueError("Action mask must allow at least one action")

        state_key = self._discretize(state)
        preferences = self.preferences.setdefault(
            state_key, np.zeros(self.action_space_size, dtype=np.float64)
        )
        valid_logits = preferences[valid_actions]
        shifted_logits = valid_logits - np.max(valid_logits)
        valid_log_probs = shifted_logits - np.log(np.exp(shifted_logits).sum())
        valid_probs = np.exp(valid_log_probs)

        if self.deterministic:
            best_indices = np.flatnonzero(valid_probs == np.max(valid_probs))
            action_index = int(self.rng.choice(best_indices))
        else:
            action_index = int(self.rng.choice(len(valid_actions), p=valid_probs))
        action = int(valid_actions[action_index])

        self.last_state = state_key
        self.last_action_index = action_index
        self.last_valid_actions = valid_actions
        self.last_probs = valid_probs
        self.last_log_probs = valid_log_probs
        return action

    def push(self, reward):
        self.last_reward = float(reward)

    def learn(self):
        baseline = self.baselines.setdefault(self.last_state, 0.0)
        advantage = self.last_reward - baseline

        policy_gradient = -self.last_probs.copy()
        policy_gradient[self.last_action_index] += 1.0
        entropy = -np.sum(self.last_probs * self.last_log_probs)
        entropy_gradient = -self.last_probs * (self.last_log_probs + entropy)

        preferences = self.preferences[self.last_state]
        preferences[self.last_valid_actions] += self.lr * (
            advantage * policy_gradient
            + self.entropy_coef * entropy_gradient
        )
        self.baselines[self.last_state] += self.baseline_lr * advantage

        loss = (
            -advantage * self.last_log_probs[self.last_action_index]
            -self.entropy_coef * entropy
        )
        self.loss.append(float(loss))
        del (
            self.last_state,
            self.last_action_index,
            self.last_valid_actions,
            self.last_probs,
            self.last_log_probs,
            self.last_reward,
        )


# Main script to run the tabular REINFORCE experiment
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--env-conf', type=str, default="config1")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--torch-seed', type=int, default=42)
    parser.add_argument('--skip-metrics', action='store_true', default=False)
    args = parser.parse_args()
    ALGORITHM = "reinforce_tabular"
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
    print(f"Algorithm seed: {torch_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")

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
    del params["desc"], env_params, task_params

    # set params as variables in this script
    for key, value in params.items():
        globals()[key] = value


    custom_network_folder = f"../networks/{network}"
    phases = [1, human_learning_episodes, int(training_eps) + human_learning_episodes]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]
    base_results_dir = os.environ.get("RESULTS_BASE_DIR", "../results")
    records_folder = os.path.join(base_results_dir, exp_id)
    plots_folder = os.path.join(records_folder, "plots")

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
        raise FileNotFoundError(f"Agents file not found: {agents_csv_path}")

    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + test_eps

    # Dump exp config to records
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
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
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)


    # Initialize the environment
    env = TrafficEnvironment(
        seed = env_seed,
        create_agents = False,
        create_paths = True,
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
            "path_gen_workers" : path_gen_workers,
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

    # Set policies for machine agents
    for idx in range(len(env.machine_agents)):
        env.machine_agents[idx].model = TabularREINFORCE(
            env.machine_agents[idx].action_space_size,
            rng=np.random.default_rng(torch_seed + idx),
            bin_width=bin_width,
            lr=lr,
            baseline_lr=baseline_lr,
            entropy_coef=entropy_coef,
        )
    agent_lookup = {str(agent.id): agent for agent in env.machine_agents}


    ### Learning phase ###
    pbar.set_description("AV learning")
    os.makedirs(plots_folder, exist_ok=True)
    for episode in range(training_eps):
        env.reset()
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                agent_lookup[agent_id].model.push(reward)
                agent_lookup[agent_id].model.learn()
                action = None
            else:
                action = agent_lookup[agent_id].model.act(observation)

            env.step(action)

        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()


    ### Testing phase ###
    for agent in env.machine_agents:
        agent.model.deterministic = True

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
    loss_records = (
        {
            "iteration": iteration,
            "agent_id": agent.id,
            "loss": loss_value,
        }
        for agent in env.machine_agents
        for iteration, loss_value in enumerate(agent.model.loss, start=1)
    )
    save_loss_records(
        records_folder,
        loss_records,
        columns=["iteration", "agent_id", "loss"],
    )

    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)
    if not args.skip_metrics:
        run_metrics_analysis(exp_id, results_folder=base_results_dir)
