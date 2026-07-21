import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
import ast
import json
import logging
import random
from pathlib import Path

import pandas as pd
import torch
import numpy as np

from routerl import TrafficEnvironment
from routerl.keychain import Keychain as kc
from tqdm import tqdm

from clustered_routes import (
    ClusteredRoutesLoader,
    resolve_route_set,
    validate_clustered_route_set,
)
from centralized_wrapper import (
    ActorCriticMLP,
    ActorCriticRNN,
    ActorCriticWithEncoder,
    CentralizedAVEnvWrapper,
    PPO,
    TRANSITION_REWARD_MODES,
    TripInfoWithETARouteCongestionEncoder,
    TripInfoWithETASumoEncoder,
)

from utils import clear_SUMO_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle', action='store_true', default=False) # shuffle the clusters to break the action space structure
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, default="config1")
    parser.add_argument('--env-conf', type=str, default="clusters-sumo-obs")
    parser.add_argument('--task-conf', type=str, default="config1")
    parser.add_argument('--net', type=str, default="ingolstadt_custom")
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--torch-seed', type=int, default=42)
    parser.add_argument(
        '--route-set',
        type=str,
        default=None,
        help="Named route-set subdirectory. Uses the network default when omitted.",
    )
    args = parser.parse_args()
    ALGORITHM = "centralized"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    torch_seed = args.torch_seed
    shuffle = args.shuffle
    route_set = resolve_route_set(network, args.route_set)
    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {ALGORITHM.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"PyTorch seed: {torch_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")
    print(f"Route set: {route_set or 'none'}")
    print(f"Shuffle: {shuffle}")

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    random.seed(env_seed)
    np.random.seed(env_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = (
        torch.device(0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("device is: ", device)

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

    training_eps = params["training_eps"]
    batch_size = params["batch_size"]
    num_epochs = params["num_epochs"]
    policy_type = params["policy_type"]
    observation_type = params[kc.OBSERVATION_TYPE]
    reward_mode = params["reward_mode"]
    rnn_type = params["rnn_type"]
    hidden_sizes = tuple(params["hidden_sizes"])
    rnn_hidden_dim = int(params["rnn_hidden_dim"])
    lr = params["lr"]
    clip_eps = params["clip_eps"]
    gamma = params["gamma"]
    gae_lambda = params["gae_lambda"]
    normalize_advantage = params["normalize_advantage"]
    entropy_coef = params["entropy_coef"]
    value_coef = params["value_coef"]
    max_grad_norm = params["max_grad_norm"]
    buffer_size = params["buffer_size"]
    use_libsumo = bool(params.get(kc.USE_LIBSUMO, False))
    human_auto_routing_key = getattr(kc, "HUMAN_AUTO_ROUTING", None)
    ratio_machines = params["ratio_machines"]
    human_learning_episodes = params["human_learning_episodes"]
    human_model = params["human_model"]
    av_behavior = params["av_behavior"]
    save_every = params[kc.SAVE_EVERY]
    smooth_by = params["smooth_by"]
    plot_choices = params["plot_choices"]
    path_gen_beta = params["path_gen_beta"]
    num_samples = params["num_samples"]
    should_humans_adapt = params["should_humans_adapt"]
    test_eps = params["test_eps"]
    step_diagnostics_every = int(params.get("step_diagnostics_every", 25))
    if step_diagnostics_every < 1:
        raise ValueError("step_diagnostics_every must be at least 1")

    custom_network_folder = f"../networks/{network}"

    # Check if we were told to use a specific high-speed scratch directory
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

    num_machines = int(num_agents * ratio_machines)
    num_training_episodes = int(training_eps)
    phases = [1, human_learning_episodes, human_learning_episodes + num_training_episodes]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]

    # Dump exp config to records
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    dump_config["network"] = network
    dump_config["env_seed"] = env_seed
    dump_config["torch_seed"] = torch_seed
    dump_config["env_config"] = env_config
    dump_config["task_config"] = task_config
    dump_config["alg_config"] = alg_config
    dump_config["num_agents"] = num_agents
    dump_config["num_machines"] = num_machines
    dump_config["algorithm"] = ALGORITHM
    dump_config["step_diagnostics_every"] = step_diagnostics_every

    # Save checkpoints for later model analysis
    checkpoint_every = max(1, num_training_episodes // 10)
    checkpoints_dir = os.path.join(records_folder, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Clustered routes: load action masks and generating paths.csv and route.rou.xml from the pregenerated routes
    use_clustered_routes = params.get("use_clustered_routes", False)
    create_paths_flag = True
    action_masks = None

    if use_clustered_routes:
        route_set_dir = (
            Path(custom_network_folder)
            / "clustered_routes"
            / route_set
        )
        route_set_dir = validate_clustered_route_set(
            network_name=network,
            route_set_dir=route_set_dir,
        )

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
        if action_masks is not None:
            create_paths_flag = False # mask loading succeeded, disable JanuX path generation

    if action_masks is None:
        raise RuntimeError(
            "Centralized controller requires clustered routes and action masks. "
            "Fallback to unclustered JanuX routes is not supported in this controller."
        )

    dump_config["use_clustered_routes"] = use_clustered_routes # intent
    dump_config["use_action_masks"] = action_masks is not None # reality
    dump_config["route_set"] = route_set
    dump_config[kc.USE_LIBSUMO] = use_libsumo
    if use_clustered_routes:
        dump_config[kc.NUMBER_OF_PATHS] = number_of_paths
        dump_config["shuffle"] = shuffle

    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    simulator_parameters = {
        kc.NETWORK_NAME : network,
        kc.CUSTOM_NETWORK_FOLDER : custom_network_folder,
        kc.SUMO_TYPE : "sumo",
        kc.USE_LIBSUMO : use_libsumo,
    }
    if human_auto_routing_key is not None:
        simulator_parameters[human_auto_routing_key] = bool(params.get(human_auto_routing_key, False))

    # Initiate the traffic environment
    env = TrafficEnvironment(
        seed = env_seed,
        create_agents = False,
        create_paths = create_paths_flag, # Clustered routes: don't create paths if using own, clustered paths
        action_masks = action_masks, # Clustered routes: use action masks if available
        generate_asgn_data = False,
        save_detectors_info = False,
        agent_parameters = {
            kc.NEW_MACHINES_AFTER_MUTATION: num_machines,
            kc.HUMAN_PARAMETERS : {
                kc.MODEL : human_model
            },
            kc.MACHINE_PARAMETERS :{
                kc.BEHAVIOR : av_behavior,
                kc.OBSERVATION_TYPE : observation_type,
            }
        },
        simulator_parameters = simulator_parameters,
        environment_parameters = {
            kc.SAVE_EVERY: save_every,
        },
        plotter_parameters = {
            kc.PHASES : phases,
            kc.PHASE_NAMES : phase_names,
            kc.SMOOTH_BY : smooth_by,
            kc.PLOT_CHOICES : plot_choices,
            kc.RECORDS_FOLDER : records_folder,
            kc.PLOTS_FOLDER : plots_folder
        },
        # Clustered routes: no paths are generated when using action masks (just a placeholder)
        path_generation_parameters = {
            kc.ORIGINS : origins,
            kc.DESTINATIONS : destinations,
            kc.NUMBER_OF_PATHS : number_of_paths,
            kc.BETA : path_gen_beta,
            kc.NUM_SAMPLES : num_samples,
            kc.VISUALIZE_PATHS : False
        }
    )

    ###################################
    ### HUMAN LEARNING AND MUTATION ###
    ###################################

    print(f"""
    Agents in the traffic:
    • Total agents           : {len(env.all_agents)}
    • Human agents           : {len(env.human_agents)}
    • AV agents              : {len(env.machine_agents)}
    """)

    env.start()
    env.reset()

    def latest_episode_records():
        records = getattr(env, "travel_times_list", None)
        if records:
            return list(records)
        return list(getattr(env, "last_episode_travel_times", []) or [])

    def travel_times_by_agent(records):
        result = {}
        for record in records:
            agent_id = record.get(kc.AGENT_ID)
            travel_time = record.get(kc.TRAVEL_TIME)
            if agent_id is None or travel_time is None:
                continue
            travel_time = float(travel_time)
            if np.isfinite(travel_time):
                result[int(agent_id)] = travel_time
        return result

    def mean_travel_time_for_agents(records, agent_ids):
        by_agent = travel_times_by_agent(records)
        values = [
            by_agent[agent_id]
            for agent_id in agent_ids
            if agent_id in by_agent
        ]
        return float(np.mean(values)) if values else np.nan

    def pre_mutation_baseline_for_agents(agent_ids):
        episode_means = []
        for episode_travel_times in pre_mutation_episode_travel_times:
            values = [
                episode_travel_times[agent_id]
                for agent_id in agent_ids
                if agent_id in episode_travel_times
            ]
            if values:
                episode_means.append(float(np.mean(values)))

        best_mean_tt = float(np.min(episode_means)) if episode_means else np.nan
        return best_mean_tt, episode_means

    def pre_mutation_agent_baselines(agent_ids):
        baselines = {}
        for agent_id in agent_ids:
            values = [
                episode_travel_times[agent_id]
                for episode_travel_times in pre_mutation_episode_travel_times
                if agent_id in episode_travel_times
            ]
            if values:
                baselines[agent_id] = float(np.min(values))
        return baselines

    def agent_relative_progress(by_agent, baselines, agent_id):
        baseline = baselines.get(agent_id, np.nan)
        current = by_agent.get(agent_id, np.nan)
        if np.isfinite(baseline) and np.isfinite(current) and baseline > 0.0:
            return float((baseline - current) / baseline)
        return np.nan

    def mean_agent_relative_progress(by_agent, baselines, agent_ids):
        values = [
            agent_relative_progress(by_agent, baselines, agent_id)
            for agent_id in agent_ids
        ]
        values = [value for value in values if np.isfinite(value)]
        return float(np.mean(values)) if values else np.nan

    def add_relative_progress_to_saved_episodes():
        episodes_folder = os.path.join(records_folder, kc.EPISODES_LOGS_FOLDER)
        if not os.path.isdir(episodes_folder):
            return

        progress_by_episode = {
            int(row["recorded_episode"]): row
            for row in relative_progress_rows
            if row.get("recorded_episode") is not None
        }

        patched = 0
        for filename in os.listdir(episodes_folder):
            if not filename.startswith("ep") or not filename.endswith(".csv"):
                continue

            episode = int(filename.split("ep", 1)[1].split(".csv", 1)[0])
            episode_path = os.path.join(episodes_folder, filename)
            if not os.path.isfile(episode_path):
                continue

            data = pd.read_csv(episode_path)
            row = progress_by_episode.get(episode, {})
            has_progress = bool(row)
            av_progress = row.get("relative_improvement_vs_best_pre_mutation_future_av", np.nan)
            human_progress = row.get("relative_improvement_vs_best_pre_mutation_remaining_human", np.nan)
            all_progress = row.get("relative_improvement_vs_best_pre_mutation_all", np.nan)
            av_agent_mean_progress = row.get("agent_relative_improvement_future_av_mean", np.nan)
            human_agent_mean_progress = row.get("agent_relative_improvement_remaining_human_mean", np.nan)
            all_agent_mean_progress = row.get("agent_relative_improvement_all_mean", np.nan)

            data["relative_progress_future_av"] = av_progress
            data["relative_progress_remaining_human"] = human_progress
            data["relative_progress_all"] = all_progress
            data["relative_progress_agent_mean_future_av"] = av_agent_mean_progress
            data["relative_progress_agent_mean_remaining_human"] = human_agent_mean_progress
            data["relative_progress_agent_mean_all"] = all_agent_mean_progress
            data["relative_progress"] = np.where(
                data[kc.AGENT_KIND] == kc.TYPE_MACHINE,
                av_progress,
                human_progress,
            )
            by_agent = travel_times_by_agent(data.to_dict("records"))
            agent_ids = data[kc.AGENT_ID].astype(int)
            if has_progress:
                data["relative_progress_agent"] = [
                    agent_relative_progress(by_agent, best_pre_mutation_agent_tt, int(agent_id))
                    for agent_id in agent_ids
                ]
            else:
                data["relative_progress_agent"] = np.nan
            data.to_csv(episode_path, index=False)
            patched += 1

        if patched:
            print(f"Added relative progress columns to {patched} saved episode files")

    # Human learning
    pre_mutation_episode_travel_times = []
    pbar = tqdm(total=human_learning_episodes, desc="Human learning")
    for episode in range(human_learning_episodes):
        env.step()
        pre_mutation_episode_travel_times.append(
            travel_times_by_agent(latest_episode_records())
        )
        pbar.update()
    pbar.close()

    # Mutation
    env.mutation(disable_human_learning = not should_humans_adapt, mutation_start_percentile=-1)
    future_av_agent_ids = {int(agent.id) for agent in env.machine_agents}
    remaining_human_agent_ids = {int(agent.id) for agent in env.human_agents}
    best_pre_mutation_future_av_mean_tt, _ = pre_mutation_baseline_for_agents(future_av_agent_ids)
    best_pre_mutation_remaining_human_mean_tt, _, = pre_mutation_baseline_for_agents(remaining_human_agent_ids)
    best_pre_mutation_all_mean_tt, _ = pre_mutation_baseline_for_agents(future_av_agent_ids | remaining_human_agent_ids)
    best_pre_mutation_agent_tt = pre_mutation_agent_baselines(future_av_agent_ids | remaining_human_agent_ids)

    print(f"""
    Agents in the traffic:
    • Total agents           : {len(env.all_agents)}
    • Human agents           : {len(env.human_agents)}
    • AV agents              : {len(env.machine_agents)}
    • Best future-AV pre-mutation mean TT: {best_pre_mutation_future_av_mean_tt:.3f}
    • Best remaining-human pre-mutation mean TT: {best_pre_mutation_remaining_human_mean_tt:.3f}
    • Best general pre-mutation mean TT: {best_pre_mutation_all_mean_tt:.3f}
    """)

    ##############################
    ### CENTRALIZED CONTROLLER ###
    ##############################

    central_env = CentralizedAVEnvWrapper(
        env,
        action_masks,
        reward_mode=reward_mode,
    )
    obs, info = central_env.reset()

    # Check whether SUMO edge observations are initialized at the right time
    obs_dim_actual = int(obs["observation"].shape[-1])
    obs_edge_ids = len(getattr(env.observation_obj, "edge_ids", []))
    obs_edge_attr_dim = len(getattr(env.observation_obj, "edge_subscription_vars", ()))
    obs_edge_vec_len = int(getattr(env.observation_obj, "edge_vec_len", 0))
    obs_route_feature_dim = int(getattr(env.observation_obj, "route_feature_dim", 0))
    sim_edge_ids = len(getattr(env.simulator, "edge_ids", []))
    sim_edge_attr_dim = len(getattr(env.simulator, "edge_subscription_vars", ()))

    print("obs shape:", obs["observation"].shape)
    print("sim edge_ids:", sim_edge_ids)
    print("obs edge_ids:", obs_edge_ids)
    print("edge_vec_len:", obs_edge_vec_len)

    uses_sumo_edge_obs = observation_type == kc.TRIP_INFO_ETA_SUMO
    uses_route_congestion_obs = observation_type in (
        kc.TRIP_INFO_ETA_ROUTE_CONGESTION,
        kc.ROUTE_CONGESTION,
    )
    includes_eta = observation_type != kc.ROUTE_CONGESTION

    if uses_sumo_edge_obs:
        assert obs_edge_ids == sim_edge_ids, (obs_edge_ids, sim_edge_ids)
        assert obs_edge_vec_len > 0, obs_edge_vec_len
        assert obs_edge_attr_dim == sim_edge_attr_dim, (obs_edge_attr_dim, sim_edge_attr_dim)
        assert obs_edge_vec_len == obs_edge_ids * obs_edge_attr_dim, (
            obs_edge_vec_len,
            obs_edge_ids,
            obs_edge_attr_dim,
        )
    elif uses_route_congestion_obs:
        assert obs_edge_ids == 0, obs_edge_ids
        assert obs_route_feature_dim > 0, obs_route_feature_dim
        assert obs_edge_vec_len == number_of_paths * obs_route_feature_dim, (
            obs_edge_vec_len,
            number_of_paths,
            obs_route_feature_dim,
        )
    else:
        assert obs_edge_ids == 0, obs_edge_ids
        assert obs_edge_vec_len == 0, obs_edge_vec_len

    action_dim = central_env.num_actions
    assert number_of_paths == action_dim, (number_of_paths, action_dim)

    eta_dim = action_dim if includes_eta else 0
    expected_without_mask = eta_dim + 3 + obs_edge_vec_len
    expected_with_mask = eta_dim + action_dim + 3 + obs_edge_vec_len
    if obs_dim_actual == expected_with_mask:
        include_action_mask_in_obs = True
    elif obs_dim_actual == expected_without_mask:
        include_action_mask_in_obs = False
    else:
        raise ValueError(
            f"Unexpected observation dimension {obs_dim_actual}; expected {expected_without_mask} or {expected_with_mask}"
        )

    if action_masks is not None and not include_action_mask_in_obs:
        raise RuntimeError("Action masks are configured, but the observation does not include mask features.")

    observation_vector = np.asarray(obs["observation"], dtype=np.float32).reshape(-1)
    assert observation_vector.shape == (obs_dim_actual,)
    assert np.isfinite(observation_vector).all(), "Initial observation contains non-finite values"
    if includes_eta:
        assert observation_vector[:action_dim].max() <= 6.0, observation_vector[:action_dim].max()
    for agent_id, mask in central_env.agent_mask_map.items():
        mask_np = np.asarray(mask, dtype=np.bool_).reshape(-1)
        assert mask_np.shape == (action_dim,), (agent_id, mask_np.shape, action_dim)
        assert mask_np.any(), f"Agent {agent_id} has no valid actions"

    # Diagnostics
    valid_action_counts = [
        int(np.asarray(mask, dtype=np.bool_).sum())
        for mask in central_env.agent_mask_map.values()
    ]

    print("[Centralized setup]")
    print("  action_dim:", action_dim)
    print("  obs_dim_actual:", obs_dim_actual)
    print("  obs_edge_ids:", obs_edge_ids)
    print("  obs_edge_attr_dim:", obs_edge_attr_dim)
    print("  obs_edge_vec_len:", obs_edge_vec_len)
    print("  obs_route_feature_dim:", obs_route_feature_dim)
    print("  include_action_mask_in_obs:", include_action_mask_in_obs)
    print("  valid_action_counts:", {
        k: valid_action_counts.count(k)
        for k in sorted(set(valid_action_counts))
    })
    print("  reward_mode:", reward_mode)
    print("  policy_type:", policy_type)
    print("  use_libsumo:", use_libsumo)
    print("  batch_size_episodes:", batch_size)
    print("  num_epochs:", num_epochs)

    encoder = None
    if observation_type == kc.TRIP_INFO_ETA_SUMO:
        encoder = TripInfoWithETASumoEncoder(
            num_paths=action_dim,
            num_edges=obs_edge_ids,
            edge_attr_dim=obs_edge_attr_dim,
            origins=origins,
            destinations=destinations, # the rest is default
            include_action_mask_in_obs=include_action_mask_in_obs
            ) # trained by PPO as well
        obs_dim = encoder.output_dim
    elif uses_route_congestion_obs:
        encoder = TripInfoWithETARouteCongestionEncoder(
            num_paths=action_dim,
            route_feature_dim=obs_route_feature_dim,
            origins=origins,
            destinations=destinations, # the rest is default
            include_action_mask_in_obs=include_action_mask_in_obs,
            include_eta=includes_eta,
            ) # trained by PPO as well
        obs_dim = encoder.output_dim
    else:
        obs_dim = obs_dim_actual

    if policy_type == "rnn":
        core = ActorCriticRNN(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_type=rnn_type,
        )
    else:
        core = ActorCriticMLP(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        )

    policy_net = ActorCriticWithEncoder(encoder, core) if encoder is not None else core

    ppo = PPO(
        policy_net=policy_net,
        action_space_size=action_dim,
        device=str(device),
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        clip_eps=clip_eps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        normalize_advantage=normalize_advantage,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm,
        buffer_size=buffer_size,
    )

    step_diag_rows = []
    update_diag_rows = []
    relative_progress_rows = []
    batch_pending_penalized = 0
    batch_unmatched_arrivals = 0

    def append_step_diagnostics(
        *,
        info,
        phase,
        episode_idx,
        reward,
        reward_by_agent,
        mask,
        policy_diag,
    ):
        assigned_rewards = list(reward_by_agent.values())
        step_diag_rows.append({
            "phase": phase,
            "episode_idx": int(episode_idx),
            "step_idx": int(info["step_idx"]),
            "agent_id": int(info["agent_id"]),
            "origin": int(info["origin"]),
            "destination": int(info["destination"]),
            "start_time": float(info["start_time"]),
            "action": int(info["action"]),
            "step_reward": float(reward),
            "num_assigned_rewards_this_step": int(len(assigned_rewards)),
            "assigned_reward_mean_this_step": (
                float(np.mean(assigned_rewards)) if assigned_rewards else np.nan
            ),
            "mask": mask.tolist(),
            **policy_diag,
        })

    def relative_progress_for_current_episode():
        records = latest_episode_records()
        by_agent = travel_times_by_agent(records)

        # Difference of means: compare current group mean to the best pre-mutation group mean
        current_av_mean_tt = mean_travel_time_for_agents(
            records,
            future_av_agent_ids,
        )
        current_human_mean_tt = mean_travel_time_for_agents(
            records,
            remaining_human_agent_ids,
        )
        current_all_mean_tt = mean_travel_time_for_agents(
            records,
            future_av_agent_ids | remaining_human_agent_ids
        ) # weighted mean because AV-human proportions may differ

        av_baseline = best_pre_mutation_future_av_mean_tt
        if np.isfinite(current_av_mean_tt) and np.isfinite(av_baseline) and av_baseline > 0.0:
            av_relative_improvement = (av_baseline - current_av_mean_tt) / av_baseline
        else:
            av_relative_improvement = np.nan

        human_baseline = best_pre_mutation_remaining_human_mean_tt
        if np.isfinite(current_human_mean_tt) and np.isfinite(human_baseline) and human_baseline > 0.0:
            human_relative_improvement = (human_baseline - current_human_mean_tt) / human_baseline
        else:
            human_relative_improvement = np.nan

        all_baseline = best_pre_mutation_all_mean_tt
        if np.isfinite(current_all_mean_tt) and np.isfinite(all_baseline) and all_baseline > 0.0:
            all_relative_improvement = (all_baseline - current_all_mean_tt) / all_baseline
        else:
            all_relative_improvement = np.nan

        # Mean of per-agent relative improvements: compare each agent to its own
        # best pre-mutation travel time, then average within the group
        av_agent_mean_relative_improvement = mean_agent_relative_progress(
            by_agent,
            best_pre_mutation_agent_tt,
            future_av_agent_ids,
        )
        human_agent_mean_relative_improvement = mean_agent_relative_progress(
            by_agent,
            best_pre_mutation_agent_tt,
            remaining_human_agent_ids,
        )
        all_agent_mean_relative_improvement = mean_agent_relative_progress(
            by_agent,
            best_pre_mutation_agent_tt,
            future_av_agent_ids | remaining_human_agent_ids,
        )

        return {
            "recorded_episode": int(env.day),
            "relative_improvement_vs_best_pre_mutation_future_av": av_relative_improvement,
            "relative_improvement_vs_best_pre_mutation_remaining_human": human_relative_improvement,
            "relative_improvement_vs_best_pre_mutation_all": all_relative_improvement,
            "agent_relative_improvement_future_av_mean": av_agent_mean_relative_improvement,
            "agent_relative_improvement_remaining_human_mean": human_agent_mean_relative_improvement,
            "agent_relative_improvement_all_mean": all_agent_mean_relative_improvement,
        }

    train_pbar = tqdm(total=num_training_episodes, desc="AV learning")
    test_pbar = tqdm(total=test_eps, desc="Test phase")
    # batch_size episodes are collected from the current policy -> ppo.learn() runs
    # these episodes are reused for num_epochs optimization epochs -> then they are discarded
    try:
        # TRAINING PHASE
        for episode_idx in range(num_training_episodes):
            obs, info = central_env.reset()
            ppo.reset_episode()
            done = False

            while not done:
                observation = np.asarray(obs["observation"], dtype=np.float32).reshape(-1)
                mask_raw = obs["action_mask"]
                mask = mask_raw.cpu().numpy() if torch.is_tensor(mask_raw) else np.asarray(mask_raw)
                mask = np.asarray(mask, dtype=np.bool_).reshape(-1)

                assert mask.shape == (action_dim,), (mask.shape, action_dim)
                assert mask.any(), "Encountered an empty action mask during training"
                assert np.isfinite(observation).all(), "Observation contained non-finite values"
                if includes_eta:
                    assert observation[:action_dim].max() <= 6.0, observation[:action_dim].max()
                assert observation.shape == (obs_dim_actual,), (observation.shape, obs_dim_actual)

                action = ppo.act(observation, mask)
                policy_diag = getattr(ppo, "last_policy_diag", {}).copy()

                obs, reward, terminated, truncated, info = central_env.step(action)
                done = terminated or truncated

                if reward_mode in TRANSITION_REWARD_MODES:
                    routed_agent_id = int(info["agent_id"])

                    initial_reward = reward if reward_mode == "transition_tt_fft_ad" else 0.0
                    ppo.push_pending(
                        agent_id=routed_agent_id,
                        done=False,
                        initial_reward=initial_reward,
                    )
                    reward_by_agent = {}

                    if done:
                        reward_by_agent = central_env.get_episode_av_rewards(reward_mode)
                        missing_reward = -float(
                            getattr(env.simulator, "simulation_length", 60.0)
                        ) / 60.0
                        ppo.finish_episode(
                            rewards_by_agent=reward_by_agent,
                            missing_reward=missing_reward,
                        )
                else:
                    reward_by_agent = {}
                    ppo.push(reward, done)

                if episode_idx % step_diagnostics_every == 0:
                    append_step_diagnostics(
                        info=info,
                        phase="train",
                        episode_idx=episode_idx,
                        reward=reward,
                        reward_by_agent=reward_by_agent,
                        mask=mask,
                        policy_diag=policy_diag,
                    )

            # Diagnostics
            if reward_mode in TRANSITION_REWARD_MODES:
                episode_diag = getattr(ppo, "last_episode_diag", {}).copy()
                batch_pending_penalized += int(episode_diag.get("episode_pending_penalized", 0))
                batch_unmatched_arrivals += int(episode_diag.get("episode_unmatched_arrivals", 0))
            relative_progress_rows.append(relative_progress_for_current_episode())

            # Learn + diagnostics
            before_updates = getattr(ppo, "update_count", 0)
            ppo.learn()
            after_updates = getattr(ppo, "update_count", 0)

            # Save model checkpoint (only for evaluation - for training resume would need: optimizer state (Adam),
            # random number states, ...)
            if (episode_idx + 1) % checkpoint_every == 0 or episode_idx + 1 == num_training_episodes:
                checkpoint_path = os.path.join(
                    checkpoints_dir,
                    f"checkpoint_ep{episode_idx + 1}.pt",
                )
                torch.save(ppo.policy_net.state_dict(), checkpoint_path)

            if after_updates > before_updates:
                update_diag = getattr(ppo, "last_update_diag", {}).copy()
                update_diag["episode_idx"] = episode_idx
                update_diag["batch_pending_penalized_sum"] = int(batch_pending_penalized)
                update_diag["batch_unmatched_arrivals_sum"] = int(batch_unmatched_arrivals)
                update_diag_rows.append(update_diag)
                batch_pending_penalized = 0
                batch_unmatched_arrivals = 0

            assert central_env.num_steps == central_env.num_machines
            train_pbar.update()

        env.plot_results()

        # TESTING PHASE
        ppo.policy_net.eval()
        ppo.deterministic = False #True
        for episode_idx in range(test_eps):
            obs, info = central_env.reset()
            ppo.reset_episode()
            done = False

            while not done:
                observation = np.asarray(obs["observation"], dtype=np.float32).reshape(-1)
                mask_raw = obs["action_mask"]
                mask = mask_raw.cpu().numpy() if torch.is_tensor(mask_raw) else np.asarray(mask_raw)
                mask = np.asarray(mask, dtype=np.bool_).reshape(-1)

                assert mask.shape == (action_dim,), (mask.shape, action_dim)
                assert mask.any(), "Encountered an empty action mask during testing"
                assert np.isfinite(observation).all(), "Observation contained non-finite values"
                if includes_eta:
                    assert observation[:action_dim].max() <= 6.0, observation[:action_dim].max()
                assert observation.shape == (obs_dim_actual,), (observation.shape, obs_dim_actual)

                action = ppo.act(observation, mask)
                policy_diag = getattr(ppo, "last_policy_diag", {}).copy()

                obs, reward, terminated, truncated, info = central_env.step(action)
                done = terminated or truncated

                if reward_mode in TRANSITION_REWARD_MODES:
                    reward_by_agent = (
                        central_env.get_episode_av_rewards(reward_mode)
                        if done
                        else {}
                    )
                else:
                    reward_by_agent = {}

                append_step_diagnostics(
                    info=info,
                    phase="test",
                    episode_idx=episode_idx,
                    reward=reward,
                    reward_by_agent=reward_by_agent,
                    mask=mask,
                    policy_diag=policy_diag,
                )

            test_pbar.update()

            relative_progress_rows.append(relative_progress_for_current_episode())

    finally:
        train_pbar.close()
        test_pbar.close()

        losses_txt_path = os.path.join(records_folder, "losses.txt")
        losses_csv_path = os.path.join(records_folder, "losses.csv")
        losses_pd = pd.DataFrame({"losses": ppo.loss})
        losses_pd.to_csv(losses_csv_path, index=False)
        with open(losses_txt_path, "w", encoding="utf-8") as losses_file:
            for loss_value in ppo.loss:
                losses_file.write(f"{loss_value}\n")

        add_relative_progress_to_saved_episodes()
        env.plot_results()

        central_env.close()

        clear_SUMO_files(
            os.path.join(records_folder, "SUMO_output"),
            os.path.join(records_folder, "episodes"),
            remove_additional_files=True,
        )

        update_diag_path = os.path.join(records_folder, "ppo_update_diagnostics.csv")
        pd.DataFrame(update_diag_rows).to_csv(update_diag_path, index=False)
        print(f"Saved PPO update diagnostics to {update_diag_path}")

        cluster_diag_path = os.path.join(records_folder, "cluster_choice_diagnostics.csv")
        pd.DataFrame(step_diag_rows).to_csv(cluster_diag_path, index=False)
        print(
            f"Saved sampled training and complete test step diagnostics to {cluster_diag_path}"
        )
