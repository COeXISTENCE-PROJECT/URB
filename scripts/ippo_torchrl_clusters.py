"""
This script is used to train IPPO agents using the TorchRL library in a traffic simulation environment.
The IPPO implementation is based on: https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html

Main differences compared to ippo_torchrl.py:
- When clustered routes are used, loads a pre-generated clustered route set and action masks.
- When clustered routes are used, writes the selected route set into the experiment folder
    as paths.csv, routes.csv, and route.rou.xml before starting SUMO.
- When clustered routes are used, allows to use route-congestion observations and encodes
    them before the TorchRL actor/critic.

Runtime behavior:
- If use_clustered_routes=true and --route-set is provided and exists, the script exports the
  clustered routes, loads action masks, disables JanuX path generation, wraps the
  env with AVMaskWrapper, and uses MaskedCategorical so invalid actions cannot be
  sampled by the policy.
- If --route-set is omitted, the default route set for the selected network is used.
- If use_clustered_routes=true but the clustered files are missing, the script
  prints a warning and falls back to a non-clustered IPPO
  run: JanuX generates paths, no action masks are used etc.
- If use_clustered_routes=false, the script behaves like a non-clustered IPPO
  run: JanuX generates paths, no action masks are used etc.
- If the observation type is trip_info_eta_route_congestion or route_congestion,
  the raw route-feature observation is encoded before it reaches the actor and
  critic. If a classic/flat observation is used, no extra encoder is applied.

Intended usage: use an env config with use_clustered_routes=true, for example
clusters or clusters-sumo-obs, and pass --route-set to override the default
clustered_routes/<route-set> directory. Non-clustered fallback exists for
compatibility, but this script is mainly for clustered IPPO experiments.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
import ast
import json
import logging

import matplotlib.pyplot as plt
import pandas as pd
import torch

from routerl import TrafficEnvironment
from routerl.keychain import Keychain as kc
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torch.distributions import Categorical
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.transforms import TransformedEnv, RewardSum
from torchrl.envs.utils import check_env_specs
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import MaskedCategorical, MultiAgentMLP, ProbabilisticActor
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from tqdm import tqdm

from clustered_routes import AVMaskWrapper, ClusteredRoutesLoader, resolve_route_set
from centralized_wrapper import TripInfoWithETARouteCongestionEncoder

from utils import clear_SUMO_files

class TorchRLObservationEncoder(torch.nn.Module):
    """
    Shape adapter for centralized-wrapper encoders used inside TorchRL.

    TorchRL can pass observations with arbitrary leading dimensions, e.g.
    (agents, obs_dim), (frames, agents, obs_dim), or minibatches. The wrapped
    encoder only cares about the last dimension, so we flatten all leading
    dimensions, encode, and then restore the original leading shape.
    """

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.output_dim = encoder.output_dim

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        leading_shape = observation.shape[:-1]
        flat_observation = observation.reshape(-1, observation.shape[-1])
        encoded = self.encoder(flat_observation)
        if encoded.dim() == 3 and encoded.shape[1] == 1:
            encoded = encoded.squeeze(1)
        return encoded.reshape(*leading_shape, self.output_dim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle', action='store_true', default=False) # shuffle the clusters to break the action space structure
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, default="config1")
    parser.add_argument('--env-conf', type=str, default="clusters-sumo-obs")
    parser.add_argument('--task-conf', type=str, default="config1")
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--torch-seed', type=int, default=42)
    parser.add_argument(
        '--route-set',
        type=str,
        default=None,
        help="Named route-set subdirectory. Uses the network default when omitted.",
    )
    args = parser.parse_args()
    ALGORITHM = "ippo_torchrl"
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
    print(f"PyTorch seed: {torch_seed}")
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

    # JanuX fallback for non-clustered configs
    use_clustered_routes = params.get("use_clustered_routes", False)
    route_set = (
        resolve_route_set(network, requested_route_set)
        if use_clustered_routes
        else None
    )
    print(f"Route set: {route_set or 'none (unclustered)'}")

    # Set params as variables in this script
    for key, value in params.items():
        globals()[key] = value

    human_auto_routing_key = getattr(kc, "HUMAN_AUTO_ROUTING", None)
    observation_type = params.get(kc.OBSERVATION_TYPE, kc.PREVIOUS_AGENTS_PLUS_START_TIME)
    custom_network_folder = f"../networks/{network}"

    # Check whether to use the scratch directory
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
    training_episodes = agent_frames_per_batch * n_iters
    frames_per_batch = num_machines * agent_frames_per_batch
    total_frames = frames_per_batch * n_iters
    phases = [1, human_learning_episodes, int(training_episodes) + human_learning_episodes]
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
    dump_config["route_set"] = route_set

    # Clustered routes: load action masks and generate paths.csv, route.rou.xml from the pregenerated routes
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
            if action_masks is not None:
                create_paths_flag = False # mask loading succeeded, disable JanuX path generation
        except FileNotFoundError as e:
            print(f"[CLUSTERED ROUTES] Warning: {e}")
            print("[CLUSTERED ROUTES] Falling back to JanuX generation\n")

    dump_config["use_clustered_routes"] = use_clustered_routes # intent
    dump_config["use_action_masks"] = action_masks is not None # reality
    if use_clustered_routes:
        dump_config["number_of_paths"] = number_of_paths
        dump_config["shuffle"] = shuffle

    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    simulator_parameters = {
        "network_name" : network,
        "custom_network_folder" : custom_network_folder,
        "sumo_type" : "sumo",
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
            "new_machines_after_mutation": num_machines,
            "human_parameters" : {
                "model" : human_model
            },
            "machine_parameters" :{
                "behavior" : av_behavior,
                "observation_type" : observation_type,
            }
        },
        simulator_parameters = simulator_parameters,
        environment_parameters = {
            "save_every" : save_every,
        },
        plotter_parameters = {
            "phases" : phases,
            "phase_names" : phase_names,
            "smooth_by" : smooth_by,
            "plot_choices" : plot_choices,
            "records_folder" : records_folder,
            "plots_folder" : plots_folder
        },
        # Clustered routes: no paths are generated when using action masks (just a placeholder)
        path_generation_parameters = {
            "origins" : origins,
            "destinations" : destinations,
            "number_of_paths" : number_of_paths,
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

    # Human learning
    pbar = tqdm(total=human_learning_episodes, desc="Human learning")
    for episode in range(human_learning_episodes):
        env.step()
        pbar.update()
    pbar.close()

    # Mutation
    env.mutation(disable_human_learning = not should_humans_adapt, mutation_start_percentile=-1)

    print(f"""
    Agents in the traffic:
    • Total agents           : {len(env.all_agents)}
    • Human agents           : {len(env.human_agents)}
    • AV agents              : {len(env.machine_agents)}
    """)

    group = {'agents': [str(machine.id) for machine in env.machine_agents]}

    # Clustered routes: add action masks to "normal" observations if available.
    # Also, change the observation keys to include the action mask.
    # MaskedCategorical distribution directly handles invalid actions
    # Actor - pass the per-agent mask to the distribution
    obs_route_feature_dim = int(getattr(env.observation_obj, "route_feature_dim", 0))
    if action_masks is not None:
        env = AVMaskWrapper(env, action_masks)
        obs_key = ("agents", "observation", "observation")
        actor_distribution_class = MaskedCategorical
        actor_in_keys = {
            "logits": ("agents", "logits"),
            "mask": ("agents", "action_mask")
        }
    else:
        obs_key = ("agents", "observation")
        actor_distribution_class = Categorical
        actor_in_keys = {
            "logits": ("agents", "logits"),
        }

    env = PettingZooWrapper(
        env=env,
        use_mask=True,
        categorical_actions=True,
        done_on_any = False,
        group_map=group,
        device=device
    )

    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )

    use_route_congestion_encoder = observation_type in {
        kc.TRIP_INFO_ETA_ROUTE_CONGESTION,
        kc.ROUTE_CONGESTION,
    }
    encoded_obs_key = ("agents", "encoded_observation")
    policy_in_keys = [obs_key]
    critic_in_keys = [obs_key]
    n_agent_inputs = env.observation_spec[obs_key].shape[-1]

    if use_route_congestion_encoder:
        if obs_route_feature_dim <= 0:
            raise ValueError(
                f"{observation_type} requires a route_feature_dim on the observation object"
            )

        include_eta = observation_type == kc.TRIP_INFO_ETA_ROUTE_CONGESTION
        encoder_kwargs = dict(
            num_paths=number_of_paths,
            route_feature_dim=obs_route_feature_dim,
            origins=origins,
            destinations=destinations,
            include_action_mask_in_obs=action_masks is not None,
            include_eta=include_eta,
        )
        policy_encoder_model = TorchRLObservationEncoder(
            TripInfoWithETARouteCongestionEncoder(**encoder_kwargs)
        ).to(device)
        critic_encoder_model = TorchRLObservationEncoder(
            TripInfoWithETARouteCongestionEncoder(**encoder_kwargs)
        ).to(device)
        policy_encoder = TensorDictModule(
            policy_encoder_model,
            in_keys=[obs_key],
            out_keys=[encoded_obs_key],
        )
        critic_encoder = TensorDictModule(
            critic_encoder_model,
            in_keys=[obs_key],
            out_keys=[encoded_obs_key],
        )
        policy_in_keys = [encoded_obs_key]
        critic_in_keys = [encoded_obs_key]
        n_agent_inputs = policy_encoder_model.output_dim
        print(
            f"Using {observation_type} encoder: raw_obs_dim="
            f"{env.observation_spec[obs_key].shape[-1]}, encoded_dim={n_agent_inputs}"
        )

    share_parameters_policy = True # False in ippo_torchrl.py

    raw_policy_net = MultiAgentMLP(
        n_agent_inputs = n_agent_inputs,
        n_agent_outputs = env.action_spec.space.n,
        n_agents = env.n_agents,
        centralised=False,
        share_params=share_parameters_policy,
        device=device,
        depth=policy_network_depth,
        num_cells=policy_network_num_cells,
        activation_class=torch.nn.Tanh,
    )

    raw_policy_module = TensorDictModule(
        raw_policy_net,
        in_keys=policy_in_keys, # Clustered routes: use different keys with action masks
        out_keys=[("agents", "logits")],
    )
    if use_route_congestion_encoder:
        policy_module = TensorDictSequential(policy_encoder, raw_policy_module)
    else:
        policy_module = raw_policy_module

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=actor_in_keys,
        out_keys=[env.action_key],
        distribution_class=actor_distribution_class,
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )


    if action_masks is not None:
        print("[CLUSTERED ROUTES] Skipping check_env_specs because the random rollout can sample masked actions and invalid SUMO routes.")
    else:
        check_env_specs(env)

    reset_td = env.reset()

    share_parameters_critic = True
    mappo = False # IPPO if False

    critic_net = MultiAgentMLP(
        n_agent_inputs=n_agent_inputs,
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralised=mappo,
        share_params=share_parameters_critic,
        device=device,
        depth=critic_network_depth,
        num_cells=critic_network_num_cells,
        activation_class=torch.nn.ReLU,
    )

    raw_critic = TensorDictModule(
        module=critic_net,
        in_keys=critic_in_keys, # Clustered routes: use different keys with action masks
        out_keys=[("agents", "state_value")],
    )
    if use_route_congestion_encoder:
        critic = TensorDictSequential(critic_encoder, raw_critic)
    else:
        critic = raw_critic

    # Collector
    collector = SyncDataCollector(
        env,
        policy,
        device=device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )

    # Replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            frames_per_batch, device=device
        ),
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,
    )

    # PPO loss function
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_eps,
        normalize_advantage=normalize_advantage,
    )
    loss_module.set_keys(
        reward=env.reward_key,
        action=env.action_key,
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )

    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
    )
    GAE = loss_module.value_estimator
    optim = torch.optim.Adam(loss_module.parameters(), lr)

    #  Training loop
    loss_values_path = os.path.join(records_folder, "losses/loss_values.txt")
    loss_entropy_path = os.path.join(records_folder, "losses/loss_entropy.txt")
    loss_objective_path = os.path.join(records_folder, "losses/loss_objective.txt")
    loss_critic_path = os.path.join(records_folder, "losses/loss_critic.txt")
    os.makedirs(os.path.dirname(loss_values_path), exist_ok=True)
    open(loss_values_path, 'w').close()
    open(loss_entropy_path, 'w').close()
    open(loss_objective_path, 'w').close()
    open(loss_critic_path, 'w').close()

    pbar = tqdm(total=n_iters, desc="Training")
    for tensordict_data in collector:
        tensordict_data.set(
            ("next", "agents", "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))), # Adjust index to start from 0
        )
        tensordict_data.set(
            ("next", "agents", "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))), # Adjust index to start from 0
        )

        # Compute GAE for all agents
        with torch.no_grad():
                GAE(
                    tensordict_data,
                    params=loss_module.critic_network_params,
                    target_params=loss_module.target_critic_network_params,
                )

        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        step_loss_values, step_loss_entropy, step_loss_objective, step_loss_critic = [], [], [], []
        ## Update the policies of the learning agents
        for _ in range(num_epochs):
            for _ in range(frames_per_batch // minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()

                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), max_grad_norm
                )

                optim.step()
                optim.zero_grad()

                step_loss_values.append(loss_value.item())
                step_loss_entropy.append(loss_vals["loss_entropy"].item())
                step_loss_objective.append(loss_vals["loss_objective"].item())
                step_loss_critic.append(loss_vals["loss_critic"].item())

        if step_loss_values:
            with open(loss_values_path, 'a') as f:
                f.write(f"{sum(step_loss_values) / len(step_loss_values)}\n")
            with open(loss_entropy_path, 'a') as f:
                f.write(f"{sum(step_loss_entropy) / len(step_loss_entropy)}\n")
            with open(loss_objective_path, 'a') as f:
                f.write(f"{sum(step_loss_objective) / len(step_loss_objective)}\n")
            with open(loss_critic_path, 'a') as f:
                f.write(f"{sum(step_loss_critic) / len(step_loss_critic)}\n")
        collector.update_policy_weights_()
        pbar.update()

    pbar.close()
    collector.shutdown()

    # Testing phase
    pbar = tqdm(total=test_eps, desc="Test phase")
    policy.eval() # set the policy into evaluation mode
    for episode in range(test_eps):
        env.rollout(len(env.machine_agents), policy=policy)
        pbar.update()
    pbar.close()

    os.makedirs(plots_folder, exist_ok=True)
    env.plot_results()

    # Visualize losses
    loss_values = list()
    with open(loss_values_path, 'r') as f:
        for line in f:
            loss_values.append(float(line.strip()))
    loss_entropy = list()
    with open(loss_entropy_path, 'r') as f:
        for line in f:
            loss_entropy.append(float(line.strip()))
    loss_objective = list()
    with open(loss_objective_path, 'r') as f:
        for line in f:
            loss_objective.append(float(line.strip()))
    loss_critic = list()
    with open(loss_critic_path, 'r') as f:
        for line in f:
            loss_critic.append(float(line.strip()))
    colors = [
        "firebrick", "teal", "peru", "navy",
        "salmon", "slategray", "darkviolet",
        "lightskyblue", "darkolivegreen", "black"]
    plt.figure(figsize=(12, 6))
    plt.plot(loss_values, label='loss_values', color=colors[0], linewidth=3)
    plt.plot(loss_entropy, label='loss_entropy', color=colors[1], linewidth=3)
    plt.plot(loss_objective, label='loss_objective', color=colors[2], linewidth=3)
    plt.plot(loss_critic, label='loss_critic', color=colors[3], linewidth=3)
    plt.legend(fontsize=12)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Losses', fontsize=18, fontweight='bold')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, 'losses.png'), dpi=300)
    plt.close()

    env.stop_simulation()

    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)
