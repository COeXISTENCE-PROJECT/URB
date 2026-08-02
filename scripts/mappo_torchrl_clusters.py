"""
This script is used to train MAPPO agents using the TorchRL library in a traffic simulation environment.
The MAPPO implementation is based on: https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html

Compared to mappo_torchrl.py, this script can load pre-generated clustered
routes, mask unavailable route actions, and encode route-congestion
observations. If clustered routes are disabled or unavailable, it falls back
to normal JanuX route generation.
"""

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
import argparse
import ast
import json
import logging

import pandas as pd
import torch

from routerl.keychain import Keychain as kc
from routerl import TrafficEnvironment
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torch.distributions import Categorical
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.transforms import TransformedEnv, RewardSum
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import MaskedCategorical, MultiAgentMLP, ProbabilisticActor
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from tqdm import tqdm

from centralized_wrapper import TripInfoWithETARouteCongestionEncoder
from clustered_routes import AVMaskWrapper, ClusteredRoutesLoader, resolve_route_set
from utils import AppendODEmbedding
from utils import clear_SUMO_files
from utils import get_od_ids_for_group
from utils import print_agent_counts
from utils import run_metrics_analysis
from utils import save_loss_records
from utils import script_path_for_config


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
    ALGORITHM = "mappo_torchrl"
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

    # Hyperparameters setting
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

    # set params as variables in this script
    for key, value in params.items():
        globals()[key] = value

    observation_type = params.get(
        kc.OBSERVATION_TYPE,
        params.get("observations", kc.PREVIOUS_AGENTS_PLUS_START_TIME),
    )
    custom_network_folder = f"../networks/{network}"
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
    dump_config["script"] = script_path_for_config(__file__)
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
        except FileNotFoundError as error:
            print(f"[CLUSTERED ROUTES] Warning: {error}")
            print("[CLUSTERED ROUTES] Falling back to JanuX generation\n")

    dump_config["use_clustered_routes"] = use_clustered_routes # intent
    dump_config["use_action_masks"] = action_masks is not None # reality
    if use_clustered_routes:
        dump_config["number_of_paths"] = number_of_paths
        dump_config["shuffle"] = shuffle

    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    # Initiate the traffic environment
    env = TrafficEnvironment(
        seed = env_seed,
        create_agents = False,
        create_paths = create_paths_flag, # Clustered routes: don't create paths if using own, clustered paths
        action_masks = action_masks, # Clustered routes: use action masks if available
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
            "machine_parameters" :{
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
            "path_gen_workers" : params.get("path_gen_workers", 1),
            "visualize_paths" : False
        } 
    )

    print_agent_counts(env)
    env.start()
    res = env.reset()
     
    # Human learning
    pbar = tqdm(total=human_learning_episodes, desc="Human learning")
    for episode in range(human_learning_episodes):
        env.step()
        pbar.update()
    pbar.close()
     
    # Mutation
    env.mutation(disable_human_learning = not should_humans_adapt, mutation_start_percentile = -1)
    print_agent_counts(env)

    
    group_agent_ids = [str(machine.id) for machine in env.machine_agents]
    group = {"agents": group_agent_ids}
    # Keep OD ids aligned with the TorchRL group order.
    od_ids = get_od_ids_for_group(group_agent_ids, env.machine_agents, len(destinations))
    num_od_pairs = len(origins) * len(destinations)

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
            "mask": ("agents", "action_mask"),
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

    reset_td = env.reset()
    use_route_congestion_encoder = observation_type in {
        kc.TRIP_INFO_ETA_ROUTE_CONGESTION,
        kc.ROUTE_CONGESTION,
    }
    network_obs_key = ("agents", "encoded_observation")

    if use_route_congestion_encoder:
        if obs_route_feature_dim <= 0:
            raise ValueError(
                f"{observation_type} requires a route_feature_dim on the observation object"
            )

        encoder_kwargs = {
            "num_paths": number_of_paths,
            "route_feature_dim": obs_route_feature_dim,
            "origins": origins,
            "destinations": destinations,
            "include_action_mask_in_obs": action_masks is not None,
            "include_eta": observation_type == kc.TRIP_INFO_ETA_ROUTE_CONGESTION,
        }
        policy_encoder_model = TorchRLObservationEncoder(
            TripInfoWithETARouteCongestionEncoder(**encoder_kwargs)
        ).to(device)
        critic_encoder_model = TorchRLObservationEncoder(
            TripInfoWithETARouteCongestionEncoder(**encoder_kwargs)
        ).to(device)
        policy_observation_encoder = TensorDictModule(
            policy_encoder_model,
            in_keys=[obs_key],
            out_keys=[network_obs_key],
        )
        critic_observation_encoder = TensorDictModule(
            critic_encoder_model,
            in_keys=[obs_key],
            out_keys=[network_obs_key],
        )
        network_obs_dim = policy_encoder_model.output_dim
        print(
            f"Using {observation_type} encoder: raw_obs_dim="
            f"{env.observation_spec[obs_key].shape[-1]}, encoded_dim={network_obs_dim}"
        )
    else:
        policy_observation_encoder = TensorDictModule(
            AppendODEmbedding(od_ids, num_od_pairs, od_embedding_dim).to(device),
            in_keys=[obs_key],
            out_keys=[network_obs_key],
        )
        critic_observation_encoder = TensorDictModule(
            AppendODEmbedding(od_ids, num_od_pairs, od_embedding_dim).to(device),
            in_keys=[obs_key],
            out_keys=[network_obs_key],
        )
        network_obs_dim = env.observation_spec[obs_key].shape[-1] + od_embedding_dim

    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs = network_obs_dim,
            n_agent_outputs = env.action_spec.space.n,
            n_agents = env.n_agents,
            centralised=False,
            share_params=share_params_agent,
            device=device,
            depth=policy_network_depth,
            num_cells=policy_network_num_cells,
            activation_class=torch.nn.Tanh,
        ),
    )

    policy_logits_module = TensorDictModule(
        policy_net,
        in_keys=[network_obs_key],
        out_keys=[("agents", "logits")],
    )

    policy_module = TensorDictSequential(policy_observation_encoder, policy_logits_module)

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=actor_in_keys,
        out_keys=[env.action_key],
        distribution_class=actor_distribution_class,
        default_interaction_type="random",
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )

    critic_net = MultiAgentMLP(
        n_agent_inputs=network_obs_dim,
        n_agent_outputs=1, 
        n_agents=env.n_agents,
        centralised=True, # MAPPO critic is centralized
        share_params=share_params_critic,
        device=device,
        depth=critic_network_depth,
        num_cells=critic_network_num_cells,
        activation_class=torch.nn.ReLU,
    )

    critic_value_module = TensorDictModule(
        module=critic_net,
        in_keys=[network_obs_key],
        out_keys=[("agents", "state_value")],
    )
    critic = TensorDictSequential(critic_observation_encoder, critic_value_module)

     
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

    # Training loop
    loss_records = []
    
    pbar = tqdm(total=n_iters, desc="Training")
    for tensordict_data in collector:
        tensordict_data.set(
            ("next", "agents", "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),  # Adjust index to start from 0
        )
        tensordict_data.set(
            ("next", "agents", "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),  # Adjust index to start from 0
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
            avg_loss_value = sum(step_loss_values) / len(step_loss_values)
            loss_records.append(
                {
                    "iteration": len(loss_records) + 1,
                    "loss": avg_loss_value,
                    "loss_entropy": sum(step_loss_entropy) / len(step_loss_entropy),
                    "loss_objective": sum(step_loss_objective) / len(step_loss_objective),
                    "loss_critic": sum(step_loss_critic) / len(step_loss_critic),
                }
            )
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
    save_loss_records(
        records_folder,
        loss_records,
        columns=["iteration", "loss", "loss_entropy", "loss_objective", "loss_critic"],
    )
    
    env.stop_simulation()

    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)
    run_metrics_analysis(exp_id, results_folder="../results")
