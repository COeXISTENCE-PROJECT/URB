# %%
import argparse
import ast
import json
import logging
import os
import torch

from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torch.distributions import Categorical
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.transforms import TransformedEnv, RewardSum
from torchrl.envs.utils import check_env_specs
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import MultiAgentMLP, ProbabilisticActor
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from routerl import TrafficEnvironment
from tqdm import tqdm

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    exp_id = args.id
    exp_config = args.conf
    network = args.net
    seed = args.seed
    print("### STARTING EXPERIMENT ###")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Seed: {seed}")

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # %%
    device = (
        torch.device(0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("device is: ", device)

    # %% [markdown]
    # #### Hyperparameters setting

    # %%
    params = json.load(open("../experiment_metadata.json"))
    params = params[exp_config]["config"]

    # %%
    # set params as variables in this notebook
    for key, value in params.items():
        globals()[key] = value

    # %%
    custom_network_folder = f"../networks/{network}"
    training_episodes = (frames_per_batch / new_machines_after_mutation) * n_iters
    total_frames = frames_per_batch * n_iters
    phases = [0, human_learning_episodes, int(training_episodes) + human_learning_episodes]
    phase_names = ["Human learning", "Mutation - Machine learning", "Testing phase"]
    records_folder = f"../records/{exp_id}"
    plots_folder = f"../plots/{exp_id}"

    # Read origin-destinations
    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    data = ast.literal_eval(content)
    origins = data['origins']
    destinations = data['destinations']

    # %%
    # Copy agents.csv from custom_network_folder to records_folder
    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    if os.path.exists(agents_csv_path):
        os.makedirs(records_folder, exist_ok=True)
        new_agents_csv_path = os.path.join(records_folder, "agents.csv")
        with open(agents_csv_path, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(new_agents_csv_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    # %%
    # Dump exp config to records
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    dump_config["network"] = network
    dump_config["seed"] = seed
    dump_config["config"] = exp_config
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    # %%
    env = TrafficEnvironment(
        seed = seed,
        create_agents = False,
        create_paths = True,
        save_detectors_info = False,
        agent_parameters = {
            "new_machines_after_mutation": new_machines_after_mutation, 
            "human_parameters" : {
                "model" : human_model, 
                "noise_weight_agent" : noise_weight_agent,
                "noise_weight_path" : noise_weight_path,
                "noise_weight_day" : noise_weight_day,
                "beta" : beta,
                "beta_k_i_variability" : beta_k_i_variability,
                "epsilon_i_variability" : epsilon_i_variability,
                "epsilon_k_i_variability" : epsilon_k_i_variability,
                "epsilon_k_i_t_variability" : epsilon_k_i_t_variability,
                "greedy" : greedy,
                "gamma_c" : gamma_c,
                "gamma_u" : gamma_u,
                "remember" : remember,
                "alpha_zero" : alpha_zero,
                "alphas" : alphas
                },
            "machine_parameters" :{
                "behavior" : av_behavior,
                }
            },
        simulator_parameters = {
            "network_name" : network,
            "custom_network_folder" : custom_network_folder,
            "sumo_type" : "sumo",
        }, 
        environment_parameters = {
            "save_every" : 5,
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

    # %%
    print("Number of total agents is: ", len(env.all_agents), "\n")
    print("Number of human agents is: ", len(env.human_agents), "\n")
    print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

    # %%
    env.start()
    res = env.reset()

    # %% [markdown]
    # #### Human learning

    # %%
    for episode in range(human_learning_episodes):
        env.step()

    # %% [markdown]
    # #### Mutation

    # %%
    env.mutation()

    # %%
    print("Number of total agents is: ", len(env.all_agents), "\n")
    print("Number of human agents is: ", len(env.human_agents), "\n")
    print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

    # %%
    group = {'agents': [str(machine.id) for machine in env.machine_agents]}

    env = PettingZooWrapper(
        env=env,
        use_mask=True,
        categorical_actions=True,
        done_on_any = False,
        group_map=group,
        device=device
    )

    # %%
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )

    # %%
    check_env_specs(env)


    # %%
    reset_td = env.reset()

    # %%
    share_parameters_policy = False 

    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs = env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs = env.action_spec.space.n,
            n_agents = env.n_agents,
            centralised=False,
            share_params=share_parameters_policy,
            device=device,
            depth=policy_network_depth,
            num_cells=policy_network_num_cells,
            activation_class=torch.nn.Tanh,
        ),
    )

    # %%
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "logits")],
    ) 

    # %%
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=[("agents", "logits")],
        out_keys=[env.action_key],
        distribution_class=Categorical,
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )

    # %%
    share_parameters_critic = True
    mappo = False  # IPPO if False

    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1, 
        n_agents=env.n_agents,
        centralised=mappo,
        share_params=share_parameters_critic,
        device=device,
        depth=critic_network_depth,
        num_cells=critic_network_num_cells,
        activation_class=torch.nn.ReLU,
    )

    critic = TensorDictModule(
        module=critic_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )

    # %% [markdown]
    # #### Collector

    # %%
    collector = SyncDataCollector(
        env,
        policy,
        device=device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    ) 

    # %% [markdown]
    # #### Replay buffer

    # %%
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            frames_per_batch, device=device
        ),  
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,
    )

    # %% [markdown]
    # #### PPO loss function

    # %%
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

    # %% [markdown]
    # #### Training loop

    # %%
    pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")

    episode_reward_mean_list = []
    loss_values = []
    loss_entropy = []
    loss_objective = []
    loss_critic = []

    for tensordict_data in collector: ##loops over frame_per_batch

        ## Generate the rollouts
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

                loss_values.append(loss_value.item())

                loss_entropy.append(loss_vals["loss_entropy"].item())

                loss_objective.append(loss_vals["loss_objective"].item())

                loss_critic.append(loss_vals["loss_critic"].item())


    
        collector.update_policy_weights_()
    
        # Logging
        done = tensordict_data.get(("next", "agents", "done"))  # Get done status for the group

        episode_reward_mean = (
            tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
        )
        episode_reward_mean_list.append(episode_reward_mean)


        pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
        pbar.update()

    # %%
    policy.eval() # set the policy into evaluation mode
    for episode in range(test_eps):
        env.rollout(len(env.machine_agents), policy=policy)

    # %%
    os.makedirs(plots_folder, exist_ok=True)
    env.plot_results()

    # %%
    env.stop_simulation()


