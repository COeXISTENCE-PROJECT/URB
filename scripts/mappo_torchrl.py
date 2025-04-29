import argparse
import ast
import json
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from routerl import TrafficEnvironment
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
from tqdm import tqdm

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--torch-seed', type=int, default=42)
    args = parser.parse_args()
    exp_id = args.id
    exp_config = args.conf
    network = args.net
    env_seed = args.env_seed
    torch_seed = args.torch_seed
    print("### STARTING EXPERIMENT ###")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"PyTorch seed: {torch_seed}")
    print(f"Experiment config: {exp_config}")

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

     
    # #### Hyperparameters setting

    
    params = json.load(open("../experiment_metadata.json"))
    params = params[exp_config]["config"]

    
    # set params as variables in this script
    for key, value in params.items():
        globals()[key] = value

    
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
    dump_config["config"] = exp_config
    dump_config["num_agents"] = num_agents
    dump_config["num_machines"] = num_machines
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    
    env = TrafficEnvironment(
        seed = env_seed,
        create_agents = False,
        create_paths = True,
        save_detectors_info = False,
        agent_parameters = {
            "new_machines_after_mutation": num_machines, 
            "human_parameters" : {
                "model" : human_model
            },
            "machine_parameters" :{
                "behavior" : av_behavior,
            }
        },
        environment_parameters = {
            "save_every" : save_every,
        },
        simulator_parameters = {
            "network_name" : network,
            "custom_network_folder" : custom_network_folder,
            "sumo_type" : "sumo",
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

    print(f"""
    Agents in the traffic:
    • Total agents           : {len(env.all_agents)}
    • Human agents           : {len(env.human_agents)}
    • AV agents              : {len(env.machine_agents)}
    """)

    
    env.start()
    res = env.reset()

     
    # #### Human learning

    pbar = tqdm(total=human_learning_episodes, desc="Human learning")
    for episode in range(human_learning_episodes):
        env.step()
        pbar.update()
    pbar.close()
     
    # #### Mutation
    env.mutation(disable_human_learning = not should_humans_adapt, mutation_start_percentile = -1)

    print(f"""
    Agents in the traffic:
    • Total agents           : {len(env.all_agents)}
    • Human agents           : {len(env.human_agents)}
    • AV agents              : {len(env.machine_agents)}
    """)

    
    group = {'agents': [str(machine.id) for machine in env.machine_agents]}

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

    
    check_env_specs(env)


    
    reset_td = env.reset()

    
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

    
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "logits")],
    )

    
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=[("agents", "logits")],
        out_keys=[env.action_key],
        distribution_class=Categorical,
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )

    
    share_parameters_critic = False
    mappo = True  # IPPO if False

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

     
    # #### Collector

    
    collector = SyncDataCollector(
        env,
        policy,
        device=device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    ) 

     
    # #### Replay buffer

    
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            frames_per_batch, device=device
        ),  
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,
    )

     
    # #### PPO loss function

    
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

     
    # #### Training loop
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


