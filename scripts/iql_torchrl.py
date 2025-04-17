# %%
import argparse
import ast
import json
import logging
import os
import torch

from torch import nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.transforms import TransformedEnv, RewardSum
from torchrl.envs.utils import check_env_specs
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import EGreedyModule, QValueModule, SafeSequential
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives.value import GAE
from torchrl.objectives import SoftUpdate, ValueEstimators, DQNLoss

from routerl import TrafficEnvironment
from tqdm import tqdm
from routerl import TrafficEnvironment


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
                "model" : human_model
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
    env.start()
    env.reset()

    # %% [markdown]
    # #### Human learning

    # %%
    for episode in range(human_learning_episodes):
        env.step()

    # %% [markdown]
    # #### Mutation
    
    # %%
    env.mutation()
    print(env)
    
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

    # %% [markdown]
    # #### Transforms

    # %%
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )

    # %% [markdown]
    # The <code style="color:white">check_env_specs()</code> function runs a small rollout and compared it output against the environment specs. It will raise an error if the specs aren't properly defined.

    # %%
    check_env_specs(env)
    env.reset()

    # %% [markdown]
    # #### Policy network

    # %% [markdown]
    # > Instantiate an `MPL` that can be used in multi-agent contexts.

    # %%
    net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=env.action_spec.space.n,
            n_agents=env.n_agents,
            centralised=False,
            share_params=False,
            device=device,
            depth=mlp_depth,
            num_cells=mlp_cells,
            activation_class=nn.ReLU,
        )

    # %%
    module = TensorDictModule(
            net, in_keys=[("agents", "observation")], out_keys=[("agents", "action_value")]
    )

    # %%
    value_module = QValueModule(
        action_value_key=("agents", "action_value"),
        out_keys=[
            env.action_key,
            ("agents", "action_value"),
            ("agents", "chosen_action_value"),
        ],
        spec=env.action_spec,
        action_space=None,
    )

    qnet = SafeSequential(module, value_module)

    # %%
    qnet_explore = TensorDictSequential(
        qnet,
        EGreedyModule(
            eps_init=eps_init,
            eps_end=eps_end,
            annealing_num_steps=int(total_frames * exploration_fraction),
            action_key=env.action_key,
            spec=env.action_spec,
        ),
    )

    # %% [markdown]
    # #### Collector

    # %%
    collector = SyncDataCollector(
            env,
            qnet_explore,
            device=device,
            storing_device=device,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
        )

    # %% [markdown]
    # #### Replay buffer

    # %%
    replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(memory_size, device=device),
            sampler=SamplerWithoutReplacement(),
            batch_size=minibatch_size,
        )

    # %% [markdown]
    # #### DQN loss function

    # %%
    loss_module = DQNLoss(qnet, delay_value=True)

    loss_module.set_keys(
            action_value=("agents", "action_value"),
            action=env.action_key,
            value=("agents", "chosen_action_value"),
            reward=env.reward_key,
            done=("agents", "done"),
            terminated=("agents", "terminated"),
    )

    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
    target_net_updater = SoftUpdate(loss_module, eps=eps)

    optim = torch.optim.Adam(loss_module.parameters(), lr)

    # %% [markdown]
    # #### Training loop

    # %%
    for i, tensordict_data in tqdm(enumerate(collector), total=n_iters, desc="Training"):
        
        current_frames = tensordict_data.numel()
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)
        
        training_tds = []

        ## Update the policies of the learning agents
        for _ in range(num_epochs):
            for _ in range(frames_per_batch // minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())

                loss_value = loss_vals["loss"]
                loss_value.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                training_tds[-1].set("grad_norm", total_norm.mean())

                optim.step()
                optim.zero_grad()
            target_net_updater.step()

        qnet_explore[1].step(frames=current_frames)  # Update exploration annealing
        collector.update_policy_weights_()
        
        training_tds = torch.stack(training_tds) 

    collector.shutdown()
    

    # %% [markdown]
    # > Testing phase

    # %%
    qnet.eval() # set the policy into evaluation mode
    for episode in range(test_eps):
        env.rollout(len(env.machine_agents), policy=qnet)
        
    # %%
    os.makedirs(plots_folder, exist_ok=True)
    env.plot_results()

    # %%
    env.stop_simulation()


