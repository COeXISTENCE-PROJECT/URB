"""
This script implements centralized DQN approach for the CAV routing task.

The implementation is built around two components:
- `GlobalObservation`: stores and updates fleet-level state information
- `cDQN`: selects routes for departing CAVs from global state snapshots

Developed: Sept-Nov 2025, updated Apr-May 2026
Primary author: M.Sudoł
"""

from __future__ import annotations

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
import math
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from routerl     import TrafficEnvironment, Keychain as kc
from tqdm        import tqdm


from baseline_models import BaseLearningModel
from utils           import (
    clear_SUMO_files,
    CSVLossLogger,
    print_agent_counts,
    run_metrics_analysis,
    script_path_for_config
)




####################################  GLOBAL OBSERVATION AND Q-NETWORK IMPLEMENTATION   #####################################################


###############################
# Global observation
################################

class GlobalObservation:
    """
    Manage the global observation of the AV fleet.

    This class maintains a global state table (pd.DataFrame) that tracks
    the state of all AV agents in the environment. Each row corresponds
    to an agent, and each column represents an observed feature such as:
    start time, origin, destination, chosen route, travel time, completion status.

    Key responsibilities:
        - Maintain agent data in a pd.DataFrame of shape (num_agents, num_features).
        - Keep track of the currently departing agent.
        - Update the state table at each environment timestep using information from the TrafficEnvironment.
        - Provide agents with a view of the global state conditioned on their start time.

    Attributes:
        features (dict):
            Mapping of column names to their default values.
        state_table (pd.DataFrame):
            Global observation table where rows correspond to agent IDs
            and columns correspond to features.
        _transitions (dict):
            Cache of per-agent episode experience. Keys are agent IDs, and values
            store (observation, action, reward) used to populate the replay
            buffer in the DQN after episode completion.
        collect_transitions (bool):
            Whether to cache transition data in _transitions during state updates.
        acting_agent_id (int):
            ID of the agent currently taking an action.
        episode_finished (bool):
            Indicates whether the current episode has finished.
    """



    ##############################
    ### Initialization & reset ###
    def __init__(self, agents: list[BaseAgent], collect_transitions:bool=True)->None:

        
        self.features = {
            'start_time': -1,
            'origin': -1,
            'destination': -1,
            'route': -1,
            'travel_time': -1.0,
            'has_finished': 0,
            'is_known': 0,
            #'is_acting': 0, #<- kept in self.acting_agent_id and appended only when generating agent's observation
        }
        self.state_table = self._initialize_state_table(agents)

        self._transitions = {agent.id : {} for agent in agents} # {agentid: {obs: ..., action: ..., reward: ...}}
        self.collect_transitions = collect_transitions

        self.acting_agent_id = None
        self.episode_finished = False

    def reset(self)->None:
        """
        Reset the global observation:
            - Fill state table (pd.DataFrame of shape (n_machine_agents,n_features)) with default values for each feature column.
            - Clear transition memory.
            - Set self.episode_finished flag to False.
            - Set currently acting agent to None.

        Args:
            None
        Returns:
            None
        """

        # Development note:
        # During training, permute agents with identical start times to avoid ordering bias.
        # Will add minor overhead but improve robustness, especially without fixed demand.

        self.state_table[:] = pd.DataFrame(self.features, index=self.state_table.index)
        self._transitions = {agentid: {} for agentid in self._transitions}
        
        self.acting_agent_id = None
        self.episode_finished = False 
        return

    def _initialize_state_table(self, agents: list[BaseAgent] )->pd.DataFrame:
        """
        Initialize state table.
        Set row indices to agent IDs (integers) sorted by start times.
        Set column names as in self.features.
        Fill with default values for each column.

        Args:
            agents (list[BaseAgent]): list of agents to be included in the global state table.

        Returns:
            pd.DataFrame indexed with agent IDs (integers) filled with default values for each column.
        """

        idx_time_sorted = [agent.id for agent in sorted(agents, key=lambda x: x.start_time)]

        empty_columns = {
            key: [value] * len(idx_time_sorted)
            for key, value in self.features.items()
        }

        return pd.DataFrame(empty_columns, index=idx_time_sorted)


    #################################################################
    ### Registering departing agent, getting agent's observation  ###
    #################################################################
    def register_starting_agent(self, agent: BaseAgent)->None:
        """
        Add agent `start_time`, `origin` and `destination` to the agent row.
        Move `self.acting_agent_id` indicator to registered agent.

        Args:
            agent (BaseAgent): An agent whose departure data is to be logged.
        Returns:
            None
        """

        assert agent.id in self.state_table.index
        assert self.state_table.at[agent.id, 'is_known'] == 0, "Trying to overwrite information for known (already registered) agent"

        # Ensure that agents are sorted by travel time and added in this order (-> start time in prev row is defined and not greater than current)
        idx_iloc, col = self.state_table.index.get_loc(agent.id), 'start_time' 
        assert isinstance(idx_iloc, int)
        assert (idx_iloc == 0) or (self.state_table.iloc[idx_iloc-1][col] != self.features[col] and self.state_table.iloc[idx_iloc-1][col] <= agent.start_time)
        

        # Register agent
        self.state_table.at[agent.id, 'is_known'] = 1
        self.state_table.loc[agent.id, ['start_time', 'origin', 'destination']] = [agent.start_time, agent.origin, agent.destination]
        self.acting_agent_id = agent.id
        return

    def register_starting_agent_action(self, agentid: int, action: int)->None:
        """
        Add action (route identifier) to 'route' column for currently acting agent.
        """
        assert agentid == self.acting_agent_id
        self.state_table.at[agentid, 'route'] = action
        return

    def generate_agent_observation(self, agentid: int) -> np.ndarray:
        """
        Generate an agent-specific snapshot of the global state table at their departure time.

        The observation is constructed as a snapshot of the global state table
        at the timestep when the specified agent departs.
        The snapshot reflects only information available at that time:
            - agents that depart later are not included (no future information),
            - earlier agents may be either completed or still in transit,
            depending on their status at that timestep.

        An additional `is_acting` indicator column is appended to mark the acting agent.

        Args:
            agentid (int):
                Identifier of the agent for whom the observation is generated.

        Returns:
            np.ndarray:
                Array representation of the state table snapshot, preserving the original
                column order, with an additional one-hot `is_acting` column appended at the end.
                This added column has value of 1 in the row corresponding to the specified agent.
        """

        if run_checks:
            assert agentid in self.state_table.index, f"Agent ID {agentid} not in self.state_table.index\nself.state_table.index: {self.state_table.index}"
            assert agentid == self.acting_agent_id

            # Development note:
            # Assumption: rows below current agent are empty.
            # May change if future scheduling logic is introduced.
            # assert _check_future_rows_empty()

        # Get agent's view of state table
        obs = self.state_table.copy()
        obs['is_acting'] = 0
        obs.at[agentid, 'is_acting'] = 1

        return obs.to_numpy()

    def get_flattened_agent_observation(self, agentid: int)->np.ndarray:
        return self.generate_agent_observation(agentid).flatten()



    ############################################
    ### Communication with TrafficEvironment ###
    ############################################
    def update_state_with_recently_finished_machines(self, env: TrafficEnvironment)->None:
        """
        Update the global observation table with changes in the environment since the last environment snapshot.

        Specifically:
            - register travel times for agents that have finished their trips since last snapshot,
            - set the 'has_finished' indicators for those agents.

        Args:
            env (TrafficEnvironment): environment for which observation is registered.
        
        Returns:
            None
        """ 

        # Get travel times for recently finished agents
        driving_agents = self.driving_agents() # <- IDs of agents that were active in the last snapshot of the observation
        finished_agents_times = { # <- Agents that finished after last snapshot - according to update from TrafficEnvironment
            info[kc.AGENT_ID] : info[kc.TRAVEL_TIME]
            for info in env.travel_times_list
            if
                info[kc.AGENT_ID] in driving_agents and
                kc.TRAVEL_TIME in info and 
                info[kc.TRAVEL_TIME] != self.features['travel_time'] # check if assigned travel time is not 'empty value'
        } 

        # Update state table
        self.state_table['travel_time'].update(pd.Series(finished_agents_times))
        self.state_table['has_finished'].update(pd.Series({agent: 1 for agent in finished_agents_times}))

        if self.state_table['has_finished'].all():
            self.episode_finished = True
        return 

    
    ###########################################
    ### Experience (s,a,r) cache management ###
    ###########################################

    def flush_transitions(self)->Iterator[Tuple[np.ndarray, int, float]]:
        """
        Export episode (s,a,r) data from _transitions dict and clear transitions cache.
        """
        if not self.collect_transitions:
            raise RuntimeError("Transition recording was disabled (collect_transitions=False). Set to True to enable recording and export.")
        assert self._transitions

        for transition in self._transitions.values():
            yield transition['observation'], transition['action'], transition['reward']
        
        # Execute when iterator exhausted
        for transition in self._transitions.values():
            transition.clear()


    def cache_transition_observation(self, agentid: int, observation: np.ndarray)->None:
        self._cache_transition_field(agentid, 'observation', observation)

    def cache_transition_action(self, agentid: int, action: int)->None:
        self._cache_transition_field(agentid, 'action', action)

    def cache_transition_reward(self, agentid: int, reward: float)->None:
        self._cache_transition_field(agentid, 'reward', reward)

    def _cache_transition_field(self, agentid: int, field: str, value) -> None:

        if not self.collect_transitions:
            raise RuntimeError("Transition recording is disabled (collect_transitions=False)")

        if field not in ['observation', 'action', 'reward']:
            raise ValueError(f"Invalid transition field name ({field}), must be one of: 'observation', 'action', 'reward' ")
        if field in self._transitions[agentid]:
            raise KeyError(f"{field} for agent {agentid} already exists")

        self._transitions[agentid][field] = value





    #################################
    ####### Auxiliary methods #######

    ### Accessing state table info ###
    @property
    def num_table_columns(self)->int:
        return len(self.features) + 1 # features + 'is_acting' column

    def driving_agents(self) -> pd.Index:
        """
        Get IDs of agents that are marked as 'started' but not 'finished' in the observation table.
        """
        df = self.state_table
        return df.index[df['is_known'] & ~df['has_finished']]

    def get_agent_feature(self, agentid: int, feature: str)->Any:
        return self.state_table.at[agentid, feature]

    def set_agent_feature(self, agentid: int, feature: str, value: Any) -> None:

        current_value = self.state_table.at[agentid, feature]
        default_value = self.features[feature]

        if current_value != default_value:
            raise valueError(f"Feature '{feature}' for agent '{agentid}' is already set (value: {current_value}).")
            
        self.state_table.at[agentid, feature] = value
        return

    def is_empty_cell(self, agentid: int, feature: str)->bool:
        """
        Check whether the table cell contains a default (empty) value.

        Args:
            agentid (int): row index.
            feature (str): column name.

        Returns:
            True if cell equals to the default empty value for that feature.
        """
        return self.state_table.at[agentid, feature] == self.features[feature]

    ### Object state ###
    def disable_transition_collection(self) -> None:
        self.collect_transitions = False
    def enable_transition_collection(self) -> None:
        self.collect_transitions = True

    ### Checking correctness ###
    def _is_column_nondescending(self, colname: str)->bool:
        """
        Check if state table column is sorted in non-descending order.
        Ignore suffix filled with default empty values.
        """

        # Get column and default value; raises error if column not present
        col = self.state_table[colname]
        default_val = self.features[colname]

        if len(col) <= 1:
            return True


        # Verify if empty and nonempty values are not mixed; get non-empty prefix
        isempty_mask = col.eq(default_val)
        if isempty_mask.any():

            first_empty = isempty_mask.idxmax()  # index of first True ( True is argmax in T/F boolean series) occurence

            # Check that all values after first empty are also empty
            clean_suffix = isempty_mask.loc[first_empty:].all()
            if not clean_suffix:
                return False
            
            # Get prefix
            prefix = col.iloc[:first_empty]

        else:
            prefix = col

        is_nondesc = (prefix.diff().iloc[1:] >= 0).all()  # drop NaN for first row
        return is_nondesc




###############################
# DQN Network
################################

### Simplified single-DQN implementation for single-step decision-making
class DQN(BaseLearningModel):
    """
    Deep Q-Network (DQN) implementation.

    Components:
        - Q-network
        - Replay buffer
        - Epsilon-greedy policy (epsilon starting value and decay rule during learning)
        - Training setup: optimizer, loss, batch handling.
    """
    def __init__(self,
                input_dim: int,
                output_dim: int,
                hidden_dims: list[int],

                batch_size=64,
                num_batches= 10,

                min_buffer_size=256,
                max_buffer_size=256, 

                eps_init=0.99,
                eps_decay=0.998,

                lr=0.003, 
                device="cpu"
                ):

        if min_buffer_size < batch_size:
            raise ValueError(f"min_buffer_size ({min_buffer_size}) must be >= batch_size ({batch_size}) to allow sampling a full batch.")
        if min_buffer_size <= 0:
            raise ValueError(f"min_buffer_size ({min_buffer_size}) must be positive.")
        if min_buffer_size > max_buffer_size:
            raise ValueError(f"min_buffer_size ({min_buffer_size}) cannot exceed max_buffer_size ({max_buffer_size}).")


        super().__init__()
        self.device = device

        # Q-network
        self.action_space_size = output_dim
        self.q_network = Network(input_dim, output_dim, hidden_dims).to(self.device)
        

        # Replay buffer
        self.min_buffer_size = min_buffer_size
        self.memory = deque(maxlen=max_buffer_size)

        # Behavior policy
        self.epsilon = eps_init
        self.epsilon_decay = eps_decay

        # Training 
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.batch_size = batch_size
        self.num_batches = num_batches

        self.training_loss_records = [] # record loss info after each `learn()` call. Type is `list[dict]`, one dict per `learn()` call

        self.is_training = True

    def reset(self)->None:
        """ Reset the model."""
        raise NotImplementedError("reset(self) not implemented for DQN class.") # reset buffer, loss logging, epsilon, all training changeable params, (weights and biases?); set training flag to true


    def set_train(self)->None:
        """
        Set the model to training mode.

        Effects:
            - enables gradient updates,
            - enables exploration via epsilon-greedy in `act()`
        """
        self.is_training = True
        self.q_network.train()

    def set_eval(self) -> None:
        """
        Set the model to evaluation mode.

        Effects:
            - set network to .eval() -> norm layers behavior,
            - only greedy policy in `act()` (Q-net argmax action),
            - prevent from running `learn()`.
        """ 
        self.is_training = False
        self.q_network.eval()


    def act(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy.

        Training mode:
            - Random action with probability epsilon
            - Greedy action (argmax Q) with probability (1 - epsilon)
        Evaluation mode:
            - Argmax Q
        """

        if not self.is_training:
            return self._argmax_action(state)

        if np.random.rand() < self.epsilon:
                return np.random.choice(self.action_space_size)

        return self._argmax_action(state)
    
    def push(self, state: np.ndarray, action: int, reward: float) -> None:
        """
        Add (s,a,r) tuple to the buffer.
        """
        if not self.is_training:
            warnings.warn("You are pushing to DQN replay memory in eval mode")

        self.memory.append((state, action, reward))
        return

    def learn(self, loss_logger: Optional[Callable[[dict], None]] = None)->None:
        """
        Update network parameters.

        Conditions:
            - skip learning if memory buffer < min_buffer_size
            - not permitted to run in evaluation mode

        Procedure:
            - sample `num_batches` from memory buffer
            - pass through the network, update parameters
            - decay epsilon after every `num_batches`
        """

        # Prevent learning in evaluation mode
        if not self.is_training:
            raise RuntimeError("Cannot call `learn()` in evaluation mode. Set `self.is_training = True` to update network.")

        # Prevent learning if not enough data in the buffer
        if len(self.memory) < self.min_buffer_size:
            raise ValueError("Insufficient memory to perform learning step") #less restrictive: print(f"Skipping learn: memory size {len(self.memory)} < min_buffer_size {self.min_buffer}")
            return


        batch_losses = []
        for _ in range(self.num_batches):

            # Get batch of states, actions and rewards
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards = zip(*batch)

            states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
            rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)


            # Predict Q-values (travel times) for actions, compare with recorded travel times
            predicted_q_values = self.q_network(states_tensor).gather(1, actions_tensor)
            target_q_values = rewards_tensor

            # Backpropagate & optimize
            loss = self.loss_fn(predicted_q_values, target_q_values)
            batch_losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        # Record loss
        avg_iteration_loss = sum(batch_losses) / len(batch_losses)
        iteration_loss_record = {
            "iteration": len(self.training_loss_records)+1,
            "loss": avg_iteration_loss,
            # "batch_losses": batch_losses #optionally
        }
        self.training_loss_records.append(iteration_loss_record)

        # Stream logging
        if loss_logger is not None:
            loss_logger(iteration_loss_record)

        self.decay_epsilon()


    def decay_epsilon(self)->None:
        self.epsilon *= self.epsilon_decay

    def _argmax_action(self, state: np.ndarray) -> int:
        """
        Return the greedy action (argmax Q-value) for a given state.

        Args:
            state (np.ndarray): state in the network input format.
        Returns:
            int: action with the highest predicted Q-value.
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Ensure shape starts from batch dim: (batch, dim1, ..., dimk)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        action = torch.argmax(q_values).item()
        return action


class Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(Network, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims)-1)])
        self.out_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.out_layer(x)
        return x

##################################################################################################################################



######################################
# Wrapper for running traffic episode
######################################

def run_episode(
    env: TrafficEnvironment,
    dqn: DQN,
    global_observation: GlobalObservation,
    agent_lookup: dict
    )->None:
    """
    Execute a full multi-agent episode in the TrafficEnvironment.

    Responsibilities:
        - Reset environment and global observation state
        - Iterate over agents using env.agent_iter() (runs two loops - in first agents are active and select actions, in second agents are terminated - reward collection)
            
            - Active agents loop:
                - update global observation with environment state change (since last agent departure timestamp)
                - update global observation with current (starting) agent
                - construct observation for current agent
                - select action for current agent via DQN policy
                - Optinally: cache agent observation and action for replay buffer (if enabled)

            - Terminated/truncated agents loop:
                - record agent travel time in global observation (if not available earlier, during env state update)
                - Optionally: cache agent reward for replay buffer (if enabled)

    Side effects:
        - Mutates global_observation
        - Steps environment

    Args:
        env (TrafficEnvironment):
            Multi-agent traffic simulation environment implemented in the Petting-Zoo style API.
            Responsible for managing agent lifecycle, state transitions, and reward signaling.
        dqn (DQN):
            Deep Q-Network object. Encapsulates policy network, experience replay buffer, and training logic.
            Provides methods for action selection (ε-greedy), learning updates, and memory sampling.
        global_observation (GlobalObservation):
            Instance of the GlobalObservation class. Maintains the global state table of all AV agents in the environment.
            Updated throughout the episode to reflect agent departures, selected actions, and recorded travel times.
            Used to construct per-agent state views used as input to DQN network. Enables optional transition caching for DQN training.
        agent_lookup (dict): 
            Mapping of TrafficEnvironment agent identifiers to agent objects. 

    Returns:
        None
    """

    global_observation.reset()
    env.reset()


    # Iterate over agents sorted by travel time
    for agent_id_str in env.agent_iter():
        
        if run_checks:
            assert isinstance(agent_id_str, str) and agent_id_str.isnumeric()

        agent = agent_lookup[agent_id_str]
        agent_id = int(agent_id_str)

        _, reward, termination, truncation, info = env.last() # observation, reward, termination, truncation, info

        if termination or truncation: # All agents finished, collect rewards

            if run_checks:
                assert isinstance(reward, (np.floating, float)) and reward<0, f"Reward: {reward} ({type(reward)}); agent: {agent_id}"

            # Update table with travel times for the last chunk of agents (arriving after the last agent's departure)
            if global_observation.is_empty_cell(agentid=agent_id, feature='travel_time'):
                travel_time = -reward
                global_observation.set_agent_feature(agentid=agent_id, feature='travel_time', value=travel_time)

            # Save reward in transitions cache
            if global_observation.collect_transitions:
                global_observation.cache_transition_reward(agent_id, reward)

            action = None

        else: # Episode in progress, select actions for agents

            # Update global observation with env state change (add travel times for recently finished agents)
            global_observation.update_state_with_recently_finished_machines(env)
            global_observation.register_starting_agent(agent)
            
            # Get observation and select action for current agent
            obs = global_observation.get_flattened_agent_observation(agent_id)
            action = dqn.act(obs) # acts epsilon-greedy or armax(q-values), depends on DQN.is_training flag
            global_observation.register_starting_agent_action(agent_id, action)

            # Save (state, action) pair in transitions cache
            if global_observation.collect_transitions:
                global_observation.cache_transition_observation(agent_id, obs)
                global_observation.cache_transition_action(agent_id, action)

        env.step(action)

    # Mark global observation as finished for episode
    global_observation.episode_finished = True
    
    # Develompent note:
    # Possibly check if state table format corresponds to finished episode
    # e.g. all travel times filled, all is_acting==False, (all is_known==True)








# Main script to run the centralized DQN experiment
logger = logging.getLogger(__name__)
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

    ALGORITHM = "centralized_dqn"
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
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    logging.basicConfig(level=logging.DEBUG)
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
        
    # Parameter setting
    params = dict()
    alg_params = json.load(open(f"../config/algo_config/{ALGORITHM}/{alg_config}.json"))
    env_params = json.load(open(f"../config/env_config/{env_config}.json"))
    task_params = json.load(open(f"../config/task_config/{task_config}.json"))
    params.update(alg_params)
    params.update(env_params)
    params.update(task_params)
    del params["desc"], env_params, task_params

    # Set debug flag to False if not passed
    if "run_checks" not in params:
        params["run_checks"] = False

    # set params as variables in this script
    for key, value in params.items():
        globals()[key] = value


    # Define input / output paths
    custom_network_folder = f"../networks/{network}"
    records_folder = f"../results/{exp_id}"
    plots_folder = f"../results/{exp_id}/plots"

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
    experience_collecting_episodes = 0 if num_machines == 0 else math.ceil(min_buffer_size/num_machines) # num of episodes needed to min fill the buffer
    total_episodes = human_learning_episodes + experience_collecting_episodes + training_episodes + test_eps
    
    # Phases for plotting
    phases = [
        1,
        human_learning_episodes,
        human_learning_episodes + int(experience_collecting_episodes),
        human_learning_episodes + int(experience_collecting_episodes) + int(training_episodes)
    ]
    phase_names = ["Human stabilization", "Mutation and experience collecting", "AV learning", "Testing"]



    # Read origin-destinations
    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    data = ast.literal_eval(content)
    origins = data['origins']
    destinations = data['destinations']

    
        
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
    dump_config["experience_collecting_episodes"] = experience_collecting_episodes # cDQN specific
    dump_config["algorithm"] = ALGORITHM
    dump_config["script"] = script_path_for_config(__file__)

    
    # Renaming 'training_episodes' parameter to match metrics.py assumptions. Previously changed from 'training_eps' to 'training_episodes' 
    # to avoid confusion with epsilon parameters: 'eps_init' and 'eps_decay'. Ideally, this handling would be moved elsewhere
    assert "training_eps" in dump_config or ( ("training_eps" not in dump_config) and ("training_episodes" in dump_config) ) 
    if ("training_eps" not in dump_config) and ("training_episodes" in dump_config):
        dump_config["training_eps"] = dump_config.pop("training_episodes")

    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    
    if run_checks:
        assert update_every_k_episodes >=1 # (mid-episode training not supported)

    # Initialize the environment
    config_machine_obs = params.get("observations")
    if config_machine_obs is not None:
        logger.debug("Ignoring observation_type=%s in this context. Using GLobalObservation instead.", config_machine_obs)

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
                # "observation_type" : # ignoring TrafficEnvironment machine observation. Using GlobalObservation.
            }
        },
        environment_parameters = {
            "save_every" : save_every,
        },
        simulator_parameters = {
            "network_name" : network,
            "custom_network_folder" : custom_network_folder,
            "sumo_type" : "sumo",
            "simulation_timesteps" : 180
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
    
    # Initialize GlobalObservation. Define Q-Net and agent lookup
    global_observation = GlobalObservation(env.machine_agents)
    q_net = DQN(
            input_dim = len(env.machine_agents) * global_observation.num_table_columns, ##
            output_dim = env.environment_params[kc.ACTION_SPACE_SIZE], ##
            hidden_dims = hidden_dims,

            batch_size = batch_size,
            num_batches = num_batches,

            min_buffer_size = min_buffer_size,
            max_buffer_size = max_buffer_size,

            eps_init = eps_init,
            eps_decay = eps_decay,

            lr=lr, 
            device=device
        )
    agent_lookup = {str(agent.id): agent for agent in env.machine_agents}




    ### Collect experience for training (random sampling of actions)
    pbar.set_description("Collecting experience samples")
    os.makedirs(plots_folder, exist_ok=True)

    global_observation.enable_transition_collection()

    for episode in range(experience_collecting_episodes):

        if run_checks:
            print(f"\nCollecting experience episodes: {episode}/{experience_collecting_episodes}")

        assert global_observation.collect_transitions == True

        # --- Run single episode and collect transitions ---
        run_episode(
            env=env,
            dqn=q_net,
            global_observation=global_observation,
            agent_lookup=agent_lookup
        )
        
        # --- Move collected (s, a, r) transitions to DQN replay buffer ---
        for (s,a,r) in global_observation.flush_transitions():
            q_net.push(s,a,r)


        # --- Plot visualization ---
        if episode % plot_every == 0:
            env.plot_results()

        pbar.update()
    
    if run_checks:
        assert len(q_net.memory) >= q_net.min_buffer_size, f"Not enough transitions in the buffer to start training ({len(q_net.memory)}). Required min_buffer_size: {q_net.min_buffer_size}"



    ### Learning phase ###
    pbar.set_description("AV learning")

    q_net.set_train()
    global_observation.enable_transition_collection()

    # Create logger for streaming loss
    loss_logger = CSVLossLogger(path=os.path.join(records_folder, "losses", "losses.csv"), columns=["iteration", "loss"])

    for episode in range(training_episodes):

        if run_checks:
            print(f"\nTraining episodes: {episode}/{training_episodes}")

        assert global_observation.collect_transitions == True

        # --- Run single episode and collect transitions ---
        run_episode(
            env=env,
            dqn=q_net,
            global_observation=global_observation,
            agent_lookup=agent_lookup
        )
        
        # --- Move collected (s, a, r) transitions to DQN replay buffer ---
        for (s,a,r) in global_observation.flush_transitions(): 
            q_net.push(s,a,r)

        # --- Network parameter update ---
        if episode > 0 and episode % update_every_k_episodes == 0:
            q_net.learn(loss_logger) 

        # --- Plot visualization ---
        if episode % plot_every == 0:
            env.plot_results()

        pbar.update()
    loss_logger.close()
    
    
    
    ### Testing phase ###
    pbar.set_description("Testing")

    q_net.set_eval()
    global_observation.disable_transition_collection()


    for episode in range(test_eps):

        if run_checks:
            print(f"\nTest episode: {episode}/{test_eps}")

        run_episode(
            env=env,
            dqn=q_net,
            global_observation=global_observation,
            agent_lookup=agent_lookup
        )

        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()

    
    # Finalize the experiment
    pbar.close()
    env.plot_results()
    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)
    if not args.skip_metrics:
        run_metrics_analysis(os.path.basename(records_folder), results_folder=os.path.dirname(records_folder))
