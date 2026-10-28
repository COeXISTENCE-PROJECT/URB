from __future__ import annotations

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
import random
from typing import Sequence, Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

from routerl.keychain import Keychain as kc

from baseline_models import BaseLearningModel

# Type alias. Hidden state can be one tensor, a tuple of two tensors, or nothing.
# This covers: GRU - one tensor; LSTM - (h, c) tuple (hidden state, cell state); MLP - None
PolicyHiddenState = torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None
TRANSITION_REWARD_MODES = (
    "transition_tt",
    "transition_tt_fft",
    "transition_tt_fft_ad" # ad - arrival, departure
)

@dataclass
class EpisodeRewardAssignment:
    """Weighted AV decision rewards and a human reward for the final decision."""

    decision_rewards: dict[int, float]
    terminal_reward: float

def _detach_hidden_state(hidden: PolicyHiddenState) -> PolicyHiddenState:
    """
    Detach the recurrent hidden state from the current autograd graph before
    storing it for the next action.

    This keeps rollout-time memory as a plain tensor value instead of a live
    gradient chain. Without detaching, PyTorch would keep linking every new
    timestep to the full past history, which would grow memory usage and make
    the next backward pass try to backpropagate through old action-selection
    steps.
    """
    if hidden is None:
        return None
    if isinstance(hidden, tuple):
        return tuple(part.detach() for part in hidden)
    return hidden.detach()

class TripInfoWithETASumoEncoder(nn.Module):
    def __init__(
        self,
        num_paths: int, # pass explicitly instead of relying on simulation params (simulator doesn't generate paths anymore)
        num_edges: int, # in an urban network graph
        edge_attr_dim: int, # number of attributes per edge (e.g. mean speed, number of vehicles)
        origins: list[int],
        destinations: list[int],

        eta_hidden_dim: int = 32,
        od_hidden_dim: int = 16,
        edge_hidden_dim: int = 16, # 2 layers though?
        start_time_hidden_dim: int = 16,

        output_dim: int = 128,
        edge_speed_scale: float = 16.67,

        include_action_mask_in_obs: bool = True,
    ):
        super().__init__()

        self.num_paths = int(num_paths)
        self.num_edges = int(num_edges)
        self.edge_attr_dim = int(edge_attr_dim)
        self.trip_dim = self.num_paths + 3 # length of trip related observations

        self.include_action_mask_in_obs = include_action_mask_in_obs
        if self.include_action_mask_in_obs:
            self.trip_dim += self.num_paths

        self.output_dim = int(output_dim)
        self.edge_speed_scale = float(edge_speed_scale)
        # self.max_start_time = float(max_start_time)
        # self.eta_scale = float(eta_scale)

        # OD embeddings
        self.origin_embedding = nn.Embedding(len(origins), od_hidden_dim)
        self.destination_embedding = nn.Embedding(len(destinations), od_hidden_dim)

        self.eta_encoder = nn.Sequential(
            nn.Linear(self.num_paths, eta_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(eta_hidden_dim)
        )

        # NO mask encoder for now

        # Start time encoder
        self.start_time_encoder = nn.Sequential(
            nn.Linear(1, start_time_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(start_time_hidden_dim)
        )

        # Shared MLP to encode edges independently
        # Take in a vector for each edge and encode it
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.edge_attr_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(edge_hidden_dim)
        )

        self.combined_dim = (
            eta_hidden_dim
            + self.num_edges * edge_hidden_dim
            + start_time_hidden_dim
            + 2 * od_hidden_dim
        )

        if self.include_action_mask_in_obs:
            self.combined_dim += self.num_paths

        self.combined_encoder = nn.Sequential(
            nn.Linear(self.combined_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # The sequence (T) is not passed all at once during rollout, but the hidden RNN state carries it forward:
        # training: full sequence [B, T, D]
        # rollout: one step [1, 1, D] + hidden state
        if obs.dim() == 2:
            obs = obs.unsqueeze(1) # (B,D) case - add a 'fake' T dimension - (B,1,D)
        if obs.dim() != 3:
            raise ValueError(f"Expected obs with shape (B,D) or (B,T,D), got {tuple(obs.shape)}")

        B, T, D = obs.shape
        x = obs.reshape(B*T, D)

        # Parse the trip related observations
        # Round and long in case the ODs are floats (2.0 ~= 1.999... -> long() -> 1, we want 2)
        eta = x[:, :self.num_paths] #/ self.eta_scale
        if self.include_action_mask_in_obs:
            mask_feat = x[:, self.num_paths : 2 * self.num_paths].float()
            offset = 2 * self.num_paths
        else:
            mask_feat = None
            offset = self.num_paths

        origin = x[:, offset].round().long()
        destination = x[:, offset + 1].round().long()
        start_time = x[:, offset + 2 : offset + 3] #/ self.max_start_time

        # Encode the trip related observations
        eta_emb = self.eta_encoder(eta)
        start_time_emb = self.start_time_encoder(start_time)
        origin_emb = self.origin_embedding(origin)
        destination_emb = self.destination_embedding(destination)
        # Encode the action masks too?

        # Encode the edges one by one:
        edges_raw = x[:, offset + 3:]
        expected_edge_dim = self.num_edges * self.edge_attr_dim
        if edges_raw.shape[-1] != expected_edge_dim:
            raise ValueError(
                f"Expected edges_raw dim {expected_edge_dim}, got {edges_raw.shape[-1]}. "
                f"num_edges={self.num_edges}, edge_attr_dim={self.edge_attr_dim}"
            )
        edge_features = edges_raw.reshape(x.shape[0], self.num_edges, self.edge_attr_dim) #(B*T, num_edges=D-6/attr_dim, attr_len=D-6/num_edges), 6=len([eta], origin, destination, start_time)
        edge_features = self._normalize_edge_features(edge_features)
        edge_emb_per_edge = self.edge_encoder(edge_features) # (B*T, num_edges, emb_dim) -> for each self.edge_attr_dim vector, learn a representation
        global_edge_emb = edge_emb_per_edge.reshape(x.shape[0], -1) # (B*T, flat embeddings for each edge)

        pieces = [
            eta_emb,
            origin_emb,
            destination_emb,
            start_time_emb,
            global_edge_emb,
        ]

        # Add the raw action masks
        if self.include_action_mask_in_obs:
            pieces.insert(1, mask_feat)

        z = self.combined_encoder(torch.cat(pieces, dim=-1)) # combine into one tensor (not stack); along the last (-1) dimension - features, not batch/

        z = z.reshape(B, T, self.output_dim)
        return z

    def _normalize_edge_features(self, edge_features: torch.Tensor) -> torch.Tensor:
        edge_features = edge_features.clone()

        if self.edge_attr_dim >= 1:
            edge_features[..., 0] = torch.log1p(torch.clamp(edge_features[..., 0], min=0.0))
        if self.edge_attr_dim >= 2:
            edge_features[..., 1] = torch.clamp(edge_features[..., 1] / self.edge_speed_scale, min=0.0, max=5.0)
        if self.edge_attr_dim >= 3:
            edge_features[..., 2] = torch.clamp(edge_features[..., 2] / 100.0, min=0.0, max=1.0)
        if self.edge_attr_dim >= 4:
            edge_features[..., 3] = torch.log1p(torch.clamp(edge_features[..., 3], min=0.0))

        return edge_features

class TripInfoWithETARouteCongestionEncoder(nn.Module):
    def __init__(
        self,
        num_paths: int, # pass explicitly instead of relying on simulation params (simulator doesn't generate paths anymore)
        route_feature_dim: int, # number of features per route (e.g. mean speed, number of vehicles)
        origins: list[int],
        destinations: list[int],

        eta_hidden_dim: int = 32,
        od_hidden_dim: int = 16,
        edge_hidden_dim: int = 16, # 2 layers though?
        start_time_hidden_dim: int = 16,

        output_dim: int = 128,
        route_speed_scale: float = 16.67,

        include_action_mask_in_obs: bool = True,
        include_eta: bool = True,
    ):
        super().__init__()

        self.num_paths = int(num_paths)
        self.route_feature_dim = int(route_feature_dim)
        self.trip_dim = 3 # O, D, start_time

        self.include_eta = bool(include_eta)
        if self.include_eta:
            self.trip_dim += self.num_paths

        # NO mask encoder for now
        self.include_action_mask_in_obs = include_action_mask_in_obs
        if self.include_action_mask_in_obs:
            self.trip_dim += self.num_paths

        self.output_dim = int(output_dim)
        self.route_speed_scale = float(route_speed_scale)

        # OD embeddings
        self.origin_embedding = nn.Embedding(len(origins), od_hidden_dim)
        self.destination_embedding = nn.Embedding(len(destinations), od_hidden_dim)

        if self.include_eta:
            self.eta_encoder = nn.Sequential(
                nn.Linear(self.num_paths, eta_hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(eta_hidden_dim)
            )

        # Start time encoder
        self.start_time_encoder = nn.Sequential(
            nn.Linear(1, start_time_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(start_time_hidden_dim)
        )

        # Shared MLP to encode route feature vectors independently
        self.route_encoder = nn.Sequential(
            nn.Linear(self.route_feature_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(edge_hidden_dim)
        )

        self.combined_dim = (
            self.num_paths * edge_hidden_dim
            + start_time_hidden_dim
            + 2 * od_hidden_dim
        )

        if self.include_eta:
            self.combined_dim += eta_hidden_dim
        if self.include_action_mask_in_obs:
            self.combined_dim += self.num_paths

        self.combined_encoder = nn.Sequential(
            nn.Linear(self.combined_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # The sequence (T) is not passed all at once during rollout, but the hidden RNN state carries it forward:
        # training: full sequence [B, T, D]
        # rollout: one step [1, 1, D] + hidden state
        if obs.dim() == 2:
            obs = obs.unsqueeze(1) # (B,D) case - add a 'fake' T dimension - (B,1,D)
        if obs.dim() != 3:
            raise ValueError(f"Expected obs with shape (B,D) or (B,T,D), got {tuple(obs.shape)}")

        B, T, D = obs.shape
        x = obs.reshape(B*T, D)

        # Parse the trip related observations
        # Round and long in case the ODs are floats (2.0 ~= 1.999... -> long() -> 1, we want 2)
        offset = 0
        if self.include_eta:
            eta = x[:, offset : offset + self.num_paths]
            offset += self.num_paths
        else:
            eta = None

        if self.include_action_mask_in_obs:
            mask_feat = x[:, offset : offset + self.num_paths].float()
            offset += self.num_paths
        else:
            mask_feat = None

        origin = x[:, offset].round().long()
        destination = x[:, offset + 1].round().long()
        start_time = x[:, offset + 2 : offset + 3]

        # Encode the trip related observations
        start_time_emb = self.start_time_encoder(start_time)
        origin_emb = self.origin_embedding(origin)
        destination_emb = self.destination_embedding(destination)

        # Encode the vectors of route features route by route:
        route_features_raw = x[:, offset + 3:]
        expected_edge_dim = self.num_paths * self.route_feature_dim
        if route_features_raw.shape[-1] != expected_edge_dim:
            raise ValueError(
                f"Expected route_features_raw dim {expected_edge_dim}, got {route_features_raw.shape[-1]}. "
                f"num_paths={self.num_paths}, route_feature_dim={self.route_feature_dim}"
            )
        route_features = route_features_raw.reshape(x.shape[0], self.num_paths, self.route_feature_dim)
        route_features = self._normalize_route_features(route_features)
        if mask_feat is not None:
            route_features = route_features * mask_feat.unsqueeze(-1)
        route_emb_per_feature = self.route_encoder(route_features)
        route_features_emb = route_emb_per_feature.reshape(x.shape[0], -1) # (B*T, flat embeddings for each edge)

        pieces = []
        if self.include_eta:
            pieces.append(self.eta_encoder(eta))
        if self.include_action_mask_in_obs:
            pieces.append(mask_feat)
        pieces.extend([
            origin_emb,
            destination_emb,
            start_time_emb,
            route_features_emb,
        ])

        z = self.combined_encoder(torch.cat(pieces, dim=-1)) # combine into one tensor (not stack); along the last (-1) dimension - features, not batch

        z = z.reshape(B, T, self.output_dim)
        return z

    def _normalize_route_features(self, route_features: torch.Tensor) -> torch.Tensor:
        route_features = route_features.clone()

        # Raw feature order from TripInfoWithETARouteCongestion:
        # vehicle_sum, halting_sum, mean_speed, mean_occupancy, max_occupancy,
        # active_edge_fraction, halted_fraction.
        if self.route_feature_dim >= 1:
            route_features[..., 0] = torch.log1p(torch.clamp(route_features[..., 0], min=0.0))
        if self.route_feature_dim >= 2:
            route_features[..., 1] = torch.log1p(torch.clamp(route_features[..., 1], min=0.0))
        if self.route_feature_dim >= 3:
            route_features[..., 2] = torch.clamp(route_features[..., 2] / self.route_speed_scale, min=0.0, max=5.0)
        if self.route_feature_dim >= 4:
            route_features[..., 3] = torch.clamp(route_features[..., 3] / 100.0, min=0.0, max=1.0)
        if self.route_feature_dim >= 5:
            route_features[..., 4] = torch.clamp(route_features[..., 4] / 100.0, min=0.0, max=1.0)
        if self.route_feature_dim >= 6:
            route_features[..., 5] = torch.clamp(route_features[..., 5], min=0.0, max=1.0)
        if self.route_feature_dim >= 7:
            route_features[..., 6] = torch.clamp(route_features[..., 6], min=0.0, max=1.0)

        return route_features

class ActorCriticBase(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        obs: torch.Tensor,
        h0: PolicyHiddenState = None,
    ) -> tuple[torch.Tensor, torch.Tensor, PolicyHiddenState]:
        """Return action logits, value estimate, and optional next hidden state."""
        raise NotImplementedError

class ActorCriticWithEncoder(ActorCriticBase):
    """
    Wrapper for ActorCriticMLP and ActorCriticRNN.
    Used with the TripInfoWithETASumo observation that includes
    a long, flat vector of per-edge attributes.
    """
    def __init__(self, encoder: nn.Module, core: ActorCriticBase):
        super().__init__()
        self.encoder = encoder
        self.core = core

    def forward(self, obs, h0: PolicyHiddenState = None):
        z = self.encoder(obs)
        return self.core(z, h0)

class ActorCriticMLP(ActorCriticBase):
    """
    An alternative to ActorCriticRNN. Both can be used with PPO.

    Input
    - centralized observation (both actor and critic)

    Output
    - actor: logits for the available routes
    - critic: one scalar estimate of the expected return from the current state

    Design (choose one)
    - two separate networks
    - shared trunk with two heads
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
    ):
        super().__init__()

        assert len(hidden_sizes) != 0

        # Input layer + hidden layers + nonlinearities (ReLU)
        # If the input is (B,T,D), the linear layers are only applied to the last
        # dimension D. It preserves the other dimensions but ignores the temporal aspect.
        # E.g.
        # episode 1, vehicle 1
        # episode 1, vehicle 2
        # episode 1, vehicle 3
        # ...
        # are treated here as independent rows that happened to be organized as (B,T,D)
        self.trunk = nn.Sequential(*[
            layer
            for in_dim, out_dim in zip([obs_dim, *hidden_sizes[:-1]], hidden_sizes)
            for layer in (nn.Linear(in_dim, out_dim), nn.ReLU())
        ])
        self.policy_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, obs: torch.Tensor, h0: PolicyHiddenState = None):
        x = self.trunk(obs)
        return self.policy_head(x), self.value_head(x).squeeze(-1), None

class ActorCriticRNN(ActorCriticBase):
    """
    observation
    -> optional observation encoder outside ActorCriticRNN
    -> feed-forward MLP encoder with ReLU
    -> GRU or LSTM
    -> two linear heads:
        policy_head -> action logits
        value_head  -> scalar value estimate
    MLP + RNN are shared by actor and critic.
    Only the final linear layers are separate.

    obs_seq: [B, T, obs_dim]; B - batch size (number of episodes in a PPO batch); T - number of AV decisions
        in the episode; obs_dim - encoded observation size
    [B, T, obs_dim]
    -> flatten to [B*T, obs_dim]
    -> MLP
    -> [B*T, hidden_sizes[-1]]
    -> reshape back to [B, T, hidden_sizes[-1]]
    out: [B, T, rnn_hidden_dim]
    hn:  final hidden state
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        rnn_hidden_dim: int,
        rnn_type: Literal["gru", "lstm"] = "gru"
    ):
        super().__init__()
        assert len(hidden_sizes) != 0
        # We first encode observations with an MLP (feature extraction),
        # then run a GRU to accumulate information over time (memory / latent state),
        # then branch into:
        # - a policy head (logits over discrete actions),
        # - a value head (V(s_t), used for advantage estimation).
        self.input_layer = nn.Linear(obs_dim, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_sizes[idx], hidden_sizes[idx + 1])
            for idx in range(len(hidden_sizes) - 1)
        )

        # rnn_hidden_dim - width of the recurrent state vector and also the width of the RNN output
        # that later feeds into the policy and value heads
        if rnn_type == "gru":
            self.rnn = nn.GRU(input_size=hidden_sizes[-1], hidden_size=rnn_hidden_dim, batch_first=True)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=hidden_sizes[-1], hidden_size=rnn_hidden_dim, batch_first=True)
        else:
            raise ValueError(f"Incorrect RNN type: {rnn_type}")

        self.policy_head = nn.Linear(rnn_hidden_dim, action_dim)
        self.value_head = nn.Linear(rnn_hidden_dim, 1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the MLP per timestep. We keep it separate so we can easily reuse it
        # in both training (batched sequences) and action selection (one-step sequence).
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return x

    def forward(self, obs_seq: torch.Tensor, h0: PolicyHiddenState = None):
        # obs_seq: [B, T, obs_dim] or [B, obs_dim] for single-step inference.
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(1)

        if obs_seq.dim() != 3:
            raise ValueError(f"Expected obs_seq with 2 or 3 dims, got shape {tuple(obs_seq.shape)}")

        b, t, d = obs_seq.shape
        # Flatten time into the batch so the MLP can process all timesteps in one go.
        x = self._encode(obs_seq.reshape(b * t, d)).reshape(b, t, -1)
        # The GRU output at each timestep is a learned summary of past observations.
        out, hn = self.rnn(x, h0)
        # Policy logits and value estimates for each timestep.
        logits = self.policy_head(out)  # [B, T, A]
        values = self.value_head(out).squeeze(-1)  # [B, T]
        return logits, values, hn

class PPO(BaseLearningModel):
    """
    PPO implementation for the centralized controller.

    This is a standard clipped PPO actor-critic with a pluggable policy/value network:
    - the policy network may be an MLP or an RNN (GRU/LSTM),
    - in case of using an RNN, the network remembers the hidden state within an episode
    (until the central controller routes all vehicles).
    - GAE(λ) for advantage estimation.
    - Clipped surrogate objective with entropy bonus and value loss.

    Shared contract
    - `policy_net(obs, hidden)` returns `(logits, value, next_hidden)`.
    - MLP policies ignore `hidden` and return `next_hidden=None`.
    - RNN policies use `hidden` and return the updated hidden state.

    Compatibility with existing OpenURB scripts:
    - `act(state)` stores last transition context.
    - `push(reward)` finalizes the last stored transition (done=True by default).
    - `learn()` performs PPO updates and clears the on-policy buffer.
    - `policy_net` attribute exists (for `.eval()` in testing phase).
    - `deterministic` flag controls greedy vs sampling actions.

    For transition-based rewards, simply use push_pending(agent_id) instead of push(reward).
    """
    def __init__(
        self,
        policy_net: ActorCriticBase, # pass an already instantiated object; this is actually both an actor and a critic - returns both logits and values
        action_space_size: int,
        device: str = "cpu",
        batch_size: int = 16,
        lr: float = 3e-4,
        num_epochs: int = 4,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
        entropy_coef: float = 0.1,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        buffer_size: int = 2048
    ):
        super().__init__()
        self.device = device
        self.action_space_size = int(action_space_size)
        # PPO is an on-policy algorithm: updates are done using trajectories
        # collected by the *current* policy. `batch_size` here counts episodes.
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        # PPO clipping parameter (Schulman et al., 2017): constrains how much
        # the policy is allowed to change per update.
        self.clip_eps = float(clip_eps)
        # Discount and GAE parameters (GAE = Generalized Advantage Estimation).
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        # Advantage normalization reduces variance and makes PPO updates more stable.
        self.normalize_advantage = bool(normalize_advantage)
        # Entropy bonus encourages exploration; value_coef balances actor vs critic loss.
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        # Gradient clipping helps with occasional large policy gradients (especially with RNNs).
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None

        # MLP or RNN actor-critic network (shared trunk + two heads).
        self.policy_net = policy_net.to(self.device)

        # One optimizer for both policy and value parameters (standard PPO implementation).
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=float(lr))

        self.loss = []
        self.deterministic = False
        self.update_count = 0
        self.optimizer_step_count = 0
        self.last_update_diag = {}

        # Store completed episodes for PPO updates (on-policy buffer).
        # Each entry holds arrays for obs/actions/log_probs/advantages/returns.
        self.memory = deque(maxlen=int(buffer_size))
        self._episode_steps = []
        # Inference-time GRU hidden state (maintained across timesteps within an episode).
        self._inference_hidden: PolicyHiddenState = None

    def reset_episode(self) -> None:
        # Call this once after env.reset(). It clears the GRU hidden state so
        # the policy doesn't "remember" information across episodes.
        self._inference_hidden = None

    def act(self, state: np.ndarray, action_mask: np.ndarray | None = None) -> int:
        # Convert to a 1-step sequence so we can reuse the same GRU forward path as training.
        state_np = np.asarray(state, dtype=np.float32)
        obs_t = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).view(1, 1, -1)
        mask_np = (
            np.ones(self.action_space_size, dtype=np.bool_)
            if action_mask is None
            else np.asarray(action_mask, dtype=np.bool_).copy()
        )
        if mask_np.shape != (self.action_space_size,):
            raise ValueError(
                f"Action mask shape {mask_np.shape} does not match action space size {self.action_space_size}"
            )

        # Forward pass updates the GRU hidden state; we keep it for the next timestep.
        with torch.no_grad():
            logits, values, hn = self.policy_net(obs_t, self._inference_hidden)

            raw_logits_t = logits[0, 0]
            value_t = values[0, 0]

            # Action masks are handled by setting invalid logits to a very negative number.
            mask = torch.as_tensor(mask_np, dtype=torch.bool, device=raw_logits_t.device)
            if not mask.any():
                raise RuntimeError(f"No valid actions. Mask: {mask}")
            logits_t = raw_logits_t.masked_fill(~mask, -1e9)

            # Categorical policy over discrete actions. (For continuous actions you'd use Normal, etc.)
            dist = torch.distributions.Categorical(logits=logits_t)
            if self.deterministic:
                # Greedy action for evaluation (argmax over logits).
                action = int(torch.argmax(logits_t).item())
            else:
                # Sample action for exploration (stochastic policy).
                action = int(dist.sample().item())

            probs_t = dist.probs.detach().cpu().numpy()

            # Store log-prob for PPO's importance sampling ratio.
            log_prob = float(dist.log_prob(torch.tensor(action, device=logits_t.device)).item())

            # Diagnostics to see e.g. whether policy collapse happens or whether it's weak (almost random)
            self.last_policy_diag = {
                "policy_entropy": float(dist.entropy().item()),
                "policy_max_prob": float(probs_t.max()),
                "policy_argmax": int(probs_t.argmax()),
                "policy_chosen_prob": float(probs_t[int(action)]),
            }

        # .no_grad() prevents PyTorch from building a graph during action selection
        # _detach_hidden_state() cuts any graph already attached to the recurrent hidden state
        self._inference_hidden = _detach_hidden_state(hn)

        # Cache the transition context so `push(reward, done)` can write it into the on-policy buffer.
        self.last_state = state_np
        self.last_action = action
        self.last_log_prob = log_prob
        self.last_value = float(value_t.item())
        self.last_mask = mask_np

        return action

    def push(self, reward: float, done: bool = True) -> None:
        # PPO collects trajectories first, then updates. This function records one timestep.
        # It pairs the most recent `act()` call with the observed reward/done flag.
        state = getattr(self, "last_state", None)
        action = getattr(self, "last_action", None)
        log_prob = getattr(self, "last_log_prob", None)
        value = getattr(self, "last_value", None)
        mask = getattr(self, "last_mask", None)
        if state is None or action is None or log_prob is None or value is None:
            raise RuntimeError("push() called before act(); use act() first or implement a push_transition().")

        self._episode_steps.append(
            {
                "obs": np.asarray(state, dtype=np.float32),
                "action": int(action),
                "reward": float(reward),
                "done": bool(done),
                "log_prob": float(log_prob),
                "value": float(value),
                "mask": np.asarray(mask, dtype=np.bool_),
            }
        )
        del self.last_state, self.last_action, self.last_log_prob, self.last_value, self.last_mask

        if done:
            # End of episode: compute advantages/returns and store it for learning.
            episode = self._finalize_episode(self._episode_steps)
            self.memory.append(episode)
            self._episode_steps = []

    def push_pending(self, agent_id: int, done: bool = False, initial_reward: float = 0.0) -> None:
        """
        Store a transition that has already been taken but has not yet received its arrival reward.
        For transition-based rewards we store the transition first and attach the actual travel-time
        reward when the episode finishes. Combined arrival/departure modes can also keep an immediate
        arrival reward in initial_reward.

        General idea reminder:
            - AV 51 acts at controller step 8
            - AV 51's travel time is available at the end of the episode
            - finish_episode() assigns its reward to step 8
        """
        state = getattr(self, "last_state", None)
        action = getattr(self, "last_action", None)
        log_prob = getattr(self, "last_log_prob", None)
        value = getattr(self, "last_value", None)
        mask = getattr(self, "last_mask", None)

        if state is None or action is None or log_prob is None or value is None:
            raise RuntimeError("push_no_reward() called before act(); use act() first or implement a push_transition().")

        self._episode_steps.append(
            {
                "agent_id": int(agent_id),
                "obs": np.asarray(state, dtype=np.float32),
                "action": int(action),
                "reward": float(initial_reward),
                "done": bool(done),
                "log_prob": float(log_prob),
                "value": float(value),
                "mask": np.asarray(mask, dtype=np.bool_),
            }
        )
        del self.last_state, self.last_action, self.last_log_prob, self.last_value, self.last_mask

    def _finalize_episode(self, steps: list[dict]) -> dict:
        """
        Calculate GAE advantages and critic return targets.
        """
        # Convert a list of step dicts into fixed arrays.
        # We keep everything per-timestep so PPO can compute a loss over the whole trajectory.
        obs = np.stack([s["obs"] for s in steps], axis=0).astype(np.float32, copy=False)
        actions = np.asarray([s["action"] for s in steps], dtype=np.int64)
        rewards = np.asarray([s["reward"] for s in steps], dtype=np.float32)
        dones = np.asarray([s["done"] for s in steps], dtype=np.float32)
        old_log_probs = np.asarray([s["log_prob"] for s in steps], dtype=np.float32)
        values = np.asarray([s["value"] for s in steps], dtype=np.float32)
        masks = np.asarray([s["mask"] for s in steps], dtype=np.bool_)

        # Compute GAE(λ) advantages (Schulman et al., 2016).
        # Intuition: advantage estimates "how much better than expected" the taken action was.
        # We assume terminal bootstrap value is 0 (standard for episodic tasks).
        adv = np.zeros_like(rewards, dtype=np.float32)
        last_adv = 0.0
        next_value = 0.0
        for t in range(rewards.shape[0] - 1, -1, -1):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * nonterminal * next_value - values[t]
            last_adv = delta + self.gamma * self.gae_lambda * nonterminal * last_adv
            adv[t] = last_adv
            next_value = values[t]
        # Returns are the value targets for the critic.
        returns = adv + values

        return {
            "obs": obs,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "advantages": adv,
            "returns": returns,
            "T": int(obs.shape[0]),
            "masks": masks
        }

    def finish_episode(
        self,
        rewards_by_agent: dict[int, float],
        missing_reward: float | None = -1.0,
        *,
        terminal_reward: float = 0.0,
    ) -> None:
        """
        Manages the end of a transition-reward episode.

        Assigns final arrival rewards to their original transitions.
        Adds the terminal reward to the final transition. With missing_reward=None,
        requires an exact match between pending decisions and supplied rewards.
        Otherwise, retains the fallback penalty for legacy callers.
        Marks the last transition as terminal.
        Produces episode diagnostics.
        Calls _finalize_episode().
        Appends the resulting episode to PPO memory.
        Clears pending transitions.
        """
        if not self._episode_steps:
            return

        step_agent_ids = {int(step["agent_id"]) for step in self._episode_steps}
        unmatched_arrivals = len(set(rewards_by_agent) - step_agent_ids)
        if missing_reward is None and step_agent_ids != set(rewards_by_agent):
            raise ValueError("Episode rewards must match the pending AV decisions")
        missing_steps = 0
        for step in self._episode_steps:
            agent_id = int(step["agent_id"])
            if agent_id in rewards_by_agent:
                step["reward"] = float(step.get("reward", 0.0)) + float(rewards_by_agent[agent_id])
            else:
                step["reward"] = float(step.get("reward", 0.0)) + float(missing_reward)
                missing_steps += 1

        if missing_steps or unmatched_arrivals:
            print(
                f"[Centralized PPO] finish_episode penalized {missing_steps} pending transitions; "
                f"{unmatched_arrivals} arrivals were unmatched."
            )

        self._episode_steps[-1]["reward"] += float(terminal_reward)
        self._episode_steps[-1]["done"] = True

        self.last_episode_diag = {
            "episode_pending_penalized": int(missing_steps),
            "episode_unmatched_arrivals": int(unmatched_arrivals),
        }

        episode = self._finalize_episode(self._episode_steps)
        self.memory.append(episode)
        self._episode_steps = []

    def learn(self) -> None:
        # Perform PPO updates using the stored on-policy episodes.
        if len(self.memory) < self.batch_size:
            return

        losses = []
        self.update_count += 1
        total_timesteps = int(sum(int(ep["T"]) for ep in self.memory))

        for _ in range(self.num_epochs):
            # PPO often reuses the same data for a few epochs ("multiple passes over data").
            batch = random.sample(self.memory, self.batch_size)
            max_t = max(int(ep["T"]) for ep in batch)

            def pad_time(x, pad_value=0.0):
                t = x.shape[0]
                if t == max_t:
                    return x
                pad_shape = (max_t - t,) + x.shape[1:]
                pad = np.full(pad_shape, pad_value, dtype=x.dtype)
                return np.concatenate([x, pad], axis=0)

            # Pad variable-length episodes so we can batch them.
            obs = torch.as_tensor(np.stack([pad_time(ep["obs"]) for ep in batch]), device=self.device)
            actions = torch.as_tensor(np.stack([pad_time(ep["actions"]) for ep in batch]), device=self.device)
            old_log_probs = torch.as_tensor(np.stack([pad_time(ep["old_log_probs"]) for ep in batch]), device=self.device)
            advantages = torch.as_tensor(np.stack([pad_time(ep["advantages"]) for ep in batch]), device=self.device)
            returns = torch.as_tensor(np.stack([pad_time(ep["returns"]) for ep in batch]), device=self.device)
            masks = torch.as_tensor(np.stack([pad_time(ep["masks"], pad_value=1) for ep in batch]), dtype=torch.bool, device=self.device)

            # Mask out padded timesteps when computing losses.
            lengths = torch.tensor([int(ep["T"]) for ep in batch], device=self.device, dtype=torch.int64)
            time_mask = (
                torch.arange(max_t, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
            ).to(dtype=torch.float32)

            if self.normalize_advantage:
                # Normalize over only real timesteps (ignoring padding).
                flat_adv = advantages[time_mask.bool()]
                advantages = (advantages - flat_adv.mean()) / (flat_adv.std(unbiased=False) + 1e-8)

            # Compute new policy and value predictions for the batch.
            logits, values, _ = self.policy_net(obs)
            # Mask invalid actions using the stored rollout mask.
            logits = logits.masked_fill(~masks, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions.long())
            entropy = dist.entropy()

            # PPO uses an importance sampling ratio between new and old policies:
            # r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Diagnostics
            with torch.no_grad():
                valid = time_mask.bool()
                approx_kl = (old_log_probs[valid] - new_log_probs[valid]).mean()
                clip_frac = ((ratio[valid] - 1.0).abs() > self.clip_eps).float().mean()
                returns_valid = returns[valid]

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            # Clipped objective: takes the pessimistic (min) of unclipped and clipped advantage terms.
            policy_loss = -(torch.min(surr1, surr2) * time_mask).sum() / time_mask.sum().clamp(min=1.0)

            # Critic loss: MSE between predicted V and computed returns.
            value_loss = (((returns - values) ** 2) * time_mask).sum() / time_mask.sum().clamp(min=1.0)
            # Entropy bonus: encourages higher-entropy (more exploratory) policies.
            entropy_bonus = (entropy * time_mask).sum() / time_mask.sum().clamp(min=1.0)

            # Full PPO loss: actor + value + entropy regularization.
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                grad_norm = nn.utils.clip_grad_norm_(
                    self.policy_net.parameters(),
                    max_norm=self.max_grad_norm,
                )
                grad_norm_value = float(grad_norm.item())
            else:
                grad_norm_value = np.nan
            self.optimizer.step()

            losses.append(float(loss.item()))

        # Diagnostics
        self.optimizer_step_count += self.num_epochs

        self.last_update_diag = {
            "ppo_update_count": int(self.update_count),
            "ppo_optimizer_step_count": int(self.optimizer_step_count),
            "ppo_total_timesteps": int(total_timesteps),
            "ppo_loss_mean": float(sum(losses) / len(losses)),
            "ppo_policy_loss": float(policy_loss.item()),
            "ppo_value_loss": float(value_loss.item()),
            "ppo_entropy": float(entropy_bonus.item()),
            "ppo_approx_kl": float(approx_kl.item()),
            "ppo_clip_frac": float(clip_frac.item()),
            "ppo_return_mean": float(returns_valid.mean().item()),
            "ppo_return_std": float(returns_valid.std(unbiased=False).item()),
            "ppo_grad_norm": float(grad_norm_value),
        }

        self.loss.append(float(sum(losses) / len(losses)))
        # On-policy buffer is cleared after an update: old data is "stale" for PPO.
        self.memory.clear()

class CentralizedAVEnvWrapper(gym.Env):
    """
    Wrapper that converts an AECEnv to a single-agent interface, useful for centralized controller settings.
    The central controller is a single agent that controls all the vehicles, departing sequentially.
    Already handles action masking.

    Gymnasium (env.step(action)):
        Takes one action, updates the environment, and returns 5 items: observation, reward, terminated, truncated, and info.
    PettingZoo (AEC API - env.step(action)):
        Takes an action for the current agent only. It advances the simulation to the next agent in the cycle, returning updated observations for that agent.

    Humans are still controlled by the classic environment.
    CentralizedAVEnvWrapper:
        chooses route only when current agent is an AV / machine agent
    RouteRL TrafficEnvironment:
        advances SUMO
        applies human Gawron behavior
        updates human route choices/learning
        handles departures, arrivals, congestion, travel times

    So the general flow looks as follows:
        1. get an observation for the AV that is about to depart next
        2. choose a route for this AV
        3. pass the action to RouteRL
        4. let RouteRL/SUMO simulate humans + general traffic until the next AV departure
        5. go to 1

    Internally, machine agents are still separate entities with their own IDs, observations, rewards etc. But the centralized wrapper reinterprets it
    as a sequential single-agent decision making setting.

    Environments (both this wrapper and the classic TrafficEnvironment):
    RouteRL has vehicles/agents internally, but PettingZoo exposes only the decision-making agents.
    machine agents / AVs - exposed to RL
    human agents - handled internally by TrafficEnvironment
    TrafficEnvironment.step(machine_action) - that action is only for the selected machine agent.
    Inside that same step() / simulation_loop(), RouteRL/SUMO handles the rest (humans, traffic, next machine agent selection).
    """
    metadata = {"render_modes": []}

    def __init__(self, env, action_masks, reward_mode: str = "transition_tt"):
        super().__init__()
        self.env = env
        self.action_masks = action_masks
        self.reward_mode = reward_mode
        self._cost_measure(reward_mode)
        self.free_flow_times = env.get_free_flow_times(invalid_pad=1e9)
        self.agent_mask_map = {}

        if action_masks is None:
            raise RuntimeError("Centralized controller requires action_masks.")
        if not env.machine_agents:
            raise ValueError("Centralized controller requires at least one AV")

        missing_agents = []
        for agent in env.machine_agents:
            mask = action_masks.get((int(agent.origin), int(agent.destination)))
            if mask is None:
                missing_agents.append(f"{agent.id} ({agent.origin} -> {agent.destination})")
                continue
            mask_np = np.asarray(mask, dtype=np.bool_).reshape(-1)
            if mask_np.ndim != 1:
                raise ValueError(f"Malformed mask for agent {agent.id}: {mask_np.shape}")
            if mask_np.shape[0] == 0 or not mask_np.any():
                raise ValueError(f"Agent {agent.id} has no valid actions in its mask")
            self.agent_mask_map[str(agent.id)] = mask_np

        if missing_agents:
            raise ValueError(
                "Missing action masks for machine agents: "
                + ", ".join(missing_agents)
            )

        self.num_actions = len(next(iter(self.agent_mask_map.values())))
        self.num_machines = len(env.machine_agents)
        self.machine_ids = {agent.id for agent in env.machine_agents}
        self.machine_ids_str = {str(agent.id) for agent in env.machine_agents}
        self.num_steps = 0
        self._travel_times_cursor = 0

        first_agent_id = str(env.machine_agents[0].id)
        base_observation_space = env.observation_space(first_agent_id)
        self.obs_shape = base_observation_space.shape
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Dict({
            "observation": base_observation_space,
            "action_mask": spaces.MultiBinary(self.num_actions),
        })

        self.num_humans = len(env.human_agents)
        self._agent_kinds = {
            int(agent.id): kind
            for agents, kind in (
                (env.machine_agents, kc.TYPE_MACHINE),
                (env.human_agents, kc.TYPE_HUMAN),
            )
            for agent in agents
        }
        self._min_fft_by_od = {}
        self._group_factors = None
        self._get_group_factors()
        self._arrival_reward_agent_ids = set()

    def _get_mask_for_od(self, origin, destination):
        mask = self.action_masks.get((int(origin), int(destination)))
        if mask is None:
            raise RuntimeError(f"Missing action mask for OD ({origin}, {destination})")
        mask_np = np.asarray(mask, dtype=np.bool_).reshape(-1)
        if mask_np.shape[0] != self.num_actions:
            raise RuntimeError(
                f"Mask length {mask_np.shape[0]} does not match action space size {self.num_actions}"
            )
        if not mask_np.any():
            raise RuntimeError(f"OD ({origin}, {destination}) has no valid actions")
        return mask_np

    def _get_machine(self, agent_id):
        agent_id = str(agent_id)
        machine = next((m for m in self.env.machine_agents if str(m.id) == agent_id), None)
        if machine is None:
            raise RuntimeError(f"Machine agent {agent_id} not found.")
        return machine

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset(seed=seed, options=options)
        self.num_steps = 0 #?
        self._travel_times_cursor = 0
        self._arrival_reward_agent_ids.clear()

        agent = self.env.agent_selection
        observation = self._build_obs(agent)
        info = {
            "current_agent": agent
        }

        if observation["observation"].shape != self.obs_shape:
            raise RuntimeError(
                f"Observation shape changed from {self.obs_shape} to "
                f"{observation['observation'].shape}"
            )

        return observation, info

    def step(self, action):
        """
        Turns RouteRL's internal state into the standard ```observation, reward, terminated, truncated, info```.
        """
        # 1. Remember which machine agent is currently selected
        routed_agent = self.env.agent_selection

        # 2. Take an action for the current agent in RouteRL
        if int(action) < 0 or int(action) >= self.num_actions:
            raise RuntimeError(f"Invalid action index {action} for action space size {self.num_actions}")
        mask = self.agent_mask_map[str(routed_agent)]
        if not bool(mask[int(action)]):
            raise RuntimeError(
                f"Invalid action selected: agent={routed_agent}, "
                f"action={action}, mask={mask}"
            )
        self.env.step(int(action))
        self.num_steps += 1

        # 3. Check done status
        terminated = all(self.env.terminations.values())
        truncated = all(self.env.truncations.values())
        done = terminated or truncated

        # 4. Compute the reward
        reward = self._compute_reward(done)

        # 5. Build the next observation
        next_agent = None
        if done:
            observation = self._terminal_obs()
        else:
            next_agent = self.env.agent_selection # already chosen by self.env.step()
            observation = self._build_obs(next_agent)

        routed_agent_object = self._get_machine(routed_agent)
        next_agent_object = self._get_machine(next_agent) if next_agent is not None else None
        info = {
            "step_idx": self.num_steps,
            # "routed_agent": routed_agent,
            "agent_id": routed_agent_object.id,
            "origin": routed_agent_object.origin,
            "destination": routed_agent_object.destination,
            "start_time": routed_agent_object.start_time,
            "action": int(action),

            "next_agent_id": None if next_agent_object is None else next_agent_object.id,
            # "num_machines": self.num_machines,
        }

        return observation, reward, terminated, truncated, info

    def get_episode_av_rewards(self, mode: str) -> dict[int, float]:
        """
        Return rewards for AV arrivals from the completed episode.
        """
        records = getattr(self.env, "last_episode_travel_times", [])

        rewards = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            if not self._is_machine_record(record):
                continue

            if kc.AGENT_ID not in record:
                continue

            agent_id = int(record[kc.AGENT_ID])
            reward = self._reward_from_record(record, mode)
            if reward is None:
                continue

            rewards[agent_id] = rewards.get(agent_id, 0.0) + reward

        return rewards

    def _min_free_flow_time(self, origin: int, destination: int) -> float:
        """Cache the minimum valid route FFT for an OD, in minutes."""
        od = (origin, destination)
        if od not in self._min_fft_by_od:
            mask = self._get_mask_for_od(origin, destination)
            free_flow_times = np.asarray(self.free_flow_times[od], dtype=np.float32)
            valid_times = free_flow_times[mask]
            valid_times = valid_times[
                np.isfinite(valid_times) & (valid_times > 0.0) & (valid_times < 1e8)
            ]
            if valid_times.size == 0:
                raise RuntimeError(f"No valid free-flow times for OD {od}")
            self._min_fft_by_od[od] = float(valid_times.min())
        return self._min_fft_by_od[od]

    def _trip_cost(self, record: dict, measure: str) -> float | None:
        """Return positive TT or absolute FFT delay, independently of behavior."""
        travel_time = record.get(kc.TRAVEL_TIME)
        if travel_time is None:
            return None
        travel_time = float(travel_time)
        if measure == "tt":
            return travel_time
        if measure != "fft_delay":
            raise ValueError(f"Unknown cost measure: {measure}")

        required_keys = (kc.AGENT_ORIGIN, kc.AGENT_DESTINATION, kc.ACTION)
        if any(key not in record for key in required_keys):
            return None
        origin = int(record[kc.AGENT_ORIGIN])
        destination = int(record[kc.AGENT_DESTINATION])
        action = int(record[kc.ACTION])
        mask = self._get_mask_for_od(origin, destination)
        if action < 0 or action >= mask.shape[0] or not mask[action]:
            raise RuntimeError(
                f"Recorded action {action} is invalid for OD ({origin}, {destination}) and mask {mask}"
            )
        return max(travel_time - self._min_free_flow_time(origin, destination), 0.0)

    def _get_group_factors(self) -> tuple[float, float]:
        """Resolve shared RouteRL behavior into per-vehicle fleet contributions."""
        if self._group_factors is None:
            coefficients = np.asarray(
                [agent.rewards_coefs for agent in self.env.machine_agents], dtype=float
            )
            if coefficients.shape != (self.num_machines, 4) or not np.isfinite(coefficients).all():
                raise ValueError("Weighted rewards require four finite coefficients per AV")
            if not np.all(coefficients == coefficients[0]):
                raise ValueError("Weighted fleet rewards require the same behavior for all AVs")
            a, b, c, d = coefficients[0]
            num_all = self.num_machines + self.num_humans
            av_factor = (a + b) / self.num_machines + d / num_all
            human_factor = c / self.num_humans + d / num_all if self.num_humans else 0.0
            self._group_factors = (float(av_factor), float(human_factor))
        return self._group_factors

    def _trip_contribution(self, record: dict, measure: str) -> float:
        """Apply signed group weights once to a valid AV or human trip cost."""
        kind = record.get(kc.AGENT_KIND)
        if kind not in (kc.TYPE_MACHINE, kc.TYPE_HUMAN):
            raise ValueError(f"Unknown vehicle kind: {kind}")
        travel_time = record.get(kc.TRAVEL_TIME)
        if travel_time is None or not np.isfinite(float(travel_time)) or float(travel_time) < 0:
            raise ValueError(f"Missing or invalid trip cost for agent {record.get(kc.AGENT_ID)}")
        cost = self._trip_cost(record, measure)
        if cost is None or not np.isfinite(cost) or cost < 0:
            raise ValueError(f"Missing or invalid trip cost for agent {record.get(kc.AGENT_ID)}")
        av_factor, human_factor = self._get_group_factors()
        return cost * (av_factor if kind == kc.TYPE_MACHINE else human_factor)

    @staticmethod
    def _cost_measure(mode: str) -> str:
        if mode in ("total_tt", "arrival_tt", "transition_tt"):
            return "tt"
        if mode in ("transition_tt_fft", "transition_tt_fft_ad"):
            return "fft_delay"
        raise ValueError(f"Unknown reward mode: {mode}")

    @classmethod
    def _transition_cost_measure(cls, mode: str) -> str:
        if mode not in TRANSITION_REWARD_MODES:
            raise ValueError(f"Expected a transition reward mode, got: {mode}")
        return cls._cost_measure(mode)

    def _trip_contributions(self, records, measure: str, *, require_complete=False) -> dict[int, float]:
        """Validate records and use fixed-population weights for any delivery mode."""
        contributions = {}
        for record in records:
            if not isinstance(record, dict) or kc.AGENT_ID not in record:
                raise ValueError("Every trip record must contain an agent ID")
            agent_id = int(record[kc.AGENT_ID])
            if agent_id in contributions:
                raise ValueError(f"Duplicate trip record for agent {agent_id}")
            if agent_id not in self._agent_kinds or record.get(kc.AGENT_KIND) != self._agent_kinds[agent_id]:
                raise ValueError(f"Unknown agent or inconsistent vehicle kind for agent {agent_id}")
            contributions[agent_id] = self._trip_contribution(record, measure)
        if require_complete:
            missing = self._agent_kinds.keys() - contributions.keys()
            if missing:
                raise ValueError(f"Missing trip records for agents: {sorted(missing)}")
        return contributions

    def get_episode_reward_assignment(self, mode: str | None = None) -> EpisodeRewardAssignment:
        """Allocate AV costs to decisions and human costs to the final decision.

        Every vehicle needs a valid record (including RouteRL teleport penalties).
        The combined mode splits the objective equally between this allocation
        and arrival rewards, so it does not count the objective twice.
        """
        mode = self.reward_mode if mode is None else mode
        measure = self._transition_cost_measure(mode)
        contributions = self._trip_contributions(
            self.env.last_episode_travel_times, measure, require_complete=True
        )
        share = 0.5 if mode == "transition_tt_fft_ad" else 1.0
        decision_rewards = {
            agent_id: share * value
            for agent_id, value in contributions.items()
            if self._agent_kinds[agent_id] == kc.TYPE_MACHINE
        }
        terminal_reward = share * sum(
            value for agent_id, value in contributions.items()
            if self._agent_kinds[agent_id] == kc.TYPE_HUMAN
        )
        return EpisodeRewardAssignment(decision_rewards, terminal_reward)

    def _reward_from_record(self, record: dict, mode: str) -> float | None:
        """Preserve the legacy unweighted, negative per-AV reward interface."""
        if record.get(kc.TRAVEL_TIME) is None:
            return None
        cost = self._trip_cost(record, self._transition_cost_measure(mode))
        return None if cost is None else -cost

    def close(self):
        self.env.stop_simulation()

    def _is_machine_record(self, record) -> bool:
        agent_id = record.get(kc.AGENT_ID)
        agent_kind = record.get(kc.AGENT_KIND)
        return (
            agent_kind == kc.TYPE_MACHINE
            or agent_id in self.machine_ids
            or str(agent_id) in self.machine_ids_str
        )

    def _compute_reward(self, done):
        """Deliver weighted fleet contributions at arrivals or episode end.

        Transition modes attach their rewards after the episode. The combined
        mode delivers half at arrivals and half through the episode assignment.
        """
        measure = self._cost_measure(self.reward_mode)
        if self.reward_mode in TRANSITION_REWARD_MODES and self.reward_mode != "transition_tt_fft_ad":
            return 0.0
        if self.reward_mode == "total_tt":
            if not done:
                return 0.0
            contributions = self._trip_contributions(
                self.env.last_episode_travel_times, measure, require_complete=True
            )
            return float(sum(contributions.values()))

        # RouteRL clears the live list on the final step. Validate the full
        # snapshot, then pay only arrivals not already rewarded earlier.
        records = (
            self.env.last_episode_travel_times if done
            else self.env.travel_times_list[self._travel_times_cursor:]
        )
        contributions = self._trip_contributions(records, measure, require_complete=done)
        if not done and self._arrival_reward_agent_ids.intersection(contributions):
            raise ValueError("Duplicate trip record across arrival batches")
        reward = sum(
            value for agent_id, value in contributions.items()
            if agent_id not in self._arrival_reward_agent_ids
        )
        self._arrival_reward_agent_ids.update(contributions)
        if not done:
            self._travel_times_cursor = len(self.env.travel_times_list)
        share = 0.5 if self.reward_mode == "transition_tt_fft_ad" else 1.0
        return float(share * reward)

    def _build_obs(self, agent):
        base_obs = self.env.observe(agent)
        if isinstance(base_obs, dict):
            base_obs = base_obs["observation"]
        base_obs = np.asarray(base_obs, dtype=np.float32).reshape(-1)

        mask = self.agent_mask_map[str(agent)].astype(np.int8, copy=False)

        return {
            "observation": base_obs,
            "action_mask": mask
        }

    def _terminal_obs(self):
        return {
            "observation": np.zeros(self.obs_shape, dtype=np.float32),
            "action_mask": np.ones(self.num_actions, dtype=np.int8)
        }
