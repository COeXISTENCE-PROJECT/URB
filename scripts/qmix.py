"""
QMIX (Value Decomposition Networks) implementation.

This follows the standard QMIX setup:
- Per-agent action-value functions `Q_i(o_i, a_i)` (here: GRU-based to support partial observability).
- A monotonic mixing network `Q_tot(s, Q_1..Q_n)` trained with a TD loss.
- Target networks (agent + mixer) for stable TD targets.
- Optional Double Q-learning for target action selection.
-
- Exploration uses a Boltzmann (softmax) policy over per-agent Q-values, controlled by a
  temperature parameter (higher = more exploration, lower = more greedy).

Practical notes:
- Single-step tasks are handled as length-1 sequences (no special-casing).
- Multi-step tasks require calling `reset_episode()` at episode start so GRU state is cleared.

Recurrent state (`reset_episode`)
- `act(...)` maintains an internal GRU hidden state for *inference-time* action selection.
- Call `reset_episode()` once after every environment `reset()` (i.e., at the beginning of each episode/trajectory).
- For OpenURB scripts (single-step per agent/day), this means calling `reset_episode()` once per day; the GRU is
  applied for a single step so it effectively behaves like a feedforward policy.
- For multi-step environments, this means calling `reset_episode()` once per episode; the GRU then
  carries information across steps within the episode.

API notes (matches existing OpenURB scripts):
- `act(obs, action_mask=None, agent_index=...)`
- `store_episode(...)` for single-step episodes (OpenURB default)
- `store_transition(..., done=...)` for multi-step rollouts
- `learn()`, `set_eval_mode()`, `set_train_mode()`
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
import random
import copy  
from collections import deque
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from routerl         import TrafficEnvironment
from tqdm            import tqdm

from baseline_models import BaseLearningModel
from utils           import add_model_snapshot_argument
from utils           import clear_SUMO_files
from utils           import model_snapshot_path
from utils           import print_agent_counts
from utils           import run_metrics_analysis
from utils           import save_loss_records
from utils           import script_path_for_config
from utils           import should_save_model_snapshot


# ===========================================================================
# QMIX (inchangé)
# ===========================================================================

def _build_mlp(in_dim: int, hidden_sizes: Sequence[int], out_dim: int) -> nn.Sequential:
    layers = []
    last_dim = int(in_dim)
    for hidden_dim in hidden_sizes:
        hidden_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, int(out_dim)))
    return nn.Sequential(*layers)


class AgentRNN(nn.Module):
    def __init__(self, obs_dim, action_dim, rnn_hidden_dim, num_hidden, widths):
        super().__init__()
        assert len(widths) == (num_hidden + 1), "QMIX widths and number of layers mismatch!"
        self.input_layer = nn.Linear(obs_dim, widths[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(widths[idx], widths[idx + 1]) for idx in range(num_hidden)
        )
        self.rnn = nn.GRU(input_size=widths[-1], hidden_size=rnn_hidden_dim, batch_first=True)
        self.out = nn.Linear(rnn_hidden_dim, action_dim)

    def _encode(self, x):
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return x

    def forward(self, obs_seq, h0=None):
        b, t, d = obs_seq.shape
        x = self._encode(obs_seq.reshape(b * t, d)).reshape(b, t, -1)
        out, hn = self.rnn(x, h0)
        q = self.out(out)
        return q, hn


class MixingNetwork(nn.Module):
    def __init__(self, num_agents, state_dim, mixing_embed_dim, hypernet_hidden_sizes, weight_clip=None):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim
        self.weight_clip = float(weight_clip) if weight_clip is not None else None
        hidden_sizes = tuple(int(x) for x in hypernet_hidden_sizes)
        self.hyper_w1 = _build_mlp(state_dim, hidden_sizes, num_agents * mixing_embed_dim)
        self.hyper_b1 = _build_mlp(state_dim, hidden_sizes, mixing_embed_dim)
        self.hyper_w2 = _build_mlp(state_dim, hidden_sizes, mixing_embed_dim)
        self.hyper_b2 = _build_mlp(state_dim, hidden_sizes, 1)

    def forward(self, agent_qs, states):
        batch_size = agent_qs.shape[0]
        w1 = F.softplus(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        if self.weight_clip is not None:
            w1 = torch.clamp(w1, max=self.weight_clip)
        w1 = w1.view(batch_size, self.num_agents, self.mixing_embed_dim)
        b1 = b1.view(batch_size, 1, self.mixing_embed_dim)
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = F.elu(hidden)
        w2 = F.softplus(self.hyper_w2(states))
        if self.weight_clip is not None:
            w2 = torch.clamp(w2, max=self.weight_clip)
        w2 = w2.view(batch_size, self.mixing_embed_dim, 1)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(batch_size)


class QMIX(BaseLearningModel):
    def __init__(
        self,
        state_size: int,
        action_space_size: int,
        num_agents: int,
        global_state_size: int,
        device: str = "cpu",
        temp_init: float = 1.0,
        temp_decay: float = 0.999,
        temp_min: float = 0.05,
        buffer_size: int = 2048,
        batch_size: int = 32,
        lr: float = 3e-4,
        num_epochs: int = 1,
        num_hidden: int = 2,
        widths: Sequence[int] = (128, 128, 128),
        rnn_hidden_dim: int = 64,
        mixing_embed_dim: int = 32,
        hypernet_embed: int = 64,
        mixing_num_hidden: Optional[int] = None,
        mixing_widths: Optional[Sequence[int]] = None,
        max_grad_norm: float = 10.0,
        gamma: float = 0.99,
        target_update_every: int = 200,
        double_q: bool = True,
        tau: float = 1.0,
        share_parameters: bool = True,
        mixing_weight_clip: Optional[float] = None,
        q_tot_clip: Optional[float] = None,
        use_huber_loss: bool = True,
    ):
        super().__init__()
        self.device = device
        self.obs_size = int(state_size)
        self.action_space_size = int(action_space_size)
        self.num_agents = int(num_agents)
        self.global_state_size = int(global_state_size)
        self.share_parameters = bool(share_parameters)
        self.temperature = float(temp_init)
        self.temp_decay = float(temp_decay)
        self.temp_min = float(temp_min)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.max_grad_norm = float(max_grad_norm)
        self.gamma = float(gamma)
        self.target_update_every = max(1, int(target_update_every))
        self.double_q = bool(double_q)
        self.tau = float(tau)
        self.q_tot_clip = float(q_tot_clip) if q_tot_clip is not None else None
        self.use_huber_loss = bool(use_huber_loss)
        self._learn_steps = 0

        if mixing_widths is not None:
            mixing_widths = tuple(int(x) for x in mixing_widths)
            if mixing_num_hidden is not None and len(mixing_widths) != int(mixing_num_hidden):
                raise AssertionError("QMIX mixing_widths and mixing_num_hidden mismatch!")
            hypernet_hidden_sizes = mixing_widths
        else:
            hypernet_hidden_sizes = (int(hypernet_embed),)

        if self.share_parameters:
            self.agent_net = AgentRNN(self.obs_size, self.action_space_size,
                                     rnn_hidden_dim, num_hidden, widths).to(self.device)
            self.target_agent_net = copy.deepcopy(self.agent_net).to(self.device)
            self.agent_nets = None
            self.target_agent_nets = None
        else:
            self.agent_net = None
            self.target_agent_net = None
            self.agent_nets = nn.ModuleList([
                AgentRNN(self.obs_size, self.action_space_size,
                         rnn_hidden_dim, num_hidden, widths).to(self.device)
                for _ in range(self.num_agents)
            ])
            self.target_agent_nets = copy.deepcopy(self.agent_nets).to(self.device)

        self.mixing_net = MixingNetwork(
            self.num_agents, self.global_state_size, mixing_embed_dim,
            hypernet_hidden_sizes, weight_clip=mixing_weight_clip,
        ).to(self.device)
        self.target_mixing_net = copy.deepcopy(self.mixing_net).to(self.device)

        self.target_mixing_net.eval()
        if self.share_parameters:
            self.target_agent_net.eval()
        else:
            self.target_agent_nets.eval()

        self.optimizer = optim.Adam(
            list(self._agent_parameters()) + list(self.mixing_net.parameters()), lr=lr,
        )
        self.loss = []
        self.memory = deque(maxlen=int(buffer_size))
        self._episode_steps = []
        self._inference_hidden = {}

    def _agent_parameters(self):
        return self.agent_net.parameters() if self.share_parameters else self.agent_nets.parameters()

    def reset_episode(self):
        self._inference_hidden = {}

    def _get_h0(self, agent_index, hidden_dim):
        h = self._inference_hidden.get(agent_index)
        if h is None:
            h = torch.zeros(1, 1, hidden_dim, device=self.device)
        return h

    def _set_h(self, agent_index, h):
        self._inference_hidden[agent_index] = h.detach()

    def _random_action(self, action_mask):
        if action_mask is None:
            return int(np.random.randint(self.action_space_size))
        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size == 0:
            return int(np.random.randint(self.action_space_size))
        return int(np.random.choice(valid_actions))

    def act(self, state, action_mask=None, agent_index=None):
        if agent_index is None:
            agent_index = 0
        obs_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, 1, -1)
        if self.share_parameters:
            hidden_dim = self.agent_net.rnn.hidden_size
            h0 = self._get_h0(agent_index, hidden_dim)
            q_seq, hn = self.agent_net(obs_t, h0)
        else:
            if agent_index < 0 or agent_index >= self.num_agents:
                raise ValueError(f"agent_index {agent_index} is out of range for {self.num_agents} agents.")
            net = self.agent_nets[agent_index]
            hidden_dim = net.rnn.hidden_size
            h0 = self._get_h0(agent_index, hidden_dim)
            q_seq, hn = net(obs_t, h0)
        self._set_h(agent_index, hn)
        q_values = q_seq.squeeze(0).squeeze(0)
        if action_mask is not None:
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=q_values.device)
            if mask.shape[0] == q_values.shape[0] and mask.any():
                q_values = q_values.masked_fill(~mask, -1e9)
            elif not mask.any():
                return self._random_action(action_mask)
        return self._boltzmann_action(q_values)

    def _boltzmann_action(self, q_values):
        temp = float(self.temperature)
        if temp <= 0.0:
            return int(torch.argmax(q_values).item())
        logits = q_values / temp
        logits = logits - torch.max(logits)
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    def store_transition(self, observations, actions, rewards, active_mask, global_state,
                         next_observations, next_active_mask, next_global_state, done,
                         action_masks=None, next_action_masks=None):
        self._episode_steps.append({
            "obs": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int64),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "active_mask": np.asarray(active_mask, dtype=np.float32),
            "state": np.asarray(global_state, dtype=np.float32),
            "next_obs": np.asarray(next_observations, dtype=np.float32),
            "next_active_mask": np.asarray(next_active_mask, dtype=np.float32),
            "next_state": np.asarray(next_global_state, dtype=np.float32),
            "done": bool(done),
            "action_masks": None if action_masks is None else np.asarray(action_masks, dtype=np.int8),
            "next_action_masks": None if next_action_masks is None else np.asarray(next_action_masks, dtype=np.int8),
        })
        if done:
            episode = self._finalize_episode(self._episode_steps)
            self.memory.append(episode)
            self._episode_steps = []

    def store_episode(self, observations, actions, rewards, active_mask, global_state):
        zeros_obs = np.zeros_like(np.asarray(observations, dtype=np.float32))
        zeros_state = np.zeros_like(np.asarray(global_state, dtype=np.float32))
        self.store_transition(
            observations=observations, actions=actions, rewards=rewards,
            active_mask=active_mask, global_state=global_state,
            next_observations=zeros_obs, next_active_mask=active_mask,
            next_global_state=zeros_state, done=True,
        )

    def _finalize_episode(self, steps):
        obs = np.stack([s["obs"] for s in steps], axis=0)
        actions = np.stack([s["actions"] for s in steps], axis=0)
        rewards = np.stack([s["rewards"] for s in steps], axis=0)
        active_mask = np.stack([s["active_mask"] for s in steps], axis=0)
        state = np.stack([s["state"] for s in steps], axis=0)
        next_obs = np.stack([s["next_obs"] for s in steps], axis=0)
        next_active_mask = np.stack([s["next_active_mask"] for s in steps], axis=0)
        next_state = np.stack([s["next_state"] for s in steps], axis=0)
        done = np.asarray([s["done"] for s in steps], dtype=np.float32)
        action_masks = None
        next_action_masks = None
        if all(s.get("action_masks") is not None for s in steps):
            action_masks = np.stack([s["action_masks"] for s in steps], axis=0)
        if all(s.get("next_action_masks") is not None for s in steps):
            next_action_masks = np.stack([s["next_action_masks"] for s in steps], axis=0)
        return {"obs": obs, "actions": actions, "rewards": rewards, "active_mask": active_mask,
                "state": state, "next_obs": next_obs, "next_active_mask": next_active_mask,
                "next_state": next_state, "done": done, "action_masks": action_masks,
                "next_action_masks": next_action_masks, "T": obs.shape[0]}

    def _masked_argmax(self, q_values, action_masks):
        if action_masks is None:
            return torch.argmax(q_values, dim=-1)
        mask = action_masks.to(dtype=torch.bool, device=q_values.device)
        if mask.shape != q_values.shape:
            return torch.argmax(q_values, dim=-1)
        masked = q_values.masked_fill(~mask, -1e9)
        no_valid = ~mask.any(dim=-1)
        argmax = torch.argmax(masked, dim=-1)
        return torch.where(no_valid, torch.zeros_like(argmax), argmax)

    def _update_targets(self):
        if self.tau >= 1.0:
            if self.share_parameters:
                self.target_agent_net.load_state_dict(self.agent_net.state_dict())
            else:
                self.target_agent_nets.load_state_dict(self.agent_nets.state_dict())
            self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
            return
        with torch.no_grad():
            if self.share_parameters:
                for t, o in zip(self.target_agent_net.parameters(), self.agent_net.parameters()):
                    t.data.mul_(1.0 - self.tau).add_(self.tau * o.data)
            else:
                for t, o in zip(self.target_agent_nets.parameters(), self.agent_nets.parameters()):
                    t.data.mul_(1.0 - self.tau).add_(self.tau * o.data)
            for t, o in zip(self.target_mixing_net.parameters(), self.mixing_net.parameters()):
                t.data.mul_(1.0 - self.tau).add_(self.tau * o.data)

    def _agent_q_values(self, obs, net, share):
        b, t, n, d = obs.shape
        if share:
            obs_bn = obs.permute(0, 2, 1, 3).reshape(b * n, t, d)
            q_bn, _ = net(obs_bn, None)
            return q_bn.reshape(b, n, t, -1).permute(0, 2, 1, 3)
        qs = []
        for idx in range(n):
            q_i, _ = net[idx](obs[:, :, idx, :], None)
            qs.append(q_i.unsqueeze(2))
        return torch.cat(qs, dim=2)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        step_losses = []
        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            max_t = max(int(ep["T"]) for ep in batch)

            def pad_time(x, pad_value=0.0):
                t = x.shape[0]
                if t == max_t:
                    return x
                pad_shape = (max_t - t,) + x.shape[1:]
                pad = np.full(pad_shape, pad_value, dtype=x.dtype)
                return np.concatenate([x, pad], axis=0)

            obs = torch.tensor(np.stack([pad_time(ep["obs"]) for ep in batch]), device=self.device)
            actions = torch.tensor(np.stack([pad_time(ep["actions"]) for ep in batch]), device=self.device)
            rewards = torch.tensor(np.stack([pad_time(ep["rewards"]) for ep in batch]), device=self.device)
            active_mask = torch.tensor(np.stack([pad_time(ep["active_mask"]) for ep in batch]), device=self.device)
            states = torch.tensor(np.stack([pad_time(ep["state"]) for ep in batch]), device=self.device)
            next_obs = torch.tensor(np.stack([pad_time(ep["next_obs"]) for ep in batch]), device=self.device)
            next_active_mask = torch.tensor(np.stack([pad_time(ep["next_active_mask"]) for ep in batch]), device=self.device)
            next_states = torch.tensor(np.stack([pad_time(ep["next_state"]) for ep in batch]), device=self.device)
            dones = torch.tensor(np.stack([pad_time(ep["done"].reshape(-1, 1)) for ep in batch]),
                                 device=self.device, dtype=torch.float32).squeeze(-1)
            lengths = torch.tensor([int(ep["T"]) for ep in batch], device=self.device, dtype=torch.int64)
            time_mask = (torch.arange(max_t, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)).to(dtype=torch.float32)
            action_masks = None
            next_action_masks = None
            if all(ep.get("action_masks") is not None for ep in batch):
                action_masks = torch.tensor(np.stack([pad_time(ep["action_masks"]) for ep in batch]), device=self.device)
            if all(ep.get("next_action_masks") is not None for ep in batch):
                next_action_masks = torch.tensor(np.stack([pad_time(ep["next_action_masks"]) for ep in batch]), device=self.device)

            if self.share_parameters:
                q_all = self._agent_q_values(obs, self.agent_net, share=True)
                next_q_online = self._agent_q_values(next_obs, self.agent_net, share=True)
                with torch.no_grad():
                    next_q_target = self._agent_q_values(next_obs, self.target_agent_net, share=True)
            else:
                q_all = self._agent_q_values(obs, self.agent_nets, share=False)
                next_q_online = self._agent_q_values(next_obs, self.agent_nets, share=False)
                with torch.no_grad():
                    next_q_target = self._agent_q_values(next_obs, self.target_agent_nets, share=False)

            safe_actions = actions.clone()
            safe_actions[active_mask == 0] = 0
            chosen_q = torch.gather(q_all, 3, safe_actions.unsqueeze(-1)).squeeze(-1)
            chosen_q = chosen_q * active_mask
            b, t, n = chosen_q.shape
            q_tot = self.mixing_net(chosen_q.reshape(b * t, n), states.reshape(b * t, -1)).reshape(b, t)
            if self.q_tot_clip is not None:
                q_tot = torch.clamp(q_tot, -self.q_tot_clip, self.q_tot_clip)
            active_counts = active_mask.sum(dim=2).clamp(min=1.0)
            team_rewards = (rewards * active_mask).sum(dim=2) / active_counts

            with torch.no_grad():
                if self.double_q:
                    next_actions = self._masked_argmax(next_q_online, next_action_masks)
                else:
                    next_actions = self._masked_argmax(next_q_target, next_action_masks)
                safe_next_actions = next_actions.clone()
                safe_next_actions[next_active_mask == 0] = 0
                next_chosen_q = torch.gather(next_q_target, 3, safe_next_actions.unsqueeze(-1)).squeeze(-1)
                next_chosen_q = next_chosen_q * next_active_mask
                q_tot_next = self.target_mixing_net(
                    next_chosen_q.reshape(b * t, n), next_states.reshape(b * t, -1)
                ).reshape(b, t)
                if self.q_tot_clip is not None:
                    q_tot_next = torch.clamp(q_tot_next, -self.q_tot_clip, self.q_tot_clip)
                targets = team_rewards + (1.0 - dones) * self.gamma * q_tot_next
                if self.q_tot_clip is not None:
                    targets = torch.clamp(targets, -self.q_tot_clip, self.q_tot_clip)

            if self.use_huber_loss:
                td = F.smooth_l1_loss(q_tot, targets, reduction="none")
            else:
                td = F.mse_loss(q_tot, targets, reduction="none")
            loss = (td * time_mask).sum() / time_mask.sum().clamp(min=1.0)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self._agent_parameters()) + list(self.mixing_net.parameters()),
                max_norm=self.max_grad_norm,
            )
            self.optimizer.step()
            self._learn_steps += 1
            if self._learn_steps % self.target_update_every == 0:
                self._update_targets()
            step_losses.append(float(loss.item()))
        self.loss.append(float(sum(step_losses) / len(step_losses)))
        self.decay_temperature()

    def decay_temperature(self):
        self.temperature = max(self.temp_min, self.temperature * self.temp_decay)

    def set_eval_mode(self):
        if self.share_parameters:
            self.agent_net.eval()
            self.target_agent_net.eval()
        else:
            self.agent_nets.eval()
            self.target_agent_nets.eval()
        self.mixing_net.eval()
        self.target_mixing_net.eval()

    def set_train_mode(self):
        if self.share_parameters:
            self.agent_net.train()
        else:
            self.agent_nets.train()
        self.mixing_net.train()


# ===========================================================================
# Runner (pattern ippo.py / reinforce_tabular.py)
# ===========================================================================

def run_episode(env, model, agent_index, learn):
    n = len(agent_index)
    obs = np.zeros((n, model.obs_size), dtype=np.float32)
    actions = np.zeros(n, dtype=np.int64)
    rewards = np.zeros(n, dtype=np.float32)
    acted = np.zeros(n, dtype=bool)
    rewarded = np.zeros(n, dtype=bool)

    model.reset_episode()

    for agent_id in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        i = agent_index[agent_id]

        if termination or truncation:
            if not acted[i]:
                raise RuntimeError(f"agent {agent_id} terminated without acting")
            rewards[i] = float(reward)
            rewarded[i] = True
            env.step(None)
        else:
            if acted[i]:
                raise RuntimeError(f"agent {agent_id} acted more than once in an episode")
            if isinstance(observation, dict):
                o = np.asarray(observation["observation"], dtype=np.float32)
                mask = observation.get("action_mask")
            else:
                o = np.asarray(observation, dtype=np.float32)
                mask = None
            actions[i] = model.act(o, mask, agent_index=i)
            obs[i] = o
            acted[i] = True
            env.step(int(actions[i]))

    if not (acted.all() and rewarded.all()):
        raise RuntimeError(
            f"{int((~acted).sum())} agents did not act, "
            f"{int((~rewarded).sum())} received no reward"
        )

    if learn:
        active_mask = np.ones(n, dtype=np.float32)
        global_state = obs.reshape(-1)  # concat
        model.store_episode(obs, actions, rewards, active_mask, global_state)
        model.learn()

    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--env-conf', type=str, default="config1")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--torch-seed', type=int, default=42)
    parser.add_argument('--skip-metrics', action='store_true', default=False)
    add_model_snapshot_argument(parser)
    args = parser.parse_args()

    ALGORITHM = "qmix"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    torch_seed = args.torch_seed
    save_model_every = args.save_model_every

    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {ALGORITHM.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Algorithm seed: {torch_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")
    print(f"Metrics will {'NOT ' if args.skip_metrics else ''}be computed after the experiment.\n")

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(env_seed)
    np.random.seed(env_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    params = dict()
    alg_params = json.load(open(f"../config/algo_config/{ALGORITHM}/{alg_config}.json"))
    env_params = json.load(open(f"../config/env_config/{env_config}.json"))
    task_params = json.load(open(f"../config/task_config/{task_config}.json"))
    params.update(alg_params)
    params.update(env_params)
    params.update(task_params)
    del params["desc"], env_params, task_params

    for key, value in params.items():
        globals()[key] = value

    custom_network_folder = f"../networks/{network}"
    phases = [1, human_learning_episodes, int(training_eps) + human_learning_episodes]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]
    base_results_dir = os.environ.get("RESULTS_BASE_DIR", "../results")
    records_folder = os.path.join(base_results_dir, exp_id)
    plots_folder = os.path.join(records_folder, "plots")

    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, 'r', encoding='utf-8') as f:
        data = ast.literal_eval(f.read())
    origins = data['origins']
    destinations = data['destinations']

    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    num_agents = len(pd.read_csv(agents_csv_path))
    os.makedirs(records_folder, exist_ok=True)
    with open(agents_csv_path, 'r', encoding='utf-8') as f:
        content = f.read()
    with open(os.path.join(records_folder, "agents.csv"), 'w', encoding='utf-8') as f:
        f.write(content)
    max_start_time = pd.read_csv(agents_csv_path)['start_time'].max()

    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + test_eps

    dump_config = params.copy()
    if args.project is not None:
        dump_config["project"] = args.project
    dump_config.update(network=network, env_seed=env_seed, torch_seed=torch_seed,
                       env_config=env_config, task_config=task_config, alg_config=alg_config,
                       script=script_path_for_config(__file__), algorithm=ALGORITHM,
                       num_agents=num_agents, num_machines=num_machines)
    if save_model_every is not None:
        dump_config["save_model_every"] = save_model_every
    with open(os.path.join(records_folder, "exp_config.json"), 'w', encoding='utf-8') as f:
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
                "observation_type": observations,
            },
        },
        environment_parameters={"save_every": save_every},
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

    env.mutation(disable_human_learning=not should_humans_adapt, mutation_start_percentile=-1)
    print_agent_counts(env)

    obs_size = env.observation_space(env.possible_agents[0]).shape[0]
    machines = list(env.machine_agents)
    agent_index = {str(a.id): i for i, a in enumerate(machines)}
    n_machines = len(machines)

    model = QMIX(
        state_size=obs_size,
        action_space_size=machines[0].action_space_size,
        num_agents=n_machines,
        global_state_size=n_machines * obs_size,   # concat
        device=str(device),
        temp_init=params["temp_init"],
        temp_decay=params["temp_decay"],
        temp_min=params["temp_min"],
        buffer_size=params["buffer_size"],
        batch_size=params["batch_size"],
        lr=params["lr"],
        num_epochs=params["num_epochs"],
        num_hidden=params["num_hidden"],
        widths=tuple(params["widths"]),
        rnn_hidden_dim=params["rnn_hidden_dim"],
        mixing_embed_dim=params["mixing_embed_dim"],
        hypernet_embed=params["hypernet_embed"],
        max_grad_norm=params["max_grad_norm"],
        gamma=params["gamma"],
        target_update_every=params["target_update_every"],
        double_q=params["double_q"],
        tau=params["tau"],
        share_parameters=params["share_parameters"],
        mixing_weight_clip=params.get("mixing_weight_clip"),
        q_tot_clip=params.get("q_tot_clip"),
        use_huber_loss=params["use_huber_loss"],
    )

    pbar.set_description("AV learning")
    os.makedirs(plots_folder, exist_ok=True)
    for episode in range(training_eps):
        env.reset()
        run_episode(env, model, agent_index, learn=True)

        completed_episode = episode + 1
        if should_save_model_snapshot(completed_episode, training_eps, save_model_every):
            payload = {"training_episode": completed_episode, "mixing": model.mixing_net.state_dict()}
            if model.share_parameters:
                payload["agent"] = model.agent_net.state_dict()
            else:
                payload["agents"] = model.agent_nets.state_dict()
            torch.save(payload, model_snapshot_path(records_folder, completed_episode, "pt"))

        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()

    model.set_eval_mode()
    model.temperature = 0.0
    pbar.set_description("Testing")
    for _ in range(test_eps):
        env.reset()
        run_episode(env, model, agent_index, learn=False)
        pbar.update()

    pbar.close()
    env.plot_results()
    save_loss_records(
        records_folder,
        ({"iteration": i, "loss": v} for i, v in enumerate(model.loss, start=1)),
        columns=["iteration", "loss"],
    )

    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"),
                     os.path.join(records_folder, "episodes"),
                     remove_additional_files=True)
    if not args.skip_metrics:
        run_metrics_analysis(exp_id, results_folder=base_results_dir)