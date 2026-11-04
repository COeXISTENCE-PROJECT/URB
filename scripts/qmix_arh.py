"""
QMIX for auto-routed humans (ARH) on URB — PyTorch port.

Functionally equivalent to the NumPy version, with the backward pass
handled by torch.autograd instead of manual gradients. Same architecture,
same features (one-hot / embedding / no identity), same training loop.

Differences from the NumPy version (numerical only, not semantic):
  - Weight init: torch's default Glorot uniform (vs _xavier). Same
    distribution, different RNG stream.
  - Optimizer: torch.optim.Adam (same math, may differ in last bits).
  - Numerics in float32 (vs float64 in NumPy), which is the torch default.
  - The manual gradient tests in test_qmix.py become moot (autograd handles
    it); no other test change needed.

Design (unchanged):
  - Agent network: shared MLP. Input identity is either:
      * one-hot agent ID (use_agent_id=True),
      * learned embedding from static per-agent features (use_agent_embedding=True),
      * nothing (both False → agents with identical observations act identically).
  - Mixer: 4 two-layer hypernetworks conditioned on concat(obs_1..obs_N).
  - Monotonicity: guaranteed because w1 and w2 are |.| of hypernet outputs
    and the ELU inside the mixer has strictly positive derivative everywhere.
  - CTDE: decentralized argmax at execution time, centralized training.
  - No bootstrap / target network: valid because each AV takes exactly one
    decision per episode.
"""

import xml.etree.ElementTree as ET
from types import MethodType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RunningNorm:
    """Per-feature running z-score. Kept in NumPy (small overhead, no need for torch)."""
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim)
        self.m2 = np.zeros(dim)

    def update(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        k = x.shape[0]
        delta = x.mean(axis=0) - self.mean
        total = self.n + k
        self.mean = self.mean + delta * k / total
        self.m2 = self.m2 + x.var(axis=0) * k + delta ** 2 * self.n * k / total
        self.n = total

    def __call__(self, x):
        if self.n < 2:
            return x
        std = np.sqrt(self.m2 / self.n)
        std = np.where(std < 1e-6, 1.0, std)
        if isinstance(x, torch.Tensor):
            mean_t = torch.from_numpy(self.mean).to(x.device, x.dtype)
            std_t = torch.from_numpy(std).to(x.device, x.dtype)
            return torch.clamp((x - mean_t) / std_t, -10.0, 10.0)
        return np.clip((x - self.mean) / std, -10.0, 10.0)


class AgentEmbedding(nn.Module):
    """Static features → learned embedding (2-layer MLP with ReLU)."""
    def __init__(self, feat_dim, hidden, out_dim):
        super().__init__()
        self.w1 = nn.Linear(feat_dim, hidden)
        self.w2 = nn.Linear(hidden, out_dim)

    def forward(self, feats):
        return self.w2(F.relu(self.w1(feats)))


class AgentNet(nn.Module):
    """Shared MLP: agent input → Q-values over actions."""
    def __init__(self, input_dim, hidden_sizes, n_actions):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MixerNet(nn.Module):
    """4 two-layer hypernetworks producing monotone mixing weights."""
    def __init__(self, state_dim, n_agents, embed, hyper_hidden=64):
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.embed = embed

        # 4 hypernetworks (Linear → ReLU → Linear).
        self.h_w1a = nn.Linear(state_dim, hyper_hidden)
        self.h_w1b = nn.Linear(hyper_hidden, n_agents * embed)
        self.h_b1a = nn.Linear(state_dim, hyper_hidden)
        self.h_b1b = nn.Linear(hyper_hidden, embed)
        self.h_w2a = nn.Linear(state_dim, hyper_hidden)
        self.h_w2b = nn.Linear(hyper_hidden, embed)
        self.h_b2a = nn.Linear(state_dim, hyper_hidden)
        self.h_b2b = nn.Linear(hyper_hidden, 1)

        # Non-zero init on the *weight* hypernets' second-layer biases.
        # |.| has zero gradient at 0, so zero biases would kill the mixer.
        bound = 1.0 / np.sqrt(state_dim)
        nn.init.uniform_(self.h_w1b.bias, -bound, bound)
        nn.init.uniform_(self.h_w2b.bias, -bound, bound)

    def forward(self, q, s):
        # q: (B, N), s: (B, state_dim) -> (B,)
        B = q.shape[0]
        w1 = torch.abs(self.h_w1b(F.relu(self.h_w1a(s)))).view(B, self.n_agents, self.embed)
        b1 = self.h_b1b(F.relu(self.h_b1a(s))).view(B, 1, self.embed)
        pre_h = q.unsqueeze(1) @ w1 + b1
        h = F.elu(pre_h)
        w2 = torch.abs(self.h_w2b(F.relu(self.h_w2a(s)))).view(B, self.embed, 1)
        b2 = self.h_b2b(F.relu(self.h_b2a(s))).view(B, 1, 1)
        q_tot = h @ w2 + b2
        return q_tot.view(B)


# ---------------------------------------------------------------------------
# QMIX model
# ---------------------------------------------------------------------------

class QMIX:
    def __init__(self, obs_dim, n_actions, n_agents, hidden=(64, 64), embed=32,
                 state_mode="concat", team_reward="mean", loss="huber",
                 use_agent_id=True, use_agent_embedding=False,
                 agent_features=None, agent_embed_dim=8, agent_embed_hidden=32,
                 normalize_obs=True, lr=3e-4, batch_size=32,
                 buffer_size=2048, max_grad_norm=10.0, eps_init=1.0,
                 eps_decay=0.995, eps_min=0.05, updates_per_episode=1,
                 hyper_hidden=64, seed=0, device="cpu"):
        if state_mode not in ("concat", "mean"):
            raise ValueError("state_mode must be 'concat' or 'mean'")
        if team_reward not in ("mean", "sum"):
            raise ValueError("team_reward must be 'mean' or 'sum'")
        if loss not in ("huber", "mse"):
            raise ValueError("loss must be 'huber' or 'mse'")
        if use_agent_id and use_agent_embedding:
            raise ValueError(
                "use_agent_id and use_agent_embedding are mutually exclusive"
            )

        self.device = torch.device(device)
        self.obs_dim, self.n_actions, self.n_agents = obs_dim, n_actions, n_agents
        self.hidden = tuple(hidden)
        self.embed = embed
        self.hyper_hidden = hyper_hidden
        self.state_mode = state_mode
        self.team_reward = team_reward
        self.huber = loss == "huber"
        self.use_agent_id = bool(use_agent_id)
        self.use_agent_embedding = bool(use_agent_embedding)
        self.agent_embed_dim = int(agent_embed_dim)
        self.agent_embed_hidden = int(agent_embed_hidden)

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.max_grad_norm = max_grad_norm
        self.updates_per_episode = updates_per_episode
        self.eps = eps_init
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.rng = np.random.default_rng(seed)

        # Static per-agent features (only used with use_agent_embedding).
        if self.use_agent_embedding:
            if agent_features is None:
                raise ValueError("agent_features required when use_agent_embedding=True")
            af = np.asarray(agent_features, dtype=np.float32)
            if af.ndim != 2 or af.shape[0] != n_agents:
                raise ValueError(
                    f"agent_features must be (N={n_agents}, D_feat), got {af.shape}"
                )
            self.agent_features = torch.from_numpy(af).to(self.device)
            self.agent_feat_dim = af.shape[1]
        else:
            self.agent_features = None
            self.agent_feat_dim = 0

        # State dim seen by the mixer.
        self.state_dim = n_agents * obs_dim if state_mode == "concat" else obs_dim

        # Agent input dim.
        if self.use_agent_embedding:
            agent_in = obs_dim + self.agent_embed_dim
        elif self.use_agent_id:
            agent_in = obs_dim + n_agents
        else:
            agent_in = obs_dim

        # Modules.
        self.agent_net = AgentNet(agent_in, hidden, n_actions).to(self.device)
        self.mixer_net = MixerNet(self.state_dim, n_agents, embed, hyper_hidden).to(self.device)
        if self.use_agent_embedding:
            self.agent_embedding = AgentEmbedding(
                self.agent_feat_dim, self.agent_embed_hidden, self.agent_embed_dim
            ).to(self.device)
        else:
            self.agent_embedding = None

        self.optimizer = torch.optim.Adam(self._parameters(), lr=lr)
        self.norm = RunningNorm(obs_dim) if normalize_obs else None

        # Cache inference-time embeddings (refreshed after every learn()).
        if self.use_agent_embedding:
            with torch.no_grad():
                self._cached_embeddings = self.agent_embedding(self.agent_features)
        else:
            self._cached_embeddings = None

        n_params = sum(p.numel() for p in self._parameters())
        if n_params > 5_000_000:
            print(f"[QMIX-torch] {n_params:,} parameters. If this is too slow, "
                  f"consider state_mode='mean' or smaller embed.")

        self.buf_obs, self.buf_act, self.buf_rew = [], [], []
        self.loss = []

    # ─── helpers ───

    def _parameters(self):
        params = list(self.agent_net.parameters()) + list(self.mixer_net.parameters())
        if self.agent_embedding is not None:
            params += list(self.agent_embedding.parameters())
        return params

    def _normalize(self, obs):
        return self.norm(obs) if self.norm is not None else obs

    def _state(self, obs):
        # obs: torch tensor (B, N, d)
        if self.state_mode == "concat":
            return obs.reshape(obs.shape[0], -1)
        return obs.mean(dim=1)

    def _agent_q_all(self, obs):
        # obs: (B, N, d) -> (B, N, A)
        B, N, d = obs.shape
        x = obs
        if self.use_agent_embedding:
            E = self.agent_embedding(self.agent_features)             # (N, k)
            E_batch = E.unsqueeze(0).expand(B, -1, -1)                # (B, N, k)
            x = torch.cat([x, E_batch], dim=-1)                       # (B, N, d+k)
        elif self.use_agent_id:
            oh = torch.eye(N, device=obs.device).unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([x, oh], dim=-1)                            # (B, N, d+N)
        return self.agent_net(x.reshape(B * N, -1)).reshape(B, N, -1)

    def _refresh_cache(self):
        if self.use_agent_embedding:
            with torch.no_grad():
                self._cached_embeddings = self.agent_embedding(self.agent_features)

    # ─── public API ───

    def act(self, obs, agent_idx, mask=None, greedy=False):
        o = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, d)
        o = self._normalize(o)
        if self.use_agent_embedding:
            e = self._cached_embeddings[agent_idx:agent_idx + 1]      # (1, k)
            o = torch.cat([o, e], dim=1)
        elif self.use_agent_id:
            oh = torch.zeros(1, self.n_agents, device=self.device)
            oh[0, agent_idx] = 1.0
            o = torch.cat([o, oh], dim=1)

        with torch.no_grad():
            q = self.agent_net(o)[0]                                  # (A,)

        valid = (np.arange(self.n_actions) if mask is None
                 else np.flatnonzero(np.asarray(mask, dtype=bool)))
        if valid.size == 0:
            raise ValueError("empty action mask")
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.choice(valid))
        return int(valid[q[valid].argmax().item()])

    def store_episode(self, obs, actions, rewards):
        obs = np.asarray(obs, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        self.buf_obs.append(obs)
        self.buf_act.append(np.asarray(actions, dtype=np.int64))
        self.buf_rew.append(float(rewards.mean() if self.team_reward == "mean"
                                   else rewards.sum()))
        if self.norm is not None:
            self.norm.update(obs)
        if len(self.buf_obs) > self.buffer_size:
            del self.buf_obs[0], self.buf_act[0], self.buf_rew[0]

    def learn(self):
        n = len(self.buf_rew)
        if n < 2:
            return None
        self.agent_net.train()
        self.mixer_net.train()
        if self.agent_embedding is not None:
            self.agent_embedding.train()

        last = None
        for _ in range(self.updates_per_episode):
            idx = self.rng.integers(0, n, size=min(self.batch_size, n))
            O = torch.tensor(np.stack([self.buf_obs[i] for i in idx]),
                             dtype=torch.float32, device=self.device)   # (B, N, d)
            A = torch.tensor(np.stack([self.buf_act[i] for i in idx]),
                             dtype=torch.long, device=self.device)      # (B, N)
            y = torch.tensor([self.buf_rew[i] for i in idx],
                             dtype=torch.float32, device=self.device)   # (B,)

            O_norm = self._normalize(O)
            state = self._state(O_norm)                                 # (B, state_dim)

            q_all = self._agent_q_all(O_norm)                           # (B, N, A)
            q_chosen = q_all.gather(2, A.unsqueeze(-1)).squeeze(-1)     # (B, N)
            q_tot = self.mixer_net(q_chosen, state)                     # (B,)

            if self.huber:
                loss = F.smooth_l1_loss(q_tot, y)
            else:
                loss = F.mse_loss(q_tot, y)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._parameters(), self.max_grad_norm)
            self.optimizer.step()

            self.loss.append(float(loss.item()))
            last = float(loss.item())

        # Refresh cached embeddings after all updates.
        self._refresh_cache()
        return last

    def decay_epsilon(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def state_dict(self):
        def _cpu(sd):
            return {k: v.detach().cpu() for k, v in sd.items()}
        out = {
            "agent_net": _cpu(self.agent_net.state_dict()),
            "mixer_net": _cpu(self.mixer_net.state_dict()),
            "eps": float(self.eps),
        }
        if self.agent_embedding is not None:
            out["agent_embedding"] = _cpu(self.agent_embedding.state_dict())
        if self.norm is not None:
            out["norm"] = {
                "n": self.norm.n,
                "mean": self.norm.mean.copy(),
                "m2": self.norm.m2.copy(),
            }
        return out


# ---------------------------------------------------------------------------
# Episode loop (unchanged)
# ---------------------------------------------------------------------------

def split_observation(observation):
    if isinstance(observation, dict):
        return (np.asarray(observation["observation"], dtype=np.float64),
                observation.get("action_mask"))
    return np.asarray(observation, dtype=np.float64), None


def run_episode(env, model, agent_index, learn):
    n = len(agent_index)
    obs = np.zeros((n, model.obs_dim))
    actions = np.zeros(n, dtype=np.int64)
    rewards = np.zeros(n)
    acted = np.zeros(n, dtype=bool)
    rewarded = np.zeros(n, dtype=bool)

    for agent_id in env.agent_iter():
        observation, reward, termination, truncation, _ = env.last()
        i = agent_index[agent_id]
        if termination or truncation:
            if not acted[i]:
                raise RuntimeError(f"agent {agent_id} terminated without acting")
            rewards[i] = reward
            rewarded[i] = True
            env.step(None)
        else:
            if acted[i]:
                raise RuntimeError(f"agent {agent_id} acted more than once in an episode")
            o, mask = split_observation(observation)
            actions[i] = model.act(o, i, mask, greedy=not learn)
            obs[i] = o
            acted[i] = True
            env.step(int(actions[i]))

    if not (acted.all() and rewarded.all()):
        raise RuntimeError(
            f"{int((~acted).sum())} agents did not act, "
            f"{int((~rewarded).sum())} received no reward")
    if learn:
        model.store_episode(obs, actions, rewards)
        model.learn()
        model.decay_epsilon()
    return rewards


# ---------------------------------------------------------------------------
# Static per-agent features
# ---------------------------------------------------------------------------

def build_agent_features_matrix(machines, agents_df, catalogs, features,
                                num_od_pairs, n_destinations, normalize=True):
    """Build (N_machines, D_feat) static features for AVs.

    Supported: "od" (one-hot of OD pair), "start_time", "free_flow_time".
    Continuous features are z-scored; one-hot is left as-is.
    """
    N = len(machines)
    blocks = []
    continuous_offsets = []
    offset = 0

    agents_lookup = agents_df.set_index("id")[["origin", "destination", "start_time"]]

    if "od" in features:
        od = np.zeros((N, num_od_pairs), dtype=np.float32)
        for i, machine in enumerate(machines):
            row = agents_lookup.loc[int(machine.id)]
            o, d = int(row["origin"]), int(row["destination"])
            od[i, o * n_destinations + d] = 1.0
        blocks.append(od)
        offset += num_od_pairs

    if "start_time" in features:
        st = np.zeros((N, 1), dtype=np.float32)
        for i, machine in enumerate(machines):
            st[i, 0] = float(agents_lookup.loc[int(machine.id)]["start_time"])
        blocks.append(st)
        continuous_offsets.append((offset, offset + 1))
        offset += 1

    if "free_flow_time" in features:
        fft = np.zeros((N, 1), dtype=np.float32)
        for i, machine in enumerate(machines):
            cat = catalogs.get(int(machine.id), [])
            if cat:
                fft[i, 0] = float(cat[0].get("free_flow_time", 0.0))
        blocks.append(fft)
        continuous_offsets.append((offset, offset + 1))
        offset += 1

    if not blocks:
        raise ValueError(f"No valid features in {features}")

    feats = np.concatenate(blocks, axis=1).astype(np.float32)

    if normalize and continuous_offsets:
        cols = np.concatenate([np.arange(a, b) for a, b in continuous_offsets])
        mean = feats[:, cols].mean(axis=0)
        std = feats[:, cols].std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        feats[:, cols] = (feats[:, cols] - mean) / std

    return feats, offset


# ---------------------------------------------------------------------------
# Auto-routed humans (same logic as iql_arh / numpy variant)
# ---------------------------------------------------------------------------

def load_human_route_catalogs(path, agent_ids):
    import pandas as pd

    routes_df = pd.read_csv(path)
    required = {"agent_id", "route_id", "probability", "edges"}
    missing_columns = required.difference(routes_df.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing_columns))}")

    has_fft = "free_flow_time" in routes_df.columns

    catalogs = {}
    for agent_id, rows in routes_df.groupby("agent_id", sort=False):
        agent_id = int(agent_id)
        rows = rows.sort_values("route_id").reset_index(drop=True)
        route_ids = [int(r) for r in rows["route_id"]]
        if route_ids != list(range(len(route_ids))):
            raise ValueError(f"Route IDs are not contiguous for agent {agent_id}")

        probabilities = rows["probability"].to_numpy(dtype=float, copy=True)
        if not np.isfinite(probabilities).all() or (probabilities <= 0).any():
            raise ValueError(f"Invalid route probabilities for agent {agent_id}")
        probabilities /= probabilities.sum()

        ffts = rows["free_flow_time"].to_numpy(dtype=float) if has_fft else np.zeros(len(rows))

        catalogs[agent_id] = []
        for route_id, probability, edges, fft in zip(route_ids, probabilities, rows["edges"], ffts):
            edge_list = tuple(str(edges).split())
            if not edge_list:
                raise ValueError(f"Empty route for agent {agent_id}")
            catalogs[agent_id].append(
                {"route_id": route_id, "probability": probability,
                 "edges": edge_list, "free_flow_time": float(fft)}
            )

    expected_ids = {int(a) for a in agent_ids}
    actual_ids = set(catalogs)
    if expected_ids != actual_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            "demand_routes.csv does not match agents.csv "
            f"(missing IDs: {missing[:10]}, extra IDs: {extra[:10]})")
    return catalogs


def install_auto_routed_humans(env, catalogs, background_ids):
    from routerl.keychain import Keychain as kc

    regular_add_vehicle = env.simulator.add_vehicle
    regular_step = env.simulator.step
    regular_record = env._record
    env.simulator.auto_routed_human_routes = {}
    probabilities = {
        agent_id: [float(route["probability"]) for route in routes]
        for agent_id, routes in catalogs.items()
    }

    def add_vehicle(simulator, action):
        if action[kc.AGENT_KIND] != kc.TYPE_HUMAN:
            return regular_add_vehicle(action)

        agent_id = int(action[kc.AGENT_ID])
        selected = simulator.auto_routed_human_routes.get(agent_id)
        if selected is None:
            raise RuntimeError(f"No imported route was selected for human {agent_id}")

        sumo_route_id = f"arh_{agent_id}_{selected['route_id']}"
        simulator.sumo_connection.route.add(sumo_route_id, list(selected["edges"]))
        simulator.sumo_connection.vehicle.add(
            vehID=str(agent_id),
            routeID=sumo_route_id,
            depart=str(action[kc.AGENT_START_TIME]),
            typeID=kc.TYPE_HUMAN,
        )
        simulator.waiting_vehicles[str(agent_id)] = 0
        action["_arh_route_id"] = selected["route_id"]

    def record(environment, episode, episode_data, agents, detectors):
        for entry in episode_data:
            if entry.get(kc.AGENT_KIND) == kc.TYPE_HUMAN:
                entry[kc.ACTION] = entry.pop("_arh_route_id")
        for agent in agents:
            if agent.kind == kc.TYPE_HUMAN:
                agent.model.cost = probabilities[int(agent.id)]
        return regular_record(episode, episode_data, agents, detectors)

    def step(simulator):
        timestep, stopped, arrivals, teleported = regular_step()
        arrivals = [v for v in arrivals if v not in background_ids]
        teleported = [v for v in teleported if v not in background_ids]
        return timestep, stopped, arrivals, teleported

    env.simulator.add_vehicle = MethodType(add_vehicle, env.simulator)
    env.simulator.step = MethodType(step, env.simulator)
    env._record = MethodType(record, env)


def sample_human_routes(env, catalogs, action_masks, env_seed, episode):
    selected_routes = {}
    for human in env.human_agents:
        options = catalogs[int(human.id)]
        probabilities = [option["probability"] for option in options]
        rng = np.random.default_rng(
            np.random.SeedSequence([int(env_seed), int(episode), int(human.id)]))
        selected_routes[int(human.id)] = options[
            int(rng.choice(len(options), p=probabilities))]
        mask = action_masks[(int(human.origin), int(human.destination))]
        human.default_action = int(np.flatnonzero(mask)[0])

    env.simulator.auto_routed_human_routes = selected_routes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    import ast
    import json
    import logging
    import os
    import pickle
    import random
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    sys.path.insert(0, os.path.abspath(os.path.join(here, "..")))

    import pandas as pd
    from routerl import TrafficEnvironment
    from tqdm import tqdm
    from utils import (add_model_snapshot_argument, clear_SUMO_files, model_snapshot_path,
                       print_agent_counts, run_metrics_analysis, save_loss_records,
                       script_path_for_config, should_save_model_snapshot)
    from clustered_routes import (AVMaskWrapper, ClusteredRoutesLoader, resolve_route_set,
                                  validate_clustered_route_set)

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--env-conf", type=str, default="config1")
    parser.add_argument("--task-conf", type=str, required=True)
    parser.add_argument("--alg-conf", type=str, required=True)
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--torch-seed", type=int, default=42,
                        help="Algorithm seed (legacy name; seeds np.random and torch).")
    parser.add_argument("--route-set", type=str, default=None)
    parser.add_argument("--skip-metrics", action="store_true", default=False)
    add_model_snapshot_argument(parser)
    args = parser.parse_args()

    algorithm = "qmix"
    network = args.net
    route_set = resolve_route_set(network, args.route_set)
    save_model_every = args.save_model_every

    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {algorithm.upper()} (auto-routed humans, torch)")
    print(f"Experiment ID: {args.id}")
    print(f"Network: {network}")
    print(f"Environment seed: {args.env_seed}")
    print(f"Algorithm seed: {args.torch_seed}")
    print(f"Algorithm config: {args.alg_conf}")
    print(f"Environment config: {args.env_conf}")
    print(f"Task config: {args.task_conf}")
    print(f"Route set: {route_set}")
    print(f"Metrics will {'NOT ' if args.skip_metrics else ''}be computed after the experiment.\n")

    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    random.seed(args.env_seed)
    np.random.seed(args.env_seed)
    torch.manual_seed(args.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    p = {}
    for path in (f"../config/algo_config/{algorithm}/{args.alg_conf}.json",
                 f"../config/env_config/{args.env_conf}.json",
                 f"../config/task_config/{args.task_conf}.json"):
        with open(path) as f:
            p.update(json.load(f))
    p.pop("desc", None)

    custom_network_folder = f"../networks/{network}"
    phases = [1,
              p["human_learning_episodes"] + 1,
              p["human_learning_episodes"] + int(p["training_eps"]) + 1]
    phase_names = ["Auto-routed human baseline", "Mutation and AV learning", "Testing phase"]
    base_results_dir = os.environ.get("RESULTS_BASE_DIR", "../results")
    records_folder = os.path.join(base_results_dir, args.id)
    plots_folder = os.path.join(records_folder, "plots")

    with open(os.path.join(custom_network_folder, f"od_{network}.txt"), "r", encoding="utf-8") as f:
        od = ast.literal_eval(f.read())
    origins, destinations = od["origins"], od["destinations"]

    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    demand_routes_path = os.path.join(custom_network_folder, "demand_routes.csv")
    for required_path in (agents_csv_path, demand_routes_path):
        if not os.path.exists(required_path):
            raise FileNotFoundError(required_path)

    agents_df = pd.read_csv(agents_csv_path)
    num_agents = len(agents_df)
    route_catalogs = load_human_route_catalogs(demand_routes_path, agents_df["id"])
    simulation_timesteps = max(180, int(agents_df["start_time"].max()) + 1)

    os.makedirs(records_folder, exist_ok=True)
    with open(agents_csv_path, "r", encoding="utf-8") as f:
        agents_content = f.read()
    with open(os.path.join(records_folder, "agents.csv"), "w", encoding="utf-8") as f:
        f.write(agents_content)

    route_set_dir = os.path.join(custom_network_folder, "clustered_routes", route_set)
    validate_clustered_route_set(network, route_set_dir)
    clustered_loader = ClusteredRoutesLoader(network, custom_network_folder,
                                              route_set_dir=route_set_dir)
    number_of_paths = clustered_loader.get_number_of_paths()
    clustered_loader.export_to_paths_csv(
        os.path.join(records_folder, "routes.csv"), origins, destinations)
    clustered_loader.export_to_sumo_rou_xml(
        os.path.join(records_folder, "route.rou.xml"), origins, destinations)

    background_path = os.path.join(custom_network_folder, "background_buses.rou.xml")
    if not os.path.exists(background_path):
        raise FileNotFoundError(background_path)
    background_root = ET.parse(background_path).getroot()
    background_vehicles = background_root.findall("vehicle")
    background_ids = {v.get("id") for v in background_vehicles}
    if len(background_ids) != len(background_vehicles) or None in background_ids:
        raise ValueError("Background bus IDs are missing or duplicated")
    if background_ids.intersection(str(a) for a in agents_df["id"]):
        raise ValueError("Background bus IDs overlap with RouteRL agent IDs")
    route_path = os.path.join(records_folder, "route.rou.xml")
    route_tree = ET.parse(route_path)
    route_tree.getroot().extend(list(background_root))
    ET.indent(route_tree, space="  ")
    route_tree.write(route_path, encoding="utf-8", xml_declaration=True)
    if background_vehicles:
        simulation_timesteps = max(
            simulation_timesteps,
            int(max(float(v.get("depart")) for v in background_vehicles)) + 1)

    action_masks = clustered_loader.create_masks(origins, destinations)
    demand_ods = set(agents_df[["origin", "destination"]].itertuples(index=False, name=None))
    missing_masks = sorted(demand_ods.difference(action_masks))
    empty_masks = sorted(od_ for od_ in demand_ods if not action_masks[od_].any())
    if missing_masks or empty_masks:
        raise ValueError(
            "Clustered route set does not cover the demand "
            f"(missing ODs: {missing_masks[:10]}, empty masks: {empty_masks[:10]})")

    num_machines = int(num_agents * p["ratio_machines"])
    total_episodes = p["human_learning_episodes"] + p["training_eps"] + p["test_eps"]

    dump_config = dict(p)
    if args.project is not None:
        dump_config["project"] = args.project
    dump_config.update(
        network=network, env_seed=args.env_seed, torch_seed=args.torch_seed,
        env_config=args.env_conf, task_config=args.task_conf, alg_config=args.alg_conf,
        script=script_path_for_config(__file__), algorithm=algorithm,
        num_agents=num_agents, num_machines=num_machines,
        human_learning=False, should_humans_adapt=False,
        human_routing="per_agent_imported_distribution",
        demand_routes_file=f"networks/{network}/demand_routes.csv",
        background_traffic_file=f"networks/{network}/background_buses.rou.xml",
        background_bus_count=len(background_ids), route_set=route_set,
        number_of_paths=number_of_paths, use_clustered_routes=True,
        use_action_masks=True, framework="torch")
    if save_model_every is not None:
        dump_config["save_model_every"] = save_model_every
    with open(os.path.join(records_folder, "exp_config.json"), "w", encoding="utf-8") as f:
        json.dump(dump_config, f, indent=4)

    env = TrafficEnvironment(
        seed=args.env_seed,
        create_agents=False,
        create_paths=False,
        action_masks=action_masks,
        save_detectors_info=False,
        agent_parameters={
            "new_machines_after_mutation": num_machines,
            "human_parameters": {
                "model": p["human_model"],
                "alpha": p["human_alpha"],
                "beta": p["human_beta"],
                "beta_randomness": p["human_beta_randomness"],
                "deterministic": p["human_deterministic"],
            },
            "machine_parameters": {
                "behavior": p["av_behavior"],
                "observation_type": p["observations"],
            },
        },
        environment_parameters={"save_every": p["save_every"]},
        simulator_parameters={
            "network_name": network,
            "custom_network_folder": custom_network_folder,
            "sumo_type": "sumo",
            "simulation_timesteps": simulation_timesteps,
        },
        plotter_parameters={
            "phases": phases,
            "phase_names": phase_names,
            "smooth_by": p["smooth_by"],
            "plot_choices": p["plot_choices"],
            "records_folder": records_folder,
            "plots_folder": plots_folder,
        },
        path_generation_parameters={
            "origins": origins,
            "destinations": destinations,
            "number_of_paths": number_of_paths,
            "beta": p.get("path_gen_beta", -1),
            "num_samples": p.get("num_samples", 1),
            "path_gen_workers": p.get("path_gen_workers", 1),
            "visualize_paths": False,
        },
    )

    install_auto_routed_humans(env, route_catalogs, background_ids)
    env.human_learning = False
    env.start()
    env.reset()
    print_agent_counts(env)

    if p.get("should_humans_adapt"):
        print("Warning: should_humans_adapt is ignored by qmix (auto-routed humans); "
              "humans never learn.")
    print(
        "Note: imported human routes do not share JanuX action indices. "
        "RouteRL observations see the first valid masked action. In saved episode "
        "files, human action is the imported route ID and human cost_table is the "
        "imported probability vector. Background buses affect traffic but are not agents.")

    # ─── Auto-routed human baseline ───
    routing_episode = 0
    pbar = tqdm(total=total_episodes, desc="Auto-routed human baseline")
    for _ in range(p["human_learning_episodes"]):
        sample_human_routes(env, route_catalogs, action_masks, args.env_seed, routing_episode)
        routing_episode += 1
        env.step()
        pbar.update()

    env.mutation(disable_human_learning=True, mutation_start_percentile=-1)
    print_agent_counts(env)
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    env = AVMaskWrapper(env, action_masks)

    machines = list(env.machine_agents)
    agent_index = {str(a.id): i for i, a in enumerate(machines)}

    # ─── Build agent features matrix if use_agent_embedding ───
    use_agent_embedding = bool(p.get("use_agent_embedding", False))
    agent_features_matrix = None
    if use_agent_embedding:
        feature_names = p.get("agent_features", ["od", "start_time", "free_flow_time"])
        print(f"[QMIX-torch] Building agent features: {feature_names}")
        agent_features_matrix, agent_feat_dim = build_agent_features_matrix(
            machines=machines,
            agents_df=agents_df,
            catalogs=route_catalogs,
            features=feature_names,
            num_od_pairs=len(origins) * len(destinations),
            n_destinations=len(destinations),
            normalize=True,
        )
        print(f"[QMIX-torch] agent_feat_dim = {agent_feat_dim}, "
              f"agent_embed_dim = {p.get('agent_embed_dim', 8)}")

    model = QMIX(
        obs_dim=obs_dim,
        n_actions=machines[0].action_space_size,
        n_agents=len(machines),
        hidden=p["hidden"],
        embed=p["embed"],
        state_mode=p["state_mode"],
        team_reward=p["team_reward"],
        loss=p["loss"],
        use_agent_id=p["use_agent_id"],
        use_agent_embedding=use_agent_embedding,
        agent_features=agent_features_matrix,
        agent_embed_dim=p.get("agent_embed_dim", 8),
        agent_embed_hidden=p.get("agent_embed_hidden", 32),
        normalize_obs=p["normalize_obs"],
        lr=p["lr"],
        batch_size=p["batch_size"],
        buffer_size=p["buffer_size"],
        max_grad_norm=p["max_grad_norm"],
        eps_init=p["eps_init"],
        eps_decay=p["eps_decay"],
        eps_min=p["eps_min"],
        updates_per_episode=p["updates_per_episode"],
        hyper_hidden=p.get("hyper_hidden", 64),
        seed=args.torch_seed,
        device=device,
    )

    # ─── AV learning ───
    pbar.set_description("AV learning")
    os.makedirs(plots_folder, exist_ok=True)
    for episode in range(p["training_eps"]):
        env.reset()
        sample_human_routes(env, route_catalogs, action_masks, args.env_seed, routing_episode)
        routing_episode += 1
        run_episode(env, model, agent_index, learn=True)

        completed = episode + 1
        if should_save_model_snapshot(completed, p["training_eps"], save_model_every):
            with open(model_snapshot_path(records_folder, completed, "pkl"), "wb") as f:
                pickle.dump({"training_episode": completed, **model.state_dict()}, f)
        if episode % p["plot_every"] == 0:
            env.plot_results()
        pbar.update()

    # ─── Testing ───
    pbar.set_description("Testing")
    for _ in range(p["test_eps"]):
        env.reset()
        sample_human_routes(env, route_catalogs, action_masks, args.env_seed, routing_episode)
        routing_episode += 1
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
        run_metrics_analysis(args.id, results_folder=base_results_dir)


if __name__ == "__main__":
    main()