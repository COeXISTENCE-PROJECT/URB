"""
QMIX for auto-routed humans (ARH) on URB.

Implementation faithful to Rashid et al. (2018) and PyMARL:
  - Agent network: shared MLP. Input identity is either:
      * one-hot agent ID (use_agent_id=True),
      * learned embedding from static per-agent features (use_agent_embedding=True),
      * nothing (both False → agents with identical observations act identically).
  - Mixer: 4 two-layer hypernetworks conditioned on concat(obs_1..obs_N).
  - Monotonicity: guaranteed because w1 and w2 are |.| of hypernet outputs and
    the ELU inside the mixer has strictly positive derivative everywhere.
  - CTDE: decentralized argmax at execution time, centralized training.
  - No bootstrap / target network: valid because each AV takes exactly one
    decision per episode.
"""

import xml.etree.ElementTree as ET
from types import MethodType

import numpy as np


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _xavier(rng, fan_in, fan_out):
    bound = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, (fan_in, fan_out))


def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))


def _elu_grad(x):
    return np.where(x > 0, 1.0, np.exp(np.minimum(x, 0.0)))


class Adam:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g * g
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class RunningNorm:
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim)
        self.m2 = np.zeros(dim)

    def update(self, x):
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
        return np.clip((x - self.mean) / std, -10.0, 10.0)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def init_params(rng, agent_in, n_actions, n_agents, state_dim, hidden, embed,
                hyper_hidden=64, agent_feat_dim=0,
                agent_embed_hidden=32, agent_embed_dim=8):
    P = {}

    # ─── Agent network ───
    sizes = [agent_in, *hidden, n_actions]
    for l in range(len(sizes) - 1):
        P[f"aW{l}"] = _xavier(rng, sizes[l], sizes[l + 1])
        P[f"ab{l}"] = np.zeros(sizes[l + 1])

    # ─── Agent embedding MLP (optional) ───
    # feature → hidden → embedding. Two linear layers with ReLU in between.
    if agent_feat_dim > 0:
        P["emb_w1"] = _xavier(rng, agent_feat_dim, agent_embed_hidden)
        P["emb_b1"] = np.zeros(agent_embed_hidden)
        P["emb_w2"] = _xavier(rng, agent_embed_hidden, agent_embed_dim)
        P["emb_b2"] = np.zeros(agent_embed_dim)

    # ─── Hypernets for the mixer (unchanged) ───
    def _init_hyper(prefix, out_dim, nonneg):
        P[f"H{prefix}a"] = _xavier(rng, state_dim, hyper_hidden)
        P[f"h{prefix}a"] = np.zeros(hyper_hidden)
        P[f"H{prefix}b"] = _xavier(rng, hyper_hidden, out_dim)
        if nonneg:
            bound = 1.0 / np.sqrt(state_dim)
            P[f"h{prefix}b"] = rng.uniform(-bound, bound, out_dim)
        else:
            P[f"h{prefix}b"] = np.zeros(out_dim)

    _init_hyper("w1", n_agents * embed, nonneg=True)
    _init_hyper("b1", embed, nonneg=False)
    _init_hyper("w2", embed, nonneg=True)
    _init_hyper("b2", 1, nonneg=False)
    return P


# ---------------------------------------------------------------------------
# Agent embedding (static features → learned embedding)
# ---------------------------------------------------------------------------

def encode_agent_features(P, feats):
    """feats: (N, D_feat) → (N, k). Returns (embedding, cache)."""
    h_pre = feats @ P["emb_w1"] + P["emb_b1"]          # (N, hidden)
    h = np.maximum(h_pre, 0.0)
    e = h @ P["emb_w2"] + P["emb_b2"]                  # (N, k)
    cache = dict(feats=feats, h_pre=h_pre, h=h)
    return e, cache


def encode_agent_features_backward(P, cache, d_e):
    """d_e: (N, k). Returns grads for emb_w1, emb_b1, emb_w2, emb_b2."""
    grads = {}
    h_pre, h = cache["h_pre"], cache["h"]
    grads["emb_w2"] = h.T @ d_e                        # (hidden, k)
    grads["emb_b2"] = d_e.sum(axis=0)                  # (k,)
    d_h = (d_e @ P["emb_w2"].T) * (h_pre > 0)          # (N, hidden)
    grads["emb_w1"] = cache["feats"].T @ d_h           # (D_feat, hidden)
    grads["emb_b1"] = d_h.sum(axis=0)                  # (hidden,)
    return grads


# ---------------------------------------------------------------------------
# Agent network
# ---------------------------------------------------------------------------

def agent_forward(P, x, n_layers):
    cache = []
    h = x
    for l in range(n_layers):
        z = h @ P[f"aW{l}"] + P[f"ab{l}"]
        cache.append((h, z))
        h = np.maximum(z, 0.0) if l < n_layers - 1 else z
    return h, cache


def agent_backward(P, cache, dq, n_layers, return_input_grad=False):
    grads = {}
    d = dq
    d_input = None
    for l in reversed(range(n_layers)):
        h_in, z = cache[l]
        if l < n_layers - 1:
            d = d * (z > 0)
        grads[f"aW{l}"] = h_in.T @ d
        grads[f"ab{l}"] = d.sum(axis=0)
        d = d @ P[f"aW{l}"].T
        if l == 0:
            d_input = d  # gradient w.r.t. the input of the agent MLP
    if return_input_grad:
        return grads, d_input
    return grads


# ---------------------------------------------------------------------------
# Mixer (unchanged)
# ---------------------------------------------------------------------------

def mixer_forward(P, q, s, n_agents, embed):
    B = q.shape[0]

    z1_w1 = s @ P["Hw1a"] + P["hw1a"];     h1_w1 = np.maximum(z1_w1, 0.0)
    pre_w1 = h1_w1 @ P["Hw1b"] + P["hw1b"]
    w1 = np.abs(pre_w1).reshape(B, n_agents, embed)

    z1_b1 = s @ P["Hb1a"] + P["hb1a"];     h1_b1 = np.maximum(z1_b1, 0.0)
    b1 = h1_b1 @ P["Hb1b"] + P["hb1b"]

    z1_w2 = s @ P["Hw2a"] + P["hw2a"];     h1_w2 = np.maximum(z1_w2, 0.0)
    pre_w2 = h1_w2 @ P["Hw2b"] + P["hw2b"]
    w2 = np.abs(pre_w2)

    z1_b2 = s @ P["Hb2a"] + P["hb2a"];     h1_b2 = np.maximum(z1_b2, 0.0)
    b2 = (h1_b2 @ P["Hb2b"] + P["hb2b"])[:, 0]

    pre_h = np.einsum("bn,bne->be", q, w1) + b1
    h = _elu(pre_h)
    q_tot = (h * w2).sum(axis=1) + b2

    cache = dict(
        q=q, s=s,
        z1_w1=z1_w1, h1_w1=h1_w1, pre_w1=pre_w1, w1=w1,
        z1_b1=z1_b1, h1_b1=h1_b1,
        z1_w2=z1_w2, h1_w2=h1_w2, pre_w2=pre_w2, w2=w2,
        z1_b2=z1_b2, h1_b2=h1_b2,
        pre_h=pre_h, h=h,
    )
    return q_tot, cache


def mixer_backward(P, c, g, n_agents, embed):
    B = g.shape[0]
    g2 = g[:, None]
    grads = {}

    grads["Hb2b"] = c["h1_b2"].T @ g2
    grads["hb2b"] = g2.sum(axis=0)
    dz1_b2 = (g2 @ P["Hb2b"].T) * (c["z1_b2"] > 0)
    grads["Hb2a"] = c["s"].T @ dz1_b2
    grads["hb2a"] = dz1_b2.sum(axis=0)

    dpre_w2 = g2 * c["h"] * np.sign(c["pre_w2"])
    grads["Hw2b"] = c["h1_w2"].T @ dpre_w2
    grads["hw2b"] = dpre_w2.sum(axis=0)
    dz1_w2 = (dpre_w2 @ P["Hw2b"].T) * (c["z1_w2"] > 0)
    grads["Hw2a"] = c["s"].T @ dz1_w2
    grads["hw2a"] = dz1_w2.sum(axis=0)

    dh = g2 * c["w2"]
    dpre_h = dh * _elu_grad(c["pre_h"])

    grads["Hb1b"] = c["h1_b1"].T @ dpre_h
    grads["hb1b"] = dpre_h.sum(axis=0)
    dz1_b1 = (dpre_h @ P["Hb1b"].T) * (c["z1_b1"] > 0)
    grads["Hb1a"] = c["s"].T @ dz1_b1
    grads["hb1a"] = dz1_b1.sum(axis=0)

    dq = np.einsum("bne,be->bn", c["w1"], dpre_h)

    dpre_w1 = np.einsum("bn,be->bne", c["q"], dpre_h)
    dpre_w1 = dpre_w1.reshape(B, -1) * np.sign(c["pre_w1"])
    grads["Hw1b"] = c["h1_w1"].T @ dpre_w1
    grads["hw1b"] = dpre_w1.sum(axis=0)
    dz1_w1 = (dpre_w1 @ P["Hw1b"].T) * (c["z1_w1"] > 0)
    grads["Hw1a"] = c["s"].T @ dz1_w1
    grads["hw1a"] = dz1_w1.sum(axis=0)

    return grads, dq


# ---------------------------------------------------------------------------
# Agent input construction
# ---------------------------------------------------------------------------

def build_agent_inputs(obs, use_agent_id):
    """obs → (B*N, d) or (B*N, d+N) if use_agent_id. NOT used with embedding."""
    B, N, d = obs.shape
    x = obs.reshape(B * N, d)
    if use_agent_id:
        x = np.concatenate([x, np.tile(np.eye(N), (B, 1))], axis=1)
    return x


def loss_and_grads(P, obs, actions, y, state, agent_features,
                   n_agents, embed, n_layers,
                   use_agent_id, use_agent_embedding, huber,
                   agent_embed_dim):
    B, N, _ = obs.shape

    # ─── Build agent inputs ───
    if use_agent_embedding:
        E, emb_cache = encode_agent_features(P, agent_features)   # (N, k)
        E_batch = np.tile(E, (B, 1))                              # (B*N, k)
        x = np.concatenate([obs.reshape(B * N, -1), E_batch], axis=1)  # (B*N, d+k)
    else:
        x = build_agent_inputs(obs, use_agent_id)
        emb_cache = None

    # ─── Agent forward ───
    Q, a_cache = agent_forward(P, x, n_layers)
    Q = Q.reshape(B, N, -1)
    q_chosen = np.take_along_axis(Q, actions[..., None], axis=2)[..., 0]

    # ─── Mixer forward ───
    q_tot, m_cache = mixer_forward(P, q_chosen, state, n_agents, embed)

    # ─── Loss ───
    err = y - q_tot
    if huber:
        loss = float(np.mean(np.where(np.abs(err) <= 1.0,
                                      0.5 * err ** 2, np.abs(err) - 0.5)))
        g = -np.clip(err, -1.0, 1.0) / B
    else:
        loss = float(np.mean(err ** 2))
        g = -2.0 * err / B

    # ─── Mixer backward ───
    m_grads, dq = mixer_backward(P, m_cache, g, n_agents, embed)

    # ─── Agent backward ───
    dQ = np.zeros_like(Q)
    np.put_along_axis(dQ, actions[..., None], dq[..., None], axis=2)
    dQ_flat = dQ.reshape(B * N, -1)

    if use_agent_embedding:
        a_grads, d_x = agent_backward(P, a_cache, dQ_flat, n_layers,
                                      return_input_grad=True)
        # gradient w.r.t. embedding: last k columns of d_x
        d_emb_batch = d_x[:, -agent_embed_dim:]                    # (B*N, k)
        d_emb = d_emb_batch.reshape(B, N, agent_embed_dim).sum(axis=0)  # (N, k)
        emb_grads = encode_agent_features_backward(P, emb_cache, d_emb)
        a_grads.update(emb_grads)
    else:
        a_grads = agent_backward(P, a_cache, dQ_flat, n_layers)

    return loss, {**a_grads, **m_grads}


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
                 hyper_hidden=64, seed=0):
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

        self.obs_dim, self.n_actions, self.n_agents = obs_dim, n_actions, n_agents
        self.hidden, self.embed = tuple(hidden), embed
        self.state_mode = state_mode
        self.team_reward = team_reward
        self.huber = loss == "huber"
        self.use_agent_id = bool(use_agent_id)
        self.use_agent_embedding = bool(use_agent_embedding)
        self.agent_embed_dim = int(agent_embed_dim)
        self.agent_embed_hidden = int(agent_embed_hidden)

        if self.use_agent_embedding:
            if agent_features is None:
                raise ValueError("agent_features required when use_agent_embedding=True")
            self.agent_features = np.asarray(agent_features, dtype=np.float64)
            if self.agent_features.ndim != 2 or self.agent_features.shape[0] != n_agents:
                raise ValueError(
                    f"agent_features must be (N={n_agents}, D_feat), "
                    f"got {self.agent_features.shape}"
                )
            self.agent_feat_dim = self.agent_features.shape[1]
        else:
            self.agent_features = None
            self.agent_feat_dim = 0

        self.batch_size, self.buffer_size = batch_size, buffer_size
        self.max_grad_norm, self.updates_per_episode = max_grad_norm, updates_per_episode
        self.eps, self.eps_decay, self.eps_min = eps_init, eps_decay, eps_min
        self.hyper_hidden = hyper_hidden
        self.rng = np.random.default_rng(seed)

        self.state_dim = n_agents * obs_dim if state_mode == "concat" else obs_dim
        self.n_layers = len(self.hidden) + 1

        if self.use_agent_embedding:
            agent_in = obs_dim + self.agent_embed_dim
        elif self.use_agent_id:
            agent_in = obs_dim + n_agents
        else:
            agent_in = obs_dim

        self.P = init_params(self.rng, agent_in, n_actions, n_agents, self.state_dim,
                             self.hidden, embed, hyper_hidden=hyper_hidden,
                             agent_feat_dim=self.agent_feat_dim,
                             agent_embed_hidden=self.agent_embed_hidden,
                             agent_embed_dim=self.agent_embed_dim)
        self.n_params = sum(v.size for v in self.P.values())
        if self.n_params > 5_000_000:
            print(f"[QMIX] {self.n_params:,} parameters. If this is too slow, "
                  f"consider state_mode='mean' or smaller embed.")
        self.opt = Adam(self.P, lr)
        self.norm = RunningNorm(obs_dim) if normalize_obs else None

        # Cache inference-time embeddings from current P
        if self.use_agent_embedding:
            self._agent_embeddings, _ = encode_agent_features(self.P, self.agent_features)
        else:
            self._agent_embeddings = None

        self.buf_obs, self.buf_act, self.buf_rew = [], [], []
        self.loss = []

    def _normalize(self, obs):
        return self.norm(obs) if self.norm is not None else obs

    def _state(self, obs):
        if self.state_mode == "concat":
            return obs.reshape(obs.shape[0], -1)
        return obs.mean(axis=1)

    def act(self, obs, agent_idx, mask=None, greedy=False):
        o = self._normalize(np.asarray(obs, dtype=np.float64))[None]
        if self.use_agent_embedding:
            e = self._agent_embeddings[agent_idx][None]            # (1, k)
            o = np.concatenate([o, e], axis=1)
        elif self.use_agent_id:
            one_hot = np.zeros((1, self.n_agents))
            one_hot[0, agent_idx] = 1.0
            o = np.concatenate([o, one_hot], axis=1)
        q = agent_forward(self.P, o, self.n_layers)[0][0]

        valid = (np.arange(self.n_actions) if mask is None
                 else np.flatnonzero(np.asarray(mask, dtype=bool)))
        if valid.size == 0:
            raise ValueError("empty action mask")
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.choice(valid))
        return int(valid[np.argmax(q[valid])])

    def store_episode(self, obs, actions, rewards):
        obs = np.asarray(obs, dtype=np.float64)
        rewards = np.asarray(rewards, dtype=np.float64)
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
        last = None
        for _ in range(self.updates_per_episode):
            idx = self.rng.integers(0, n, size=min(self.batch_size, n))
            O = self._normalize(np.stack([self.buf_obs[i] for i in idx]))
            A = np.stack([self.buf_act[i] for i in idx])
            y = np.array([self.buf_rew[i] for i in idx])
            loss, grads = loss_and_grads(
                self.P, O, A, y, self._state(O),
                self.agent_features,
                self.n_agents, self.embed, self.n_layers,
                self.use_agent_id, self.use_agent_embedding, self.huber,
                self.agent_embed_dim)
            norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
            if norm > self.max_grad_norm:
                grads = {k: g * (self.max_grad_norm / norm) for k, g in grads.items()}
            self.opt.step(self.P, grads)
            self.loss.append(loss)
            last = loss

        # Refresh cached embeddings since P just changed
        if self.use_agent_embedding:
            self._agent_embeddings, _ = encode_agent_features(self.P, self.agent_features)

        return last

    def decay_epsilon(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def state_dict(self):
        out = {"params": {k: v.copy() for k, v in self.P.items()}, "eps": self.eps}
        if self.norm is not None:
            out["norm"] = {"n": self.norm.n, "mean": self.norm.mean.copy(),
                           "m2": self.norm.m2.copy()}
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

    Supported features:
      - "od"               : one-hot of (origin, destination) pair,
      - "start_time"       : agent's start time (from agents.csv),
      - "free_flow_time"   : free-flow travel time of the agent's demand route
                             (from demand_routes.csv, single route per agent).

    Continuous features are z-scored; one-hot features are left as-is.
    """
    N = len(machines)
    blocks = []
    continuous_offsets = []  # (start_col, end_col) of continuous blocks
    offset = 0

    agents_lookup = agents_df.set_index("id")[["origin", "destination", "start_time"]]

    if "od" in features:
        od = np.zeros((N, num_od_pairs), dtype=np.float64)
        for i, machine in enumerate(machines):
            row = agents_lookup.loc[int(machine.id)]
            o, d = int(row["origin"]), int(row["destination"])
            od[i, o * n_destinations + d] = 1.0
        blocks.append(od)
        offset += num_od_pairs

    if "start_time" in features:
        st = np.zeros((N, 1), dtype=np.float64)
        for i, machine in enumerate(machines):
            st[i, 0] = float(agents_lookup.loc[int(machine.id)]["start_time"])
        blocks.append(st)
        continuous_offsets.append((offset, offset + 1))
        offset += 1

    if "free_flow_time" in features:
        fft = np.zeros((N, 1), dtype=np.float64)
        for i, machine in enumerate(machines):
            cat = catalogs.get(int(machine.id), [])
            if cat:
                fft[i, 0] = float(cat[0].get("free_flow_time", 0.0))
        blocks.append(fft)
        continuous_offsets.append((offset, offset + 1))
        offset += 1

    if not blocks:
        raise ValueError(f"No valid features in {features}")

    feats = np.concatenate(blocks, axis=1).astype(np.float64)

    if normalize and continuous_offsets:
        cols = np.concatenate([np.arange(a, b) for a, b in continuous_offsets])
        mean = feats[:, cols].mean(axis=0)
        std = feats[:, cols].std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        feats[:, cols] = (feats[:, cols] - mean) / std

    return feats, offset


# ---------------------------------------------------------------------------
# Auto-routed humans (same logic as iql_arh)
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
    parser.add_argument("--torch-seed", type=int, default=42)
    parser.add_argument("--route-set", type=str, default=None)
    parser.add_argument("--skip-metrics", action="store_true", default=False)
    add_model_snapshot_argument(parser)
    args = parser.parse_args()

    algorithm = "qmix"
    network = args.net
    route_set = resolve_route_set(network, args.route_set)
    save_model_every = args.save_model_every

    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {algorithm.upper()} (auto-routed humans)")
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
        use_action_masks=True)
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
        print(f"[QMIX] Building agent features: {feature_names}")
        agent_features_matrix, agent_feat_dim = build_agent_features_matrix(
            machines=machines,
            agents_df=agents_df,
            catalogs=route_catalogs,
            features=feature_names,
            num_od_pairs=len(origins) * len(destinations),
            n_destinations=len(destinations),
            normalize=True,
        )
        print(f"[QMIX] agent_feat_dim = {agent_feat_dim}, "
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