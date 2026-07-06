from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from torch.distributions import Categorical


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ManagerOutput:
    subgoal: int
    log_prob: float
    entropy: float
    probs: torch.Tensor


class FeudalManager(nn.Module):
    """High-level policy that picks a discrete subgoal.

    First pass design:
    - discrete subgoals only
    - optional cluster embedding hook
    - PPO-style update support through log-prob caching in caller
    """

    def __init__(
        self,
        obs_dim: int,
        num_subgoals: int,
        hidden_dims: Iterable[int] = (128, 128),
        use_cluster_embedding: bool = False,
        num_clusters: int = 0,
        cluster_embed_dim: int = 8,
    ):
        super().__init__()
        self.num_subgoals = int(num_subgoals)
        self.use_cluster_embedding = bool(use_cluster_embedding)
        self.cluster_embedding = None
        mlp_input_dim = int(obs_dim)

        if self.use_cluster_embedding:
            if num_clusters <= 0:
                raise ValueError("num_clusters must be > 0 when use_cluster_embedding=True")
            self.cluster_embedding = nn.Embedding(int(num_clusters), int(cluster_embed_dim))
            mlp_input_dim += int(cluster_embed_dim)

        self.policy = MLP(mlp_input_dim, list(hidden_dims), self.num_subgoals)

    def _augment_obs(
        self,
        obs: torch.Tensor,
        cluster_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_cluster_embedding:
            return obs
        if cluster_ids is None:
            raise ValueError("cluster_ids are required when cluster embeddings are enabled")
        cluster_emb = self.cluster_embedding(cluster_ids.long())
        return torch.cat([obs, cluster_emb], dim=-1)

    def forward(self, obs: torch.Tensor, cluster_ids: torch.Tensor | None = None) -> torch.Tensor:
        x = self._augment_obs(obs, cluster_ids)
        return self.policy(x)

    def dist(self, obs: torch.Tensor, cluster_ids: torch.Tensor | None = None) -> Categorical:
        logits = self.forward(obs, cluster_ids)
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        cluster_ids: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> ManagerOutput:
        distribution = self.dist(obs, cluster_ids)
        if deterministic:
            subgoal = torch.argmax(distribution.probs, dim=-1)
        else:
            subgoal = distribution.sample()
        log_prob = distribution.log_prob(subgoal)
        entropy = distribution.entropy()
        return ManagerOutput(
            subgoal=int(subgoal.item()),
            log_prob=float(log_prob.item()),
            entropy=float(entropy.mean().item()),
            probs=distribution.probs.detach().cpu(),
        )
