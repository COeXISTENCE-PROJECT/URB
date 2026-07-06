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
class ControllerOutput:
    action: int
    log_prob: float
    entropy: float
    probs: torch.Tensor


class FeudalController(nn.Module):
    """Low-level policy conditioned on a discrete manager subgoal.

    This first version uses a learned subgoal embedding and optional masking.
    The mask is expected to be 1 for valid actions and 0 for invalid actions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_subgoals: int,
        hidden_dims: Iterable[int] = (128, 128),
        subgoal_embed_dim: int = 16,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.subgoal_embedding = nn.Embedding(int(num_subgoals), int(subgoal_embed_dim))
        self.policy = MLP(int(obs_dim) + int(subgoal_embed_dim), list(hidden_dims), self.action_dim)

    def forward(self, obs: torch.Tensor, subgoals: torch.Tensor) -> torch.Tensor:
        subgoal_emb = self.subgoal_embedding(subgoals.long())
        x = torch.cat([obs, subgoal_emb], dim=-1)
        return self.policy(x)

    def dist(
        self,
        obs: torch.Tensor,
        subgoals: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> Categorical:
        logits = self.forward(obs, subgoals)
        if action_mask is not None:
            very_negative = torch.finfo(logits.dtype).min
            logits = torch.where(action_mask > 0, logits, torch.full_like(logits, very_negative))
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        subgoals: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> ControllerOutput:
        distribution = self.dist(obs, subgoals, action_mask)
        if deterministic:
            action = torch.argmax(distribution.probs, dim=-1)
        else:
            action = distribution.sample()
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return ControllerOutput(
            action=int(action.item()),
            log_prob=float(log_prob.item()),
            entropy=float(entropy.mean().item()),
            probs=distribution.probs.detach().cpu(),
        )
