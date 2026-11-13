"""Multi-Agent Transformer for clustered route choice.

Wen et al., NeurIPS 2022 (https://arxiv.org/abs/2205.14953).
Each traffic day is one autoregressive sequence with a terminal fleet reward.
"""

import argparse
import ast
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from routerl import TrafficEnvironment
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from tqdm import tqdm

from clustered_routes import ClusteredRoutesLoader, resolve_route_set, validate_clustered_route_set
from utils import (
    add_model_snapshot_argument,
    clear_SUMO_files,
    model_snapshot_path,
    print_agent_counts,
    run_metrics_analysis,
    save_loss_records,
    script_path_for_config,
    should_save_model_snapshot,
)


class MATEnvironment(TrafficEnvironment):
    decision_observation = None

    def observe(self, agent):
        observation = super().observe(agent)
        # RouteRL observes again at route submission. Record the input used to choose the route.
        if self.decision_observation is not None and agent == self.agent_selection:
            return self.decision_observation
        return observation

    def step(self, machine_action=None, *, decision_observation=None):
        self.decision_observation = decision_observation
        try:
            super().step(machine_action)
        finally:
            self.decision_observation = None


def linear(input_size, output_size, *, activate=False, bias=True):
    layer = nn.Linear(input_size, output_size, bias=bias)
    nn.init.orthogonal_(layer.weight, gain=2**0.5 if activate else 0.01)
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


class Attention(nn.Module):
    def __init__(self, width, heads, causal=False):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.query = linear(width, width)
        self.key = linear(width, width)
        self.value = linear(width, width)
        self.output = linear(width, width)

    def split_heads(self, x):
        return x.unflatten(-1, (self.heads, -1)).transpose(1, 2)

    def forward(self, query, source, cache=None, position=None):
        q = self.split_heads(self.query(query))
        k = self.split_heads(self.key(source))
        v = self.split_heads(self.value(source))
        if cache is not None:
            # A single rollout query attends to the cached prefix including its new token.
            keys, values = cache
            keys[:, :, position:position + 1] = k
            values[:, :, position:position + 1] = v
            k, v = keys[:, :, :position + 1], values[:, :, :position + 1]
        output = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal and cache is None)
        return self.output(output.transpose(1, 2).flatten(-2))


class EncoderBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attention = Attention(width, heads)
        self.attention_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(linear(width, width, activate=True), nn.GELU(), linear(width, width))
        self.mlp_norm = nn.LayerNorm(width)

    def forward(self, x):
        x = self.attention_norm(x + self.attention(x, x))
        return self.mlp_norm(x + self.mlp(x))


class DecoderBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.action_attention = Attention(width, heads, causal=True)
        self.observation_attention = Attention(width, heads, causal=True)
        self.action_norm = nn.LayerNorm(width)
        self.observation_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(linear(width, width, activate=True), nn.GELU(), linear(width, width))
        self.mlp_norm = nn.LayerNorm(width)

    def forward(self, x, observations, cache=None, position=None):
        action_cache, observation_cache = (None, None) if cache is None else cache
        x = self.action_norm(x + self.action_attention(x, x, action_cache, position))
        x = self.observation_norm(
            observations + self.observation_attention(observations, x, observation_cache, position)
        )
        return self.mlp_norm(x + self.mlp(x))


class MAT(nn.Module):
    def __init__(self, observation_size, action_size, n_embd, n_head, n_block):
        super().__init__()
        if min(n_embd, n_head, n_block) < 1 or n_embd % n_head:
            raise ValueError("MAT needs positive block and head counts and a width divisible by the head count")
        self.action_size = action_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.encoder = nn.Sequential(
            nn.LayerNorm(observation_size),
            linear(observation_size, n_embd, activate=True),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            *(EncoderBlock(n_embd, n_head) for _ in range(n_block)),
        )
        self.action_encoder = nn.Sequential(
            linear(action_size + 1, n_embd, activate=True, bias=False),
            nn.GELU(),
            nn.LayerNorm(n_embd),
        )
        self.decoder = nn.ModuleList(DecoderBlock(n_embd, n_head) for _ in range(n_block))
        self.value_head = nn.Sequential(
            linear(n_embd, n_embd, activate=True), nn.GELU(), nn.LayerNorm(n_embd), linear(n_embd, 1)
        )
        self.policy_head = nn.Sequential(
            linear(n_embd, n_embd, activate=True), nn.GELU(), nn.LayerNorm(n_embd), linear(n_embd, action_size)
        )
        self.live_projection = linear(observation_size, n_embd, activate=True, bias=False)
        # Exponential moving moments and their debiasing weight (MAT ValueNorm).
        self.register_buffer("value_moments", torch.zeros(3))

    def encode(self, observations):
        # Balance feature scales before per-token LayerNorm, especially raw OD indices.
        scale = observations.abs().amax(dim=1, keepdim=True).clamp(min=1.0)
        return self.encoder(observations / scale), scale

    def add_live_observation(self, encoded, reset_obs, live_obs, scale):
        # Use only reset features to set the scale. Compress large congestion counts.
        change = ((live_obs - reset_obs) / scale).asinh()
        return encoded + self.live_projection(change)

    def decode(self, observations, shifted_actions, caches=None, position=None):
        x = self.action_encoder(shifted_actions)
        for index, block in enumerate(self.decoder):
            cache = None if caches is None else caches[index]
            x = block(x, observations, cache, position)
        return self.policy_head(x)

    def forward(self, reset_obs, live_obs, actions, action_masks):
        """Evaluate stored conditional probabilities in parallel using shifted actions."""
        encoded, scale = self.encode(reset_obs)
        # Add live observations after fleet attention to keep later observations out of earlier tokens.
        encoded = self.add_live_observation(encoded, reset_obs, live_obs, scale)
        previous_actions = F.one_hot(actions[:, :-1], self.action_size)
        shifted_actions = F.pad(previous_actions, (1, 0, 1, 0)).to(encoded.dtype)
        shifted_actions[:, 0, 0] = 1.0
        logits = self.decode(encoded, shifted_actions)
        policy = Categorical(logits=logits.masked_fill(~action_masks, -torch.inf))
        return policy.log_prob(actions), self.value_head(encoded).squeeze(-1), policy.entropy()

    def start_decoding(self, encoded):
        batch_size, num_agents, _ = encoded.shape
        # One key/value cache for each of the two causal attentions per decoder block.
        caches = encoded.new_empty(
            (len(self.decoder), 2, 2, batch_size, self.n_head, num_agents, self.n_embd // self.n_head)
        )
        token = encoded.new_zeros((batch_size, 1, self.action_size + 1))
        token[:, 0, 0] = 1.0
        return caches, token

    def act(self, observation, action_mask, token, caches, position, deterministic=False):
        logits = self.decode(observation, token, caches, position).squeeze(1)
        logits = logits.masked_fill(~action_mask, -torch.inf)
        policy = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else policy.sample()
        token = F.pad(F.one_hot(action, self.action_size), (1, 0)).unsqueeze(1).to(observation.dtype)
        return action, policy.log_prob(action), token

    def value_stats(self):
        mean, mean_sq = self.value_moments[:2] / self.value_moments[2].clamp(min=1e-5)
        return mean, (mean_sq - mean.square()).clamp(min=1e-2).sqrt()


def learn(model, optimizer, rollout, action_masks, params):
    reset_obs, live_obs, actions, old_log_probs, old_values, returns = map(torch.cat, zip(*rollout))
    # Every route decision uses the undiscounted terminal team return and its own baseline.
    mean, scale = model.value_stats()
    advantages = returns - (old_values * scale + mean)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-5)
    clip_eps = params["clip_eps"]
    losses = []

    for _ in range(params["num_epochs"]):
        indices = torch.randperm(len(rollout), device=reset_obs.device)
        # Every sample is a complete day in its rollout order, including all AV tokens.
        for batch in indices.split(params["batch_size"]):
            log_probs, values, entropy = model(reset_obs[batch], live_obs[batch], actions[batch], action_masks)
            ratio = (log_probs - old_log_probs[batch]).exp()
            advantage = advantages[batch]
            policy_loss = -torch.minimum(
                ratio * advantage, ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage
            ).mean()

            # As in the reference trainer, normalize targets and clip the value update.
            rewards = returns[batch]
            moments = torch.stack((rewards.mean(), rewards.square().mean(), rewards.new_ones(())))
            model.value_moments.lerp_(moments, 1e-5)
            mean, scale = model.value_stats()
            targets = ((rewards - mean) / scale).expand_as(values)
            clipped_values = old_values[batch] + (values - old_values[batch]).clamp(-clip_eps, clip_eps)
            # Use standard Huber loss on both tails of the value error.
            value_loss = torch.maximum(
                F.huber_loss(values, targets, reduction="none", delta=10.0),
                F.huber_loss(clipped_values, targets, reduction="none", delta=10.0),
            ).mean()

            entropy = entropy.mean()
            loss = policy_loss + params["value_loss_coef"] * value_loss - params["entropy_coef"] * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), params["max_grad_norm"])
            optimizer.step()
            losses.append(torch.stack((loss, policy_loss, value_loss, entropy)).detach())

    return torch.stack(losses).mean(dim=0).tolist()


@torch.no_grad()
def run_episode(env, model, agent_ids, action_masks, device, prepare_day, deterministic=False):
    observations, _ = env.reset()
    prepare_day()
    reset_obs = torch.as_tensor(
        np.stack([observations[agent_id] for agent_id in agent_ids]), dtype=torch.float32, device=device,
    ).unsqueeze(0)
    encoded, scale = model.encode(reset_obs)
    caches, token = model.start_decoding(encoded)
    live_obs = torch.empty_like(reset_obs)
    actions = torch.empty(reset_obs.shape[:2], dtype=torch.long, device=device)
    log_probs = reset_obs.new_empty(actions.shape)
    position = 0
    rewards = {}
    for agent_id in env.agent_iter():
        _, reward, termination, truncation, _ = env.last(observe=False)
        if termination or truncation:
            rewards[agent_id] = reward
            env.step(None)
        else:
            if position >= len(agent_ids) or agent_id != agent_ids[position]:
                raise RuntimeError("RouteRL agent order differs from the MAT sequence")
            observation = np.array(env.observe(agent_id), copy=True)
            live_obs[0, position] = torch.as_tensor(observation, dtype=reset_obs.dtype, device=device)
            current = slice(position, position + 1)
            conditioned = model.add_live_observation(
                encoded[:, current], reset_obs[:, current], live_obs[:, current], scale,
            )
            action, log_prob, token = model.act(
                conditioned, action_masks[:, position], token, caches, position, deterministic=deterministic,
            )
            actions[:, position] = action
            log_probs[:, position] = log_prob
            env.step(action.item(), decision_observation=observation)
            position += 1
    if position != len(agent_ids):
        raise RuntimeError("RouteRL finished before every MAT agent chose a route")
    conditioned = model.add_live_observation(encoded, reset_obs, live_obs, scale)
    values = model.value_head(conditioned).squeeze(-1)
    returns = reset_obs.new_tensor([[np.mean([rewards[agent_id] for agent_id in agent_ids])]])
    if not torch.isfinite(returns).all():
        raise ValueError(f"Non-finite MAT team return on day {env.day}")
    return reset_obs, live_obs, actions, log_probs, values, returns


def main(script_file=__file__, prepare_humans=None):
    os.chdir(Path(__file__).resolve().parent)
    auto_routed = prepare_humans is not None
    algorithm = "mat_arh" if auto_routed else "mat"
    parser = argparse.ArgumentParser(description="Run MAT with clustered AV routes.")
    parser.add_argument("--id", required=True)
    parser.add_argument("--project")
    parser.add_argument("--env-conf", default="clusters-eta-mask")
    parser.add_argument("--task-conf", required=True)
    parser.add_argument("--alg-conf", required=True)
    parser.add_argument("--net", required=True)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--torch-seed", type=int, default=42)
    parser.add_argument("--route-set", help="Named clustered route set. Uses the network default when omitted.")
    if not auto_routed:
        parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    add_model_snapshot_argument(parser)
    args = parser.parse_args()

    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {algorithm.upper()} | Experiment ID: {args.id} | Network: {args.net}")
    print(f"Environment seed: {args.env_seed} | Torch seed: {args.torch_seed}")
    print(f"Algorithm config: {args.alg_conf} | Environment config: {args.env_conf} | Task config: {args.task_conf}")

    random.seed(args.env_seed)
    np.random.seed(args.env_seed)
    torch.manual_seed(args.torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    params = {}
    for directory, name in (("algo_config/mat", args.alg_conf),
                            ("env_config", args.env_conf), ("task_config", args.task_conf)):
        with open(f"../config/{directory}/{name}.json", encoding="utf-8") as file:
            params.update(json.load(file))
    params.pop("desc", None)
    if min(params[key] for key in ("update_every", "batch_size", "num_epochs")) < 1:
        raise ValueError("MAT rollout length, batch size and PPO epochs must be positive")

    network_folder = Path("../networks") / args.net
    results_folder = Path(os.environ.get("RESULTS_BASE_DIR", "../results"))
    records_folder = results_folder / args.id
    plots_folder = records_folder / "plots"
    route_set = resolve_route_set(args.net, args.route_set)
    route_set_dir = validate_clustered_route_set(args.net, network_folder / "clustered_routes" / route_set)
    shuffle = getattr(args, "shuffle", False)

    od_data = ast.literal_eval((network_folder / f"od_{args.net}.txt").read_text(encoding="utf-8"))
    origins, destinations = od_data["origins"], od_data["destinations"]
    agents_path = network_folder / "agents.csv"
    agents = pd.read_csv(agents_path)
    num_machines = int(len(agents) * params["ratio_machines"])
    if num_machines == 0 and not (agents["kind"] == "AV").any():
        raise ValueError("MAT needs at least one AV after mutation")

    loader = ClusteredRoutesLoader(args.net, network_folder, shuffle, args.env_seed, route_set_dir=route_set_dir)
    number_of_paths = loader.get_number_of_paths()
    masks_by_od = loader.create_masks(origins, destinations)
    for od in set(agents[["origin", "destination"]].itertuples(index=False, name=None)):
        mask = masks_by_od.get(od)
        if mask is None or mask.shape != (number_of_paths,) or not np.isin(mask, (0, 1)).all() or not mask.any():
            raise ValueError(f"Invalid or missing clustered action mask for demand OD {od}")

    records_folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(agents_path, records_folder / "agents.csv")
    loader.export_paths_routes(str(records_folder), origins, destinations)

    config = dict(params, network=args.net, env_seed=args.env_seed, torch_seed=args.torch_seed,
                  env_config=args.env_conf, task_config=args.task_conf, alg_config=args.alg_conf,
                  algorithm=algorithm, script=script_path_for_config(script_file),
                  num_agents=len(agents), num_machines=num_machines, route_set=route_set,
                  number_of_paths=number_of_paths, use_clustered_routes=True, use_action_masks=True,
                  shuffle=shuffle, parameter_sharing=True, team_reward="mean_av_reward",
                  observation_timing="joint_reset_and_agent_turn")
    if args.project is not None:
        config["project"] = args.project
    if args.save_model_every is not None:
        config["save_model_every"] = args.save_model_every

    human_days = params["human_learning_episodes"]
    training_days = params["training_eps"]
    testing_days = params["test_eps"]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]
    human_params = {"model": params["human_model"]}
    simulator_params = {"network_name": args.net, "custom_network_folder": str(network_folder), "sumo_type": "sumo"}
    if auto_routed:
        catalogs, background_ids = prepare_humans(network_folder, records_folder, agents, config)
        simulator_params["simulation_timesteps"] = config["simulation_timesteps"]
        phase_names[0] = "Auto-routed human baseline"
        for key in ("alpha", "beta", "beta_randomness", "deterministic"):
            human_params[key] = params[f"human_{key}"]
        if params["should_humans_adapt"]:
            print("Warning: should_humans_adapt is ignored by mat_arh because human routes are imported.")

    with (records_folder / "exp_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    env = MATEnvironment(
        seed=args.env_seed,
        create_agents=False,
        create_paths=False,
        action_masks=masks_by_od,
        save_detectors_info=False,
        agent_parameters={
            "new_machines_after_mutation": num_machines,
            "human_parameters": human_params,
            "machine_parameters": {"behavior": params["av_behavior"], "observation_type": params["observations"]},
        },
        environment_parameters={"save_every": params["save_every"]},
        simulator_parameters=simulator_params,
        plotter_parameters={
            "phases": [1, human_days, human_days + training_days],
            "phase_names": phase_names,
            "smooth_by": params["smooth_by"],
            "plot_choices": params["plot_choices"],
            "records_folder": str(records_folder),
            "plots_folder": str(plots_folder),
        },
        path_generation_parameters={"origins": origins, "destinations": destinations, "number_of_paths": number_of_paths},
    )

    if auto_routed:
        from iql_arh import install_auto_routed_humans, sample_human_routes

        install_auto_routed_humans(env, catalogs, background_ids)
        env.human_learning = False

    def prepare_day():
        if auto_routed:
            sample_human_routes(env, catalogs, masks_by_od, args.env_seed, env.day)

    env.start()
    loss_records = []
    try:
        env.reset()
        print_agent_counts(env)
        with tqdm(total=human_days + training_days + testing_days, desc=phase_names[0]) as pbar:
            for _ in range(human_days):
                prepare_day()
                env.step()
                pbar.update()

            env.mutation(disable_human_learning=not config["should_humans_adapt"], mutation_start_percentile=-1)
            print_agent_counts(env)
            agent_ids = env.possible_agents
            agents_by_id = {str(agent.id): agent for agent in env.machine_agents}
            action_masks = torch.as_tensor(
                np.stack([masks_by_od[(agents_by_id[aid].origin, agents_by_id[aid].destination)] for aid in agent_ids]),
                dtype=torch.bool, device=device,
            ).unsqueeze(0)
            model = MAT(env.observation_space(agent_ids[0]).shape[0], number_of_paths,
                        params["n_embd"], params["n_head"], params["n_block"]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], eps=1e-5)
            rollout = []

            pbar.set_description("AV learning")
            for episode in range(1, training_days + 1):
                rollout.append(run_episode(env, model, agent_ids, action_masks, device, prepare_day))
                if len(rollout) == params["update_every"] or episode == training_days:
                    loss, policy_loss, value_loss, entropy = learn(model, optimizer, rollout, action_masks, params)
                    loss_records.append(dict(iteration=len(loss_records) + 1, agent_id="shared", loss=loss,
                                             episode=episode, policy_loss=policy_loss, value_loss=value_loss,
                                             entropy=entropy))
                    rollout.clear()

                if should_save_model_snapshot(episode, training_days, args.save_model_every):
                    torch.save({"training_episode": episode, "model": model.state_dict(), "agent_ids": agent_ids},
                               model_snapshot_path(str(records_folder), episode, "pt"))
                if (episode - 1) % params["plot_every"] == 0:
                    env.plot_results()
                pbar.update()

            pbar.set_description("Testing")
            model.eval()
            for _ in range(testing_days):
                run_episode(env, model, agent_ids, action_masks, device, prepare_day, deterministic=True)
                pbar.update()
    finally:
        env.stop_simulation()

    env.plot_results()
    save_loss_records(str(records_folder), loss_records,
                      columns=["iteration", "agent_id", "loss", "episode", "policy_loss", "value_loss", "entropy"])
    clear_SUMO_files(records_folder / "SUMO_output", records_folder / "episodes", remove_additional_files=True)
    if not args.skip_metrics:
        run_metrics_analysis(args.id, results_folder=str(results_folder))


if __name__ == "__main__":
    main()
