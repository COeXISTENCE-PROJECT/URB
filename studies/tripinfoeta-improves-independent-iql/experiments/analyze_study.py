import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from iql_eta_ablation import Network


SCRIPT_DIR = Path(__file__).resolve().parent
SEEDS = [40, 41, 42]


def run_id(mode, seed):
    return f"eta_iql_ing2_{mode}_s{seed}"


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def human_episode_hash(run_folder, last_human_day):
    digest = hashlib.sha256()
    for path in sorted(
        run_folder.glob("episodes/ep*.csv"),
        key=lambda item: int(item.stem[2:]),
    ):
        if int(path.stem[2:]) <= last_human_day:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def cav_ids(run_folder):
    paths = sorted(
        run_folder.glob("episodes/ep*.csv"),
        key=lambda item: int(item.stem[2:]),
    )
    frame = pd.read_csv(paths[-1], usecols=["id", "kind"])
    return sorted(frame.loc[frame["kind"] == "AV", "id"].astype(int).tolist())


def load_run(results_folder, mode, seed):
    exp_id = run_id(mode, seed)
    folder = results_folder / exp_id
    config = read_json(folder / "exp_config.json")
    daily = pd.read_csv(folder / "metrics" / "DailyMetrics.csv")
    training = daily[daily["phase"] == "training"]
    testing = daily[daily["phase"] == "testing"]
    if len(training) != config["training_eps"] or len(testing) != config["test_eps"]:
        raise ValueError(f"{exp_id} has incomplete daily metrics")

    row = {
        "exp_id": exp_id,
        "mode": mode,
        "seed": seed,
        "t_CAV_test": testing["t_CAV"].mean(),
        "t_HDV_test": testing["t_HDV"].mean(),
        "t_system_test": testing["t_system"].mean(),
        "t_CAV_training": training["t_CAV"].mean(),
        "t_CAV_training_first_250": training.iloc[:250]["t_CAV"].mean(),
        "t_CAV_training_final_250": training.iloc[-250:]["t_CAV"].mean(),
    }
    benchmark = pd.read_csv(folder / "metrics" / "BenchmarkMetrics.csv").iloc[0]
    for name in ["t_CAV", "t_HDV_test", "t_test", "CAV_advantage", "cost_of_learning"]:
        row[f"benchmark_{name}"] = benchmark[name]
    return folder, config, row


def sensitivity(run_folder, config, seed):
    test_start = config["human_learning_episodes"] + config["training_eps"] + 1
    frames = []
    for path in sorted(
        run_folder.glob("episodes/ep*.csv"),
        key=lambda item: int(item.stem[2:]),
    ):
        if int(path.stem[2:]) >= test_start:
            frame = pd.read_csv(
                path,
                usecols=["id", "kind", "origin", "destination", "start_time", "observation"],
            )
            frames.append(frame[frame["kind"] == "AV"])
    test_data = pd.concat(frames, ignore_index=True)

    routes = pd.read_csv(run_folder / "routes.csv")
    freeflows = {
        (int(origin), int(destination)): group["free_flow_time"].to_numpy(np.float32)
        for (origin, destination), group in routes.groupby(
            ["origins", "destinations"], sort=False
        )
    }
    policies = torch.load(
        run_folder / "models" / "policies.pt", map_location="cpu", weights_only=True
    )

    rows = []
    for agent_id, agent_data in test_data.groupby("id", sort=False):
        observations = np.stack(
            agent_data["observation"].map(lambda value: np.fromstring(value, sep=","))
        ).astype(np.float32)
        origin = int(agent_data["origin"].iloc[0])
        destination = int(agent_data["destination"].iloc[0])
        fixed = observations.copy()
        fixed[:, : config["number_of_paths"]] = freeflows[(origin, destination)]

        model = Network(
            observations.shape[1],
            config["number_of_paths"],
            config["num_hidden"],
            config["widths"],
        )
        model.load_state_dict(policies[str(int(agent_id))])
        model.eval()
        with torch.no_grad():
            q_dynamic = model(torch.as_tensor(observations)).numpy()
            q_fixed = model(torch.as_tensor(fixed)).numpy()

        eta_changed = np.any(
            ~np.isclose(
                observations[:, : config["number_of_paths"]],
                fixed[:, : config["number_of_paths"]],
            ),
            axis=1,
        )
        action_changed = np.argmax(q_dynamic, axis=1) != np.argmax(q_fixed, axis=1)
        rows.append(
            {
                "seed": seed,
                "agent_id": int(agent_id),
                "start_time": int(agent_data["start_time"].iloc[0]),
                "eta_changed_fraction": eta_changed.mean(),
                "action_changed_fraction": action_changed.mean(),
                "mean_absolute_q_change": np.abs(q_dynamic - q_fixed).mean(),
            }
        )

    by_agent = pd.DataFrame(rows).sort_values(["start_time", "agent_id"]).reset_index(drop=True)
    group_names = np.array(["early", "middle", "late"])
    by_agent["departure_group"] = group_names[
        np.minimum(2, np.arange(len(by_agent)) * 3 // len(by_agent))
    ]
    summary = (
        by_agent.groupby("departure_group", as_index=False)[
            ["eta_changed_fraction", "action_changed_fraction", "mean_absolute_q_change"]
        ]
        .mean()
        .assign(seed=seed)
    )
    overall = pd.DataFrame(
        [{
            "departure_group": "all",
            "eta_changed_fraction": by_agent["eta_changed_fraction"].mean(),
            "action_changed_fraction": by_agent["action_changed_fraction"].mean(),
            "mean_absolute_q_change": by_agent["mean_absolute_q_change"].mean(),
            "seed": seed,
        }]
    )
    return by_agent, pd.concat([overall, summary], ignore_index=True)


def analyze(results_folder, evidence_folder):
    evidence_folder.mkdir(parents=True, exist_ok=True)
    runs = {}
    run_rows = []
    for seed in SEEDS:
        for mode in ["dynamic", "static"]:
            runs[(mode, seed)] = load_run(results_folder, mode, seed)
            run_rows.append(runs[(mode, seed)][2])

    checks = []
    pairs = []
    sensitivity_agents = []
    sensitivity_summaries = []
    for seed in SEEDS:
        dynamic_folder, dynamic_config, dynamic = runs[("dynamic", seed)]
        static_folder, static_config, static = runs[("static", seed)]
        dynamic_invariants = {key: value for key, value in dynamic_config.items() if key != "eta_mode"}
        static_invariants = {key: value for key, value in static_config.items() if key != "eta_mode"}
        check = {
            "seed": seed,
            "routes_match": file_hash(dynamic_folder / "routes.csv") == file_hash(static_folder / "routes.csv"),
            "route_xml_matches": file_hash(dynamic_folder / "route.rou.xml") == file_hash(static_folder / "route.rou.xml"),
            "human_learning_matches": human_episode_hash(dynamic_folder, dynamic_config["human_learning_episodes"])
            == human_episode_hash(static_folder, static_config["human_learning_episodes"]),
            "cav_ids_match": cav_ids(dynamic_folder) == cav_ids(static_folder),
            "settings_match": dynamic_invariants == static_invariants,
        }
        check["fair_pair"] = all(value for key, value in check.items() if key != "seed")
        checks.append(check)

        pairs.append(
            {
                "seed": seed,
                "t_CAV_dynamic": dynamic["t_CAV_test"],
                "t_CAV_static": static["t_CAV_test"],
                "test_improvement_percent": 100 * (static["t_CAV_test"] - dynamic["t_CAV_test"]) / static["t_CAV_test"],
                "learning_improvement_percent": 100 * (static["t_CAV_training"] - dynamic["t_CAV_training"]) / static["t_CAV_training"],
                "early_250_improvement_percent": 100 * (static["t_CAV_training_first_250"] - dynamic["t_CAV_training_first_250"]) / static["t_CAV_training_first_250"],
                "final_250_improvement_percent": 100 * (static["t_CAV_training_final_250"] - dynamic["t_CAV_training_final_250"]) / static["t_CAV_training_final_250"],
            }
        )
        by_agent, summary = sensitivity(dynamic_folder, dynamic_config, seed)
        sensitivity_agents.append(by_agent)
        sensitivity_summaries.append(summary)

    checks = pd.DataFrame(checks)
    pairs = pd.DataFrame(pairs)
    test_effects = pairs["test_improvement_percent"]
    learning_effects = pairs["learning_improvement_percent"]
    fair = checks["fair_pair"].all()

    if fair and test_effects.mean() >= 1 and (test_effects > 0).all():
        assessment = "supported"
    elif fair and test_effects.mean() <= 0 and (test_effects <= 0).sum() >= 2:
        assessment = "falsified"
    else:
        assessment = "inconclusive"
    if fair and learning_effects.mean() >= 1 and (learning_effects > 0).all():
        learning_assessment = "better"
    elif fair and learning_effects.mean() <= -1 and (learning_effects < 0).all():
        learning_assessment = "worse"
    else:
        learning_assessment = "unclear_or_practically_similar"

    decision = {
        "assessment": assessment,
        "learning_assessment": learning_assessment,
        "mean_test_improvement_percent": float(test_effects.mean()),
        "pairs_favoring_dynamic": int((test_effects > 0).sum()),
        "mean_learning_improvement_percent": float(learning_effects.mean()),
        "all_pairs_fair": bool(fair),
    }
    pd.DataFrame(run_rows).to_csv(evidence_folder / "run_metrics.csv", index=False)
    checks.to_csv(evidence_folder / "pair_checks.csv", index=False)
    pairs.to_csv(evidence_folder / "pair_results.csv", index=False)
    pd.concat(sensitivity_agents).to_csv(evidence_folder / "sensitivity_by_agent.csv", index=False)
    pd.concat(sensitivity_summaries).to_csv(evidence_folder / "sensitivity_summary.csv", index=False)
    with open(evidence_folder / "decision.json", "w", encoding="utf-8") as file:
        json.dump(decision, file, indent=4)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-folder", default=str(SCRIPT_DIR / "results"))
    parser.add_argument("--evidence-folder", default=str(SCRIPT_DIR / "evidence"))
    args = parser.parse_args()
    analyze(Path(args.results_folder), Path(args.evidence_folder))
