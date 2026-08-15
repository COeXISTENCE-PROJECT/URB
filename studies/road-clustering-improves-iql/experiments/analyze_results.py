"""Summarize the predeclared paired road-clustering experiment."""

from pathlib import Path

import pandas as pd


STUDY_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = STUDY_DIR / "results"
SEEDS = (40, 41, 42)
CONTROLLERS = ("aon", "iql")
ROUTE_KINDS = ("janux", "clustered")
METRICS = ("t_CAV", "t_test", "t_HDV_test", "Effect_of_change", "cost_of_learning")


def experiment_id(seed: int, controller: str, route_kind: str) -> str:
    return f"rciql_ing2_s{seed}_{controller}_{route_kind}"


def load_results() -> pd.DataFrame:
    rows = []
    missing = []

    for seed in SEEDS:
        for controller in CONTROLLERS:
            for route_kind in ROUTE_KINDS:
                exp_id = experiment_id(seed, controller, route_kind)
                metrics_path = RESULTS_DIR / exp_id / "metrics" / "BenchmarkMetrics.csv"
                if not metrics_path.is_file():
                    missing.append(metrics_path)
                    continue

                metrics = pd.read_csv(metrics_path)
                if len(metrics) != 1:
                    raise ValueError(f"Expected one metrics row in {metrics_path}, found {len(metrics)}")

                row = {
                    "seed": seed,
                    "controller": controller,
                    "route_kind": route_kind,
                }
                row.update(metrics.loc[0, list(METRICS)].to_dict())
                rows.append(row)

    if missing:
        missing_text = "\n".join(f"- {path.relative_to(STUDY_DIR)}" for path in missing)
        raise SystemExit(f"Study is incomplete. Missing metrics:\n{missing_text}")

    return pd.DataFrame(rows)


def paired_route_effect(data: pd.DataFrame, controller: str) -> pd.DataFrame:
    subset = data[data["controller"] == controller]
    paired = subset.pivot(index="seed", columns="route_kind", values="t_CAV")
    paired["delta_minutes"] = paired["clustered"] - paired["janux"]
    paired["delta_percent"] = 100 * paired["delta_minutes"] / paired["janux"]
    return paired


def main() -> None:
    data = load_results()
    print("Per-run benchmark metrics")
    print(data.to_string(index=False))

    for controller in CONTROLLERS:
        paired = paired_route_effect(data, controller)
        print(f"\nPaired clustered - JanuX effect for {controller.upper()}")
        print(paired.to_string())
        print(
            "mean delta: "
            f"{paired['delta_minutes'].mean():.4f} min "
            f"({paired['delta_percent'].mean():.2f}%)"
        )

    cav = data.pivot(index="seed", columns=["controller", "route_kind"], values="t_CAV")
    gain_janux = cav["aon", "janux"] - cav["iql", "janux"]
    gain_clustered = cav["aon", "clustered"] - cav["iql", "clustered"]
    interaction = gain_clustered - gain_janux
    decomposition = pd.DataFrame(
        {
            "iql_gain_janux": gain_janux,
            "iql_gain_clustered": gain_clustered,
            "learning_interaction": interaction,
        }
    )
    print("\nAON-control decomposition")
    print(decomposition.to_string())
    print(f"mean learning interaction: {interaction.mean():.4f} min")


if __name__ == "__main__":
    main()
