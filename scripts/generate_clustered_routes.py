import argparse
import json
import os
from pathlib import Path

import pandas as pd
from path_clustering import generate_named_route_set
from path_clustering.generation import make_route_id

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def resolve_config(config: str) -> Path | str:
    config_path = Path(config)
    if config_path.exists() or config_path.parent != Path("."):
        return config_path

    config_name = config if config.endswith(".json") else f"{config}.json"
    urb_config_path = Path("../config/clustering_config") / config_name
    if urb_config_path.is_file():
        return urb_config_path

    return config


def load_human_routes(network_folder: Path) -> pd.DataFrame | None:
    demand_path = network_folder / "demand_routes.csv"
    if not demand_path.is_file():
        return None

    try:
        demand = pd.read_csv(demand_path)
    except pd.errors.EmptyDataError:
        return None
    if demand.empty:
        return None

    if not {"edges", "free_flow_time"}.issubset(demand):
        raise ValueError("Invalid demand_routes.csv")

    routes = demand.copy()
    routes["edges"] = routes["edges"].map(lambda path: str(path).split())
    routes["origins"] = routes["edges"].str[0]
    routes["destinations"] = routes["edges"].str[-1]
    routes["path"] = routes["edges"].map(",".join)
    routes["free_flow_time"] = pd.to_numeric(routes["free_flow_time"])
    routes["free_flow_time"] /= 60.0  # Extractor uses seconds; RouteRL uses minutes.
    routes = routes.sort_values("free_flow_time").drop_duplicates(
        ["origins", "destinations"]
    )
    routes["route_id"] = [
        make_route_id(origin, destination, path)
        for origin, destination, path in routes[
            ["origins", "destinations", "path"]
        ].itertuples(index=False, name=None)
    ]
    routes["cluster"] = 0
    return routes[
        ["origins", "destinations", "path", "free_flow_time", "route_id", "cluster"]
    ]


def insert_human_routes(outputs, network_name: str, human_routes: pd.DataFrame) -> None:
    representatives = pd.read_csv(outputs.representants)
    masks = pd.read_csv(outputs.action_masks)

    mask_ods = set(
        masks[["origins", "destinations"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    human_ods = set(
        human_routes[["origins", "destinations"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    if mask_ods != human_ods:
        raise ValueError("Human and generated routes cover different OD pairs")

    representatives = representatives[representatives["cluster"] != 0]
    human_routes = human_routes.reindex(columns=representatives.columns)
    representatives = pd.concat([human_routes, representatives], ignore_index=True)
    representatives.sort_values(["origins", "destinations", "cluster"]).to_csv(
        outputs.representants, index=False
    )

    masks["mask_0"] = 1
    masks.to_csv(outputs.action_masks, index=False)

    metadata_path = outputs.representants.parent / f"{network_name}_route_set_config.json"
    with metadata_path.open() as file:
        metadata = json.load(file)
    metadata["human_route_action"] = 0
    with metadata_path.open("w") as file:
        json.dump(metadata, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="ingolstadt_custom")
    parser.add_argument("--route-set", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Config name in config/clustering_config, explicit path, or bundled path-clustering config.",
    )
    parser.add_argument(
        "--prepend-human-route",
        action="store_true",
        help="Use the lowest-free-flow imported human route as action 0.",
    )
    args = parser.parse_args()

    network_folder = Path("../networks") / args.net
    if not network_folder.is_dir():
        raise FileNotFoundError(f"Network folder not found: {network_folder}")

    human_routes = load_human_routes(network_folder) if args.prepend_human_route else None
    outputs = generate_named_route_set(
        network_folder=network_folder,
        route_set=args.route_set,
        config=resolve_config(args.config),
    )
    if human_routes is not None:
        insert_human_routes(outputs, args.net, human_routes)

    print(f"Route set directory: {outputs.representants.parent}")
    print(f"Representants: {outputs.representants}")
    print(f"Action masks: {outputs.action_masks}")
    print(f"Clustering config: {outputs.clustering_config}")
    if outputs.diagnostics is not None:
        print(f"Diagnostics: {outputs.diagnostics}")


if __name__ == "__main__":
    main()
