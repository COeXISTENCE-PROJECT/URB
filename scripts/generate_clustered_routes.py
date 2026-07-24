import argparse
import os
from pathlib import Path
from path_clustering import generate_named_route_set

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
    args = parser.parse_args()

    network_folder = Path("../networks") / args.net
    if not network_folder.is_dir():
        raise FileNotFoundError(f"Network folder not found: {network_folder}")

    outputs = generate_named_route_set(
        network_folder=network_folder,
        route_set=args.route_set,
        config=resolve_config(args.config),
    )

    print(f"Route set directory: {outputs.representants.parent}")
    print(f"Representants: {outputs.representants}")
    print(f"Action masks: {outputs.action_masks}")
    print(f"Clustering config: {outputs.clustering_config}")
    if outputs.diagnostics is not None:
        print(f"Diagnostics: {outputs.diagnostics}")


if __name__ == "__main__":
    main()
