"""Run clustered MAT with per-agent imported human routes and background buses."""

import xml.etree.ElementTree as ET

from mat import main


def prepare_auto_routed_humans(network_folder, records_folder, agents, config):
    from iql_arh import load_human_route_catalogs

    demand_routes_path = network_folder / "demand_routes.csv"
    catalogs = load_human_route_catalogs(demand_routes_path, agents["id"])
    background_path = network_folder / "background_buses.rou.xml"
    background_root = ET.parse(background_path).getroot()
    background_vehicles = background_root.findall("vehicle")
    background_ids = {vehicle.get("id") for vehicle in background_vehicles}
    if len(background_ids) != len(background_vehicles) or None in background_ids:
        raise ValueError("Background bus IDs are missing or duplicated")
    if background_ids.intersection(agents["id"].astype(str)):
        raise ValueError("Background bus IDs overlap with RouteRL agent IDs")

    route_path = records_folder / "route.rou.xml"
    route_tree = ET.parse(route_path)
    route_tree.getroot().extend(background_root)
    ET.indent(route_tree, space="  ")
    route_tree.write(route_path, encoding="utf-8", xml_declaration=True)

    last_bus_departure = max((float(vehicle.get("depart")) for vehicle in background_vehicles), default=0)
    config.update(
        human_learning=False,
        human_routing="per_agent_imported_distribution",
        demand_routes_file=str(demand_routes_path.relative_to("..")),
        background_traffic_file=str(background_path.relative_to("..")),
        background_bus_count=len(background_ids),
        should_humans_adapt=False,
        simulation_timesteps=max(180, int(agents["start_time"].max()) + 1, int(last_bus_departure) + 1),
    )
    return catalogs, background_ids


if __name__ == "__main__":
    main(__file__, prepare_auto_routed_humans)
