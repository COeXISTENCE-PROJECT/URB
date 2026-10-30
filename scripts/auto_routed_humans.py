"""Imported human route distributions shared by IQL and centralized routing."""

import os
import xml.etree.ElementTree as ET
from types import MethodType

import numpy as np
import pandas as pd
from routerl.keychain import Keychain as kc


def load_human_route_catalogs(path, agent_ids):
    """Load and validate the per-agent route distributions produced by the extractor."""
    routes_df = pd.read_csv(path)
    required = {"agent_id", "route_id", "probability", "edges"}
    missing_columns = required.difference(routes_df.columns)
    if missing_columns:
        raise ValueError(
            f"{path} is missing columns: {', '.join(sorted(missing_columns))}"
        )

    catalogs = {}
    for agent_id, rows in routes_df.groupby("agent_id", sort=False):
        agent_id = int(agent_id)
        rows = rows.sort_values("route_id")
        route_ids = [int(route_id) for route_id in rows["route_id"]]
        if route_ids != list(range(len(route_ids))):
            raise ValueError(f"Route IDs are not contiguous for agent {agent_id}")

        probabilities = rows["probability"].to_numpy(dtype=float, copy=True)
        if not np.isfinite(probabilities).all() or (probabilities <= 0).any():
            raise ValueError(f"Invalid route probabilities for agent {agent_id}")
        probabilities /= probabilities.sum()

        catalogs[agent_id] = []
        for route_id, probability, edges in zip(
            route_ids, probabilities, rows["edges"]
        ):
            edge_list = tuple(str(edges).split())
            if not edge_list:
                raise ValueError(f"Empty route for agent {agent_id}")
            catalogs[agent_id].append(
                {
                    "route_id": route_id,
                    "probability": probability,
                    "edges": edge_list,
                }
            )

    expected_ids = {int(agent_id) for agent_id in agent_ids}
    actual_ids = set(catalogs)
    if expected_ids != actual_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            "demand_routes.csv does not match agents.csv "
            f"(missing IDs: {missing[:10]}, extra IDs: {extra[:10]})"
        )
    return catalogs


def install_auto_routed_humans(env, catalogs, background_ids):
    """Use imported routes while preserving RouteRL's normal episode columns.

    Human ``action`` is the imported route ID and ``cost_table`` is that
    agent's imported route-probability vector, ordered by route ID.
    """
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

        # This private field travels with the episode snapshot and is removed
        # before RouteRL writes the standard CSV schema.
        action["_arh_route_id"] = selected["route_id"]

    def record(environment, episode, episode_data, agents, detectors):
        # RouteRL uses the clustered action internally so its observations stay
        # valid. Only the saved copy receives the imported human route ID.
        for entry in episode_data:
            if entry.get(kc.AGENT_KIND) == kc.TYPE_HUMAN:
                entry[kc.ACTION] = entry.pop("_arh_route_id")
        for agent in agents:
            if agent.kind == kc.TYPE_HUMAN:
                agent.model.cost = probabilities[int(agent.id)]
        return regular_record(episode, episode_data, agents, detectors)

    def step(simulator):
        timestep, stopped, arrivals, teleported = regular_step()
        arrivals = [vehicle_id for vehicle_id in arrivals if vehicle_id not in background_ids]
        teleported = [vehicle_id for vehicle_id in teleported if vehicle_id not in background_ids]
        return timestep, stopped, arrivals, teleported

    env.simulator.add_vehicle = MethodType(add_vehicle, env.simulator)
    env.simulator.step = MethodType(step, env.simulator)
    env._record = MethodType(record, env)


def sample_human_routes(env, catalogs, action_masks, env_seed, episode):
    """Sample every current human route reproducibly for one episode."""
    selected_routes = {}
    for human in env.human_agents:
        options = catalogs[int(human.id)]
        probabilities = [option["probability"] for option in options]
        rng = np.random.default_rng(
            np.random.SeedSequence([int(env_seed), int(episode), int(human.id)])
        )
        selected_routes[int(human.id)] = options[
            int(rng.choice(len(options), p=probabilities))
        ]
        mask = action_masks[(int(human.origin), int(human.destination))]
        human.default_action = int(np.flatnonzero(mask)[0])

    env.simulator.auto_routed_human_routes = selected_routes


def prepare_imported_human_demand(network_folder, records_folder, agents, action_masks):
    """Validate imported demand and add background buses to exported AV routes.

    Return route catalogs, background IDs and the required simulation horizon.
    Call once, after exporting the clustered route XML and before starting SUMO.
    """
    catalogs = load_human_route_catalogs(
        os.path.join(network_folder, "demand_routes.csv"), agents["id"]
    )
    demand_ods = set(agents[["origin", "destination"]].itertuples(index=False, name=None))
    missing = sorted(demand_ods.difference(action_masks))
    empty = sorted(od for od in demand_ods if od in action_masks and not action_masks[od].any())
    if missing or empty:
        raise ValueError(
            "Clustered route set does not cover the demand "
            f"(missing ODs: {missing[:10]}, empty masks: {empty[:10]})"
        )

    background_path = os.path.join(network_folder, "background_buses.rou.xml")
    background_root = ET.parse(background_path).getroot()
    vehicles = background_root.findall("vehicle")
    background_ids = {vehicle.get("id") for vehicle in vehicles}
    if len(background_ids) != len(vehicles) or None in background_ids:
        raise ValueError("Background bus IDs are missing or duplicated")
    if background_ids.intersection(str(agent_id) for agent_id in agents["id"]):
        raise ValueError("Background bus IDs overlap with RouteRL agent IDs")

    horizon = max(180, int(agents["start_time"].max()) + 1)
    if vehicles:
        horizon = max(horizon, int(max(float(vehicle.get("depart")) for vehicle in vehicles)) + 1)

    route_path = os.path.join(records_folder, "route.rou.xml")
    route_tree = ET.parse(route_path)
    route_tree.getroot().extend(list(background_root))
    ET.indent(route_tree, space="  ")
    route_tree.write(route_path, encoding="utf-8", xml_declaration=True)
    return catalogs, background_ids, horizon


