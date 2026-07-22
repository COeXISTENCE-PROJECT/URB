import os
import pandas as pd
import numpy as np
import torch
from gymnasium.spaces import Dict, MultiBinary
from pettingzoo.utils.wrappers import BaseWrapper

class ClusteredRoutesLoader:
    """
    Loads clustered route representatives and exports them in RouteRL/SUMO formats.
    """
    
    def __init__(self, network_name: str, network_folder: str, shuffle: bool = False, seed: int = 42):
        """
        Initialize the loader.
        
        Args:
            network_name: Name of the network (e.g., 'saint_arnoult')
            network_folder: Path to the network folder in URB
            shuffle: Randomly shuffles the clusters if True
        """
        self.network_name = network_name
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        self.clustering_csv_path = os.path.join(
            network_folder, "clustered_routes",
            f"{network_name}_clusters_representants.csv"
        )

        self.masks_csv_path = os.path.join(
            network_folder, "clustered_routes",
            f"{network_name}_action_masks.csv"
        )
        
        if not os.path.exists(self.clustering_csv_path):
            raise FileNotFoundError(
                f"[ClusteredRoutesLoader] Clustered routes not found at {self.clustering_csv_path}\n"
                f"Copy the clustering results to this location first."
            )
        
        self.df = pd.read_csv(self.clustering_csv_path)
        self._build_route_index()

        print(f"[ClusteredRoutesLoader] Loaded {len(self.routes_by_od)} OD pairs "
              f"with {self.num_clusters} routes each")

    def _build_route_index(self):
        """Build lookup dictionaries for routes."""
        self.routes_by_od = {}

        for (origin, dest), group in self.df.groupby(['origins', 'destinations']):
            group_sorted = group.sort_values('cluster')
            routes = {}
            for row in group_sorted.itertuples():
                path = str(row.path).split(',')
                fft = row.free_flow_time
                routes[int(row.cluster)] = (path, fft)

            if self.shuffle:
                valid_keys = list(routes.keys()) # non-masked keys, e.g. 0, 3, 4
                values = list(routes.values()) # corresponding routes+ffts e.g. A, B, C
                shuffled_values = self.rng.permutation(len(values)) # e.g. 2, 0, 1
                routes = {
                    valid_keys[i]: values[shuffled_values[i]]
                    for i in range(len(valid_keys))
                } # e.g. 0: C (2), 3: A (0), 4: B (1)

            self.routes_by_od[(str(origin), str(dest))] = routes

        self.num_clusters = self.df['cluster'].nunique()
        self.od_pairs = list(self.routes_by_od.keys()) 

    def get_number_of_paths(self) -> int:
        """Get number of route options per agent (= number of clusters)."""
        return self.num_clusters

    def create_masks(self, origins, destinations):
        action_masks = {}

        if not os.path.exists(self.masks_csv_path):
            raise FileNotFoundError(
                f"[ClusteredRoutesLoader] Action masks not found at {self.masks_csv_path}\n"
                f"Copy the action masks to this location first."
            )
        
        origin_to_id = {str(edge): idx for idx, edge in enumerate(origins)}
        destination_to_id = {str(edge): idx for idx, edge in enumerate(destinations)}

        masks_df = pd.read_csv(self.masks_csv_path)
        mask_cols = [c for c in masks_df.columns if c.startswith("mask")]

        missing = []
        for row in masks_df.itertuples():
            # integer tuples (not edge strings) to match agents.csv ids!
            origin_edge = str(row.origins)
            destination_edge = str(row.destinations)

            if origin_edge not in origin_to_id:
                missing.append(f"Origin {origin_edge}")
                continue
            if destination_edge not in destination_to_id:
                missing.append(f"Destination {destination_edge}")
                continue

            o_idx = origin_to_id[origin_edge]
            d_idx = destination_to_id[destination_edge]

            mask = np.array(
                [int(getattr(row, c)) for c in mask_cols], dtype=np.int8
            )
            action_masks[(o_idx, d_idx)] = mask

        if missing:
            print(f"[ClusteredRoutesLoader] WARNING - {len(missing)} mask edges not found in OD lists: {missing[:5]}...")

        print(f"[ClusteredRoutesLoader] Loaded action masks for {len(action_masks)} agents from {self.masks_csv_path}")
        return action_masks

    def export_to_paths_csv(self, output_path: str, origins: list[str], destinations: list[str]):
        """
        Export routes to RouteRL's paths.csv format.
        
        Args:
            output_path: Where to save the CSV (typically records_folder/paths.csv)
            origins: List of origin edge IDs (in index order from od_*.txt)
            destinations: List of destination edge IDs (in index order from od_*.txt)
        """
        rows = []

        for o_idx, origin in enumerate(origins):
            for d_idx, destination in enumerate(destinations):
                key = (str(origin), str(destination))
                if key not in self.routes_by_od:
                    continue

                routes = self.routes_by_od[key]
                for cluster, (path, fft) in routes.items():
                    rows.append({
                        'origins': o_idx,
                        'destinations': d_idx,
                        'path': ' '.join(path), # space-separated!
                        'free_flow_time': fft,
                        'cluster': cluster # NEW - for padding missing ffts
                    })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"[ClusteredRoutesLoader] Exported {len(rows)} routes to {output_path}")

    def export_to_sumo_rou_xml(self, output_path: str, origins: list[str], destinations: list[str]):
        """
        Export routes to SUMO .rou.xml format.
        
        RouteRL expects this file to be named "route.rou.xml" in the records folder.
        
        Args:
            output_path: Where to save the .rou.xml file (typically records_folder/route.rou.xml)
            origins: List of origin edge IDs (in index order)
            destinations: List of destination edge IDs (in index order)
        """
        with open(output_path, 'w') as f:
            f.write('<routes>\n')
            f.write('<vType id="Human" color="red" guiShape="passenger/sedan"/>\n')
            f.write('<vType id="AV" color="yellow"/>\n')

            route_count = 0
            for o_idx, origin in enumerate(origins):
                for d_idx, destination in enumerate(destinations):
                    key = (str(origin), str(destination))
                    if key not in self.routes_by_od:
                        continue
                    
                    routes = self.routes_by_od[key]
                    for cluster, (path, _) in routes.items():
                        route_id = f"{o_idx}_{d_idx}_{cluster}"
                        edges = ' '.join(path)
                        f.write(f'<route id="{route_id}" edges="{edges}"/>\n')
                        route_count += 1
            
            f.write('</routes>\n')
        
        print(f"[ClusteredRoutesLoader] Exported {route_count} routes to {output_path}")

    def export_paths_routes(self, records_folder: str, origins: list[str], destinations: list[str]):
        """
        Convenience method to export both files to the records folder.
        """
        os.makedirs(records_folder, exist_ok=True)
        
        # Export CSV as both "paths.csv" AND "routes.csv" (RouteRL might use either)
        for filename in ["paths.csv", "routes.csv"]:
            csv_path = os.path.join(records_folder, filename)
            self.export_to_paths_csv(csv_path, origins, destinations)
        
        # Export route.rou.xml
        rou_xml = os.path.join(records_folder, "route.rou.xml")
        self.export_to_sumo_rou_xml(rou_xml, origins, destinations)

class AVMaskWrapper(BaseWrapper):
    """
    Env wrapper adding action masks to normal observations.
    """
    def __init__(self, env, action_masks):
        super().__init__(env)
        self.action_masks = action_masks
        self.agent_mask_map = {}
        missing_agents = []
        for agent in env.machine_agents:
            mask = action_masks.get((agent.origin, agent.destination))
            if mask is None:
                missing_agents.append(f"{agent.id} ({agent.origin} -> {agent.destination})")
                continue
            self.agent_mask_map[str(agent.id)] = mask

        if missing_agents:
            raise ValueError(
                "Missing action masks for machine agents: "
                + ", ".join(missing_agents)
            )

    def observe(self, agent):
        obs = self.env.observe(agent)
        mask = self.agent_mask_map[str(agent)] # TorchRL passes the agent id string here, not the agent object
        mask = torch.as_tensor(mask, dtype=torch.bool)

        if isinstance(obs, dict):
            obs["action_mask"] = mask
            return obs
        
        return {
            "observation": obs,
            "action_mask": mask
        }

    def observation_space(self, agent):
        return Dict(
            {
                "observation": self.env.observation_space(agent),
                "action_mask": MultiBinary(self.env.action_space(agent).n),
            }
        )