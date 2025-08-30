import random
from typing import Optional, Literal

from statistics import mean

from routerl import Keychain as kc




#############################################
# Initializing & updating experiment records
#############################################


def initialize_route_records(free_flow_time: float)->dict:
    """
    Initialize a route records dictionary.

    Args:
        free_flow_time (float): route free-flow travel time.

    Returns:
        dict: traffic records dictionary for route with the following keys:
            - free_flow (float): free flow travel time for route (available in TrafficEnvironment object).
            - min_duration (float | None): shortest observed duration during the experiment.
            - min_duration_ep (float | None): episode index when min_duration was observed.
            - n_drives (int | None): total number of recorded route choices (all AV agents across all episodes).
            - latest_duration (float | None): duration of the most recent drive on this route.
            - latest_ep (float | None): episode index of the most recent drive on this route.
    """

    return {
        'free_flow': free_flow_time,
        'n_drives': 0,
        'min_duration': None,
        'min_duration_ep': None,
        'latest_duration': None,
        'latest_ep': None
    }


def initialize_od_timestamp_records(od_free_flows: dict[tuple[int,int],list])->dict:
    return { i: initialize_route_records(ff_time) for i,ff_time in enumerate(od_free_flows)}


def initialize_experiment_records(traffic_environment)->dict:
    experiment_records = dict()
    free_flows = traffic_environment.get_free_flow_times() # {(origin, destination): [t0,t1,..tn]}
    
    # Initialize records for each (origin-destination)-timepoint combination present in agents
    for agent in traffic_environment.machine_agents:
        od = (agent.origin, agent.destination)
        timestamp = agent.start_time

        if timestamp not in experiment_records.setdefault(od, dict()):
            origin_destination_free_flows = free_flows[od]
            experiment_records[od][timestamp] = initialize_od_timestamp_records(od_free_flows=origin_destination_free_flows)

    return experiment_records


######  Updating traffic records  ######

def update_records(od, timestamp, route:int, duration:float, episode:int, records:dict):
    """
    Update origin-destination records for given timestamp and route with latest travel duration and episode.
    """
    rr = records[od][timestamp][route]

    # Update latest record values
    rr['latest_duration'] = duration
    rr['latest_ep'] = episode
    rr['n_drives'] += 1


    # Update if new minimal duration achieved
    if rr['min_duration'] is None or duration < rr['min_duration']:
        rr['min_duration'] = duration
        rr['min_duration_ep'] = episode
    return



#############################################
# Accessing records & choosing agent action
#############################################

def _min_route_duration(records, od, timestamp, route)->float:
    """
    Return the shortest reported time for a route if available. 
    If no time is reported, return the route's free flow time.
    """
    min_duration = records[od][timestamp][route]['min_duration']
    if min_duration is None:
        min_duration = records[od][timestamp][route]['free_flow']
    return min_duration

def route_with_smallest_min_time(records, od, timestamp)->int:
    estimated_min_times = [
        _min_route_duration(records=records, od=od, timestamp=timestamp, route=i)
        for i in range(len(records[od][timestamp]))
    ]
    #print(f"Estimated min times for od={od} routes: {estimated_min_times}")
    min_route = random.choice([rou for rou, time_est in enumerate(estimated_min_times) if time_est == (min_time := min(estimated_min_times))])
    #print(f"Route with minimal time for od {od}: {min_route}")
    return min_route

def select_agent_action(agent: 'Agent', records:dict)->int:
    od = (agent.origin, agent.destination)
    timestamp = agent.start_time
    action = route_with_smallest_min_time(records, od, timestamp)
    return action