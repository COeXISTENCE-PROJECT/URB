import random
from typing import Optional, Literal

from statistics import mean

from routerl import Keychain as kc


##############################################################
# TODO: 
#   - [x] incorporate in greedy.py
#   - [x] test, fix potential bugs
#   - [] fix plots (what's going on with no data for humans)
#   - [] geerate results for parameters
#       - [] same city, 'large' params
#       - [] different city
# (o co by mogło chodzić z 'greedy' bez historii??) - np. to samo, tylko naszymi records jest bieący dzień (i ew brak ograniczenia do 'common time'?) -> i wtedy mamy po prostu radsze estymacje dla prawdiwych danych)
##############################################################



# def initialize_records_dict(agents: list[int])->dict:
#     """
#     Initializes a dictionary for storing episode records indexed with agent ids (integers).
#     """
#     columns = ['kind', 'origin', 'destination', 'route', 'travel_time', 'time_start', 'time_end']
#     records = {aid : {key: None for key in columns} for aid in agents}
#     return records



def _in_range(value, low, high)->bool:
    if (low is not None) and (value < low):
        return False
    if (high is not None) and (value > high):
        return False
    return True


##############################################
# Subsetting episode and experiment records
##############################################

def episode_records_subset(
    episode_records: dict[int,dict],
    origin: Optional[int]=None,
    destination: Optional[int]=None,
    route: Optional[int] = None,
    time_start: tuple[Optional[int], Optional[int]] = (None, None),
    time_end: tuple[Optional[int], Optional[int]] = (None, None),
    kind: Literal['machine', 'human', 'all']='machine'
    )->dict:

    """
    Filter episode records by criteria:
      - origin, destination, route
      - time_start and time_end ranges
      - driver type ('machine', 'human', 'all')

    If a bound is None, it is ignored.

    'episode_records' is dict[int,dict] with per-agent records for given episode (day).
    """

    if kind != 'machine':
        raise NotImplementedError("Implemented only for machine records.")

    else:
        type_machine = kc.TYPE_MACHINE

        return {
            aid: record
            for aid, record in episode_records.items()

            if (origin is None or record['origin'] == origin)
            and (destination is None or record['destination'] == destination)
            and (route is None or record['route'] == route)

            and _in_range(record['time_start'], *time_start)
            and _in_range(record['time_end'], *time_end)

            and (kind=='all' or record['kind'] == type_machine)                   
        }

def experiment_records_subset(
    experiment_records: dict[int, dict[int,dict]],
    days: tuple[Optional[int], Optional[int]]=(None, None),
    origin: Optional[int]=None,
    destination: Optional[int]=None,
    route: Optional[int] = None,
    time_start: tuple[Optional[int], Optional[int]] = (None, None),
    time_end: tuple[Optional[int], Optional[int]] = (None, None),
    kind: Literal['machine', 'human', 'all'] = 'machine'
    )->dict:

    """
    Filter experiment traffic records (episode records grouped by day).
    - Restrict days range (if provided).
    - For each included day, apply `episode_records_subset`.
    """

    day_filtered_experiment_records = {
        day : records 
        for day, records in experiment_records.items()
        if _in_range(day, *days)
    }

    return {
        day : episode_records_subset(
                episode_records=day_records,
                origin=origin,
                destination=destination,
                route=route,
                time_start=time_start,
                time_end=time_end,
                kind=kind
                )
        for day, day_records in day_filtered_experiment_records.items()
    }


###########################################
# Auxiliary functions
###########################################

def _closest_timepoint_records(timepoint: int|float, episode_records:dict)->dict:
    """
    Get records whose 'time_start' is closest to the given timepoint.
    If multiple such records, all are returned.
    """
    delta_min = min( abs(record['time_start'] - timepoint) for record in episode_records.values() )
    return {
        aid : record
        for aid, record in episode_records.items()
        if abs(record['time_start'] - timepoint) == delta_min
    }


def _get_latest_nonempty_episode_records(experiment_records: dict[int, dict[int, dict]])->dict:
    """
    For given experiment records, return episode records for the latest episode (day) with nonempty records (used for filtered experiment data).

    experiment_records: {day_i: {agentid_j : {'travel_time': ..., ....}, ... }, ... }, may be {day_i : {}, ...} after filtering.
    """
    for day in sorted(experiment_records.keys(), reverse=True):  # check days in descending order
        episode_records = experiment_records[day]
        if episode_records:  # not empty
            return episode_records
    return {}  # if all empty


def _mean_travel_time(records: dict[int, dict])->float:
    """
    Return mean travel time for episode records subset.
    
    records: {agentid_1 : {'travel_time': ..., ....}, ... , agentid_k: {'travel_time': ..., ....}}
    """
    return mean(record['travel_time'] for record in records.values())


###########################################
# Route travel time estimation
###########################################

def estimate_agent_routes_travel_time(agent: 'Agent', day: int, experiment_records: dict, free_flow_times: dict, span:int|None) -> float:
    """
    Using driving history, estimate travel time for all routes available for an agent.

    Estimation order:
    1. Machine with exact start time.
    2. Last machine departed before agent (shared driving time).
    3. Free flow time.

    'day' is the current day of the experiment.
    'span' is the number of past days (episodes) to consider in estimation; span None means using all available history.
    """

    assert span is None or span > 0

    if span == 0:
        raise NotImplementedError("span=0 (current day) is not supported. Only span>=1 (past days) is implemented.")

    
    days_range = (None, day) if span is None else (max(0,day-span), day)

    #################################################################################
    # Filter experiment records to origin, destination and history span of interest
    #################################################################################
    od_history_filter = {
        'origin': agent.origin,
        'destination': agent.destination,
        'days': days_range,
        'kind': 'machine'
    }

    od_history_records = experiment_records_subset(
        experiment_records=experiment_records,
        **od_history_filter
    )
    # If none of days available, return free flow times
    if not od_history_records:
        print(f"No records for day={day}, span={span}. Returning free flow times for agent {agent.id}, for all routes.")
        return free_flow_times[(agent.origin, agent.destination)]

    ####################################################################
    # Estimate travel time for all routes available for given agent
    ####################################################################
    num_routes = len(free_flow_times[(agent.origin, agent.destination)])
    route_time_estimations = []

    for i in range(num_routes):
        time_est = estimate_agent_route_travel_time(
            agent=agent,
            route=i,
            day=day,
            od_history_experiment_records=od_history_records,
            free_flow_times=free_flow_times)
        route_time_estimations.append(time_est)

    return route_time_estimations

def estimate_agent_route_travel_time(agent: 'Agent', route: int, day: int, od_history_experiment_records: dict, free_flow_times: dict) -> float:
    """
    Estimate travel time on a route for an agent using past episode(s).

    Estimation order:
    1. Machine with exact start time.
    2. Last machine departed before agent (shared driving time).
    3. Free flow time.

    'day' is the current day of the experiment.
    """


    base_filter = {
        'route': route

        # filters applied in 'estimate_agent_routes_travel_time'
        # 'origin': agent.origin,
        # 'destination': agent.destination,
        # 'days': (min(0,day-span), day),
        # 'kind': 'machine'
    }

    # Define estimation filters 
    estimation_filters = [ # filters for time estimations
        {'time_start': (agent.start_time, agent.start_time)},
        {'time_start': (None, agent.start_time), 'time_end': (agent.start_time, None)}
    ]
    filter_descriptions = [
        "Exact start time.",
        "Closest deparing before, shared travel time."
    ]

    ##################################################################################
    # Estimate travel time - take the first available result in order of significance
    ##################################################################################

    # 1-2. Try to estimate route travel time from existing data (predefined cases)
    for estimation_args, descr in zip(estimation_filters,filter_descriptions):

        filter_args = base_filter.copy()
        filter_args.update(estimation_args)

        estimation_records = experiment_records_subset( # {day: {agentid: {'travel_time: ..., ...} } }
            experiment_records=od_history_experiment_records,
            **filter_args
        )



        # Take the records from the latest day possible (latest day in span range with nonempty data for current filtering); avg if >1 agent in such day
        latest_nonempty_episode_records = _get_latest_nonempty_episode_records(experiment_records=estimation_records)
        if latest_nonempty_episode_records: #{agentid_1: {....}, ...., agentid_k: {}}
            print(f"Estimation data for agent {agent.id}, route {route}: {descr}")
            return _mean_travel_time(latest_nonempty_episode_records)

   

    # 3. Free-flow fallback for empty results
    print(f"No prior driver data for this route ({route}). Best estimation for agent {agent.id}: free flow time.")
    return free_flow_times[(agent.origin, agent.destination)][route]


###########################################
# Action selection
###########################################


def choose_agent_action(agent: 'Agent', day: int, experiment_records: dict[int, dict[int,dict]], free_flow_times: dict, span:int)->int:
    """
    Selects route for the agent based on past episode(s) driving data using greedy strategy.
    """
 
    # Get travel time estimations for each route acrssible for the agent, choose one with minimal estimation
    num_routes = len(free_flow_times[(agent.origin, agent.destination)])
    route_time_estimations = estimate_agent_routes_travel_time(
        agent=agent,
        day=day,
        experiment_records=experiment_records,
        free_flow_times=free_flow_times,
        span=span)
    action = random.choice([rou for rou, time_est in enumerate(route_time_estimations) if time_est == (min_time := min(route_time_estimations))])
    return action