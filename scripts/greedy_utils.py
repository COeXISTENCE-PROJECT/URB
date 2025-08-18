import random
from typing import Optional, Literal

from statistics import mean




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
    origin: Optional[int],
    destination: Optional[int],
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

    return {
        aid: record
        for aid, record in episode_records.items()

        if (origin is None or record['origin'] == origin)
        and (destination is None or record['destination'] == destination)
        and (route is None or record['route'] == route)

        and _in_range(record['time_start'], time_start)
        and _in_range(record['time_end'], time_end)

        and (kind=='all' or record['kind'] == kind)                   
    }

def experiment_records_subset(
    experiment_records: dict[int, dict[int,dict]],
    days: tuple[Optional[int], Optional[int]],
    origin: Optional[int],
    destination: Optional[int],
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
        if _in_range(day, days)
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

def _mean_travel_time(records: dict[int, dict])->float:
    """Mean travel time for episode records subset."""
    return mean(record['travel_time'] for record in records.values())


###########################################
# Route travel time estimation
###########################################

def estimate_route_travel_time(agent: 'Agent', route: int, day: int, experiment_records: dict, free_flow_times: dict, span:int=1) -> float:
    """
    Estimate travel time on a route for an agent using past episode(s).

    Estimation order:
    1. Machine with exact start time.
    2. Last machine departed before agent (shared driving time).
    3. Free flow time.

    'day' is the current day of the experiment.
    'span' is the number of past days (episodes) to consider in estimation; 'span' curently only supports 1
    """


    if span != 1:
        raise NotImplementedError("Curent implementation only for span=1 history.")
    


    previous_day_records = experiment_records.get(day-1, {})
    if not previous_day_records:
        print(f"No records for previous day day: {day-1}. Returning free flow times for agent {agent.id}.")
        return free_flow_times[(agent.origin, agent.destination)][route]




    base_filter_args = { # origin, destination, route
        'origin': agent.origin,
        'destination': agent.destination,
        'route': route,
        'kind': 'machine'
    }

    estimation_filters = [ # filters for time estimations
        {'time_start': (agent.start_time, agent.start_time), 'description': "Exact start time."},
        {'time_start': (None, agent.start_time), 'time_end': (agent.start_time, None), 'description': "Earlier start time, shared travel time."}
    ]


    od_route_records = episode_records_subset(
        episode_records=previous_day_records,
        **base_filter_args
    )


    ##############################################
    # Estimate travel time - in predefined order
    ##############################################

    # 1-2. Try to estimate route travel time from existing data (predefined cases)
    for estimation_args in estimation_filters:

        filter_args = base_filter_args.copy().update(estimation_args)
        estimation_records = episode_records_subset(
            episode_records=od_route_records,
            **filter_args
        )

        if estimation_records:
            print(f"Estimation data for agent {agent.id}: {filter_args['description']}")
            return _mean_travel_time(filtered_records)

   

    # 3. Free-flow fallback for empty results
    print(f"No prior driver data for this route. Best estimation for agent {agent.id}: free flow time.")
    return free_flow_times[(agent.origin, agent.destination)][route]


###########################################
# Action selection
###########################################


def choose_agent_action(agent: 'Agent', day: int, experiment_records: dict[int, dict[int,dict]], free_flow_times: dict)->int:
    """
    Selects route for the agent based on past episode(s) driving data using greedy strategy.
    """
 
    # Get travel time estimations for each route acrssible for the agent, choose one with minimal estimation
    num_routes = len(free_flow_times[(agent.origin, agent.destination)])
    route_time_estimations = [ estimate_route_travel_time(
                                    agent=agent,
                                    route=i,
                                    day=day,
                                    experiment_records=experiment_records,
                                    free_flow_times=free_flow_times
                                )
                                for i in range(num_routes)]
    action = random.choice([rou for rou, time_est in enumerate(route_time_estimations) if time_est == (min_time := min(route_time_estimations))])
    return action