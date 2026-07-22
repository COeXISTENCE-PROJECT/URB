import glob
import os
import sys
import argparse
import json
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_episodes(ep_path: str) -> list[int]:
    """Get the episodes data

    Args:
        ep_path (str): the path to the episodes folder
    Returns:
        sorted_episodes (list[int]): the sorted episodes data
    Raises:
        FileNotFoundError: If the episodes folder does not exist
    """

    if not os.path.exists(ep_path):
        raise FileNotFoundError(f"Episodes folder does not exist at: {ep_path}")

    eps = list()
    for file in os.listdir(ep_path):
        episode = int(file.split("ep")[1].split(".csv")[0])
        eps.append(episode)

    return sorted(eps)


def flatten_by_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten a DataFrame by agent ID to create a single row DataFrame for single episode.

    Transform the DataFrame into a one row dataframe with all columns renamed to "agent_<id>_<original_column_name>" for each id

    Args:
        df (pd.DataFrame): The DataFrame to flatten.
    Returns:
        pd.DataFrame: The flattened DataFrame.
    """
    if df.empty or "id" not in df.columns:
       return pd.DataFrame()
    df_indexed = df.set_index("id")
    s = df_indexed.stack()
    new_names = s.index.map(lambda idx: f"agent_{idx[0]}_{idx[1]}")

    flattened_df = pd.DataFrame([s.values], columns=new_names)

    return flattened_df


def load_general_SUMO(file) -> pd.DataFrame:
    """
    Load general SUMO output data and return a DataFrame.

    Args:
        file (str): The path to the SUMO output file.
    Returns:
        pd.DataFrame: A DataFrame containing the SUMO output data.
    Raises:
        FileNotFoundError: If the file does not exist.
    """

    if not os.path.exists(file):
        raise FileNotFoundError(f"SUMO output file not found at: {file}")

    try:
        tree = ET.parse(file)
        root = tree.getroot()
    except ET.ParseError:
        print(f"Error parsing XML file: {file}")
        return pd.DataFrame()

    # ----- Extract needed elements -----

    flat_data = {
        f"{child.tag}_{k}": v for child in root for k, v in child.attrib.items()
    }

    df = pd.DataFrame([flat_data])

    REQUIRED_COLUMNS = [
        "teleports_total",
        "teleports_jam",
        "teleports_yield",
        "teleports_wrongLane",
        "vehicleTripStatistics_count",
        "vehicleTripStatistics_routeLength",
        "vehicleTripStatistics_speed",
        "vehicleTripStatistics_duration",
        "vehicleTripStatistics_waitingTime",
        "vehicleTripStatistics_timeLoss",
        "vehicleTripStatistics_departDelay",
        "vehicleTripStatistics_totalTravelTime",
        "vehicleTripStatistics_totalDepartDelay",
    ]

    present_columns = [col for col in REQUIRED_COLUMNS if col in df.columns]

    if len(present_columns) < len(REQUIRED_COLUMNS):
        print(f"Warning: Some columns are missing in the file: {file}, returning an empty DataFrame.")
        return pd.DataFrame()

    df = df[present_columns]

    df = df.apply(pd.to_numeric, errors="coerce")

    return df

def load_detailed_SUMO(file: str) -> pd.DataFrame:
    """
    Load detailed SUMO output data and return a DataFrame.

    Args:
        file (str): The path to the SUMO output file.
    Returns:
        pd.DataFrame: A DataFrame containing the SUMO output data.
    Raises:
        FileNotFoundError: If the file does not exist.
    """

    if not os.path.exists(file):
        raise FileNotFoundError(f"SUMO detailed output file not found at: {file}")
    
    try:
        tree = ET.parse(file)
        root = tree.getroot()
    except ET.ParseError:
        print(f"Error parsing XML file: {file}")
        return pd.DataFrame()

    # ----- Extract needed tripinfo elements and their attributes -----

    data = [trip.attrib for trip in root.findall("tripinfo")]
    df = pd.DataFrame(data)

    if df.empty:
        return pd.DataFrame()

    USED_NUMERICAL_COLUMNS = [
        "id",
        "depart",
        "departDelay",
        "arrival",
        "routeLength",
        "duration",
        "waitingTime",
        "timeLoss",
        "speedFactor",
    ]

    USED_STRING_COLUMNS = [
        "vType",
    ]

    USED_COLUMNS = USED_NUMERICAL_COLUMNS + USED_STRING_COLUMNS

    present_columns = [col for col in USED_COLUMNS if col in df.columns]

    if len(present_columns) < len(USED_COLUMNS):
        print(f"Warning: Some columns are missing in the file: {file}, returning an empty DataFrame.")
        return pd.DataFrame()

    df = df[present_columns]

    df_num_subset = df[USED_NUMERICAL_COLUMNS]
    df_num_subset = df_num_subset.apply(pd.to_numeric, errors="coerce")
    df[USED_NUMERICAL_COLUMNS] = df_num_subset
    
    df = flatten_by_id(df)

    return df


def load_routeRL(file) -> pd.DataFrame:
    """
    Load RouteRL output file and return a DataFrame.

    Args:
        file (str): The path to the RouteRL output file.
    Returns:
        pd.DataFrame: A DataFrame containing the RouteRL output data.
    """

    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        print(f"RouteRL output file not found at: {file}")
        return pd.DataFrame()

    except pd.errors.ParserError as e:
        print(f"Error parsing file: {file}: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # ----- Ensure numeric columns are properly typed -----
    NUMERIC_COLUMNS = [
        "travel_time",
        "id",
        "action",
        "origin",
        "destination",
        "start_time",
        "reward",
    ]

    numeric_df = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df[NUMERIC_COLUMNS] = numeric_df

    return flatten_by_id(df)


def load_episode(results_path: str, episode: int, verbose: bool) -> pd.DataFrame:
    """
    Load, merge, and return all data for a single episode as a single-row DataFrame.

    This function searches for SUMO and RouteRL output files for the given episode,
    loads them using helper functions, and concatenates them horizontally.

    This function can be easily extended to include detector data in the future.
    Currently, detector data loading is commented out.

    Args:
        results_path (str): The path to the main results folder.
        episode (int): The episode number to load.
        verbose (bool): If True, print progress and warning messages.

    Returns:
        pd.DataFrame: A single-row DataFrame containing all merged data for the episode.
                      Returns an empty DataFrame if no valid data files are found.
    """

    sumo_path = os.path.join(results_path, "SUMO_output")
    routerl_path = os.path.join(results_path, "episodes")
    # detectors_path = os.path.join(results_path, "detectors")

    sumo_files = glob.glob(os.path.join(sumo_path, f"*_{episode}.xml"))
    routerl_files = glob.glob(os.path.join(routerl_path, f"*ep{episode}.csv"))
    # detectors_files = glob.glob(os.path.join(detectors_path, f"*ep{episode}.csv"))


    routerl_file = next((file for file in routerl_files if "ep" in file), None)
    detailed_sumo_file = next((file for file in sumo_files if "detailed" in file), None)
    general_sumo_file = next((file for file in sumo_files if "detailed" not in file), None)

    if not detailed_sumo_file or not general_sumo_file or not routerl_file:
        if verbose:
            print(f"Skipping episode {episode} due to missing data files.")
        return pd.DataFrame()

    df_detailed = load_detailed_SUMO(detailed_sumo_file)
    df_general = load_general_SUMO(general_sumo_file)
    df_route_rl = load_routeRL(routerl_file)

    if df_detailed.empty or df_general.empty or df_route_rl.empty:
        if verbose:
            print(f"Skipping episode {episode} due to empty data file.")
        return pd.DataFrame()

    merged_df = pd.concat([df_detailed, df_general, df_route_rl], axis=1)
    merged_df.insert(0, "episode", episode)
    if verbose:
        print(f"Loaded episode {episode} with shape {merged_df.shape}")
    return merged_df

def collect_to_single_CSV(
    path: str, save_path: str = "metrics.csv", verbose: bool = False
) -> pd.DataFrame:
    """
    Collect results of the experiment to the single CSV file.

    Args:
        path (str): The path to the results folder, 'episodes' and 'SUMO_output' should be a subdirectories.
        save_path (str): The path to the output file.
        verbose (bool): If True, print the loading progress.
    Returns:
        pd.DataFrame: A DataFrame containing the episode data. This dataframe has one row for each episode and all columns from the SUMO and RouteRL files.
    """

    # ----- Get the episodes ids from the episodes folder -----
    episodes_path = os.path.join(path, "episodes")
    episodes = get_episodes(episodes_path)

    dfs = []

    if verbose:
        print(f"Loading {len(episodes)} episodes...")

    # ----- Each episode is loaded and merged into a single row DataFrame -----
    for i in tqdm(episodes) if verbose else episodes:
        episode_df = load_episode(path, i, verbose)
        if not episode_df.empty:
           dfs.append(episode_df)

    if verbose:
        print(f"Loaded {len(dfs)} episodes.")
        print(f"Final shape of the DataFrame: {pd.concat(dfs, axis=0, ignore_index=True).shape if dfs else (0,0)}")

        
    if not dfs:
        if verbose:
            print("No data loaded from any episodes. Returning empty DataFrame.")
        return pd.DataFrame()
    
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df["episode"] = df["episode"].astype("int32")

    df.to_csv(save_path, index=False, float_format="%.2f")    

    return df


def plot_vector_values(df: pd.DataFrame, path: str, title: str, ylabel: str) -> None:
    """
    Make plots of the vector metrics. 
    The plot always has episode on the x-axis and the values of all other columns on the y-axis.

    Args:
        df (pd.DataFrame): The DataFrame to plot.
        path (str): The path to the output folder.
        title (str): The title of the plot.
        ylabel (str): The y-axis label.
    Returns:
        None
    """

    plt.figure(figsize=(10, 6))
    for column in df.columns:
        if column != "episode":
            plt.plot(df["episode"], df[column], label=column)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)

    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(path, title + ".png"))
    plt.close()


def add_benchmark_columns(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Add benchmark columns to the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to add columns to.
        params (dict): The parameters for the benchmark script.
    Returns:
        pd.DataFrame: The DataFrame with the new columns.
    """

    if df.empty:
        return df

    # Infer agent IDs from actual per-agent columns instead of relying on
    # vehicleTripStatistics_count, which may be lower in sliced frames.
    duration_ids = sorted({
        int(col.split("_")[1])
        for col in df.columns
        if col.startswith("agent_") and col.endswith("_travel_time")
    })
    action_ids = sorted({
        int(col.split("_")[1])
        for col in df.columns
        if col.startswith("agent_") and col.endswith("_action")
    })

    new_columns = {}
    
    new_columns.update({
        f"agent_{i}_action_change": (df[f"agent_{i}_action"] != df[f"agent_{i}_action"].shift(1)).astype(int)
        for i in action_ids
    })

    avg_times_pre = params.get("avg_times_pre", {})

    new_columns.update({
        f"agent_{i}_time_lost": df[f"agent_{i}_travel_time"] - avg_times_pre.get(i, 0) 
        for i in duration_ids
    })

    new_df = pd.DataFrame(new_columns)
    df = pd.concat([df, new_df], axis=1)

    return df


def get_type_ids(df: pd.DataFrame, type: str) -> list:
    """
    Helper function to get the IDs of the agents of a given type.
    Args:
        df (pd.DataFrame): The DataFrame to search in.
        type (str): The type of the agents to search for.
    Returns:
        list: A list of the IDs of the agents of the given type.
    """
    
    if df.empty:
        return []

    last_row = df.iloc[-1]

    vtype_cols = [col for col in df.columns if col.startswith("agent_") and col.endswith("vType")]

    vtype_data = last_row[vtype_cols]
    
    matching_cols = vtype_data[vtype_data == type].index.tolist()

    type_IDs = [int(col.split("_")[1]) for col in matching_cols]

    return type_IDs


def slice_episodes(df: pd.DataFrame, config: dict) -> dict:
    """
    Slice the DataFrame into periods of interest.

    Output data frames:
        - 'before_mutation' - last 50 episodes of human learning phase (before mutation); or all human learning episodes if less than 50.
        - 'after_mutation' - all episodes with CAVS - starting from episode after mutation, ending with the end of the experiment.
        - 'testing_frames' - episodes from testing phase only.
        - 'training_frames' - episodes from CAV training period only.

    Args:
        df (pd.DataFrame): The DataFrame to slice.
        config (dict): The configuration dictionary.
    Returns:
        dict: A dictionary containing the sliced DataFrames.
    """


    # Development note:
    #   TODO: generalize phase handling.
    #   Move phase definition and boundary computation (e.g., hl, experience_collecting, training, testing) to the experiment scripts (URB/scripts/<algo>.py),
    #   store boundaries explictky in exp_config, and consume them here (metrics).

    # Phase lengths
    hl_episodes = int(config["human_learning_episodes"])
    experience_collecting_eps = int(config["experience_collecting_episodes"])
    training_eps = int(config["training_eps"])

    # Assuming episode indexing starts from 1
    trainphase_start_episode = hl_episodes + experience_collecting_eps + 1
    testphase_start_episode = hl_episodes + experience_collecting_eps + training_eps + 1


    return {
        "before_mutation": df[
            (df["episode"] <= hl_episodes)
            & (df["episode"] > hl_episodes - 50)  # Last 50 days of simulation taken as human policy testing period
        ].copy(),
        "after_mutation": df[df["episode"] > hl_episodes].copy(),

        "training_frames": df[ 
            (df["episode"] >= trainphase_start_episode)
            & (df["episode"] < testphase_start_episode)
        ].copy(),
        "testing_frames": df[df["episode"] >= testphase_start_episode].copy(),
    }


def extract_metrics(path, config, verbose=False):
    """
    Extract metrics from the DataFrame.
    """

    # ----- Config validation -----

    try:
        df = pd.read_csv(path)
    except Exception as e:
        if verbose:
            print(f"Error reading CSV file at {path}: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
    if df.empty:
        if verbose:
            print(f"The DataFrame loaded from {path} is empty. Returning empty metrics.")
        return pd.DataFrame(), pd.DataFrame()
    
    periods = slice_episodes(df, config)
    testing_frames = periods["testing_frames"]
    before_mutation = periods["before_mutation"]
    after_mutation = periods["after_mutation"]
    training_frames = periods["training_frames"]

    if verbose:
        print(f"Before mutation: {before_mutation.shape}")
        print(f"After mutation: {after_mutation.shape}")
        print(f"Testing frames: {testing_frames.shape}")
        print(f"Training frames: {training_frames.shape}")

    # ----- Identify agent types and IDs ---

    CAV_ids = get_type_ids(df, "AV")
    human_ids = get_type_ids(df, "Human")
    all_ids = human_ids + CAV_ids
    n_agents = len(all_ids)

    if n_agents == 0:
        if verbose:
            print("No agents found in the data. Returning empty metrics.")
        return pd.DataFrame(), pd.DataFrame()
   
    AV_only = len(human_ids) == 0 or before_mutation.empty
    if verbose and AV_only:
        print("AV only experiment, no human learning period found.")

    # ----- Average travel times before mutation -----
    avg_times_pre = {}

    if not before_mutation.empty:
        pre_cols = [f"agent_{id}_travel_time" for id in all_ids if f"agent_{id}_travel_time" in before_mutation.columns]
        if pre_cols:
            pre_means_series = before_mutation[pre_cols].mean()
            avg_times_pre = {
                int(col.split("_")[1]): pre_means_series[col]
                for col in pre_means_series.index
                if not pd.isna(pre_means_series[col]) # Only store non-NaN means
            }

    params = {"avg_times_pre": avg_times_pre}

    # ----- Add benchmark columns -----
    if not AV_only:
        before_mutation = add_benchmark_columns(before_mutation, params)

    if not after_mutation.empty:   
        after_mutation = add_benchmark_columns(after_mutation, params)

    if not training_frames.empty:
        training_frames = add_benchmark_columns(training_frames, params)
    if not testing_frames.empty:
        testing_frames = add_benchmark_columns(testing_frames, params)

    # ----- Calculate metrics (Average travel times) -----

    def get_agent_avg_travel_time(df_slice, ids, suffix="_travel_time"):
        if df_slice.empty or not ids:
            return np.nan # Use NaN for safe propagation if slice or ID list is empty

        # Ensure columns exist before selection
        cols = [f"agent_{id}{suffix}" for id in ids if f"agent_{id}{suffix}" in df_slice.columns]
        if len(cols) == 0:
            return np.nan
        
        return df_slice[cols].mean(axis=1).mean()


    t_CAV = get_agent_avg_travel_time(testing_frames, CAV_ids)

    t_train = get_agent_avg_travel_time(training_frames, all_ids)

    t_test = get_agent_avg_travel_time(testing_frames, all_ids)
    
    t_sumo = testing_frames["vehicleTripStatistics_duration"].mean() if "vehicleTripStatistics_duration" in testing_frames.columns else np.nan
    t_sumo /= 60.0

    t_HDV_pre, t_pre, t_HDV_test = np.nan, np.nan, np.nan
    if not AV_only:
        t_HDV_pre = get_agent_avg_travel_time(before_mutation, human_ids)

        t_pre = get_agent_avg_travel_time(before_mutation, all_ids)

        t_HDV_test = get_agent_avg_travel_time(testing_frames, human_ids)

    def get_df_mean(df_slice, column):
        return df_slice[column].mean() if column in df_slice.columns and not df_slice.empty else np.nan

    avg_mileage_pre, avg_speed_pre = np.nan, np.nan
    if not AV_only:
        avg_mileage_pre = get_df_mean(before_mutation, "vehicleTripStatistics_routeLength")
        avg_speed_pre = get_df_mean(before_mutation, "vehicleTripStatistics_speed")

    avg_mileage_test = get_df_mean(testing_frames, "vehicleTripStatistics_routeLength")
    avg_speed_test = get_df_mean(testing_frames, "vehicleTripStatistics_speed")

    # ----- Calculate metrics (Cost of learning) -----

    def get_cost_of_learning(df_slice, ids):
        if df_slice.empty or not ids:
            return np.nan
        
        cols = [f"agent_{id}_travel_time" for id in ids if f"agent_{id}_travel_time" in df_slice.columns]
        if len(cols) == 0:
            return np.nan

        durations = df_slice[cols]
        max_times = durations.max()
        min_times = durations.min()
        
        # Calculate (max - min) for each agent and then take the mean of those differences
        cost_series = max_times - min_times
        return cost_series.mean()

    cost_of_learning_humans = get_cost_of_learning(training_frames, human_ids)
    cost_of_learning_CAVs = get_cost_of_learning(training_frames, CAV_ids)

    total_cost_of_learning = (
            (cost_of_learning_humans * len(human_ids) if not pd.isna(cost_of_learning_humans) else 0) + 
            (cost_of_learning_CAVs * len(CAV_ids) if not pd.isna(cost_of_learning_CAVs) else 0)
        ) / n_agents if n_agents > 0 else np.nan

    # ----- Calculate metrics (Time lost) -----

    average_time_lost, average_human_time_lost, average_CAV_time_lost = np.nan, np.nan, np.nan

    time_lost_cols = [f"agent_{id}_time_lost" for id in all_ids if f"agent_{id}_time_lost" in after_mutation.columns]
    
    if time_lost_cols:
        total_time_lost_series = after_mutation[time_lost_cols].sum()

        average_time_lost = total_time_lost_series.mean()

        human_time_lost_cols = [c for c in time_lost_cols if int(c.split("_")[1]) in human_ids]
        average_human_time_lost = after_mutation[human_time_lost_cols].values.mean() if human_time_lost_cols else np.nan

        CAV_time_lost_cols = [c for c in time_lost_cols if int(c.split("_")[1]) in CAV_ids]
        average_CAV_time_lost = after_mutation[CAV_time_lost_cols].values.mean() if CAV_time_lost_cols else np.nan

    # ----- Calculate metrics (Winrate) -----

    winrate = np.nan

    if (not np.isnan(t_CAV)) and (not np.isnan(t_pre)):
        winrate = float(t_CAV < t_pre) # return bool as 1.0 or 0.0 for single experiment


    # ----- Compile metrics into DataFrames -----
    
    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator and not pd.isna(denominator) else None
    metrics = {}

    metrics["t_pre"] = None if AV_only else t_pre
    metrics["t_test"] = t_test
    metrics["t_train"] = t_train
    metrics["t_CAV"] = t_CAV
    metrics["t_HDV_pre"] = None if AV_only else t_HDV_pre
    metrics["t_HDV_test"] = None if AV_only else t_HDV_test

    metrics["CAV_advantage"] = None if AV_only else safe_divide(t_HDV_test, t_CAV)
    metrics["Effect_of_change"] = None if AV_only else safe_divide(t_HDV_pre, t_CAV)
    metrics["Effect_of_remaining"] = None if AV_only else safe_divide(t_HDV_pre, t_HDV_test)

    # Convert speed from m/s to km/h
    metrics["avg_speed_pre"] = None if AV_only else avg_speed_pre * 3.6 if not pd.isna(avg_speed_pre) else None
    metrics["avg_speed_test"] = avg_speed_test * 3.6 if not pd.isna(avg_speed_test) else None

    # Convert mileage from m to km
    metrics["avg_mileage_pre"] = None if AV_only else avg_mileage_pre / 1000.0 if not pd.isna(avg_mileage_pre) else None
    metrics["avg_mileage_test"] = avg_mileage_test / 1000.0 if not pd.isna(avg_mileage_test) else None

    metrics["winrate"] = winrate

    metrics["cost_of_learning"] = total_cost_of_learning
    metrics["cost_of_learning_humans"] = cost_of_learning_humans
    metrics["cost_of_learning_CAVs"] = cost_of_learning_CAVs

    metrics["avg_time_lost"] = average_time_lost
    metrics["avg_human_time_lost"] = average_human_time_lost
    metrics["avg_CAV_time_lost"] = average_CAV_time_lost
    
    metrics["diff_sumo_routerl"] = t_sumo - t_test

    metrics_df = pd.DataFrame([metrics])

    # ----- Vector metrics -----
    if after_mutation.empty:
        vector_metrics_df = pd.DataFrame()
    else:

        def get_instability(df_slice, ids): 
            cols = [f"agent_{id}_action_change" for id in ids if f"agent_{id}_action_change" in df_slice.columns]
            return df_slice[cols].sum(axis=1)

        instability_humans = get_instability(after_mutation, human_ids)
        instability_CAVs = get_instability(after_mutation, CAV_ids)
        
        avg_time_lost = pd.Series(np.nan, index=after_mutation.index)
        if "vehicleTripStatistics_timeLoss" in after_mutation.columns and "vehicleTripStatistics_departDelay" in after_mutation.columns:
            avg_time_lost = (
                after_mutation["vehicleTripStatistics_timeLoss"]
                + after_mutation["vehicleTripStatistics_departDelay"]
            )
        
        time_excess = pd.Series(np.nan, index=after_mutation.index)
        agent_time_lost_cols = [f"agent_{id}_time_lost" for id in all_ids if f"agent_{id}_time_lost" in after_mutation.columns]
        if len(agent_time_lost_cols):
            time_excess = after_mutation[agent_time_lost_cols].sum(axis=1)


        # check if there are nonnumerical values in the series and convert them to NaN
        vector_metrics_df = pd.DataFrame({
            "episode": after_mutation["episode"],
            "instability_humans": pd.Series(instability_humans, index=after_mutation.index),
            "instability_CAVs": pd.Series(instability_CAVs, index=after_mutation.index),
            "avg_time_lost": pd.Series(avg_time_lost, index=after_mutation.index),
            "time_excess": pd.Series(time_excess, index=after_mutation.index),
        }).astype(
            {
                "episode": int,
                "instability_humans": int,
                "instability_CAVs": int,
                "avg_time_lost": float,
                "time_excess": float,
            }
        )

        if not human_ids:
            vector_metrics_df["instability_humans"] = np.zeros(len(vector_metrics_df), dtype=float)
        if not CAV_ids:
            vector_metrics_df["instability_CAVs"] = np.zeros(len(vector_metrics_df), dtype=float)


    return metrics_df, vector_metrics_df


RESULTS_DEFAULT_DIR = f"./results"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process experiment results and generate metrics.")
    parser.add_argument("--id", type=str, required=True, help="Experiment ID to process.")
    parser.add_argument("--skip-collecting", type=bool, default=False, help="Skip collecting episodes into the combined CSV.")
    parser.add_argument("--results-folder", type=str, default=RESULTS_DEFAULT_DIR, help="Root folder where experiment results are stored.")
    parser.add_argument("--verbose", type=bool, default=False, help="Enable verbose output.")

    args = parser.parse_args()

    exp_id = args.id
    skip_collecting = args.skip_collecting
    results_folder = args.results_folder
    verbose = args.verbose
    if verbose:
        print(f"Experiment ID: {exp_id}")
        print(f"Skip collecting: {skip_collecting}")
        print(f"results folder: {results_folder}")

    data_path = ""
    for root, dirs, files in os.walk(results_folder):
        if exp_id in dirs:
            data_path = os.path.join(root, exp_id)
            break

    if data_path == "":
        print(f"Experiment ID {exp_id} not found in {results_folder}")
        sys.exit(1)

    metrics_path = os.path.join(data_path, "metrics")
    plot_path = os.path.join(metrics_path, "plots")
    combined_csv_path = os.path.join(metrics_path, "combined_data.csv")
    benchmark_csv_path = os.path.join(metrics_path, "BenchmarkMetrics.csv")
    vector_csv_path = os.path.join(metrics_path, "VectorMetrics.csv")
    config_json_path = os.path.join(data_path, "exp_config.json")

    try:
        os.makedirs(plot_path, exist_ok=True)
        if verbose:
            print(f"Ensured metrics directory structure exists at: {metrics_path}")
    except OSError as e:
        print(f"Error creating directories: {e}")
        sys.exit(1)

    try:
        with open(config_json_path, "r") as f:
            exp_config = json.load(f)
        if verbose:
            print(f"Loaded configuration from: {config_json_path}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_json_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_json_path}")
        sys.exit(1)

    if not skip_collecting:
        try:
            collect_to_single_CSV(data_path, combined_csv_path, verbose)
            if verbose:
                print(f"Collected data to {combined_csv_path}")
        except Exception as e:
            print(f"Error during data collection: {e}")
            sys.exit(1)

    if not os.path.exists(combined_csv_path):
        print(f"Error: Combined data file not found at {combined_csv_path}. Cannot extract metrics.")
        sys.exit(1)

    if "training_eps" in exp_config:
        computed_training_eps = exp_config["training_eps"]
    elif "n_iters" in exp_config and "agent_frames_per_batch" in exp_config:
        computed_training_eps = exp_config["n_iters"] * exp_config["agent_frames_per_batch"]
    else:
        # Default to a safe value or raise an error if critical
        print("Warning: Could not determine 'training_eps'. Assuming 0.")
        computed_training_eps = 0   

    metric_config = {
        "algorithm": exp_config["algorithm"],

        "human_learning_episodes": exp_config["human_learning_episodes"],
        "training_eps": computed_training_eps,
        "experience_collecting_episodes": exp_config.get("experience_collecting_episodes", 0), # Use .get for non-critical keys
        "test_eps": exp_config.get("test_eps", 0), # Use .get for non-critical keys
    }

    metrics, vector_metrics = extract_metrics(combined_csv_path, metric_config, verbose)
    # try:
    #     if verbose:
    #         print("Successfully extracted metrics.")
    # except Exception as e:
    #     print(f"Error during metric extraction: {e}")
    #     sys.exit(1)

    # save metrics to csv
    metrics.to_csv(benchmark_csv_path, index=False)
    vector_metrics.to_csv(vector_csv_path, index=False)

    if verbose:
        print(f"Saved metrics to {benchmark_csv_path}")
        print(f"Saved vector metrics to {vector_csv_path}")

    # make plots of the vector metrics
    if vector_metrics.empty:
        if verbose:
            print("Vector metrics are empty. Skipping plots.")
    else:
        # Avoid creating new DataFrames if possible, just pass slice
        if verbose:
            print("Generating plots...")

        plot_vector_values(
            vector_metrics[["episode", "instability_humans", "instability_CAVs"]].copy(),
            plot_path,
            "action change count",
            "Instability",
        )

        plot_vector_values(
            vector_metrics[["episode", "avg_time_lost"]].copy(),
            plot_path,
            "avg time lost",
            "Average time lost",
        )

        plot_vector_values(
            vector_metrics[["episode", "time_excess"]].copy(),
            plot_path,
            "time excess",
            "Time excess",
        )
        if verbose:
            print("Plots successfully generated.")
