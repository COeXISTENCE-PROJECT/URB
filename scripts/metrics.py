import argparse
import json
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

def get_episodes(ep_path: str) -> list[int]:
    """Get the episodes data

    Args:
        ep_path (str): the path to the episodes folder
    Returns:
        sorted_episodes (list[int]): the sorted episodes data
    Raises:
        FileNotFoundError: If the episodes folder does not exist
    """

    eps = list()
    if os.path.exists(ep_path):
        for file in os.listdir(ep_path):
            episode = int(file.split("ep")[1].split(".csv")[0])
            eps.append(episode)
    else:
        raise FileNotFoundError(f"Episodes folder does not exist!")

    eps = [ep for ep in eps if ep % 5 == 0]  # faster

    return sorted(eps)


def flatten_by_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten a DataFrame by agent ID to create a single row DataFrame for single episode.

    return one row dataframe with all columns renamed to "agent_<id>_<original_column_name>" for each id


    Args:
        df (pd.DataFrame): The DataFrame to flatten.
    Returns:
        pd.DataFrame: The flattened DataFrame.
    """
   
    flattened_df = pd.DataFrame()

    columns = []
    values = []

    for id in df["id"]:
        # get the row with the id
        row = df[df["id"] == id]

        # rename the columns
        for column in row.columns:
            if column != "id":
                columns.append(f"agent_{id}_{column}")
                values.append(row[column].values[0])

    # create a new row with the values
    new_row = {}
    for i in range(len(columns)):
        new_row[columns[i]] = values[i]

    flattened_df = pd.DataFrame([new_row])

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
    try:
        tree = ET.parse(file)
    except ET.ParseError:
        print(f"Error parsing XML file: {file}")
        return pd.DataFrame()
    root = tree.getroot()

    # Flatten the XML into a single dictionary
    flat_data = {}
    for child in root:
        for key, value in child.attrib.items():
            flat_data[f"{child.tag}_{key}"] = value

    # Convert to a single-row DataFrame
    df = pd.DataFrame([flat_data])

    # remove the columns that are not needed
    cols = [
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

    try:
        df = df[cols]
    except KeyError:
        print(f"Error: Some columns are missing in the file: {file}")
        return pd.DataFrame()

    try:
        df = df.apply(pd.to_numeric)
    except ValueError:
        pass


    return df


def load_detailed_SUMO(file) -> pd.DataFrame:
    """
    Load detailed SUMO output data and return a DataFrame.

    Args:
        file (str): The path to the SUMO output file.
    Returns:
        pd.DataFrame: A DataFrame containing the SUMO output data.
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        tree = ET.parse(file)
    except ET.ParseError:
        print(f"Error parsing XML file: {file}")
        return pd.DataFrame()
    root = tree.getroot()

    # Extract all tripinfo elements and their attributes
    data = [trip.attrib for trip in root.findall("tripinfo")]
    # Convert to DataFrame
    df = pd.DataFrame(data)

    # filter out the columns that are not needed
    cols = [
        "id",
        "depart",
        "departDelay",
        "arrival",
        "routeLength",
        "duration",
        "waitingTime",
        "timeLoss",
        "vType",
        "speedFactor",
    ]
    
    try:
        df = df[cols]
    except KeyError:
        print(f"Error: Some columns are missing in the file: {file}")
        return pd.DataFrame()
    # print(df.shape)

    df = flatten_by_id(df)

    # print(df.shape)

    # keep only the columns that contain words from the cols

    return df


def load_routeRL(file) -> pd.DataFrame:
    """
    Load RouteRL output file and return a DataFrame.

    Args:
        file (str): The path to the RouteRL output file.
    Returns:
        pd.DataFrame: A DataFrame containing the RouteRL output data.
    Raises:
        FileNotFoundError: If the file does not exist.
    """

    # load the csv file
    try:
        df = pd.read_csv(file)
    except pd.errors.ParserError:
        print(f"Error parsing file: {file}")
        return pd.DataFrame()

    # convert to numeric
    try:
        df = df.apply(pd.to_numeric)
    except ValueError:
        pass

    df = flatten_by_id(df)

    # print(df.shape)

    return df


def load_episode(results_path: str, episode: int, verbose: bool) -> pd.DataFrame:
    """
    Load the full episode data and return a DataFrame.
    Args:
        results_path (str): The path to the results folder.
        episode (int): The episode number.
        verbose (bool): If True, print the loading progress.
    
    Returns:
        pd.DataFrame: A DataFrame containing the episode data. This dataframe has one row and all columns from the SUMO and RouteRL files.
    """

    SUMO_path = os.path.join(results_path, "SUMO_output")
    RouteRL_path = os.path.join(results_path, "episodes")
    Detectors_path = os.path.join(results_path, "detectors")

    SUMO_files = []
    RouteRL_files = []
    Detectors_files = []

    # find files in the directories

    for root, dirs, files in os.walk(SUMO_path):
        for file in files:
            if file.endswith("_" + str(episode) + ".xml"):
                SUMO_files.append(os.path.join(root, file))

    for root, dirs, files in os.walk(RouteRL_path):
        for file in files:
            if file.endswith("ep" + str(episode) + ".csv"):
                RouteRL_files.append(os.path.join(root, file))

    for root, dirs, files in os.walk(Detectors_path):
        for file in files:
            if file.endswith("ep" + str(episode) + ".csv"):
                Detectors_files.append(os.path.join(root, file))

    dfs = []
    for file in SUMO_files:
        if "detailed" in file:
            df = load_detailed_SUMO(file)
            # print("Detailed SUMO file loaded.")
            if len(df) > 0:
                dfs.append(df)
            else:
                print("Detailed SUMO file loaded, but no other files found.")
        else:
            df = load_general_SUMO(file)
            # print("General SUMO file loaded.")
            if len(df) > 0:
                dfs.append(df)

    for file in RouteRL_files:
        df = load_routeRL(file)
        # print("RouteRL file loaded.")
        dfs.append(df)

    for file in Detectors_files:
        pass

    for i in range(len(dfs)):
        if i == 0:
            df = dfs[i]
        else:
            df = pd.concat([df, dfs[i]], axis=1)

    # print(df.shape)
    # add first column - "episode"
    df.insert(0, "episode", episode)

    return df


def collect_to_single_CSV(path: str, save_path: str ="metrics.csv", verbose: bool =False) -> pd.DataFrame:
    """
    Collect results of the experiment to the single CSV file.

    Args:
        path (str): The path to the results folder.
        save_path (str): The path to the output file.
        verbose (bool): If True, print the loading progress.
    Returns:
        pd.DataFrame: A DataFrame containing the episode data. This dataframe has one row for each episode and all columns from the SUMO and RouteRL files.
    """

    df = pd.DataFrame()

    episodes = get_episodes(os.path.join(path, "episodes"))

    dfs = []
    if verbose:
        print(f"Loading {len(episodes)} episodes...")
    for i in tqdm(episodes):
        # add new rows to the DataFrame
        df = load_episode(path, i, verbose)
        if len(df) > 0:
            dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df.to_csv(save_path, index=False)

    return df


def plot_vector_values(df: pd.DataFrame, path: str, title: str, ylabel: str) -> None:
    """
    Make plots of the vector metrics.

    Args:
        df (pd.DataFrame): The DataFrame to plot.
        path (str): The path to the output folder.
        title (str): The title of the plot.
        ylabel (str): The y-axis label.
    Returns:
        None
    """

    # make one plots with all columns of df 
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
    n_agents = df["vehicleTripStatistics_count"].values[0]
    
    new_columns = {}
    for i in range(n_agents):
        col = (df[f"agent_{i}_action"] != df[f"agent_{i}_action"].shift(1)).astype(int)
        new_columns[f"agent_{i}_action_change"] = col

    avg_times_pre = params["avg_times_pre"]
    for i in range(n_agents):  
        col = (
            df[f"agent_{i}_duration"] - avg_times_pre[i]
        )
        new_columns[f"agent_{i}_time_lost"] = col



    # add the new columns to the DataFrame
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

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
    df = df.iloc[[-1]]

    type_IDs = [
        col.split("_")[1]
        for col in df.columns
        if col.startswith("agent_")
        and col.endswith("vType")
        and (df[col] == type).any()
    ]

    # Cast to int
    type_IDs = [int(id) for id in type_IDs]

    # print(f"IDs of {type} agents: {type_IDs}")

    return type_IDs


def slice_episodes(df: pd.DataFrame, config: dict) -> dict:
    """
    Slice the DataFrame into periods of interest.
    Args:
        df (pd.DataFrame): The DataFrame to slice.
        config (dict): The configuration dictionary.
    Returns:
        dict: A dictionary containing the sliced DataFrames.
    """
    training_duration = config["human_learning_episodes"] + config["training_eps"]
    return {
        "before_mutation": df[(df["episode"] <= config["human_learning_episodes"]) & 
                              (df["episode"] > config["human_learning_episodes"] - 50)],
        "after_mutation": df[df["episode"] > config["human_learning_episodes"]],
        "testing_frames": df[df["episode"] > training_duration],
        "training_frames": df[(df["episode"] > config["human_learning_episodes"]) & 
                              (df["episode"] <= training_duration)]
    }


def extract_metrics(path: str, config: dict, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract metrics from the DataFrame.

    Args:
        path (str): The path to the CSV file.
        config (dict): The configuration dictionary.
    Returns:
        pd.DataFrame: A DataFrame containing the metrics.
        pd.DataFrame: A DataFrame containing the vector metrics.
    """

    
    df = pd.read_csv(path)


    CAV_ids = get_type_ids(df, "AV")
    human_ids = get_type_ids(df, "Human")
    AV_only = len(human_ids) == 0

    if verbose and AV_only:
        print("AV only experiment, no human learning period found.")

    periods = slice_episodes(df, config, verbose)
    testing_frames = periods["testing_frames"]
    before_mutation = periods["before_mutation"]
    after_mutation = periods["after_mutation"]
    training_frames = periods["training_frames"]

    if verbose:
        print(f"Before mutation: {before_mutation.shape}")
        print(f"After mutation: {after_mutation.shape}")
        print(f"Testing frames: {testing_frames.shape}")
        print(f"Training frames: {training_frames.shape}")

    avg_times_pre = {}
    for id in human_ids + CAV_ids:
        avg_times_pre[id] = before_mutation[f"agent_{id}_duration"].mean()
    
    params = {"avg_times_pre": avg_times_pre}

    if not AV_only:
        before_mutation = add_benchmark_columns(before_mutation, params)
    after_mutation = add_benchmark_columns(after_mutation, params)
    training_frames = add_benchmark_columns(training_frames, params)
    testing_frames = add_benchmark_columns(testing_frames, params)

    t_CAV = 0
    for id in CAV_ids:
        t_CAV += testing_frames[f"agent_{id}_duration"].mean()

    t_CAV /= len(CAV_ids)

 
    if not AV_only:
        t_HDV_pre = 0
        for id in human_ids:
            t_HDV_pre += before_mutation[f"agent_{id}_duration"].mean()
        t_HDV_pre /= len(human_ids)

        t_pre = np.sum(
        [before_mutation[f"agent_{id}_duration"].mean() for id in human_ids + CAV_ids]
        )
        t_pre = t_pre / (len(CAV_ids) + len(human_ids))

        t_HDV_test = 0
        for id in human_ids:
            t_HDV_test += testing_frames[f"agent_{id}_duration"].mean()
        t_HDV_test /= len(human_ids)

    t_train = np.sum(
        [training_frames[f"agent_{id}_duration"].mean() for id in human_ids + CAV_ids]
    )
    t_train = t_train / (len(CAV_ids) + len(human_ids))

    t_test = np.sum(
        [testing_frames[f"agent_{id}_duration"].mean() for id in human_ids + CAV_ids]
    )
    t_test = t_test / (len(CAV_ids) + len(human_ids))

    t_sumo = np.mean(testing_frames["vehicleTripStatistics_totalTravelTime"]) 
    t_sumo = t_sumo / (len(CAV_ids) + len(human_ids))

    if not AV_only:
        avg_mileage_pre = np.mean(before_mutation["vehicleTripStatistics_routeLength"])
    avg_mileage_test = np.mean(testing_frames["vehicleTripStatistics_routeLength"])

    if not AV_only:
        avg_speed_pre = np.mean(before_mutation["vehicleTripStatistics_speed"])
    avg_speed_test = np.mean(testing_frames["vehicleTripStatistics_speed"])
    
    avg_times_pre = [before_mutation[f"agent_{id}_duration"].mean() for id in human_ids + CAV_ids]

    if not AV_only:
        cost_of_learning_humans = np.mean([ sum(training_frames[f"agent_{id}_duration"] - avg_times_pre[id]) for id in human_ids])
    cost_of_learning_CAVs = np.mean([ sum(training_frames[f"agent_{id}_duration"] - avg_times_pre[id]) for id in CAV_ids])

      
 
    total_time_lost = {}
    for id in human_ids + CAV_ids:
        total_time_lost[id] = after_mutation[f"agent_{id}_time_lost"].sum()

    average_time_lost = np.mean(
        [total_time_lost[id] for id in human_ids + CAV_ids]
    )
    if not AV_only:
        average_human_time_lost = np.mean(
            [total_time_lost[id] for id in human_ids]
        )
    average_CAV_time_lost = np.mean(
        [total_time_lost[id] for id in CAV_ids]
    )

    average_time_CAVs = training_frames[
        [f"agent_{id}_duration" for id in CAV_ids]
    ].mean(axis=1)
    average_time_humans = training_frames[
        [f"agent_{id}_duration" for id in human_ids]
    ].mean(axis=1)

    timeDiff = average_time_humans - average_time_CAVs
    timeDiff = timeDiff.tolist()
    isTimeDiffPositive = [1 if i > 0 else 0 for i in timeDiff]

    winrate = np.mean(isTimeDiffPositive)


    metrics = {}

    metrics["t_pre"] = None if AV_only else t_pre
    metrics["t_test"] = t_test
    metrics["t_train"] = t_train
    metrics["t_CAV"] = t_CAV
    metrics["t_HDV_pre"] = None if AV_only else t_HDV_pre
    metrics["t_HDV_test"] = None if AV_only else t_HDV_test
    metrics["CAV_advantage"] = None if AV_only else t_HDV_test / t_CAV
    metrics["Effect_of_change"] = None if AV_only else t_HDV_pre / t_CAV
    metrics["Effect_of_remaining"] = None if AV_only else t_HDV_pre / t_HDV_test #!
    metrics["diff_sumo_routerl"] = t_sumo - t_test
    metrics["avg_speed_pre"] = None if AV_only else avg_speed_pre #!
    metrics["avg_speed_test"] = avg_speed_test
    metrics["avg_mileage_pre"] = None if AV_only else avg_mileage_pre #!
    metrics["avg_mileage_test"] = avg_mileage_test
    metrics["cost_of_learning"] = (cost_of_learning_humans * len(human_ids) + cost_of_learning_CAVs * len(CAV_ids)) / (len(human_ids) + len(CAV_ids))
    metrics["cost_of_learning_humans"] = cost_of_learning_humans
    metrics["cost_of_learning_CAVs"] = cost_of_learning_CAVs
    metrics["avg_time_lost"] = average_time_lost
    metrics["avg_human_time_lost"] = None if AV_only else average_human_time_lost
    metrics["avg_CAV_time_lost"] = average_CAV_time_lost
    metrics["winrate"] = winrate

    #metrics to dataframe
    metrics_df = pd.DataFrame([metrics])
    
    # now metrics that are not a single value but a list of values

    instability_humans = after_mutation[
        [f"agent_{id}_action_change" for id in human_ids]
    ].sum(axis=1).tolist()
    instability_CAVs = after_mutation[
        [f"agent_{id}_action_change" for id in CAV_ids]
    ].sum(axis=1).tolist()

    avg_time_lost = after_mutation["vehicleTripStatistics_timeLoss"] + after_mutation["vehicleTripStatistics_departDelay"]
    avg_time_lost = avg_time_lost.tolist()

    time_excess = after_mutation[[f"agent_{id}_time_lost" for id in human_ids + CAV_ids]].sum(axis=1).tolist()

    vector_metrics_df = pd.DataFrame({
    "episode": after_mutation["episode"],
    "instability_humans": instability_humans,
    "instability_CAVs": instability_CAVs,
    "avg_time_lost": avg_time_lost,
    "time_excess": time_excess,
    }).astype({
    "episode": int,
    "instability_humans": int,
    "instability_CAVs": int,
    "avg_time_lost": float,
    "time_excess": float,
})

    return metrics_df, vector_metrics_df


def clear_SUMO_files(sumo_path: str, ep_path:str, remove_additional_files: bool =False) -> None:
    """
    Clear SUMO files that are empty.
    If remove_additional_files=True, remove also files that are not in the episodes folder.
    Works only for the consecutive files with the same name.
    The file naming convention is: <file_name>_<episode>.xml

    This is a destructive function, it will remove files from the directory!

    Args:
        sumo_path (str): The path to the SUMO output folder.
        ep_path (str): The path to the episodes folder.
        remove_additional_files (bool): If True, remove additional files that are not in the episodes folder.
    Returns:
        None
    """
    file_id = 1
    episode = 1

    file_name = "detailed_sumo_stats"
    
    while True:
        # check if file exists
        file_path = os.path.join(sumo_path, f"{file_name}_{episode}.xml")
        if os.path.exists(file_path):
            # read xml file and check if <tripinfos> is empty (no <tripinfo> elements)
            try:
                tree = ET.parse(file_path)
            except ET.ParseError:
                print(f"Error parsing XML file: {file_path}")
                break
            root = tree.getroot()
            if len(root.findall("tripinfo")) == 0:
                # remove the file
                os.remove(file_path)
                # print(f"Removed empty file: {file_path}")
            else:
                # rename to the next file_id
                new_file_path = os.path.join(sumo_path, f"{file_name}_{file_id}.xml")
                os.rename(file_path, new_file_path)
                # print(f"Renamed file {file_path} to {new_file_path}")
                file_id += 1
        else:
            break
        episode += 1

    file_id = 1
    episode = 1

    file_name = "sumo_stats"

    while True:
        # check if file exists
        file_path = os.path.join(sumo_path, f"{file_name}_{episode}.xml")
        if os.path.exists(file_path):
            # read xml file and check if <vehicle loaded=0>
            try:
                tree = ET.parse(file_path)
            except ET.ParseError:
                print(f"Error parsing XML file: {file_path}")
                break
            root = tree.getroot()
            vehicle = root.find("vehicles")
            if vehicle is not None and vehicle.attrib.get("loaded") == "0":
                # remove the file
                os.remove(file_path)
                # print(f"Removed empty file: {file_path}")
            else:
                # rename to the next file_id
                new_file_path = os.path.join(sumo_path, f"{file_name}_{file_id}.xml")
                os.rename(file_path, new_file_path)
                # print(f"Renamed file {file_path} to {new_file_path}")
                file_id += 1
        else:
            break
        episode += 1
    if remove_additional_files:
        episodes = get_episodes(ep_path)
        # remove SUMO files that are not in the episodes
        for file in os.listdir(sumo_path):
            if file.endswith(".xml"):
                episode = int(file.split("_")[-1].split(".")[0])
                if episode not in episodes:
                    os.remove(os.path.join(sumo_path, file))
                    # print(f"Removed file: {file}")


results_folder = f"./results"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--skip_clearing", type=bool, default=False)
    parser.add_argument("--skip_collecting", type=bool, default=False)
    parser.add_argument("--results_folder", type=str, default=results_folder)
    parser.add_argument("--verbose", type=bool, default=False)
    args = parser.parse_args()
    exp_id = args.id
    skip_clearing = args.skip_clearing
    skip_collecting = args.skip_collecting
    results_folder = args.results_folder
    verbose = args.verbose
    if verbose:
        print(f"Experiment ID: {exp_id}")
        print(f"Skip clearing: {skip_clearing}")
        print(f"Skip collecting: {skip_collecting}")
        print(f"results folder: {results_folder}")

    data_path = None
    for root, dirs, files in os.walk(results_folder):
        if exp_id in dirs:
            data_path = os.path.join(root, exp_id)
            break


    metrics_path = os.path.join(data_path, "metrics")
    plot_path = os.path.join(metrics_path, "plots")

    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)
        os.makedirs(plot_path)
        if verbose:
            print(f"Created directory for metrics and plots: {metrics_path}")

    exp_config_path = os.path.join(data_path, "exp_config.json")

    exp_config = json.load(open(exp_config_path, "r"))


    if not skip_clearing:
        clear_SUMO_files(
            os.path.join(data_path, "SUMO_output"), os.path.join(data_path, "episodes"), True
        )
        if verbose:
            print(f"Cleared SUMO files in {os.path.join(data_path, 'SUMO_output')}")

    if not skip_collecting:
        collect_to_single_CSV(data_path, os.path.join(metrics_path, "combined_data.csv"), verbose)
        if verbose:
            print(f"Collected data to {os.path.join(metrics_path, 'combined_data.csv')}")

    metrics, vector_metrics = extract_metrics(
        os.path.join(metrics_path, "combined_data.csv"),
        {
            "human_learning_episodes": exp_config["human_learning_episodes"],
            "training_eps": exp_config["training_eps"] if "training_eps" in exp_config else exp_config["n_iters"] * exp_config["agent_frames_per_batch"],
            "test_eps": exp_config["test_eps"],
        },
    )
    if verbose:
        print(f"Extracted metrics")

    # save metrics to csv
    metrics.to_csv(os.path.join(metrics_path, "BenchmarkMetrics.csv"), index=False)
    vector_metrics.to_csv(os.path.join(metrics_path, "VectorMetrics.csv"), index=False)
    
    if verbose:
        print(f"Saved metrics to {os.path.join(metrics_path, 'BenchmarkMetrics.csv')}")
        print(f"Saved vector metrics to {os.path.join(metrics_path, 'VectorMetrics.csv')}")


    # make plots of the vector metrics
    plot_vector_values(
        vector_metrics[["episode", "instability_humans", "instability_CAVs"]],
        plot_path,
        "action change count",
        "Instability",
    )

    plot_vector_values(
        vector_metrics[["episode", "avg_time_lost"]],
        plot_path,
        "avg time lost",
        "Average time lost",
    )

    plot_vector_values(
        vector_metrics[["episode", "time_excess"]],
        plot_path,
        "time excess",
        "Time excess",
    )
