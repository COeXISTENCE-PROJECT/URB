# %%
import argparse
import json
import logging
import os

import routerl

from routerl import Keychain as kc

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--conf", type=str, required=True)
    args = parser.parse_args()
    exp_id = args.id
    exp_config = args.conf
    print("### STARTING COLLECTING KPIS ###")
    print(f"Experiment ID: {exp_id}")

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # %%
    params = json.load(open("../experiment_metadata.json"))
    params = params[exp_config]["config"]
    exp_length = (
        params["human_learning_episodes"] + params["training_eps"] + params["test_eps"]
    )

    # %% [markdown]
    # #### Collecting experiment data into one file

    # %%
    records_folder = f"../records/{exp_id}"
    if not os.path.exists(records_folder):
        raise FileNotFoundError(f"Records folder {records_folder} does not exist.")

    records_sumo = os.path.join(records_folder, "SUMO_OUTPUT")

    KPIs_location = os.path.join(records_folder, "KPIs.csv")

    print("Clearing additional SUMO output files...")
    # clear_SUMO_files(records_sumo)

    print("Collecting KPIs from records folder...")
    # collect_to_single_CSV(records_folder, exp_length, KPIs_location)

    print("Extracting data...")
    # extract_KPIs(KPIs_location)

    # %%
    print("Data extraction completed.")

    print("### END OF KPIs COLLECTION ###")
