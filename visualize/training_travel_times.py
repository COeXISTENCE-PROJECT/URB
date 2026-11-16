import argparse
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from common import experiment_episodes, load_style, save_plot


def rolling_interval(episode_means, window, confidence):
    rolling = episode_means.rolling(window, min_periods=1)
    mean = rolling.mean()
    count = rolling.count().to_numpy()
    # A saved episode is the sampling unit. Trips within an episode share traffic conditions.
    half_width = t.ppf((1 + confidence) / 2, count - 1) * rolling.std().to_numpy() / np.sqrt(count)
    return mean, mean - half_width, mean + half_width


def seed_interval(seed_means, confidence):
    # Each column is one run's mean for the same cohort and episode.
    count = seed_means.shape[1]
    mean = seed_means.mean(axis=1)
    half_width = t.ppf((1 + confidence) / 2, count - 1) * seed_means.std(axis=1) / np.sqrt(count)
    return mean, mean - half_width, mean + half_width


def load_episode_means(experiment_id, episodes, training_start, test_start, last_episode):
    means = {}
    for day, file in episodes:
        if day > last_episode:
            break
        data = pd.read_csv(file, usecols=["kind", "travel_time"])
        if data.empty or not np.isfinite(data["travel_time"]).all():
            raise ValueError(f"Expected finite travel times in {file}")
        means[day] = data.groupby("kind")["travel_time"].mean()
    if not means:
        raise ValueError(f"{experiment_id} has no saved episodes to plot")
    daily = pd.DataFrame.from_dict(means, orient="index")
    if not ((daily.index >= training_start) & (daily.index < test_start)).any():
        raise ValueError(f"{experiment_id} has no saved training episodes")
    return daily


def main():
    parser = argparse.ArgumentParser(description="Plot mean travel times with intervals across episodes or matched seeds.")
    parser.add_argument("experiment_ids", nargs="+")
    args = parser.parse_args()
    if len(set(args.experiment_ids)) != len(args.experiment_ids):
        parser.error("Provide a different experiment ID for each seed")

    style = load_style()
    options = style["training_curve"]
    confidence = options["confidence_level"]
    if not 0 < confidence < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    first_id = args.experiment_ids[0]
    config, episodes, *boundaries = experiment_episodes(first_id)
    pre_end, training_start, test_start, test_end = boundaries
    last_episode = max(pre_end, test_start - 1, test_end)
    runs = [load_episode_means(first_id, episodes, training_start, test_start, last_episode)]
    for experiment_id in args.experiment_ids[1:]:
        _, other_episodes, *other_boundaries = experiment_episodes(experiment_id)
        if other_boundaries != boundaries:
            parser.error(f"{experiment_id} has different phase boundaries from {first_id}")
        runs.append(load_episode_means(experiment_id, other_episodes, training_start, test_start, last_episode))

    multi_seed = len(runs) > 1
    if multi_seed:
        common_days = runs[0].index
        for daily in runs[1:]:
            common_days = common_days.intersection(daily.index)
        if not ((common_days >= training_start) & (common_days < test_start)).any():
            parser.error("The seeds have no saved training episodes in common")
        print(f"Using {len(common_days)} shared episode numbers across {len(runs)} runs")
        daily = pd.concat([run.loc[common_days] for run in runs], axis=1, keys=args.experiment_ids)
        daily = daily.swaplevel(axis=1)
        subtitle = f"{len(runs)} seeds · {first_id} and replicates"
        legend_title = f"{confidence:.0%} pointwise CI across seeds"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
        filename = f"{first_id}_training_travel_times_{len(runs)}_seeds_{stamp}.png"
    else:
        daily = runs[0]
        window = options["window_saved_episodes"]
        if window is None:
            smooth_by = int(config.get("smooth_by", 10))
            # RouteRL averages the current saved episode plus smooth_by previous ones.
            window = max(1, len(episodes)) if smooth_by == 0 else max(1, smooth_by + 1)
        if not isinstance(window, int) or window < 1:
            raise ValueError("window_saved_episodes must be a positive integer or null")
        subtitle = first_id
        legend_title = f"{confidence:.0%} pointwise CI across episodes" if window > 1 else "Per-episode means"
        filename = f"{first_id}_training_travel_times_ci.png"

    fig, ax = plt.subplots(figsize=style["figsize"]["training"], layout="constrained")
    ax.set_title(f"Mean travel times over episodes\n{subtitle}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean travel time (min)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=style["grid_alpha"])

    # Restart the rolling window at phase boundaries so it cannot mix different cohorts or policies.
    phases = [daily.loc[daily.index <= pre_end],
              daily.loc[(daily.index > pre_end) & (daily.index < training_start)],
              daily.loc[(daily.index >= training_start) & (daily.index < test_start)],
              daily.loc[daily.index >= test_start]]
    for kind, label, key in [("Human", "Humans", "human"), ("AV", "AVs", "av")]:
        if kind not in daily:
            continue
        color = style["colors"][key]
        for phase in phases:
            episode_means = phase[kind]
            if multi_seed:
                complete = episode_means.dropna()
                if complete.empty or complete.shape[1] != len(runs):
                    continue
                # Leave gaps where any seed is missing this cohort.
                mean, lower, upper = seed_interval(complete, confidence)
                mean, lower, upper = (series.reindex(phase.index) for series in (mean, lower, upper))
            else:
                episode_means = episode_means.dropna()
                if episode_means.empty:
                    continue
                mean, lower, upper = rolling_interval(episode_means, window, confidence)
            ax.plot(mean.index, mean, color=color, linewidth=options["linewidth"], label=label,
                    marker="." if len(mean) == 1 else None)
            ax.fill_between(mean.index, lower, upper, color=color, alpha=options["band_alpha"], linewidth=0)
            label = "_nolegend_"
    if pre_end > 0 and daily.index.min() <= pre_end < daily.index.max():
        ax.axvline(pre_end, label="AV deployment", **options["phase_line"])
    if daily.index.min() < test_start <= daily.index.max():
        ax.axvline(test_start - 1, label="Testing begins", **options["phase_line"])
    ax.legend(title=legend_title)
    save_plot(fig, filename, style)


if __name__ == "__main__":
    main()
