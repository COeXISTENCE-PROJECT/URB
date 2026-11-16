import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
PLOTS = HERE / "plots"


def load_style():
    with (HERE / "config.json").open(encoding="utf-8") as file:
        style = json.load(file)
    plt.rcParams.update(style["rc_params"])
    return style


def phase_means(files):
    episodes = []
    for file in files:
        episode = pd.read_csv(file, usecols=["id", "kind", "travel_time"]).set_index("id").sort_index()
        if episode.empty or episode.index.hasnans or not episode.index.is_unique:
            raise ValueError(f"Expected one row per agent in {file}")
        if not episode["kind"].isin(["Human", "AV"]).all():
            raise ValueError(f"Unexpected agent type in {file}")
        if not np.isfinite(episode["travel_time"]).all() or (episode["travel_time"] <= 0).any():
            raise ValueError(f"Expected positive, finite travel times in {file}")
        if episodes and not episode["kind"].equals(episodes[0]["kind"]):
            raise ValueError(f"Agent IDs or types changed within the phase at {file}")
        episodes.append(episode)
    means = pd.concat(episodes)["travel_time"].groupby(level="id").mean()
    return episodes[0][["kind"]].assign(travel_time=means)


def experiment_episodes(experiment_id):
    folder = RESULTS / experiment_id
    with (folder / "exp_config.json").open(encoding="utf-8") as file:
        config = json.load(file)

    pre_end = int(config["human_learning_episodes"])
    if "training_eps" in config:
        training_days = int(config["training_eps"])
    else:
        training_days = int(config["n_iters"]) * int(config["agent_frames_per_batch"])
    training_start = pre_end + int(config.get("experience_collecting_episodes", 0)) + 1
    test_start = training_start + training_days
    test_end = test_start + int(config.get("test_eps", 0)) - 1
    episodes = sorted((int(path.stem[2:]), path) for path in (folder / "episodes").glob("ep*.csv"))
    return config, episodes, pre_end, training_start, test_start, test_end


def load_travel_times(experiment_id, baseline_days):
    config, episodes, pre_end, _, test_start, test_end = experiment_episodes(experiment_id)
    pre_files = [path for day, path in episodes if pre_end - baseline_days < day <= pre_end]
    test_files = [path for day, path in episodes if test_start <= day <= test_end]
    if not pre_files or not test_files:
        raise ValueError(f"{experiment_id} needs saved episodes in both the baseline and testing windows")

    before, after = phase_means(pre_files), phase_means(test_files)
    if not before.index.equals(after.index) or len(before) != int(config.get("num_agents", len(before))):
        raise ValueError(f"{experiment_id} has incomplete or unmatched agent IDs across phases")

    times = before.join(after, lsuffix="_before", rsuffix="_after")
    print(f"{experiment_id}: {len(pre_files)} saved baseline episodes, {len(test_files)} saved test episodes")
    return times


def inlier_mask(values, style):
    tail = style["outlier_tail_fraction"]
    if not 0 <= tail < 0.5:
        raise ValueError("outlier_tail_fraction must be at least 0 and less than 0.5")
    if tail == 0 or len(values) < 3:
        return pd.Series(True, index=values.index)
    lower, upper = values.quantile([tail, 1 - tail])
    return values.between(lower, upper)


def experiment_label(experiment_id, style):
    label = style["experiment_labels"].get(experiment_id, experiment_id)
    return fill(label.replace("_", "_ "), width=style["label_wrap"]).replace("_ ", "_")


def describe_change(percent_change, subject, reference):
    change = round(percent_change, 1)
    if change == 0:
        return f"{subject} have similar mean travel times to {reference}."
    direction = "shorter" if change < 0 else "longer"
    return f"{subject} average {abs(change):.1f}% {direction} travel times than {reference}."


def plot_groups(ax, data, colors, style, plot_kind, labels=None):
    if any(len(values) == 0 for values in data):
        raise ValueError("Every plotted group needs at least one agent. Try lowering outlier_tail_fraction.")
    if plot_kind == "kde":
        import seaborn as sns

        for values, color, label in zip(data, colors, labels):
            label = label.replace("\n", " ")
            if values.nunique() == 1:
                ax.axvline(values.mean(), color=color, label=label, **style["kde_mean"])
            else:
                sns.kdeplot(x=values, color=color, label=label, ax=ax, **style["kde"])
                ax.axvline(values.mean(), color=color, **style["kde_mean"])
    else:
        boxes = ax.boxplot(data, patch_artist=True, **style["boxplot"])
        for box, color in zip(boxes["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(style["box_alpha"])
        if labels is not None:
            ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=style["grid_alpha"])


def comparison_colors(style, count):
    palette = style["colors"]["comparison_palette"]
    if isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        if count > 1 and (not hasattr(cmap, "colors") or cmap.N >= 256 or count > cmap.N):
            return cmap(np.linspace(0, 1, count))
        return [cmap(index) for index in range(count)]
    return [palette[index % len(palette)] for index in range(count)]


def save_plot(fig, filename, style):
    PLOTS.mkdir(parents=True, exist_ok=True)
    path = PLOTS / filename
    fig.savefig(path, dpi=style["dpi"])
    plt.close(fig)
    print(f"Saved {path}")
