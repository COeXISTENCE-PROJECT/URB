import argparse
from datetime import datetime, timezone

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from common import comparison_colors, experiment_label, inlier_mask, load_style, load_travel_times, plot_groups, save_plot


def main(plot_kind="boxplot"):
    parser = argparse.ArgumentParser(description="Compare post-training travel times and the change for converted AVs.")
    parser.add_argument("experiment_ids", nargs="+")
    args = parser.parse_args()

    style = load_style()
    labels, humans, avs, changes = [], [], [], []
    for experiment_id in args.experiment_ids:
        times = load_travel_times(experiment_id, style["baseline_days"])
        labels.append(experiment_label(experiment_id, style))
        humans.append(times.loc[times["kind_after"] == "Human", "travel_time_after"])
        avs.append(times.loc[times["kind_after"] == "AV", "travel_time_after"])
        converted = times[(times["kind_before"] == "Human") & (times["kind_after"] == "AV")]
        before = converted["travel_time_before"]
        changes.append((before - converted["travel_time_after"]) / before)

    width, height = style["figsize"]["comparison"]
    kde = plot_kind == "kde"
    if not kde:
        width = max(width, style["comparison_width_per_experiment"] * len(labels))
    fig, axes = plt.subplots(3, 1, sharex=not kde, figsize=(width, height), layout="constrained")
    title = "Travel times across experiments" if len(labels) > 1 else "Travel times for one experiment"
    colors = comparison_colors(style, len(labels))
    changes = [group[inlier_mask(group, style)] for group in changes]
    humans = [group[inlier_mask(group, style)] for group in humans]
    avs = [group[inlier_mask(group, style)] for group in avs]
    for ax, data, panel_title in zip(axes, [humans, avs, changes],
                                    ["Humans after training", "AVs after training", "Travel time change for converted AVs"]):
        plot_groups(ax, data, colors, style, plot_kind, labels if kde else None)
        ax.set_title(panel_title)
    if len(labels) == 1:
        change = -100 * changes[0].mean()
        if round(change, 1) == 0:
            summary = "Converted AV users saw little average change."
        else:
            direction = "reduction" if change < 0 else "increase"
            summary = f"Converted AV users saw a {abs(change):.1f}% average {direction} in travel time."
        title = f"{title}\n{summary}"
    fig.suptitle(title)
    if kde:
        human_limits, av_limits = (ax.get_xlim() for ax in axes[:2])
        for ax in axes[:2]:
            ax.set_xlim(min(human_limits[0], av_limits[0]), max(human_limits[1], av_limits[1]))
        for ax in axes[:2]:
            ax.set_xlabel("Mean per agent (min)")
        for ax in axes:
            ax.set_ylabel("Density")
        axes[0].legend(title="Dashed lines show means")
        axes[2].set_xlabel("Travel time reduction")
        axes[2].xaxis.set_major_formatter(PercentFormatter(xmax=1))
        axes[2].axvline(0, **style["zero_line"])
    else:
        axes[1].sharey(axes[0])
        axes[0].set_ylabel("Mean per agent (min)")
        axes[1].set_ylabel("Mean per agent (min)")
        axes[2].set_ylabel("Travel time reduction")
        axes[2].yaxis.set_major_formatter(PercentFormatter(xmax=1))
        axes[2].axhline(0, **style["zero_line"])
        axes[2].set_xticks(range(1, len(labels) + 1), labels)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    suffix = "_kde" if kde else ""
    save_plot(fig, f"travel_time_comparison{suffix}_{stamp}.png", style)


if __name__ == "__main__":
    main()
