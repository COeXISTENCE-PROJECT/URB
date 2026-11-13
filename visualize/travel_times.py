import argparse

import matplotlib.pyplot as plt

from common import describe_change, inlier_mask, load_style, load_travel_times, plot_groups, save_plot


def main(plot_kind="boxplot"):
    parser = argparse.ArgumentParser(description="Compare baseline human and test-phase human and AV travel times.")
    parser.add_argument("experiment_id")
    args = parser.parse_args()

    style = load_style()
    times = load_travel_times(args.experiment_id, style["baseline_days"])
    before = times.loc[times["kind_before"] == "Human", "travel_time_before"]
    humans = times.loc[times["kind_after"] == "Human", "travel_time_after"]
    avs = times.loc[times["kind_after"] == "AV", "travel_time_after"]

    before, humans, avs = [group[inlier_mask(group, style)] for group in (before, humans, avs)]
    fig, ax = plt.subplots(figsize=style["figsize"]["travel_times"], layout="constrained")
    fig.suptitle("Travel times before deployment and after training")
    labels = [f"Humans before mutation\n(n = {len(before):,})",
              f"Humans after training\n(n = {len(humans):,})",
              f"AVs after training\n(n = {len(avs):,})"]
    colors = [style["colors"][key] for key in ("before", "human", "av")]
    plot_groups(ax, [before, humans, avs], colors, style, plot_kind, labels)
    change = 100 * (avs.mean() / humans.mean() - 1)
    summary = describe_change(change, "AV users", "humans after training")
    ax.set_title(f"{args.experiment_id}\n{summary}")
    if plot_kind == "kde":
        ax.set(xlabel="Mean travel time per agent (min)", ylabel="Density")
        ax.legend(title="Dashed lines show means")
    else:
        ax.set_ylabel("Mean travel time per agent (min)")
    suffix = "kde" if plot_kind == "kde" else "boxplots"
    save_plot(fig, f"{args.experiment_id}_travel_time_{suffix}.png", style)


if __name__ == "__main__":
    main()
