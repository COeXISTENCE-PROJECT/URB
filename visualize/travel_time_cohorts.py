import argparse

import matplotlib.pyplot as plt

from common import describe_change, inlier_mask, load_style, load_travel_times, plot_groups, save_plot


def main(plot_kind="boxplot"):
    parser = argparse.ArgumentParser(description="Compare travel times before and after deployment for matched cohorts.")
    parser.add_argument("experiment_id")
    args = parser.parse_args()

    style = load_style()
    times = load_travel_times(args.experiment_id, style["baseline_days"])
    humans_before = times[times["kind_before"] == "Human"]
    stayed_human = humans_before[humans_before["kind_after"] == "Human"]
    converted = humans_before[humans_before["kind_after"] == "AV"]

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=style["figsize"]["cohorts"], layout="constrained")
    fig.suptitle(f"Travel times for matched cohorts\n{args.experiment_id}")
    if plot_kind == "kde":
        axes[1].sharex(axes[0])
        fig.supxlabel("Mean travel time per agent (min)")
        fig.supylabel("Density")
    else:
        fig.supylabel("Mean travel time per agent (min)")
    for ax, cohort, title, kind in zip(axes, [stayed_human, converted], ["Remained human", "Converted to AV"], ["human", "av"]):
        before, after = cohort["travel_time_before"], cohort["travel_time_after"]
        # Keep the same agents before and after within each cohort.
        keep = inlier_mask(before, style) & inlier_mask(after, style)
        before, after = before[keep], after[keep]
        colors = [style["colors"]["before"], style["colors"][kind]]
        plot_groups(ax, [before, after], colors, style, plot_kind, ["Before deployment", "After training"])
        change = 100 * (after.mean() / before.mean() - 1)
        subject = "Non-converting humans" if kind == "human" else "AV users"
        summary = describe_change(change, subject, "before")
        ax.set_title(f"{title} (n = {len(before):,})\n{summary}", fontsize=style["rc_params"]["font.size"])
        if plot_kind == "kde":
            ax.set(xlabel="", ylabel="")
            ax.legend(title="Dashed lines show means")
    suffix = "_kde" if plot_kind == "kde" else ""
    save_plot(fig, f"{args.experiment_id}_travel_time_cohorts{suffix}.png", style)


if __name__ == "__main__":
    main()
