import os
import json
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import paz
import paz.utils.plot as plot


def collect_results(results_dir, max_seeds_per_way):
    aggregated_results = defaultdict(lambda: defaultdict(list))
    dir_pattern = re.compile(r".*_(\d+)-WAY_RUN-\d+_SEED-\d+")

    try:
        dir_list = sorted(os.listdir(results_dir))
    except FileNotFoundError:
        print(f"Error: The directory '{results_dir}' was not found.")
        return None

    # Keep track of how many seeds we've processed for each 'way'
    seeds_processed_count = defaultdict(int)

    print(f"Scanning for results in '{os.path.abspath(results_dir)}'...")

    for dirname in dir_list:
        dir_path = os.path.join(results_dir, dirname)
        if not os.path.isdir(dir_path):
            continue

        match = dir_pattern.match(dirname)
        if not match:
            continue

        train_ways = int(match.group(1))

        # Respect the --num_seeds limit for each 'way' configuration
        if seeds_processed_count[train_ways] >= max_seeds_per_way:
            continue

        json_path = os.path.join(dir_path, "results.json")
        if not os.path.exists(json_path):
            print(f"Warning: 'results.json' not found in {dirname}")
            continue

        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            for test_scenario, accuracy in data.items():
                aggregated_results[test_scenario][train_ways].append(accuracy)

            seeds_processed_count[train_ways] += 1

        except (json.JSONDecodeError, TypeError) as e:
            print(
                f"Warning: Could not read or parse 'results.json' in {dirname}. Error: {e}"
            )

    print(
        f"\nProcessed data for {len(seeds_processed_count)} training configurations."
    )
    for way, count in sorted(seeds_processed_count.items()):
        print(f" - Found {count} runs for {way}-WAY training.")

    return aggregated_results


def plot_results(results, num_seeds, filepath=None):
    config = plot.build_configuration(
        "max", fontsize=25, label_pads=(10, 10)
    )
    plt.rcParams.update(
        {
            # "font.family": "serif",
            # "font.serif": ["Georgia"],  # A clean, readable serif font
            "axes.labelsize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 11,
        }
    )
    figure, axis = plt.subplots(figsize=config.figsize)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = colors[: len(results)][::-1]
    for i, (test_scenario, data_by_ways) in enumerate(sorted(results.items())):
        sorted_ways = sorted(data_by_ways.keys())
        mean_accuracies = [np.mean(data_by_ways[way]) for way in sorted_ways]
        stdv_accuracies = [np.std(data_by_ways[way]) for way in sorted_ways]
        mean_accuracies = np.array(mean_accuracies)
        stdv_accuracies = np.array(stdv_accuracies)
        test_scenario = " ".join(test_scenario.split("_")[0:2])

        axis.plot(
            sorted_ways,
            mean_accuracies,
            marker="o",
            linestyle="-",
            linewidth=3,
            markersize=9,
            color=colors[i],
            label=test_scenario,
        )

        axis.fill_between(
            sorted_ways,
            mean_accuracies - stdv_accuracies,
            mean_accuracies + stdv_accuracies,
            color=colors[i],
            alpha=0.15,
        )

    axis.yaxis.grid(
        True, linestyle="--", which="major", color="lightgrey", alpha=0.7
    )
    axis.set_axisbelow(True)

    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        handles[::-1],
        labels[::-1],
        frameon=True,
        edgecolor="white",
    )

    plot.hide_axes(axis)
    plot.set_label_pads(axis, config)
    axis.set_ylim(45, 100)
    axis.set_xlabel("Number of Training Classes")
    axis.set_ylabel("Test Accuracy (\\%)")
    plt.tight_layout()
    plot.write_or_show(figure, filepath)


parser = argparse.ArgumentParser(description="Plot accuracies across classes.")
parser.add_argument("--results_dir", type=str, default="experiments")
parser.add_argument("--num_seeds", type=int, default=20)
parser.add_argument(
    "--output_file",
    type=str,
    default="omniglot_test-accuracy_vs_training-classes.pdf",
)
args = parser.parse_args()

data = collect_results(args.results_dir, args.num_seeds)


def remove_between_alphabet_results(data):
    for key in list(data.keys()):
        if "between" in key:
            data.pop(key)
    return data


data = remove_between_alphabet_results(data)

if data:
    plot_results(data, args.num_seeds, args.output_file)
