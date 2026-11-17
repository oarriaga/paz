import argparse

import paz
import paz.utils.plot as plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discretize distributions")
    parser.add_argument("--normal_steps", type=int, default=31)
    parser.add_argument("--skewed_steps", type=int, default=25)
    args = parser.parse_args()

    normal = paz.distributions.Normal(0.0, 1.0)
    skewed = paz.distributions.LogNormal(0.0, 0.6)

    plot.configure(fontsize=12)
    figure, axes = plot.subplots(ncols=2, figsize=(12, 4))
    normal_args = (normal, -3.0, 3.0, args.normal_steps, axes[0])
    skewed_args = (skewed, 0.0, 6.0, args.skewed_steps, axes[1])
    plot.discretized_distribution(*normal_args, "N discrete")
    plot.discretized_distribution(*skewed_args, "LogNormal discrete")
    figure.tight_layout()
    plot.show()
