import argparse
import pickle
from pathlib import Path

import paz
import figures
from draw_program import write_prior_program


if __name__ == "__main__":
    description = "Redraw the prior figures of an optimize_bijectors run"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--directory", type=str, required=True)
    args = parser.parse_args()

    directory = Path(args.directory)
    priors = paz.inference.load(directory / "priors").inputs
    with open(directory / "figure_data.pkl", "rb") as filedata:
        figure_data = pickle.load(filedata)
    figures_directory = directory / "figures"
    figures.configure()
    figures.write_prior_figures(figures_directory, priors, *figure_data)
    sample_paths = (directory / "prior_predictive_samples").glob("*.png")
    write_prior_program(figures_directory, sorted(sample_paths))
