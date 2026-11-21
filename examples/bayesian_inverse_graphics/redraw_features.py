import argparse
import pickle
from pathlib import Path

import figures


if __name__ == "__main__":
    description = "Redraw the feature figures of an extract_features run"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--directory", type=str, required=True)
    args = parser.parse_args()

    directory = Path(args.directory)
    with open(directory / "figure_data.pkl", "rb") as filedata:
        figure_data = pickle.load(filedata)
    figures.configure()
    figures.write_feature_figures(directory / "figures", *figure_data)
