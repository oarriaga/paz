import os
import csv
import json
import glob
from pathlib import Path
from datetime import datetime

import keras
import jax


def make_timestamped_directory(root="experiments", label=None):
    """Builds and makes directory with time date and user given label.

    # Arguments:
        root: String with partial or full path.
        label: String user label.

    # Returns
        Full directory path
    """

    def timestamp_directory(root, label=None):
        directory_name = [datetime.now().strftime("%d-%m-%Y_%H-%M-%S")]
        if label is not None:
            directory_name.extend([label])
        directory_name = "_".join(directory_name)
        return os.path.join(root, directory_name)

    directory_name = timestamp_directory(root, label)
    return make_directory(directory_name)


def make_directory(directory_name):
    """Makes directory.

    # Arguments:
        directory_name: String. Directory name.
    """
    Path(directory_name).mkdir(parents=True, exist_ok=True)
    return directory_name


def write_dictionary(dictionary, filepath, indent=4):
    """Writes a dictionary to a JSON file.

    Args:
        dictionary: The dictionary to write to the file.
        filepath: The full path to the output file.
        indent: The number of spaces to use for indentation in the JSON file.
    """
    with open(filepath, "w") as filedata:
        json.dump(dictionary, filedata, indent=indent)


def write_weights(model, directory, name=None):
    """Writes Keras weights in memory.

    # Arguments:
        model: Keras model.
        directory: String. Directory name.
        name: String or `None`. Weights filename.
    """
    name = model.name if name is None else name
    weights_path = os.path.join(directory, name + "_weights.hdf5")
    model.save_weights(weights_path)


def find_path(wildcard):
    filenames = glob.glob(wildcard)
    filepaths = []
    for filename in filenames:
        if os.path.isdir(filename):
            filepaths.append(filename)
    return max(filepaths, key=os.path.getmtime)


def load_latest(wildcard, filename):
    filepath = find_path(wildcard)
    filepath = os.path.join(filepath, filename)
    filedata = open(filepath, "r")
    parameters = json.load(filedata)
    return parameters


# def setup(args, model=None, label=None, root="experiments"):
#     labels = [item for item in [model, label] if item]
#     root = make_timestamped_directory(root, "_".join(labels))
#     write_dictionary(args.__dict__, root, "parameters.json")
#     keras.utils.set_random_seed(args.seed)
#     key = jax.random.PRNGKey(args.seed)
#     return root, key


def warn(message):
    print("\033[93m" + message + "\033[0m")


def setup(args):
    defaults = {"model": None, "label": None, "root": "log", "seed": 777}
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
            warn(f"`{key}` not found in `args`. Using default `{value}`.")

    keras.utils.set_random_seed(args.seed)
    labels = [item for item in [args.model, args.label] if item]
    experiment_name = "_".join(labels)
    experiment_root = make_timestamped_directory(args.root, experiment_name)
    filepath = os.path.join(experiment_root, "parameters.json")
    write_dictionary(args.__dict__, filepath)
    return experiment_root, jax.random.PRNGKey(args.seed)


def load_csv(filepath):

    def check_column_size(row_arg, row_values, num_columns):
        if len(row_values) != num_columns:
            raise ValueError(f"Invalid column size at row {row_arg + 1}")

    def initialize_data(header):
        return {column_name: [] for column_name in header}

    def process_row_value(value_str, column_arg, column_name):
        return int(value_str) if column_name == "epoch" else float(value_str)

    def build_header_names(header):
        return [column_name.strip() for column_name in header]

    try:
        with open(filepath, mode="r", newline="") as filedata:
            reader = csv.reader(filedata)
            header = build_header_names(next(reader, None))
            data = initialize_data(header)
            for row_arg, row_values in enumerate(reader, 1):
                check_column_size(row_arg, row_values, len(header))
                for column_arg, value in enumerate(row_values):
                    column_name = header[column_arg]
                    value = process_row_value(value, column_arg, column_name)
                    data[column_name].append(value)
    except FileNotFoundError:
        raise FileNotFoundError(f"The log file '{filepath}' was not found.")
    return data
