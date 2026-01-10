import os
import csv
import json

import keras


def write_json(dictionary, filepath, indent=4):
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
    weights_path = os.path.join(directory, name + ".weights.h5")
    model.save_weights(weights_path)


def load_latest(wildcard, filename):
    from paz.backend import directory

    filepath = directory.find_latest(wildcard)
    filepath = os.path.join(filepath, filename)
    filedata = open(filepath, "r")
    parameters = json.load(filedata)
    return parameters


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
