import os
import json
import glob
from pathlib import Path
from datetime import datetime


def build_directory(root='experiments', label=None):
    """Builds and makes directory with time date and user given label.

    # Arguments:
        root: String with partial or full path.
        label: String user label.

    # Returns
        Full directory path
    """
    directory_name = build_directory_name(root, label)
    make_directory(directory_name)
    return directory_name


def build_directory_name(root, label=None):
    """Build directory name with time date and user label

    # Arguments:
        root: String with partial or full path.
        label: String user label.

    # Returns
        Full directory path
    """
    directory_name = [datetime.now().strftime("%d-%m-%Y_%H-%M-%S")]
    if label is not None:
        directory_name.extend([label])
    directory_name = '_'.join(directory_name)
    return os.path.join(root, directory_name)


def make_directory(directory_name):
    """Makes directory.

    # Arguments:
        directory_name: String. Directory name.
    """
    Path(directory_name).mkdir(parents=True, exist_ok=True)


def write_dictionary(dictionary, directory, filename, indent=4):
    """Writes dictionary as json file.

    # Arguments:
        dictionary: Dictionary to write in memory.
        directory: String. Directory name.
        filename: String. Filename.
        indent: Number of spaces between keys.
    """
    fielpath = os.path.join(directory, filename)
    filedata = open(fielpath, 'w')
    json.dump(dictionary, filedata, indent=indent)


def write_weights(model, directory, name=None):
    """Writes Keras weights in memory.

    # Arguments:
        model: Keras model.
        directory: String. Directory name.
        name: String or `None`. Weights filename.
    """
    name = model.name if name is None else name
    weights_path = os.path.join(directory, name + '_weights.hdf5')
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
    filedata = open(filepath, 'r')
    parameters = json.load(filedata)
    return parameters
