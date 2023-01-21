import os
import json
import glob
from pathlib import Path
from datetime import datetime


def build_directory(root='experiments', label=None):
    directory_name = build_directory_name(root, label)
    make_directory(directory_name)
    return directory_name


def build_directory_name(root, label=None):
    directory_name = [datetime.now().strftime("%d-%m-%Y_%H-%M-%S")]
    if label is not None:
        directory_name.extend([label])
    directory_name = '_'.join(directory_name)
    return os.path.join(root, directory_name)


def make_directory(directory_name):
    Path(directory_name).mkdir(parents=True, exist_ok=True)


def write_dictionary(dictionary, directory, filename, indent=4):
    fielpath = os.path.join(directory, filename)
    filedata = open(fielpath, 'w')
    json.dump(dictionary, filedata, indent=indent)


def write_argparse(args, directory, filename='parameters.json', indent=4):
    write_dictionary(args.__dict__, directory, filename, indent)


def write_weights(model, directory, name=None):
    name = model.name if name is None else name
    weights_path = os.path.join(directory, name + '.hdf5')
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
