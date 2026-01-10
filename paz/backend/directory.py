import os
import glob
import shutil
from pathlib import Path
from datetime import datetime


def make_timestamped(root="experiments", label=None):
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
    return make(directory_name)


def make(directory_name):
    """Makes directory.

    # Arguments:
        directory_name: String. Directory name.
    """
    Path(directory_name).mkdir(parents=True, exist_ok=True)
    return directory_name


def find_latest(wildcard):
    filenames = glob.glob(wildcard)
    filepaths = []
    for filename in filenames:
        if os.path.isdir(filename):
            filepaths.append(filename)
    return max(filepaths, key=os.path.getmtime)


def exists(path):
    return os.path.isdir(path)


def is_empty(path):
    return len(os.listdir(path)) == 0


def list_subdirectories(path):
    entries = os.listdir(path)
    subdirectories = []
    for entry in entries:
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            subdirectories.append(full_path)
    return subdirectories


def list_files(path, pattern=None):
    if pattern is None:
        pattern = "*"
    search_pattern = os.path.join(path, pattern)
    all_matches = glob.glob(search_pattern)
    files = []
    for match in all_matches:
        if os.path.isfile(match):
            files.append(match)
    return files


def remove(path):
    shutil.rmtree(path)


def size(path):
    total = 0
    for root, subdirectories, files in os.walk(path):
        for filename in files:
            filepath = os.path.join(root, filename)
            total += os.path.getsize(filepath)
    return total
