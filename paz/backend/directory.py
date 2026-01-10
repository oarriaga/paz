import glob
import shutil
from pathlib import Path
from datetime import datetime


def make_timestamped(root="experiments", label=None):
    """Builds and makes directory with time date and user given label.

    # Arguments:
        root: String or Path with partial or full path.
        label: String user label.

    # Returns
        Full directory path as string
    """
    root = Path(root)
    directory_name = [datetime.now().strftime("%d-%m-%Y_%H-%M-%S")]
    if label is not None:
        directory_name.extend([label])
    directory_name = "_".join(directory_name)
    return make(root / directory_name)


def make(directory_name):
    """Makes directory.

    # Arguments:
        directory_name: String or Path. Directory name.

    # Returns
        Full directory path as string
    """
    path = Path(directory_name)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def find_latest(wildcard):
    wildcard = str(wildcard) if isinstance(wildcard, Path) else wildcard
    filenames = glob.glob(wildcard)
    filepaths = []
    for filename in filenames:
        path = Path(filename)
        if path.is_dir():
            filepaths.append(filename)
    return max(filepaths, key=lambda p: Path(p).stat().st_mtime)


def exists(path):
    return Path(path).is_dir()


def is_empty(path):
    path = Path(path)
    return len(list(path.iterdir())) == 0


def list_subdirectories(path):
    path = Path(path)
    subdirectories = []
    for entry in path.iterdir():
        if entry.is_dir():
            subdirectories.append(str(entry))
    return subdirectories


def list_files(path, pattern=None):
    path = Path(path)
    if pattern is None:
        pattern = "*"
    files = []
    for match in path.glob(pattern):
        if match.is_file():
            files.append(str(match))
    return files


def remove(path):
    shutil.rmtree(path)


def size(path):
    path = Path(path)
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total
