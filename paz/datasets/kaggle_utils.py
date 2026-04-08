import shutil
import tarfile
import zipfile
from pathlib import Path


def build_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as error:
        raise ImportError(build_import_message()) from error
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as error:
        raise RuntimeError(build_credentials_message()) from error
    return api


def download_competition(competition_name, dataset_root, overwrite=False):
    api = build_api()
    dataset_root = Path(dataset_root)
    archive_path = dataset_root / f"{competition_name}.zip"
    if overwrite and archive_path.exists():
        archive_path.unlink()
    path = str(dataset_root)
    try:
        api.competition_download_files(competition_name, path=path, quiet=False)
    except Exception as error:
        message = build_download_message(competition_name)
        raise RuntimeError(message) from error
    return archive_path


def extract_archive(archive_path, output_path):
    archive_path = Path(archive_path)
    output_path = Path(output_path)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(output_path)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            try:
                archive.extractall(output_path, filter="data")
            except TypeError:
                archive.extractall(output_path)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def download_dataset(
    competition_name,
    dataset_root,
    csv_path,
    extracted_path,
    nested_archive=None,
    overwrite=False,
):
    dataset_root = Path(dataset_root)
    csv_path = Path(csv_path)
    extracted_path = Path(extracted_path)
    if csv_path.exists() and not overwrite:
        return dataset_root
    dataset_root.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_root / f"{competition_name}.zip"
    if overwrite or not archive_path.exists():
        archive_path = download_competition(
            competition_name, dataset_root, overwrite
        )
    extract_archive(archive_path, dataset_root)
    if nested_archive is not None:
        extract_archive(dataset_root / nested_archive, dataset_root)
    shutil.copyfile(extracted_path, csv_path)
    return dataset_root


def build_import_message():
    return "Please install kaggle to download datasets: pip install kaggle"


def build_credentials_message():
    message = "Kaggle credentials not found. "
    message += "Export KAGGLE_USERNAME and KAGGLE_KEY in your bashrc."
    return message


def build_download_message(competition_name):
    message = f"Failed to download {competition_name} from Kaggle. "
    message += "Accept the competition rules and retry the download."
    return message
