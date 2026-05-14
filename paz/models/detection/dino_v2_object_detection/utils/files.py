import requests
from tqdm import tqdm


def download_file(url, filename):
    """Download a file from *url* and save it to *filename*.

    Streams the response in 1 KiB chunks and displays a tqdm
    progress bar during the download.

    Args:
        url (str): Source URL.
        filename (str): Local file path to write to.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)
