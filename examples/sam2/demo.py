"""Demo image for the SAM 2 example, fetched on first use."""
from keras.utils import get_file

import paz

URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


def fetch_image():
    path = get_file("sam2_cats.jpg", URL, cache_subdir="paz/examples/sam2")
    return paz.image.load(path)
