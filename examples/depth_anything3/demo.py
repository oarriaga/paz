"""Demo images for the Depth Anything 3 examples, fetched on first use.

The multi-view pair are the Sydney Opera House frames shipped with the
official Depth Anything 3 repository; the indoor image is a COCO sample.
"""
from keras.utils import get_file

import paz

SOH = "https://raw.githubusercontent.com/ByteDance-Seed/Depth-Anything-3/main/assets/examples/SOH/"  # fmt: skip
URLS = {
    "opera_house_0": SOH + "000.png",
    "opera_house_1": SOH + "010.png",
    "indoor": "http://images.cocodataset.org/val2017/000000039769.jpg",
}


def fetch_image(name):
    url = URLS[name]
    path = get_file(name + url[-4:], url, cache_subdir="paz/examples/da3")
    return paz.image.load(path)
