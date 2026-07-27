"""Download and build the pretrained GEAR-WBC lower-body actors.

Weights are Model Derivatives licensed by NVIDIA Corporation under the
NVIDIA Open Model License (see the release's gear_wbc_LICENSE.txt /
gear_wbc_NOTICE.txt at GEAR_WBC_ASSETS_URL). By calling GearWBC(weights=
"pretrained") you agree to that Agreement and to NVIDIA's Trustworthy AI
terms (https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/).
"""

from collections import namedtuple

from keras.utils import get_file

from paz.models.foundation.gear_wbc.model import build_actor

GEAR_WBC_ASSETS_URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.30/"  # fmt: skip
GEAR_WBC_CACHE_SUBDIR = "paz/models/gear_wbc"

# The release ships two experts over one architecture: "balance" holds a
# stance under near-zero velocity commands, "walk" locomotes.
GearWBCModels = namedtuple("GearWBCModels", "balance walk")


def GearWBC(weights="pretrained"):
    balance, walk = build_actor(), build_actor()
    if weights == "pretrained":
        load_pretrained_weights(balance, walk)
    return GearWBCModels(balance, walk)


def load_pretrained_weights(balance, walk):
    balance.load_weights(fetch_asset("gear_wbc_balance.weights.h5"))
    walk.load_weights(fetch_asset("gear_wbc_walk.weights.h5"))


def fetch_asset(filename):
    url = GEAR_WBC_ASSETS_URL + filename
    return get_file(filename, url, cache_subdir=GEAR_WBC_CACHE_SUBDIR)
