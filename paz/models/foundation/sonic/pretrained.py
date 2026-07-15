"""Download and build the pretrained SONIC deploy actor.

The v0.27 release assets are not uploaded yet: run
paz.models.foundation.sonic.conversion against a real release directory
(see conversion.py's main()) to produce them, then publish that output
directory's contents to SONIC_ASSETS_URL before Sonic(weights="pretrained")
can succeed.
"""

from collections import namedtuple

from keras.utils import get_file

from paz.models.foundation.sonic.layout import load_release_observation_layout
from paz.models.foundation.sonic.model import build_sonic_actor
from paz.models.foundation.sonic.model import build_sonic_decoder
from paz.models.foundation.sonic.model import build_sonic_encoder

SONIC_ASSETS_URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.27/"  # fmt: skip
SONIC_CACHE_SUBDIR = "paz/models/sonic"

SonicModels = namedtuple("Sonic", "layout encoder decoder actor")


def Sonic(weights="pretrained"):
    layout = build_layout()
    encoder = build_sonic_encoder(layout)
    decoder = build_sonic_decoder(layout)
    actor = build_sonic_actor(layout, encoder, decoder)
    if weights == "pretrained":
        load_pretrained_weights(encoder, decoder)
    return SonicModels(layout, encoder, decoder, actor)


def build_layout():
    obs_config = fetch_sonic_asset("observation_config.yaml")
    return load_release_observation_layout(obs_config)


def load_pretrained_weights(encoder, decoder):
    encoder.load_weights(fetch_sonic_asset("sonic_encoder.weights.h5"))
    decoder.load_weights(fetch_sonic_asset("sonic_decoder.weights.h5"))


def fetch_sonic_asset(filename):
    return get_file(filename, SONIC_ASSETS_URL + filename,
                     cache_subdir=SONIC_CACHE_SUBDIR)
