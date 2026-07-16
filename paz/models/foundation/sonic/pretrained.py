"""Download and build the pretrained SONIC deploy actor.

The v0.27 weight assets are not uploaded yet: run
paz.models.foundation.sonic.conversion against a real release directory
(see conversion.py's main()) to produce them, then publish that output
directory's contents to SONIC_ASSETS_URL before SONIC(weights="pretrained")
can succeed.
"""

from collections import namedtuple

from keras.utils import get_file

from paz.models.foundation.sonic.layout import build_observation_layout
from paz.models.foundation.sonic.model import build_actor
from paz.models.foundation.sonic.model import build_decoder
from paz.models.foundation.sonic.model import build_encoder

SONIC_ASSETS_URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.27/"  # fmt: skip
SONIC_CACHE_SUBDIR = "paz/models/sonic"

SonicModels = namedtuple("SonicModels", "layout encoder decoder actor")

# The released observation contract (encoder branches, history spans, and
# their fixed dimensions). Small and release-specific, so it ships as a
# literal here instead of a separately downloaded/parsed YAML file.
RELEASE_OBSERVATION_CONFIG = {
    "observations": (
        {"name": "token_state", "enabled": True},
        {"name": "his_base_angular_velocity_10frame_step1", "enabled": True},
        {"name": "his_body_joint_positions_10frame_step1", "enabled": True},
        {"name": "his_body_joint_velocities_10frame_step1", "enabled": True},
        {"name": "his_last_actions_10frame_step1", "enabled": True},
        {"name": "his_gravity_dir_10frame_step1", "enabled": True},
    ),
    "encoder": {
        "dimension": 64,
        "encoder_observations": (
            {"name": "encoder_mode_4", "enabled": True},
            {"name": "motion_joint_positions_10frame_step5",
             "enabled": True},
            {"name": "motion_joint_velocities_10frame_step5",
             "enabled": True},
            {"name": "motion_root_z_position_10frame_step5",
             "enabled": True},
            {"name": "motion_root_z_position", "enabled": True},
            {"name": "motion_anchor_orientation", "enabled": True},
            {"name": "motion_anchor_orientation_10frame_step5",
             "enabled": True},
            {"name": "motion_joint_positions_lowerbody_10frame_step5",
             "enabled": True},
            {"name": "motion_joint_velocities_lowerbody_10frame_step5",
             "enabled": True},
            {"name": "vr_3point_local_target", "enabled": True},
            {"name": "vr_3point_local_orn_target", "enabled": True},
            {"name": "smpl_joints_10frame_step1", "enabled": True},
            {"name": "smpl_anchor_orientation_10frame_step1",
             "enabled": True},
            {"name": "motion_joint_positions_wrists_10frame_step1",
             "enabled": True},
        ),
        "encoder_modes": (
            {
                "name": "g1",
                "mode_id": 0,
                "required_observations": (
                    "encoder_mode_4",
                    "motion_joint_positions_10frame_step5",
                    "motion_joint_velocities_10frame_step5",
                    "motion_anchor_orientation_10frame_step5",
                ),
            },
            {
                "name": "teleop",
                "mode_id": 1,
                "required_observations": (
                    "encoder_mode_4",
                    "motion_joint_positions_lowerbody_10frame_step5",
                    "motion_joint_velocities_lowerbody_10frame_step5",
                    "vr_3point_local_target",
                    "vr_3point_local_orn_target",
                    "motion_anchor_orientation",
                ),
            },
            {
                "name": "smpl",
                "mode_id": 2,
                "required_observations": (
                    "encoder_mode_4",
                    "smpl_joints_10frame_step1",
                    "smpl_anchor_orientation_10frame_step1",
                    "motion_joint_positions_wrists_10frame_step1",
                ),
            },
        ),
    },
}


def SONIC(weights="pretrained"):
    layout = build_observation_layout(RELEASE_OBSERVATION_CONFIG)
    encoder = build_encoder(layout)
    decoder = build_decoder(layout)
    actor = build_actor(layout, encoder, decoder)
    if weights == "pretrained":
        load_pretrained_weights(encoder, decoder)
    return SonicModels(layout, encoder, decoder, actor)


def load_pretrained_weights(encoder, decoder):
    encoder.load_weights(fetch_asset("sonic_encoder.weights.h5"))
    decoder.load_weights(fetch_asset("sonic_decoder.weights.h5"))


def fetch_asset(filename):
    url = SONIC_ASSETS_URL + filename
    return get_file(filename, url, cache_subdir=SONIC_CACHE_SUBDIR)
