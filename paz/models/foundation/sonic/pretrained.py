"""Download and build the pretrained SONIC deploy actor.

Weights are Model Derivatives licensed by NVIDIA Corporation under the
NVIDIA Open Model License (see the release's sonic_LICENSE.txt /
sonic_NOTICE.txt at SONIC_ASSETS_URL). By calling SONIC(weights=
"pretrained") you agree to that Agreement and to NVIDIA's Trustworthy AI
terms (https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/).

The G1 MuJoCo scene, meshes, and example reference motions fetched by
fetch_scene_assets/fetch_motion_assets are a separate, Apache-2.0-licensed
part of the same release (see sonic_scene_LICENSE.txt / sonic_scene_
NOTICE.txt at SONIC_ASSETS_URL); no agreement is required for those.
"""

from collections import namedtuple
from pathlib import Path

from keras.utils import get_file

from paz.models.foundation.sonic.layout import build_observation_layout
from paz.models.foundation.sonic.model import build_actor
from paz.models.foundation.sonic.model import build_decoder
from paz.models.foundation.sonic.model import build_encoder
from paz.utils import extract

SONIC_ASSETS_URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.28/"  # fmt: skip
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


def fetch_scene_assets():
    scene_dir = extract(fetch_asset("sonic_scene.zip"))
    return Path(scene_dir) / "scene_43dof.xml"


def fetch_motion_assets():
    return Path(extract(fetch_asset("sonic_motions.zip")))


def fetch_asset(filename):
    url = SONIC_ASSETS_URL + filename
    return get_file(filename, url, cache_subdir=SONIC_CACHE_SUBDIR)
