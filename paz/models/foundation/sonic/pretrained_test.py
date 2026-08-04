import numpy as np

from paz.models.foundation.sonic.layout import compute_encoder_input_dim
from paz.models.foundation.sonic.pretrained import SONIC
from paz.models.foundation.sonic.pretrained import fetch_motion_assets
from paz.models.foundation.sonic.pretrained import fetch_scene_assets


def test_pretrained_sonic_downloads_and_runs():
    sonic = SONIC(weights="pretrained")
    layout = sonic.layout
    x = np.zeros((1, compute_encoder_input_dim(layout)), dtype="float32")
    tokens = np.array(sonic.encoder(x, training=False))
    assert tokens.shape == (1, layout.token_dim)
    assert np.isfinite(tokens).all()


def test_fetch_scene_assets_downloads_and_extracts():
    scene_path = fetch_scene_assets()
    assert scene_path.name == "scene_43dof.xml"
    assert scene_path.exists()
    assert (scene_path.parent / "meshes").is_dir()


def test_fetch_motion_assets_downloads_and_extracts():
    motion_dir = fetch_motion_assets()
    clip_dirs = [path for path in motion_dir.iterdir() if path.is_dir()]
    assert len(clip_dirs) > 0
    assert (clip_dirs[0] / "joint_pos.csv").exists()
