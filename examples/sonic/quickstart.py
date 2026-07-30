"""Standalone SONIC smoke test: no MuJoCo scene or motion clips required.

Downloads the released encoder/decoder weights and runs one actor step on
zeroed observations, proving the install works before wiring up
examples/sonic/mujoco_demo.py.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

from paz.models.foundation.sonic.layout import compute_encoder_input_dim
from paz.models.foundation.sonic.layout import compute_policy_tail_dim
from paz.models.foundation.sonic.pretrained import SONIC


if __name__ == "__main__":
    print("Downloading SONIC weights and building the PAZ actor...")
    sonic = SONIC(weights="pretrained")
    encoder_obs = np.zeros(
        (1, compute_encoder_input_dim(sonic.layout)), dtype="float32")
    policy_tail = np.zeros(
        (1, compute_policy_tail_dim(sonic.layout)), dtype="float32")
    inputs = {"encoder_obs": encoder_obs, "policy_obs_tail": policy_tail}
    action = np.array(sonic.actor(inputs, training=False))
    print(f"SONIC actor ran: action shape {action.shape}, "
          f"finite={np.isfinite(action).all()}")
