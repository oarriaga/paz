import os
from collections import namedtuple

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax.numpy as jp
import numpy as np
from jax import random as jr

import randomize

FIELDS = "geom_friction, geom_solref, body_mass, nbody, body_ipos, actuator_gainprm, actuator_biasprm, dof_frictionloss, dof_armature"  # fmt: skip
FakeModel = namedtuple("FakeModel", FIELDS)

NUM_JOINTS, TORSO = 4, 1


def build_model():
    num_dofs = 6 + NUM_JOINTS
    args = jp.ones((3, 3)), jp.ones((3, 2)), jp.full(3, 2.0), 3
    args = args + (jp.zeros((3, 3)), jp.full((NUM_JOINTS, 10), 40.0))
    args = args + (jp.full((NUM_JOINTS, 10), -2.0), jp.zeros(num_dofs))
    return FakeModel(*args, jp.zeros(num_dofs))


def test_randomized_ranges():
    model = build_model()
    for seed in range(10):
        values = randomize.physics_model(jr.key(seed), model, NUM_JOINTS, TORSO)  # fmt: skip
        friction = np.asarray(values["dof_frictionloss"])[6:]
        assert friction.min() >= 0.0 and friction.max() <= 0.05
        armature = np.asarray(values["dof_armature"])[6:]
        assert armature.min() >= 0.0 and armature.max() <= 0.005
        mass = np.asarray(values["body_mass"])
        assert mass[0] / 2.0 >= 0.9 and mass[0] / 2.0 <= 1.1
        assert 0.9 * 1.0 <= mass[TORSO] <= 1.1 * 5.0
        gain = np.asarray(values["actuator_gainprm"])[:, 0] / 40.0
        assert gain.min() >= 0.8 and gain.max() <= 1.2
        offset = np.asarray(values["body_ipos"])[TORSO]
        assert np.all(np.abs(offset) <= 0.03)


def test_payload_only_changes_torso():
    model = build_model()
    body_mass = randomize.payload(jr.key(0), model, TORSO)
    changed = np.asarray(body_mass) != np.asarray(model.body_mass)
    assert changed.tolist() == [False, True, False]
    added = float(body_mass[TORSO] - model.body_mass[TORSO])
    assert -1.0 <= added <= 3.0
