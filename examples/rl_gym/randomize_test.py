import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
from jax import random as jr
from mujoco import mjx

import randomize
from robots.g1 import G1DoF29


def build_randomized(num_envs=8):
    robot = G1DoF29()
    model = mjx.put_model(robot.mjspec.compile())
    keys = jr.split(jr.key(0), num_envs)
    torso_arg = robot.bodies.torso_link.arg
    randomized, axes = randomize.physics(keys, model, robot.num_actuators, torso_arg)  # fmt: skip
    return model, randomized, axes, torso_arg


def test_randomized_fields_carry_a_leading_environment_axis():
    model, randomized, axes, _ = build_randomized()
    for name in randomize.get_randomized_fields():
        assert getattr(axes, name) == 0
        assert getattr(randomized, name).shape[0] == 8
        assert getattr(randomized, name).shape[1:] == getattr(model, name).shape  # fmt: skip


def test_friction_and_gains_stay_inside_the_reference_ranges():
    model, randomized, _, _ = build_randomized()
    friction = np.asarray(randomized.geom_friction[:, :, 0])
    assert friction.min() >= 0.2 and friction.max() <= 1.25
    gains = np.asarray(randomized.actuator_gainprm[:, :, 0])
    scale = gains / np.asarray(model.actuator_gainprm[:, 0])
    assert scale.min() >= 0.8 and scale.max() <= 1.2
    assert np.allclose(randomized.actuator_biasprm[:, :, 1], -gains)


def test_payload_only_touches_the_torso_and_inertia_follows_mass():
    model, randomized, _, torso_arg = build_randomized()
    scale = np.asarray(randomized.body_mass) / np.asarray(model.body_mass)
    others = np.delete(scale, torso_arg, axis=1)[:, 1:]
    assert others.min() >= 0.9 and others.max() <= 1.1
    torso = np.asarray(randomized.body_mass[:, torso_arg])
    base = float(model.body_mass[torso_arg])
    assert torso.min() >= (base - 1.0) * 0.9 and torso.max() <= (base + 3.0) * 1.1  # fmt: skip
    inertia_scale = np.asarray(randomized.body_inertia)[:, 1:] / np.asarray(model.body_inertia)[1:]  # fmt: skip
    assert np.allclose(inertia_scale, scale[:, 1:, None], atol=1e-5)


def test_joint_friction_and_armature_are_added_to_joints_only():
    model, randomized, _, _ = build_randomized()
    added = np.asarray(randomized.dof_frictionloss) - np.asarray(model.dof_frictionloss)  # fmt: skip
    assert np.allclose(added[:, :6], 0.0)
    assert added[:, 6:].min() >= 0.0 and added[:, 6:].max() <= 0.05
    armature = np.asarray(randomized.dof_armature) - np.asarray(model.dof_armature)  # fmt: skip
    assert np.allclose(armature[:, :6], 0.0)
    assert armature[:, 6:].min() >= 0.0 and armature[:, 6:].max() <= 0.005
