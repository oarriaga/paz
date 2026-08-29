from collections import namedtuple

import jax.numpy as jp
from mujoco import mjx

from terrain import add_heightfield

World = namedtuple("World", "robot, dynamics, physics_template, terrain")


def build(robot, terrain, backend, num_envs, num_contacts=32, num_constraints=32):  # fmt: skip
    mjmodel = build_mjmodel(robot, terrain)
    dynamics = mjx.put_model(mjmodel, impl=backend)
    template_args = mjmodel, backend, num_envs, num_contacts, num_constraints
    physics_template = build_physics_template(*template_args)
    terrain = move_origins_to_jax(terrain)
    return World(robot, dynamics, physics_template, terrain)


def build_mjmodel(robot, terrain):
    mjspec = robot.mjspec.copy()
    add_heightfield(mjspec, terrain)
    return robot.configure(mjspec.compile())


def build_physics_template(mjmodel, backend, num_envs, num_contacts, num_constraints):  # fmt: skip
    contacts = num_contacts * num_envs
    constraints = num_constraints * num_envs
    kwargs = dict(impl=backend, naconmax=contacts, njmax=constraints)
    return mjx.make_data(mjmodel, **kwargs)


def move_origins_to_jax(terrain):
    return terrain._replace(origins=jp.asarray(terrain.origins))
