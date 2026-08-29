from collections import namedtuple

import jax.numpy as jp
from mujoco import mjx

from terrain import add_heightfield

World = namedtuple("World", "robot, dynamics, physics_template, terrain")


def build(robot, terrain, backend, num_envs, device, num_contacts=32, num_constraints=256):  # fmt: skip
    mjmodel = build_mjmodel(robot, terrain)
    dynamics = mjx.put_model(mjmodel, impl=backend, device=device)
    template_args = mjmodel, backend, num_envs, device
    budget = num_contacts, num_constraints
    physics_template = build_physics_template(*template_args, *budget)
    terrain = move_origins_to_jax(terrain)
    return World(robot, dynamics, physics_template, terrain)


def build_mjmodel(robot, terrain):
    mjspec = robot.mjspec.copy()
    add_heightfield(mjspec, terrain)
    return robot.configure(mjspec.compile())


def build_physics_template(mjmodel, backend, num_envs, device, num_contacts, num_constraints):  # fmt: skip
    # warp budgets naconmax over the whole batch but njmax per environment,
    # so only the contact budget scales with the number of environments
    contacts = num_contacts * num_envs
    kwargs = dict(impl=backend, naconmax=contacts, njmax=num_constraints)
    return mjx.make_data(mjmodel, device=device, **kwargs)


def move_origins_to_jax(terrain):
    return terrain._replace(origins=jp.asarray(terrain.origins))
