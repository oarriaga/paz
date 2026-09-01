import paz

import jax
import jax.numpy as jp
from jax import random as jr


def physics(keys, model, num_joints, torso_arg):

    def build_model_in_axes(model):
        model_in_axes = jax.tree.map(lambda _: None, model)
        randomized_in_axes = {name: 0 for name in get_randomized_fields()}
        return model_in_axes.tree_replace(randomized_in_axes)

    args = (model, num_joints, torso_arg)
    physics_models = jax.vmap(paz.lock(physics_model, *args))(keys)
    model = model.tree_replace(physics_models)
    return model, build_model_in_axes(model)


def get_randomized_fields():
    return (
        "geom_friction",
        "geom_solref",
        "body_mass",
        "body_inertia",
        "body_ipos",
        "actuator_gainprm",
        "actuator_biasprm",
        "dof_frictionloss",
        "dof_armature",
    )


def physics_model(key, model, num_joints, torso_arg):
    keys = jr.split(key, 9)
    gain = actuator_gain(keys[4], model, num_joints)
    damping = actuator_velocity_damping(keys[5], model, num_joints)
    body_mass = mass(keys[2], payload(keys[8], model, torso_arg))
    randomizations = [
        friction(keys[0], model),
        contact_damping_ratio(keys[1], model),
        body_mass,
        scale_inertia(model, body_mass),
        torso_CoM(keys[3], model, torso_arg),
        build_actuator_gainprm(model, gain),
        build_actuator_biasprm(model, gain, damping),
        joint_dry_friction(keys[6], model),
        armature(keys[7], model, num_joints),
    ]
    return dict(zip(get_randomized_fields(), randomizations))


def friction(key, model, minval=0.2, maxval=1.25):
    friction = jr.uniform(key, (), minval=minval, maxval=maxval)
    return model.geom_friction.at[:, 0].set(friction)


def contact_damping_ratio(key, model, minval=0.9, maxval=1.0):
    damping_ratio = jr.uniform(key, (), minval=minval, maxval=maxval)
    return model.geom_solref.at[:, 1].set(damping_ratio)


def payload(key, model, torso_arg, minval=-1.0, maxval=3.0):
    added = jr.uniform(key, (), minval=minval, maxval=maxval)
    return model.body_mass.at[torso_arg].add(added)


def mass(key, body_mass, minval=0.9, maxval=1.1):
    scale = jr.uniform(key, body_mass.shape, minval=minval, maxval=maxval)
    return body_mass * scale


def scale_inertia(model, body_mass):
    # the reference recomputes body inertia with the randomized mass
    scale = body_mass / jp.maximum(model.body_mass, 1e-9)
    return model.body_inertia * scale[:, None]


def torso_CoM(key, model, torso_arg, minval=-0.03, maxval=0.03):
    offset = jr.uniform(key, (3,), minval=minval, maxval=maxval)
    return model.body_ipos.at[torso_arg].add(offset)


def actuator_gain(key, model, num_joints=29, minval=0.8, maxval=1.2):
    scale = jr.uniform(key, (num_joints,), minval=minval, maxval=maxval)
    return model.actuator_gainprm[:, 0] * scale


def actuator_velocity_damping(key, model, num_joints=29, minval=0.8, maxval=1.2):  # fmt: skip
    scale = jr.uniform(key, (num_joints,), minval=minval, maxval=maxval)
    return model.actuator_biasprm[:, 2] * scale


def joint_dry_friction(key, model, minval=0.0, maxval=0.05):
    kwargs = {"minval": minval, "maxval": maxval}
    added_friction = jr.uniform(key, model.dof_frictionloss[6:].shape, **kwargs)
    return model.dof_frictionloss.at[6:].add(added_friction)


def armature(key, model, num_joints=29, minval=0.0, maxval=0.005):
    armature = jr.uniform(key, (num_joints,), minval=minval, maxval=maxval)
    return model.dof_armature.at[6:].add(armature)


def build_actuator_gainprm(model, gain):
    return model.actuator_gainprm.at[:, 0].set(gain)


def build_actuator_biasprm(model, gain, damping):
    biasprm = model.actuator_biasprm.at[:, 1].set(-gain)
    return biasprm.at[:, 2].set(damping)
