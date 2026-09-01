import math
from collections import namedtuple

import jax.numpy as jp
import keras

Parameters = namedtuple("Parameters", "stdv, actor, critic")


def Actor(shapes, num_actions=29, hidden_units=(512, 256, 128)):
    inputs = build_inputs(shapes)
    x = concatenate_inputs(inputs)
    x = dense_block(x, hidden_units, "actor")
    x = dense_layer(x, num_actions, None, "actor_output")
    return keras.Model(inputs, x, name="actor")


def Critic(shapes, hidden_units=(512, 256, 128)):
    inputs = build_inputs(shapes)
    x = concatenate_inputs(inputs)
    x = dense_block(x, hidden_units, "critic")
    x = dense_layer(x, 1, None, "critic_output")
    return keras.Model(inputs, x, name="critic")


def build_inputs(shapes):
    inputs = []
    for name, shape in zip(shapes._fields, shapes):
        inputs.append(keras.Input(shape=shape, name=name))
    return inputs


def concatenate_inputs(inputs):
    flattened = []
    for tensor in inputs:
        flattened.append(keras.layers.Flatten()(tensor))
    return keras.layers.Concatenate()(flattened)


def compute_shapes(observations):
    shapes = []
    for term in observations:
        shapes.append(term.shape[1:])
    return type(observations)(*shapes)


def dense_block(inputs, units_per_layer, prefix):
    x = inputs
    for layer_arg, units in enumerate(units_per_layer):
        x = dense_layer(x, units, "elu", f"{prefix}_{layer_arg}")
    return x


def dense_layer(inputs, units, activation, name):
    fan_in = int(inputs.shape[-1])
    bound = 1.0 / math.sqrt(fan_in)
    initializer = keras.initializers.RandomUniform(-bound, bound)
    bias_initializer = keras.initializers.RandomUniform(-bound, bound)
    args = (units, activation, True, initializer, bias_initializer)
    return keras.layers.Dense(*args, name=name)(inputs)


def PPO(actor_shapes, critic_shapes, num_actions=29):
    actor = Actor(actor_shapes, num_actions)
    critic = Critic(critic_shapes)
    # the exploration noise is state independent and learned directly, as
    # in the reference implementation; read_stdv guards its positivity
    stdv = keras.Variable(jp.ones(num_actions), name="stdv")
    return actor, critic, stdv


def read_stdv(parameters, floor=1e-2):
    # the floor keeps the action distribution valid if the optimizer ever
    # drives the learned deviation to zero or below
    return jp.maximum(parameters.stdv, floor)


def call_actor(actor, actor_parameters, observations):
    inputs = list(observations)
    outputs, _ = actor.stateless_call(actor_parameters, [], inputs)
    return outputs


def call_critic(critic, critic_parameters, observations):
    inputs = list(observations)
    outputs, _ = critic.stateless_call(critic_parameters, [], inputs)
    return jp.squeeze(outputs, axis=-1)


def pack_parameters(parameters):
    return [parameters.stdv] + list(parameters.actor) + list(parameters.critic)  # fmt: skip


def unpack_parameters(variables):
    stdv = variables[0]
    num_network_variables = (len(variables) - 1) // 2
    actor_end = num_network_variables + 1
    actor_parameters = variables[1:actor_end]
    return Parameters(stdv, actor_parameters, variables[actor_end:])


def Optimizer(actor, critic, stdv, learning_rate):
    optimizer = keras.optimizers.Adam(learning_rate, epsilon=1e-8)
    trainable = actor.trainable_variables + critic.trainable_variables
    optimizer.build([stdv] + trainable)
    optimizer_state = [variable.value for variable in optimizer.variables]
    return optimizer, optimizer_state


def read_variables(model):
    variables = []
    for variable in model.trainable_variables:
        variables.append(variable.value)
    return variables


def snapshot_parameters(actor, critic, stdv):
    actor_parameters = read_variables(actor)
    return Parameters(stdv.value, actor_parameters, read_variables(critic))
