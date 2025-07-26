import os

os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import paz
import numpy as np
import jax.numpy as jp
import matplotlib.pyplot as plt


def Sinusoid(
    RNG, num_shots, min_amplitude=0.1, max_amplitude=5.0, min_x=-5.0, max_x=5.0
):
    """Creates a sampler for a single sinusoid task."""
    amplitude = RNG.uniform(min_amplitude, max_amplitude)
    phase = RNG.uniform(0, np.pi)

    def sample():
        """Samples support and query sets for the task."""
        x = RNG.uniform(min_x, max_x, num_shots * 2)
        y = amplitude * np.sin(x - phase)
        # Split the sampled points into support and query sets
        support_x, query_x = np.split(x, 2)
        support_y, query_y = np.split(y, 2)
        return (support_x, support_y), (query_x, query_y)

    return sample


def MLP(hidden_dimensions=[40, 40]):
    x = inputs = keras.Input(shape=(1,))
    for dimension in hidden_dimensions:
        x = keras.layers.Dense(dimension, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)
    return keras.Model(inputs, outputs)


def meta_step(state, data, model, compute_loss, optimizer, fast_learning_rate):

    def compute_task_loss(theta, theta_static, x, y):
        y_pred, _ = model.stateless_call(theta, theta_static, x)
        y_pred = jp.reshape(y_pred, y.shape)  # match loss function labels
        return compute_loss(y, y_pred)

    def gradient_step(theta, gradients):
        return theta - fast_learning_rate * gradients

    def compute_meta_loss(theta, theta_static, s_x, s_y, q_x, q_y):
        gradients = jax.grad(compute_task_loss)(theta, theta_static, s_x, s_y)
        fast_weights = jax.tree.map(gradient_step, theta, gradients)
        meta_loss = compute_task_loss(fast_weights, theta_static, q_x, q_y)
        return meta_loss

    compute_meta_gradients = jax.vmap(
        jax.value_and_grad(compute_meta_loss),
        in_axes=(None, None, 0, 0, 0, 0),
        out_axes=0,
    )

    theta, theta_static, theta_optimizer = state
    loss, gradients = compute_meta_gradients(theta, theta_static, *data)
    gradients = jax.tree.map(lambda gradient: jp.mean(gradient, 0), gradients)
    theta, optimizer_theta = optimizer.stateless_apply(
        theta_optimizer, gradients, theta
    )
    return jp.mean(loss), (theta, theta_static, optimizer_theta)


META_LEARNING_RATE = 1e-3
FAST_LR = 0.1
TRAIN_STEPS = 20_000
# TASKS_PER_BATCH = 16
TASKS_PER_BATCH = 32
SHOTS_PER_TASK = 20
seed = 777
# (16, 10 ,1)

RNG = np.random.default_rng(seed)
model = MLP()
compute_loss = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=META_LEARNING_RATE)
optimizer.build(model.trainable_variables)

state = (
    model.trainable_variables,
    model.non_trainable_variables,
    optimizer.variables,
)

step = jax.jit(paz.lock(meta_step, model, compute_loss, optimizer, FAST_LR))
losses, progress_bar = [], keras.utils.Progbar(TRAIN_STEPS)
for train_step in range(TRAIN_STEPS):
    # Generate a new batch of tasks for each step
    task_samplers = [
        Sinusoid(RNG, SHOTS_PER_TASK) for _ in range(TASKS_PER_BATCH)
    ]
    support_x, support_y, query_x, query_y = [], [], [], []
    for sampler in task_samplers:
        (sx, sy), (qx, qy) = sampler()
        support_x.append(sx.reshape(SHOTS_PER_TASK, 1))
        support_y.append(sy.reshape(SHOTS_PER_TASK, 1))
        query_x.append(qx.reshape(SHOTS_PER_TASK, 1))
        query_y.append(qy.reshape(SHOTS_PER_TASK, 1))

    data_batch = (
        np.array(support_x),
        np.array(support_y),
        np.array(query_x),
        np.array(query_y),
    )

    loss, state = step(state, data_batch)
    losses.append(float(loss))
    progress_bar.update(train_step + 1, [("loss", float(loss))])

plt.plot(losses)
plt.show()

theta, theta_static, optimizer_variables = state
for variable, value in zip(model.trainable_variables, theta):
    variable.assign(value)
