import os

os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import paz
import jax.numpy as jp
import matplotlib.pyplot as plt
import sinusoidal


def MLP(hidden_dimensions=[40, 40]):
    x = inputs = keras.Input(shape=(1,))
    for dimension in hidden_dimensions:
        x = keras.layers.Dense(dimension, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)
    return keras.Model(inputs, outputs)


def meta_step(state, data, model, compute_loss, optimizer, fast_learning_rate):

    def compute_task_loss(theta, theta_static, x, y):
        y_pred, _ = model.stateless_call(theta, theta_static, x)
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
FAST_LR = 0.01
TRAIN_STEPS = 20_000
TASKS_PER_BATCH = 4
SHOTS_PER_TASK = 20
seed = 777
min_x = -5.0
max_x = 5.0
min_y = 0.1
max_y = 5.0

key = jax.random.PRNGKey(seed)
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
batch = jax.jit(
    paz.lock(
        sinusoidal.sample_batch,
        TASKS_PER_BATCH,
        SHOTS_PER_TASK,
        min_x,
        max_x,
        min_y,
        max_y,
    )
)

losses, progress_bar = [], keras.utils.Progbar(TRAIN_STEPS)
for step_arg, step_key in enumerate(jax.random.split(key, TRAIN_STEPS)):
    loss, state = step(state, batch(step_key)[0])
    losses.append(loss)
    progress_bar.update(step_arg + 1, [("loss", float(loss))])

plt.plot(losses)
plt.show()

theta, theta_static, optimizer_variables = state
for variable, value in zip(model.trainable_variables, theta):
    variable.assign(value)
