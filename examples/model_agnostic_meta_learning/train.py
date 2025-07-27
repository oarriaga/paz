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


print("\nVisualizing model adaptation on new, unseen tasks...")


def forward_pass(theta, theta_static, x, model):
    y_pred, _ = model.stateless_call(theta, theta_static, x)
    return y_pred


def adapt_step(state, support_data, model, compute_loss, fast_learning_rate):
    theta, theta_static = state
    support_x, support_y = support_data

    def compute_task_loss(theta, theta_static, x, y):
        y_pred, _ = model.stateless_call(theta, theta_static, x)
        return compute_loss(y, y_pred)

    gradients = jax.grad(compute_task_loss)(
        theta, theta_static, support_x, support_y
    )

    fast_weights = jax.tree.map(
        lambda t, g: t - fast_learning_rate * g, theta, gradients
    )
    return fast_weights


jitted_forward_pass = jax.jit(forward_pass, static_argnames=("model",))
jitted_adapt_step = jax.jit(
    adapt_step, static_argnames=("model", "compute_loss", "fast_learning_rate")
)

trained_theta, trained_theta_static, _ = state
trained_state = (trained_theta, trained_theta_static)

key, viz_key = jax.random.split(key)
(s_x, s_y, q_x, q_y), (amplitudes, phases) = sinusoidal.sample_batch(
    viz_key, TASKS_PER_BATCH, SHOTS_PER_TASK, min_x, max_x, min_y, max_y
)

figure, axes = plt.subplots(2, 2, figsize=(16, 14))
figure.suptitle("MAML Adaptation to New Sinusoid Tasks", fontsize=18)

x_curve = jp.linspace(min_x, max_x, 200).reshape(-1, 1)

for i, axis in enumerate(axes.flat):
    # Select data for the i-th task
    support_data = (s_x[i], s_y[i])
    query_data = (q_x[i], q_y[i])

    # --- Pre-Update Prediction (using meta-parameters) ---
    y_pred_pre_update = jitted_forward_pass(
        trained_theta, trained_theta_static, x_curve, model
    )

    # --- Adaptation Step ---
    fast_weights = jitted_adapt_step(
        trained_state, support_data, model, compute_loss, FAST_LR
    )

    # --- Post-Update Prediction (using fast weights) ---
    y_pred_post_update = jitted_forward_pass(
        fast_weights, trained_theta_static, x_curve, model
    )

    # --- Plotting ---
    # Plot ground truth and data points
    y_curve_true = amplitudes[i] * jp.sin(x_curve - phases[i])
    axis.plot(
        x_curve,
        y_curve_true,
        label="Ground Truth",
        color="gray",
        linestyle="--",
    )
    axis.scatter(
        support_data[0],
        support_data[1],
        s=80,
        label="Support Set (for adaptation)",
    )
    axis.scatter(
        query_data[0],
        query_data[1],
        s=60,
        label="Query Set (for evaluation)",
        marker="x",
    )

    # Plot model predictions
    axis.plot(
        x_curve, y_pred_pre_update, label="Pre-Update Prediction", linestyle=":"
    )
    axis.plot(
        x_curve,
        y_pred_post_update,
        label="Post-Update Prediction (1 step)",
        linewidth=2.5,
    )

    # Make it pretty
    axis.set_title(f"Task {i + 1}", fontsize=12)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.legend()
    axis.grid(True, linestyle=":", alpha=0.6)
    axis.set_ylim(min_y - max_y - 1, max_y + 1)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
