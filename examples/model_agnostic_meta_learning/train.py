import os
from functools import partial

os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import jax.numpy as jp
import matplotlib.pyplot as plt
import sinusoidal
import maml


def MLP(hidden_dimensions=[40, 40]):
    x = inputs = keras.Input(shape=(1,))
    for dimension in hidden_dimensions:
        x = keras.layers.Dense(dimension, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)
    return keras.Model(inputs, outputs)


META_LEARNING_RATE = 1e-3
FAST_LR = 0.01
TRAIN_STEPS = 20_000
TASKS_PER_BATCH = 4
SHOTS_PER_TASK = 20
num_steps = 1
seed = 777
min_x = -5.0
max_x = 5.0
min_y = 0.1
max_y = 5.0

key = jax.random.PRNGKey(seed)
model = MLP()
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=META_LEARNING_RATE)
optimizer.build(model.trainable_variables)

parameters = (model.trainable_variables, model.non_trainable_variables)
state = (parameters, optimizer.variables)

train_step_args = (model, loss_fn, optimizer, FAST_LR, num_steps)
train_step = jax.jit(partial(maml.train_step, *train_step_args))

batch = jax.jit(
    partial(
        sinusoidal.sample_batch,
        batch_size=TASKS_PER_BATCH,
        num_shots=SHOTS_PER_TASK,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
    )
)

losses, progress_bar = [], keras.utils.Progbar(TRAIN_STEPS)
for step_arg, step_key in enumerate(jax.random.split(key, TRAIN_STEPS)):
    loss, state = train_step(state, batch(step_key)[0])
    progress_bar.update(step_arg + 1, [("loss", float(loss))])
    losses.append(loss)

plt.plot(losses)
plt.show()

(variables, static_parameters), optimizer_variables = state
for variable, value in zip(model.trainable_variables, variables):
    variable.assign(value)


print("\nVisualizing model adaptation on new, unseen tasks...")
parameters, _ = state
key, subkey = jax.random.split(key)
data_args = subkey, TASKS_PER_BATCH, SHOTS_PER_TASK, min_x, max_x, min_y, max_y
data, (amplitudes, phases) = sinusoidal.sample_batch(*data_args)
figure, axes = plt.subplots(2, 2, figsize=(16, 14))
figure.suptitle("MAML Adaptation to New Sinusoid Tasks", fontsize=18)
x = jp.linspace(min_x, max_x, 200).reshape(-1, 1)
for task_arg, axis in enumerate(axes.flat):
    support_data = data[0][task_arg], data[1][task_arg]
    queries_data = data[2][task_arg], data[3][task_arg]
    y_pred_pre_update = maml.call(model, parameters, x)
    y_pred_post_update = maml.predict(
        model, loss_fn, FAST_LR, num_steps, parameters, support_data, x
    )
    y = amplitudes[task_arg] * jp.sin(x - phases[task_arg])
    axis.plot(x, y, label="Ground Truth", color="gray", linestyle="--")
    axis.scatter(*support_data, s=80, label="Support Set (for adaptation)")
    axis.scatter(
        *queries_data, s=60, label="Query Set (for evaluation)", marker="x"
    )
    axis.plot(
        x, y_pred_pre_update, label="Pre-Update Prediction", linestyle=":"
    )
    axis.plot(
        x,
        y_pred_post_update,
        label="Post-Update Prediction (1 step)",
        linewidth=2.5,
    )
    axis.set_title(f"Task {task_arg + 1}", fontsize=12)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.legend()
    axis.grid(True, linestyle=":", alpha=0.6)
    axis.set_ylim(min_y - max_y - 1, max_y + 1)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
