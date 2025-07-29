import os
from functools import partial
import argparse

os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import jax.numpy as jp
import matplotlib.pyplot as plt
import sinusoidal
import maml

parser = argparse.ArgumentParser(description="MAML Training")
parser.add_argument("--learning_rate", default=1e-3, type=float)
parser.add_argument("--inner_learning-rate", default=0.01, type=float)
parser.add_argument("--train_steps", default=20_000, type=int)
parser.add_argument("--tasks_per_batch", default=4, type=int)
parser.add_argument("--shots_per_task", default=20, type=int)
parser.add_argument("--num_steps", default=1, type=int)
parser.add_argument("--seed", default=777, type=int)
parser.add_argument("--min_x", default=-5.0, type=float)
parser.add_argument("--max_x", default=5.0, type=float)
parser.add_argument("--min_y", default=0.1, type=float)
parser.add_argument("--max_y", default=5.0, type=float)
parser.add_argument("--hidden_dims", nargs="+", default=[40, 40], type=int)
args = parser.parse_args()


def MLP(hidden_dimensions):
    x = inputs = keras.Input(shape=(1,))
    for dimension in hidden_dimensions:
        x = keras.layers.Dense(dimension, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)
    return keras.Model(inputs, outputs)


key = jax.random.PRNGKey(args.seed)
model = MLP(args.hidden_dims)
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(args.learning_rate)
optimizer.build(model.trainable_variables)

state = maml.build_state(model, optimizer)

train_step = jax.jit(
    partial(
        maml.train_step,
        model,
        loss_fn,
        optimizer,
        args.inner_learning_rate,
        args.num_steps,
    )
)

batch = jax.jit(
    partial(
        sinusoidal.sample_batch,
        batch_size=args.tasks_per_batch,
        num_shots=args.shots_per_task,
        min_x=args.min_x,
        max_x=args.max_x,
        min_y=args.min_y,
        max_y=args.max_y,
    )
)

losses, progress_bar = [], keras.utils.Progbar(args.train_steps)
for step_arg, step_key in enumerate(jax.random.split(key, args.train_steps)):
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
data_args = (
    subkey,
    args.tasks_per_batch,
    args.shots_per_task,
    args.min_x,
    args.max_x,
    args.min_y,
    args.max_y,
)
data, (amplitudes, phases) = sinusoidal.sample_batch(*data_args)
figure, axes = plt.subplots(2, 2, figsize=(16, 14))
figure.suptitle("MAML Adaptation to New Sinusoid Tasks", fontsize=18)
x = jp.linspace(args.min_x, args.max_x, 200).reshape(-1, 1)
predict = partial(maml.predict, model, loss_fn, args.inner_learning_rate)
for task_arg, axis in enumerate(axes.flat):
    support_data = data[0][task_arg], data[1][task_arg]
    queries_data = data[2][task_arg], data[3][task_arg]
    y_pred_0 = maml.call(model, parameters, x)
    y = amplitudes[task_arg] * jp.sin(x - phases[task_arg])
    axis.plot(x, y, label="Ground Truth", color="gray", linestyle="--")
    label = "Query Set (for evaluation)"
    axis.scatter(*support_data, s=80, label=label, marker="o", color="C0")
    label = "Support Set (for adaptation)"
    axis.scatter(*queries_data, s=60, label=label, marker="x", color="C1")
    color, label, alpha = "C1", "Pre-Update Prediction", 0.4
    axis.plot(x, y_pred_0, label=label, linestyle=":", color=color, alpha=alpha)
    for num_steps, alpha in zip([1, 5], [alpha, 1.0]):
        y_pred = predict(num_steps, parameters, support_data, x)
        label = f"({num_steps} step{'s' if num_steps > 1 else ''})"
        label = f"Post-Update Prediction {label}"
        axis.plot(
            x, y_pred, label=label, linewidth=2.5, color=color, alpha=alpha
        )
    axis.set_title(f"Task {task_arg + 1}", fontsize=12)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.legend()
    axis.grid(True, linestyle=":", alpha=0.6)
    axis.set_ylim(args.min_y - args.max_y - 1, args.max_y + 1)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
