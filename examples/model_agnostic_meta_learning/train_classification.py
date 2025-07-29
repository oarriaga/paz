import os
from functools import partial
import argparse

os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import numpy as np
import matplotlib.pyplot as plt
from paz.datasets import omniglot
import maml


parser = argparse.ArgumentParser(description="MAML Training on Omniglot")
parser.add_argument("--learning_rate", default=1e-3, type=float)
parser.add_argument("--inner_learning_rate", default=0.4, type=float)
parser.add_argument("--train_steps", default=10_000, type=int)
parser.add_argument("--tasks_per_batch", default=32, type=int)
parser.add_argument("--num_ways", default=5, type=int)
parser.add_argument("--num_shots", default=1, type=int)
parser.add_argument("--num_queries", default=1, type=int)
parser.add_argument("--num_inner_steps", default=1, type=int)
parser.add_argument("--seed", default=777, type=int)
args = parser.parse_args()


def CNN(input_shape=(28, 28, 1), num_classes=5):
    """Defines a 4-layer CNN, as described in the MAML paper."""
    x = inputs = keras.Input(shape=input_shape)
    for i in range(4):
        x = keras.layers.Conv2D(filters=64, kernel_size=3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling2D(pool_size=2)(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(num_classes)(x)
    return keras.Model(inputs, outputs)


def omniglot_batch_generator(
    dataset, num_ways, num_shots, num_queries, tasks_per_batch, rng
):
    while True:
        s_x_batch, s_y_batch, q_x_batch, q_y_batch = [], [], [], []
        for _ in range(tasks_per_batch):
            (s_x, s_y), (q_x, q_y) = omniglot.sample_between_alphabet(
                rng, dataset, num_ways, num_shots, num_queries
            )
            s_x_batch.append(s_x)
            s_y_batch.append(s_y)
            q_x_batch.append(q_x)
            q_y_batch.append(q_y)

        s_x_batch = np.stack(s_x_batch)
        s_y_batch = np.stack(s_y_batch)
        q_x_batch = np.stack(q_x_batch)
        q_y_batch = np.stack(q_y_batch)

        s_x_batch = np.expand_dims(s_x_batch, axis=-1)
        q_x_batch = np.expand_dims(q_x_batch, axis=-1)

        yield (s_x_batch, s_y_batch, q_x_batch, q_y_batch)


RNG = np.random.default_rng(args.seed)
key = jax.random.PRNGKey(args.seed)
train_data = omniglot.load(split="train", shape=(28, 28), flat=True)

model = CNN(input_shape=(28, 28, 1), num_classes=args.num_ways)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
optimizer.build(model.trainable_variables)

state = maml.build_state(model, optimizer)
train_step = jax.jit(
    partial(
        maml.train_step,
        model,
        loss_fn,
        optimizer,
        args.inner_learning_rate,
        args.num_inner_steps,
    )
)

data_generator = omniglot_batch_generator(
    train_data,
    args.num_ways,
    args.num_shots,
    args.num_queries,
    args.tasks_per_batch,
    RNG,
)

print(f"Starting MAML training for {args.train_steps} steps...")
losses, progress_bar = [], keras.utils.Progbar(args.train_steps)
for step_arg in range(args.train_steps):
    data_batch = next(data_generator)
    loss, state = train_step(state, data_batch)
    progress_bar.update(step_arg + 1, [("loss", float(loss))])
    losses.append(loss)

plt.plot(losses)
plt.title("MAML Training Loss on Omniglot")
plt.xlabel("Training Step")
plt.ylabel("Sparse Categorical Crossentropy")
plt.show()


print("\nVisualizing model adaptation on new, unseen tasks...")

test_data = omniglot.load(split="test", shape=(28, 28), flat=True)
parameters, _ = state

jitted_adapt = jax.jit(
    partial(
        maml.adapt,
        model,
        loss_fn,
        args.inner_learning_rate,
        args.num_inner_steps,
    )
)
jitted_call = jax.jit(partial(maml.call, model))

NUM_TASKS_TO_PLOT = 4
NUM_QUERIES_VIZ = 1

figure, axes = plt.subplots(
    NUM_TASKS_TO_PLOT,
    args.num_ways + 2,
    figsize=(3 * (args.num_ways + 2), 3 * NUM_TASKS_TO_PLOT),
)
figure.suptitle("MAML Adaptation on Unseen Omniglot Tasks", fontsize=16)

for task_arg in range(NUM_TASKS_TO_PLOT):
    (s_x, s_y), (q_x, q_y) = omniglot.sample_between_alphabet(
        RNG, test_data, args.num_ways, args.num_shots, NUM_QUERIES_VIZ
    )

    support_data = (np.expand_dims(s_x, axis=-1), s_y)
    query_images = np.expand_dims(q_x, axis=-1)
    query_image_to_plot = query_images[0:1]
    true_label = q_y[0]

    logits_pre = jitted_call(parameters, query_images)
    probs_pre = jax.nn.softmax(logits_pre)

    adapted_params = jitted_adapt(parameters, support_data)
    logits_post = jitted_call(adapted_params, query_images)
    probs_post = jax.nn.softmax(logits_post)
    for i in range(args.num_ways):
        image = s_x[s_y == i][0]
        axes[task_arg, i].imshow(image, cmap="gray")
        axes[task_arg, i].set_title(f"Support: Class {i}")
        axes[task_arg, i].axis("off")

    axes[task_arg, args.num_ways].imshow(
        query_image_to_plot.squeeze(), cmap="gray"
    )
    axes[task_arg, args.num_ways].set_title(f"Query (True: {true_label})")
    axes[task_arg, args.num_ways].axis("off")

    bar_axis = axes[task_arg, args.num_ways + 1]
    bar_positions = np.arange(args.num_ways)

    probs_pre_to_plot = probs_pre[0]
    probs_post_to_plot = probs_post[0]

    bar_axis.bar(
        bar_positions - 0.2,
        np.asarray(probs_pre_to_plot),
        width=0.4,
        label="Pre-Update",
        color="C0",
        alpha=0.6,
    )
    bar_axis.bar(
        bar_positions + 0.2,
        np.asarray(probs_post_to_plot),
        width=0.4,
        label=f"Post-Update ({args.num_inner_steps} step)",
        color="C2",
    )

    bar_axis.get_children()[true_label].set_color("C3")
    bar_axis.get_children()[true_label + args.num_ways].set_color("C3")

    bar_axis.set_xticks(bar_positions)
    bar_axis.set_ylim([0, 1])
    bar_axis.set_title("Predictions")
    bar_axis.set_xlabel("Class")
    bar_axis.set_ylabel("Probability")
    bar_axis.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
