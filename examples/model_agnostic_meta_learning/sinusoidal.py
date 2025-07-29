import jax
import jax.numpy as jp


def sample_task(key, num_shots, min_x, max_x, min_y, max_y):
    keys = jax.random.split(key, 3)
    amplitude = jax.random.uniform(keys[0], (), jp.float32, min_y, max_y)
    phase = jax.random.uniform(keys[1], (), jp.float32, 0.0, jp.pi)
    x = jax.random.uniform(keys[2], (num_shots * 2,), jp.float32, min_x, max_x)
    y = amplitude * jp.sin(x - phase)
    x_support, x_queries = jp.split(x, 2)
    y_support, y_queries = jp.split(y, 2)
    x_support = jp.reshape(x_support, (num_shots, 1))
    y_support = jp.reshape(y_support, (num_shots, 1))
    x_queries = jp.reshape(x_queries, (num_shots, 1))
    y_queries = jp.reshape(y_queries, (num_shots, 1))
    return ((x_support, y_support), (x_queries, y_queries)), (amplitude, phase)


def sample_batch(key, batch_size, num_shots, min_x, max_x, min_y, max_y):
    task_keys = jax.random.split(key, batch_size)
    in_axes = (0, None, None, None, None, None)
    _sample_batch = jax.vmap(sample_task, in_axes=in_axes)
    batch = _sample_batch(task_keys, num_shots, min_x, max_x, min_y, max_y)
    ((x_support, y_support), (x_queries, y_queries)), (amplitude, phase) = batch
    return (x_support, y_support, x_queries, y_queries), (amplitude, phase)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    key = jax.random.PRNGKey(777)
    NUM_TASKS_TO_PLOT = 4
    SHOTS_PER_TASK = 10
    MIN_X, MAX_X = -5.0, 5.0
    MIN_Y, MAX_Y = 0.1, 5.0

    (x_support, y_support, x_queries, y_queries), (amplitudes, phases) = (
        sample_batch(
            key, NUM_TASKS_TO_PLOT, SHOTS_PER_TASK, MIN_X, MAX_X, MIN_Y, MAX_Y
        )
    )

    figure, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=True)
    figure.suptitle("Sampled Sinusoid Tasks (Shared Y-Axis)", fontsize=18)

    y_limit = MAX_Y + 0.5
    axes[0, 0].set_ylim(-y_limit, y_limit)

    for axis_arg, axis in enumerate(axes.flat):
        x_support_task = x_support[axis_arg]
        y_support_task = y_support[axis_arg]
        x_queries_task = x_queries[axis_arg]
        y_queries_task = y_queries[axis_arg]
        amplitude = amplitudes[axis_arg]
        phase = phases[axis_arg]

        x_curve = jp.linspace(MIN_X, MAX_X, 200).reshape(-1, 1)
        y_curve = amplitude * jp.sin(x_curve - phase)

        axis.plot(
            x_curve, y_curve, label="Ground Truth", color="gray", linestyle="--"
        )
        axis.scatter(
            x_support_task,
            y_support_task,
            s=80,
            label=f"Support Set ({SHOTS_PER_TASK}-shots)",
            color="C0",
            zorder=3,
        )
        axis.scatter(
            x_queries_task,
            y_queries_task,
            s=60,
            label="Query Set",
            color="C1",
            marker="X",
            zorder=3,
        )

        axis.set_title(
            f"Task {axis_arg + 1} (Amp={amplitude:.2f}, Phase={phase:.2f})",
            fontsize=12,
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.legend()
        axis.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
