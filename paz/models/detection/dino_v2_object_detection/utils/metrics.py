import functools
import os
import time
from types import SimpleNamespace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.compat.proto.event_pb2 import Event
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import wandb
except ImportError:
    wandb = None

plt.ioff()

PLOT_FILE_NAME = "metrics_plot.png"
TENSORBOARD_LOSS_TAGS = {"train_loss": "Loss/Train", "test_loss": "Loss/Test"}  # fmt: skip
SERIES_KEYS = ("AP50", "AP", "AR")
SERIES_VALUES = ("Average Precision @0.50", "Average Precision @0.50:0.95", "Average Recall @0.50:0.95")  # fmt: skip
SERIES_TITLES = dict(zip(SERIES_KEYS, SERIES_VALUES))


def read_index(values, index):
    return values[index] if 0 <= index < len(values) else None


def MetricsPlotSink(output_dir):
    ns = SimpleNamespace()
    ns.output_dir = output_dir
    ns.history = []
    ns.update = functools.partial(update_plot_history, ns)
    ns.save = functools.partial(save_metrics_plot, ns)
    return ns


def update_plot_history(ns, values):
    ns.history.append(values)


def read_history_array(ns, key):
    return np.array([h[key] for h in ns.history if key in h])


def read_history_list(ns, key):
    return [h[key] for h in ns.history if key in h]


def read_coco_metric(coco_eval, index):
    values = [read_index(x, index) for x in coco_eval if x is not None]
    return np.array(values, dtype=np.float32)


def read_coco_series(ns, key):
    coco_eval = read_history_list(ns, key)
    indexes = (0, 1, 8)
    return [read_coco_metric(coco_eval, index) for index in indexes]


def draw_metrics_figure(ns, epochs, base, ema):
    figure, axes = plt.subplots(2, 2, figsize=(18, 12))
    train_loss = read_history_array(ns, 'train_loss')
    test_loss = read_history_array(ns, 'test_loss')
    plot_loss_axes(axes[0][0], epochs, train_loss, test_loss)
    plot_metric_series(axes[0][1], epochs, base[1], ema[1], 'AP50')
    plot_metric_series(axes[1][0], epochs, base[0], ema[0], 'AP')
    plot_metric_series(axes[1][1], epochs, base[2], ema[2], 'AR')
    return figure


def save_metrics_plot(ns):
    if not ns.history:
        print("No data to plot.")
    else:
        epochs = read_history_array(ns, 'epoch')
        base = read_coco_series(ns, 'test_coco_eval_bbox')
        ema = read_coco_series(ns, 'ema_test_coco_eval_bbox')
        figure = draw_metrics_figure(ns, epochs, base, ema)
        plt.tight_layout()
        plt.savefig(f"{ns.output_dir}/{PLOT_FILE_NAME}")
        plt.close(figure)
        print(f"Results saved to {ns.output_dir}/{PLOT_FILE_NAME}")


def plot_loss_axes(ax, epochs, train_loss, test_loss):
    if len(epochs) > 0:
        if len(train_loss):
            style = dict(label='Training Loss', marker='o', linestyle='-')
            ax.plot(epochs, train_loss, **style)
        if len(test_loss):
            style = dict(label='Validation Loss', marker='o', linestyle='--')
            ax.plot(epochs, test_loss, **style)
        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epoch Number')
        ax.set_ylabel('Loss Value')
        ax.legend()
        ax.grid(True)


def plot_metric_series(ax, epochs, base, ema, ylabel):
    if base.size > 0 or ema.size > 0:
        if base.size > 0:
            style = dict(marker='o', linestyle='-', label='Base Model')
            ax.plot(epochs[:len(base)], base, **style)
        if ema.size > 0:
            style = dict(marker='o', linestyle='--', label='EMA Model')
            ax.plot(epochs[:len(ema)], ema, **style)
        ax.set_title(SERIES_TITLES[ylabel])
        ax.set_xlabel('Epoch Number')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)


def MetricsTensorBoardSink(output_dir):
    ns = SimpleNamespace()
    ns.output_dir = output_dir
    ns.writer = None
    if HAS_TENSORBOARD:
        try:
            os.makedirs(output_dir, exist_ok=True)
            ns.writer = EventFileWriter(output_dir)
            print("TensorBoard logging initialized.")
        except Exception:
            ns.writer = None
            msg = "Unable to initialize TensorBoard. Logging is turned off."
            print(msg)
    else:
        print("TensorBoard package not installed. Logging is turned off.")
    ns.add_scalar = functools.partial(add_tensorboard_scalar, ns)
    ns.update = functools.partial(update_tensorboard_sink, ns)
    ns.close = functools.partial(close_tensorboard_sink, ns)
    return ns


def add_tensorboard_scalar(ns, tag, value, step):
    if ns.writer is not None:
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
        event = Event(summary=summary, wall_time=time.time(), step=step)
        ns.writer.add_event(event)


def update_tensorboard_sink(ns, values):
    if ns.writer is not None:
        epoch = values.get('epoch', 0)
        for key, tag in TENSORBOARD_LOSS_TAGS.items():
            if key in values:
                ns.add_scalar(tag, values[key], epoch)
        coco_eval = values.get('test_coco_eval_bbox')
        if coco_eval is not None and read_index(coco_eval, 0) is not None:
            ns.add_scalar("Metrics/Base/AP50_90", coco_eval[0], epoch)
        ns.writer.flush()


def close_tensorboard_sink(ns):
    if ns.writer is not None:
        ns.writer.close()
        ns.writer = None


def MetricsWandBSink(output_dir, project=None, run=None, config=None):
    ns = SimpleNamespace()
    ns.output_dir = output_dir
    if wandb:
        keys = ("project", "name", "config", "dir")
        values = (project, run, config, output_dir)
        ns.run = wandb.init(**dict(zip(keys, values)))
        print("W&B logging initialized.")
    else:
        ns.run = None
        print("Unable to initialize W&B. Logging is turned off.")
    ns.update = functools.partial(update_wandb_sink, ns)
    ns.close = functools.partial(close_wandb_sink, ns)
    return ns


def update_wandb_sink(ns, values):
    if wandb and ns.run:
        wandb.log(values)


def close_wandb_sink(ns):
    if wandb and ns.run:
        ns.run.finish()
