import keras
import matplotlib.pyplot as plt

from paz.utils import plot


class PlotMetrics(keras.callbacks.Callback):
    """Draws every logged metric against the epoch, one panel each.

    The figure is rewritten at the end of every epoch, so an interrupted run
    still leaves one behind next to its CSV log.
    """

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        for name, value in (logs or {}).items():
            entry = (epoch + 1, float(value))
            self.history.setdefault(name, []).append(entry)
        draw_metrics(self.history, self.path)


def draw_metrics(history, path):
    """Writes one panel per training metric, beside its validation twin."""
    names = [name for name in history if not name.startswith("val_")]
    kwargs = dict(figsize=(4.0 * len(names), 3.0), squeeze=False)
    figure, axes = plt.subplots(1, len(names), **kwargs)
    for axis, name in zip(axes[0], names):
        draw_metric(axis, name, history)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def draw_metric(axis, name, history):
    palette = plot.DEFAULT_PALETTE
    draw_series(axis, history[name], name, palette.primary)
    validation = history.get(f"val_{name}")
    if validation is not None:
        draw_series(axis, validation, f"val_{name}", palette.secondary)
    plot.set_labels(axis, x="epoch", y=name)
    plot.clean(axis)
    plot.legend(axis, fontsize=8)


def draw_series(axis, entries, label, color):
    """A metric logged every few epochs is drawn at the epochs it has."""
    epochs = [epoch for epoch, _ in entries]
    values = [value for _, value in entries]
    plot.line(epochs, values, axis=axis, color=color, label=label)
