import functools
import fnmatch
from collections import namedtuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    _IS_MATPLOTLIB_AVAILABLE = True
except ImportError:
    _IS_MATPLOTLIB_AVAILABLE = False
    plt, ticker = None, None


def matplotlib_required(function):
    """Decorator to ensure matplotlib is installed before function is called."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if not _IS_MATPLOTLIB_AVAILABLE:
            raise ImportError(
                f"Function '{function.__name__}' requires matplotlib. "
                "Please install it by running: pip install matplotlib"
            )
        return function(*args, **kwargs)

    return wrapper


@matplotlib_required
def build_configuration(
    mode="max",
    y_units=r"\%",
    figsize=(640, 480),
    fontsize=20,
    label_pads=(5, 5),
):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["font.family"] = "ptm"
    plt.rcParams["font.serif"] = "phv"
    yellow = (1.0, 0.65, 0.0)
    gray = (0.662, 0.647, 0.576)
    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    figsize = (figsize[0] * px, figsize[1] * px)
    Configuration = namedtuple(
        "Configuration",
        [
            "color_1",
            "color_2",
            "palette",
            "fontsize",
            "figsize",
            "x_labelpad",
            "y_labelpad",
            "mode",
            "y_units",
        ],
    )
    return Configuration(
        yellow,
        gray,
        [yellow, "tab:blue"],
        20,
        figsize,
        *label_pads,
        mode,
        y_units,
    )


@matplotlib_required
def hide_axes(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)


@matplotlib_required
def set_minor_ticks(axis):
    axis.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axis.xaxis.set_minor_locator(ticker.AutoMinorLocator())


@matplotlib_required
def set_label_pads(axis, config):
    axis.xaxis.labelpad = config.x_labelpad
    axis.yaxis.labelpad = config.y_labelpad


@matplotlib_required
def compute_line_coordinates(data, y_max, config):
    if config.mode == "max":
        x_best = np.argmax(data)
    else:
        x_best = np.argmin(data)
    y_best = data[x_best]
    return x_best, y_best


@matplotlib_required
def set_vertical_line(axis, data, y_max, config):
    x, y = compute_line_coordinates(data, y_max, config)
    y_line = (y / y_max) + 0.05
    y_shift = 0.075 * y_max
    axis.axvline(x, color=config.color_2, linestyle="--", ymax=y_line)
    text = f"{y:.2f}" + f" {config.y_units}"
    x_shift = -2.5
    axis.text(
        x + x_shift,
        y + y_shift,
        text,
        color=config.color_2,
        fontsize=config.fontsize,
    )


@matplotlib_required
def write_or_show(figure, filepath=None):
    if filepath is None:
        plt.show()
    else:
        figure.savefig(filepath, bbox_inches="tight")
        plt.close()


@matplotlib_required
def set_axis(axis, x_label=None, y_label=None, x_range=None, y_range=None):
    if x_label is not None:
        axis.set_xlabel(x_label)
    if y_label is not None:
        axis.set_ylabel(y_label)
    if x_range is not None:
        axis.set_xlim(x_range)
    if y_range is not None:
        axis.set_ylim(y_range)


@matplotlib_required
def plot_same_axis(axis, ys, y_max, legends, config):
    # ensures first element shows up on top
    for y_arg in reversed(range(len(ys))):
        y = y_max * np.array(ys[y_arg])
        axis.plot(y, "-o", color=config.palette[y_arg])
    # legends must be reversed to match the order of the lines
    if config.mode == "max":
        location = "upper left"
    elif config.mode == "min":
        location = "upper right"
    else:
        raise ValueError(f"Invalid mode: {config.mode}. Use 'max' or 'min'.")
    axis.legend(legends[::-1], prop={"size": 10}, frameon=False, loc=location)


@matplotlib_required
def loss(ys, legends, y_max=None, filepath=None):
    if y_max is None:
        y_max = np.array(ys).max()
    config = build_configuration("min")
    figure, axis = plt.subplots(figsize=config.figsize)
    plot_same_axis(axis, ys, y_max, legends, config)
    set_axis(axis, "Epoch", r"Loss (\%)", None, (0, y_max))
    hide_axes(axis)
    set_minor_ticks(axis)
    set_label_pads(axis, config)
    write_or_show(figure, filepath)


def find_values_by_wildcard(dictionary, wildcard):
    matching_values = []
    for key in dictionary.keys():
        if fnmatch.fnmatch(key, wildcard):
            matching_values.append(dictionary[key])
    return matching_values


# @matplotlib_required
# def history(wildcard, fit, legends, filepath=None):
#     name_to_data = find_values_by_wildcard(fit.history, wildcard)
#     return None


@matplotlib_required
def trace(
    x,
    filepath=None,
    y_label=None,
    x_label="iteration",
    x_range=None,
    y_range=None,
):
    config = build_configuration()
    figure, axis = plt.subplots(figsize=config.figsize)
    hide_axes(axis)
    set_minor_ticks(axis)
    set_label_pads(axis, config)
    set_axis(axis, x_label, y_label, x_range, y_range)
    axis.plot(x, color="black")
    write_or_show(figure, filepath)
