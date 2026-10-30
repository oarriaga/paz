import functools
import time
import datetime
from collections import defaultdict, deque
from types import SimpleNamespace
import keras.ops as k
import numpy as np

INTERPOLATION_MODES = {"bilinear": "bilinear", "bicubic": "bicubic"}
SMOOTHED_VALUE_METHODS = ("update", "synchronize_between_processes", "median", "avg", "global_avg", "max", "value", "to_str")  # fmt: skip
METRIC_LOGGER_METHODS = ("update", "to_str", "synchronize_between_processes", "add_meter", "log_every")  # fmt: skip


def SmoothedValue(window_size=20, fmt=None):
    ns = SimpleNamespace()
    ns.deque = deque(maxlen=window_size)
    ns.total = 0.0
    ns.count = 0
    ns.fmt = fmt or "{median:.4f} ({global_avg:.4f})"
    functions = (update_smoothed_value, synchronize_smoothed_value, read_smoothed_median, read_smoothed_average, read_smoothed_global_average, read_smoothed_max, read_smoothed_value, format_smoothed_value)  # fmt: skip
    for name, function in zip(SMOOTHED_VALUE_METHODS, functions):
        setattr(ns, name, functools.partial(function, ns))
    return ns


def update_smoothed_value(ns, value, n=1):
    ns.deque.append(value)
    ns.count += n
    ns.total += value * n


def synchronize_smoothed_value(ns):
    # Single-process training keeps every meter local; the hook exists so
    # callers can stay backend-agnostic.
    return None


def read_smoothed_median(ns):
    return np.median(np.array(list(ns.deque)))


def read_smoothed_average(ns):
    return np.mean(np.array(list(ns.deque), dtype="float32"))


def read_smoothed_global_average(ns):
    return ns.total / ns.count


def read_smoothed_max(ns):
    return max(ns.deque)


def read_smoothed_value(ns):
    return ns.deque[-1]


def format_smoothed_value(ns):
    keys = ("median", "avg", "global_avg", "max", "value")
    values = (ns.median(), ns.avg(), ns.global_avg(), ns.max(), ns.value())
    return ns.fmt.format(**dict(zip(keys, values)))


def MetricLogger(delimiter="\t"):
    ns = SimpleNamespace()
    ns.meters = defaultdict(SmoothedValue)
    ns.delimiter = delimiter
    functions = (update_metrics, format_metrics, synchronize_metrics, add_metric_meter, log_every)  # fmt: skip
    for name, function in zip(METRIC_LOGGER_METHODS, functions):
        setattr(ns, name, functools.partial(function, ns))
    return ns


def update_metrics(ns, **kwargs):
    for key, value in kwargs.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (float, int)):
            ns.meters[key].update(value)


def format_metrics(ns):
    parts = [f"{name}: {meter.to_str()}" for name, meter in ns.meters.items()]
    return ns.delimiter.join(parts)


def synchronize_metrics(ns):
    for meter in ns.meters.values():
        meter.synchronize_between_processes()


def add_metric_meter(ns, name, meter):
    ns.meters[name] = meter


def build_log_message(delimiter, header, total):
    space = ":" + str(len(str(total))) + "d"
    counter = "[{0" + space + "}/{1}]"
    parts = (header, counter, "eta: {eta}", "{meters}", "time: {time}", "data: {data}")  # fmt: skip
    return delimiter.join(parts)


def print_progress(ns, message, index, total, iter_time, data_time):
    eta_seconds = iter_time.global_avg() * (total - index)
    keys = ("eta", "meters", "time", "data")
    values = (str(datetime.timedelta(seconds=int(eta_seconds))), ns.to_str(), iter_time.to_str(), data_time.to_str())  # fmt: skip
    print(message.format(index, total, **dict(zip(keys, values))))


def report_total_time(header, elapsed, total):
    formatted = str(datetime.timedelta(seconds=int(elapsed)))
    per_step = elapsed / total
    print("{} Total time: {} ({:.4f} s / it)".format(header, formatted, per_step))  # fmt: skip


def log_every(ns, iterable, print_freq, header=None):
    header = header or ""
    total = len(iterable)
    message = build_log_message(ns.delimiter, header, total)
    iter_time = SmoothedValue(fmt="{avg:.4f}")
    data_time = SmoothedValue(fmt="{avg:.4f}")
    start_time = end = time.time()
    for index, item in enumerate(iterable):
        data_time.update(time.time() - end)
        yield item
        iter_time.update(time.time() - end)
        if index % print_freq == 0 or index == total - 1:
            args = (ns, message, index, total, iter_time, data_time)
            print_progress(*args)
        end = time.time()
    report_total_time(header, time.time() - start_time, total)


def NestedTensor(tensors, mask=None):
    ns = SimpleNamespace()
    ns.tensors = tensors
    ns.mask = mask
    ns.to = functools.partial(move_nested_tensor, ns)
    ns.decompose = functools.partial(decompose_nested_tensor, ns)
    return ns


# device is unused: this mirrors the torch NestedTensor API so ported call
# sites keep working, and Keras/JAX place tensors implicitly.
def move_nested_tensor(ns, device):
    return ns


def decompose_nested_tensor(ns):
    return ns.tensors, ns.mask


def pad_image_to_size(image, height, width):
    shape = k.shape(image)
    height_pad = height - shape[1]
    width_pad = width - shape[2]
    padded = k.pad(image, [[0, 0], [0, height_pad], [0, width_pad]])
    # Mask is False for valid pixels and True for padding
    mask = k.zeros((shape[1], shape[2]), dtype="bool")
    mask_paddings = [[0, height_pad], [0, width_pad]]
    return padded, k.pad(mask, mask_paddings, constant_values=True)


def nested_tensor_from_tensor_list(tensor_list):
    if k.ndim(tensor_list[0]) != 3:
        raise ValueError("not supported")
    max_size = max_by_axis([list(k.shape(image)) for image in tensor_list])
    height, width = max_size[1], max_size[2]
    # Keras tensors are immutable, so pad each image and stack the batch
    padded = [pad_image_to_size(image, height, width) for image in tensor_list]
    tensor = k.stack([image for image, _ in padded], axis=0)
    mask = k.stack([mask for _, mask in padded], axis=0)
    return NestedTensor(tensor, mask)


def max_by_axis(shapes):
    maxes = shapes[0][:]
    for shape in shapes[1:]:
        for index, item in enumerate(shape):
            maxes[index] = max(maxes[index], item)
    return maxes


def resolve_resize_size(x, size, scale_factor):
    if size is not None:
        new_size = size
    else:
        shape = k.shape(x)
        height, width = shape[1], shape[2]
        new_size = [int(height * scale_factor), int(width * scale_factor)]
    return new_size


# align_corners is unused; kept for API compatibility with torch callers.
def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):  # fmt: skip
    if size is None and scale_factor is None:
        raise ValueError("Either size or scale_factor must be defined")
    # Resize expects NHWC, so transpose out of and back into NCHW
    x = k.transpose(input, (0, 2, 3, 1))
    new_size = resolve_resize_size(x, size, scale_factor)
    method = INTERPOLATION_MODES.get(mode, "nearest")
    resized = k.image.resize(x, new_size, interpolation=method)
    return k.transpose(resized, (0, 3, 1, 2))


def inverse_sigmoid(x, eps=1e-5):
    x = k.clip(x, 0, 1)
    numerator = k.maximum(x, eps)
    denominator = k.maximum(1 - x, eps)
    return k.log(numerator / denominator)


def accuracy(output, target, topk=(1,)):
    if k.size(target) == 0:
        result = [k.zeros([])]
    else:
        batch_size = k.shape(target)[0]
        predictions = k.transpose(k.top_k(output, max(topk))[1], (1, 0))
        expanded = k.repeat(k.expand_dims(target, 0), max(topk), axis=0)
        correct = k.equal(predictions, k.cast(expanded, predictions.dtype))
        counts = [k.sum(k.cast(correct[:top], "float32")) for top in topk]
        result = [count * (100.0 / batch_size) for count in counts]
    return result


def is_dist_avail_and_initialized():
    try:
        import jax
        available = jax.process_count() > 1
    except Exception:
        available = False
    return available


def get_world_size():
    size = 1
    if is_dist_avail_and_initialized():
        import jax
        size = jax.process_count()
    return size


def get_rank():
    rank = 0
    if is_dist_avail_and_initialized():
        import jax
        rank = jax.process_index()
    return rank


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        import keras
        keras.saving.save_model(*args, **kwargs)


def setup_for_distributed(is_master):
    import builtins
    args = (is_master, builtins.print)
    builtins.print = functools.partial(print_on_master, *args)


def print_on_master(is_master, builtin_print, *args, **kwargs):
    force = kwargs.pop("force", False)
    if is_master or force:
        builtin_print(*args, **kwargs)
