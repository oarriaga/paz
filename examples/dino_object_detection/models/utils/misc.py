import datetime
import os
import time
import pickle
import subprocess
from collections import defaultdict, deque
from typing import Optional, List

import keras
from keras import ops
import numpy as np


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Synchronization is backend-specific in Keras 3.
        Current implementation assumes single-process (World Size = 1).
        """
        if not is_dist_avail_and_initialized():
            return
        pass

    @property
    def median(self):
        d = ops.convert_to_tensor(list(self.deque))
        return float(ops.median(d))

    @property
    def avg(self):
        d = ops.convert_to_tensor(list(self.deque), dtype="float32")
        return float(ops.mean(d))

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data.
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    return [data]


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    return input_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t", wandb_logging=False):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        if wandb_logging:
            import wandb

            self.wandb = wandb
        else:
            self.wandb = None

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(v, "numpy"):
                v = float(v)
            if isinstance(v, (list, tuple)) and len(v) == 1:
                v = v[0]
            if not isinstance(v, (float, int)):
                try:
                    v = float(v)
                except:
                    raise TypeError(f"Metric {k} must be scalar, got {type(v)}")

            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"

        log_msg = self.delimiter.join(
            [
                header,
                "[{0" + space_fmt + "}/{1}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
            ]
        )

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if self.wandb:
                    if is_main_process():
                        log_dict = {k: v.value for k, v in self.meters.items()}
                        self.wandb.log(log_dict)

                print(
                    log_msg.format(
                        i,
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                    )
                )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[any]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors
        mask = self.mask
        if mask is not None:
            cast_mask = mask
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list):
    """
    Equivalent to the PyTorch implementation using Keras Ops.
    Assumes tensor_list contains tensors of shape (C, H, W).
    """
    if len(ops.shape(tensor_list[0])) == 3:
        shapes = [ops.shape(t) for t in tensor_list]
        max_size = _max_by_axis([list(s) for s in shapes])

        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape

        padded_tensors = []
        padded_masks = []

        for img in tensor_list:
            img_shape = ops.shape(img)
            pad_h = h - img_shape[1]
            pad_w = w - img_shape[2]
            padding = [[0, 0], [0, pad_h], [0, pad_w]]
            padded_img = ops.pad(img, padding)
            padded_tensors.append(padded_img)

            valid_h = img_shape[1]
            valid_w = img_shape[2]
            mask_valid = ops.zeros((valid_h, valid_w), dtype="bool")
            mask_padding = [[0, pad_h], [0, pad_w]]
            padded_mask = ops.pad(mask_valid, mask_padding, constant_values=True)
            padded_masks.append(padded_mask)

        tensor = ops.stack(padded_tensors, axis=0)
        mask = ops.stack(padded_masks, axis=0)

    else:
        raise ValueError("not supported")

    return NestedTensor(tensor, mask)


def setup_for_distributed(is_master):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    return False


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return 1


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return 0


def is_main_process():
    return get_rank() == 0


def save_on_master(obj, f, *args, **kwargs):
    if is_main_process():
        with open(f, "wb") as file_obj:
            pickle.dump(obj, file_obj)


def init_distributed_mode(args):
    print("Not using distributed mode (Keras 3 Port Defaults)")
    args.distributed = False
    args.rank = 0
    args.world_size = 1
    args.gpu = 0
    return


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = ops.shape(target)[0]

    _, pred = ops.top_k(output, k=maxk)
    pred = ops.transpose(pred)
    target_reshaped = ops.reshape(target, (1, -1))
    correct = ops.equal(pred, ops.cast(target_reshaped, pred.dtype))

    res = []
    for k in topk:
        correct_k = ops.sum(ops.cast(ops.reshape(correct[:k], (-1,)), "float32"))
        res.append(correct_k * (100.0 / batch_size))
    return res


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    """
    Wrapper for resizing. Matches PyTorch behavior.
    """
    if size is None and scale_factor is None:
        raise ValueError("Either size or scale_factor must be defined")

    shape = ops.shape(input)
    # Shape is [B, C, H, W]

    if size is not None:
        target_h, target_w = size
    else:
        current_h, current_w = shape[2], shape[3]
        target_h = int(float(current_h) * scale_factor)
        target_w = int(float(current_w) * scale_factor)

    if mode == "nearest":
        # Get dimensions
        h = shape[2]
        w = shape[3]

        # Calculate scales using floats
        scale_h = ops.cast(h, "float32") / ops.cast(target_h, "float32")
        scale_w = ops.cast(w, "float32") / ops.cast(target_w, "float32")

        # Generate target grid
        grid_y = ops.arange(target_h, dtype="float32")
        grid_x = ops.arange(target_w, dtype="float32")

        # Calculate source indices
        src_y = ops.floor(grid_y * scale_h)
        src_x = ops.floor(grid_x * scale_w)

        # Cast to int
        src_y = ops.cast(src_y, "int32")
        src_x = ops.cast(src_x, "int32")

        # Clip to be safe
        src_y = ops.clip(src_y, 0, h - 1)
        src_x = ops.clip(src_x, 0, w - 1)

        # Gather (Input is B, C, H, W)
        # Gather along axis 2 (H)
        out = ops.take(input, src_y, axis=2)
        # Gather along axis 3 (W)
        out = ops.take(out, src_x, axis=3)
        return out

    x = ops.transpose(input, (0, 2, 3, 1))  # B H W C

    x = ops.image.resize(
        x,
        (target_h, target_w),
        interpolation=mode,
        data_format="channels_last",
    )

    x = ops.transpose(x, (0, 3, 1, 2))  # B C H W
    return x


def inverse_sigmoid(x, eps=1e-5):
    x = ops.clip(x, 0, 1)
    x1 = ops.clip(x, eps, 1.0)
    x2 = ops.clip(1 - x, eps, 1.0)
    return ops.log(x1 / x2)


def strip_checkpoint(checkpoint):
    with open(checkpoint, "rb") as f:
        state_dict = pickle.load(f)

    new_state_dict = {
        "model": state_dict.get("model", None),
        "args": state_dict.get("args", None),
    }

    with open(checkpoint, "wb") as f:
        pickle.dump(new_state_dict, f)
