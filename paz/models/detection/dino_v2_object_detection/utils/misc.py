import time
import datetime
from collections import defaultdict, deque
import keras
import keras.ops as k
from keras import random
import numpy as np

class SmoothedValue(object):
    """Track a series of values and provide smoothed statistics.

    Maintains a fixed-size sliding window of recent values together with
    a running global total/count for computing windowed and global
    averages.

    Attributes:
        deque (deque): Sliding window of recent values.
        total (float): Running sum of all recorded values.
        count (int): Total number of recorded values.
        fmt (str): Format string used by ``__str__``.
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
        """Placeholder for distributed synchronization.

        In a multi-process setup this would aggregate the global
        total and count across workers. Currently a no-op.
        """
        return

    @property
    def median(self):
        d = np.array(list(self.deque))
        return np.median(d)

    @property
    def avg(self):
        d = np.array(list(self.deque), dtype="float32")
        return np.mean(d)

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
            value=self.value)


class MetricLogger(object):
    """Aggregate and display multiple ``SmoothedValue`` meters.

    Attributes:
        meters (dict): Mapping from metric name to ``SmoothedValue``.
        delimiter (str): Separator used when formatting output.
    """
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """Record one or more named scalar values."""
        for k, v in kwargs.items():
            if hasattr(v, "item"):
                v = v.item()
            if isinstance(v, (float, int)):
                self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """Yield items from *iterable*, printing progress periodically.

        Args:
            iterable: Data source to iterate over.
            print_freq (int): Print every *print_freq* iterations.
            header (str | None): Prefix for log lines.

        Yields:
            Items from *iterable*.
        """
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ])
        
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class NestedTensor(object):
    """Pair of a batched image tensor and a corresponding padding mask.

    Wraps a ``(B, C, H, W)`` tensor and a ``(B, H, W)`` boolean mask
    where ``True`` marks padded (invalid) pixels.

    Attributes:
        tensors: Batched image tensor.
        mask: Boolean padding mask.
    """
    def __init__(self, tensors, mask=None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        """Device placement stub (no-op under Keras)."""
        return self

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list):
    """Pad a list of 3-D image tensors into a batched ``NestedTensor``.

    Each image ``(C, H, W)`` is zero-padded to the maximum spatial
    dimensions found in *tensor_list*.  The returned mask is ``True``
    for padded positions.

    Args:
        tensor_list: List of tensors, each of shape ``(C, H, W)``.

    Returns:
        NestedTensor: Batched ``(B, C, H_max, W_max)`` tensor with
            an accompanying ``(B, H_max, W_max)`` boolean padding mask.

    Raises:
        ValueError: If input tensors are not 3-D.
    """
    if k.ndim(tensor_list[0]) == 3:
        # Determine the maximum spatial extent across all images
        max_size = _max_by_axis([list(k.shape(img)) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype

        tensor = k.zeros(batch_shape, dtype=dtype)
        mask = k.ones((b, h, w), dtype="bool")

        # Build padded tensors and masks for each image, then stack
        # into a single batch. Direct element assignment is not
        # supported on immutable Keras tensors.
        padded_tensors = []
        padded_masks = []
        
        for img in tensor_list:
            img_shape = k.shape(img)
            # Compute padding needed to reach (C, H_max, W_max)
            h_pad = h - img_shape[1]
            w_pad = w - img_shape[2]
            
            # Pad along spatial dimensions only (channel dim unchanged)
            paddings = [[0, 0], [0, h_pad], [0, w_pad]]
            padded_img = k.pad(img, paddings)
            padded_tensors.append(padded_img)
            
            # Mask: False for valid pixels, True for padding
            m = k.zeros((img_shape[1], img_shape[2]), dtype="bool")
            m_paddings = [[0, h_pad], [0, w_pad]]
            padded_mask = k.pad(m, m_paddings, constant_values=True)
            padded_masks.append(padded_mask)
            
        tensor = k.stack(padded_tensors, axis=0)
        mask = k.stack(padded_masks, axis=0)

    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def _max_by_axis(the_list):
    """Element-wise maximum across a list of equal-length lists.

    Args:
        the_list: List of lists (e.g. ``[[C, H, W], ...]``).

    Returns:
        list: Element-wise maxima.
    """
    maxes = the_list[0][:]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """Resize a 4-D tensor (NCHW) using the specified interpolation mode.

    The input is transposed to channels-last layout for the resize
    operation and transposed back to channels-first before returning.

    Args:
        input: Tensor of shape ``(N, C, H, W)``.
        size (tuple | None): Target ``(H, W)``.
        scale_factor (float | None): Multiplicative spatial factor.
            Exactly one of *size* or *scale_factor* must be given.
        mode (str): ``'nearest'``, ``'bilinear'``, or ``'bicubic'``.
        align_corners: Unused; accepted for API compatibility.

    Returns:
        Tensor of shape ``(N, C, H', W')``.

    Raises:
        ValueError: If neither *size* nor *scale_factor* is provided.
    """
    if size is None and scale_factor is None:
        raise ValueError("Either size or scale_factor must be defined")
        
    # Transpose from NCHW to NHWC for the resize operation
    x = k.transpose(input, (0, 2, 3, 1))
    
    if size is not None:
        # size is (H, W)
        new_size = size
    else:
        # scale_factor
        shape = k.shape(x)
        h, w = shape[1], shape[2]
        new_size = [int(h * scale_factor), int(w * scale_factor)]
        
    if mode == 'bilinear':
        method = 'bilinear'
    elif mode == 'bicubic':
        method = 'bicubic'
    else:
        method = 'nearest'
        
    x = k.image.resize(x, new_size, interpolation=method)
    
    # Transpose back to NCHW
    return k.transpose(x, (0, 3, 1, 2))


def inverse_sigmoid(x, eps=1e-5):
    """Compute the inverse sigmoid (logit) of a probability tensor.

    Values are clipped to ``[0, 1]`` and clamped away from zero before
    taking the log to ensure numerical stability.

    Args:
        x: Tensor of probabilities.
        eps (float): Minimum value for clamping.

    Returns:
        Tensor of logits.
    """
    x = k.clip(x, 0, 1)
    x1 = k.maximum(x, eps)
    x2 = k.maximum(1 - x, eps)
    return k.log(x1/x2)


def accuracy(output, target, topk=(1,)):
    """Compute top-k classification accuracy.

    Args:
        output: Logits tensor of shape ``(N, C)``.
        target: Ground-truth class indices of shape ``(N,)``.
        topk (tuple[int]): Tuple of *k* values to evaluate.

    Returns:
        list: One accuracy percentage per requested *k*.
    """
    if k.size(target) == 0:
        return [k.zeros([])]
        
    maxk = max(topk)
    batch_size = k.shape(target)[0]

    # Retrieve the top-k predicted class indices per sample
    _, pred = k.top_k(output, maxk)  # (N, maxk)
    pred = k.transpose(pred, (1, 0))  # (maxk, N)
    
    # Broadcast target to (maxk, N) for element-wise comparison
    target_expand = k.expand_dims(target, 0)  # (1, N)
    target_expand = k.repeat(target_expand, maxk, axis=0)  # (maxk, N)
    
    correct = k.equal(pred, k.cast(target_expand, pred.dtype))

    res = []
    for k_val in topk:
        correct_k = k.sum(k.cast(correct[:k_val], "float32"))
        res.append(correct_k * (100.0 / batch_size))
    return res


# ---------------------------------------------------------------------------
# Distributed training utilities
# ---------------------------------------------------------------------------


def is_dist_avail_and_initialized():
    """Check whether JAX multi-host distributed mode is active.

    Returns True when ``jax.process_count() > 1``, meaning the
    program was launched via ``jax.distributed.initialize()``.
    """
    try:
        import jax
        return jax.process_count() > 1
    except Exception:
        return False


def get_world_size():
    """Return the number of JAX processes (1 if single-device)."""
    if not is_dist_avail_and_initialized():
        return 1
    import jax
    return jax.process_count()


def get_rank():
    """Return the current process index (0 if single-device)."""
    if not is_dist_avail_and_initialized():
        return 0
    import jax
    return jax.process_index()


def is_main_process():
    """Return True if this is rank 0 (or single-device)."""
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """Call ``keras.saving.save_model`` only on the master process."""
    if is_main_process():
        import keras
        keras.saving.save_model(*args, **kwargs)


def setup_for_distributed(is_master):
    """Suppress print output on non-master processes.

    After calling this, ``print()`` becomes a no-op on workers
    with ``is_master=False`` unless ``force=True`` is passed.
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
