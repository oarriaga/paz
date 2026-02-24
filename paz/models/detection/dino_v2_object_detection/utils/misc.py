import time
import datetime
from collections import defaultdict, deque
import keras
import keras.ops as k
from keras import random
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
        Warning: does not synchronize the deque!
        """
        # Distributed synchronization stub
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
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(v, "item"): # Handle tensor-like objects
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
    def __init__(self, tensors, mask=None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # Keras handles device placement differently, stub for now
        return self

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list):
    if k.ndim(tensor_list[0]) == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(k.shape(img)) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype

        tensor = k.zeros(batch_shape, dtype=dtype)
        mask = k.ones((b, h, w), dtype="bool")

        # Keras ops doesn't allow in-place assignment like tensor[i] = x
        # We need to construct the list of padded tensors and stack them.
        padded_tensors = []
        padded_masks = []
        
        for img in tensor_list:
            img_shape = k.shape(img)
            # Pad img to max_size
            # img is (C, H, W)
            # pad width to (C, max_h, max_w)
            # padding needed: (0, 0), (0, max_h - h), (0, max_w - w)
            h_pad = h - img_shape[1]
            w_pad = w - img_shape[2]
            
            # Use k.pad
            # k.pad expects [[top, bottom], [left, right]] for 2D, but we have 3D
            # It expects padding_width relative to each dimension.
            paddings = [[0, 0], [0, h_pad], [0, w_pad]]
            padded_img = k.pad(img, paddings)
            padded_tensors.append(padded_img)
            
            # Mask: 0 for valid pixels, 1 for padding
            # Create mask of shape (h, w)
            m = k.zeros((img_shape[1], img_shape[2]), dtype="bool")
            m_paddings = [[0, h_pad], [0, w_pad]]
            # Pad with True (1)
            padded_mask = k.pad(m, m_paddings, constant_values=True)
            padded_masks.append(padded_mask)
            
        tensor = k.stack(padded_tensors, axis=0)
        mask = k.stack(padded_masks, axis=0)

    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def _max_by_axis(the_list):
    # the_list is list of [C, H, W]
    maxes = the_list[0][:] # Copy
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """
    Equivalent to nn.functional.interpolate.
    Supported modes: 'nearest', 'bilinear', 'bicubic'
    """
    if size is None and scale_factor is None:
        raise ValueError("Either size or scale_factor must be defined")
        
    # Input shape: (N, C, H, W) usually in PyTorch code ported here
    # Keras usually expects (N, H, W, C). 
    # RF-DETR uses (N, C, H, W) throughout? 
    # Let's assume input is (N, C, H, W) as per standard PyTorch vision models.
    # We must transpose to (N, H, W, C) for resize, then back.
    
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
    
    return k.transpose(x, (0, 3, 1, 2))


def inverse_sigmoid(x, eps=1e-5):
    x = k.clip(x, 0, 1)
    x1 = k.maximum(x, eps)
    x2 = k.maximum(1 - x, eps)
    return k.log(x1/x2)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if k.size(target) == 0:
        return [k.zeros([])]
        
    maxk = max(topk)
    batch_size = k.shape(target)[0]

    # output: (N, C)
    # topk indices
    # k.top_k returns values, indices
    _, pred = k.top_k(output, maxk) # (N, maxk)
    pred = k.transpose(pred, (1, 0)) # (maxk, N)
    
    target_expand = k.expand_dims(target, 0) # (1, N)
    target_expand = k.repeat(target_expand, maxk, axis=0) # (maxk, N)
    
    correct = k.equal(pred, k.cast(target_expand, pred.dtype))

    res = []
    for k_val in topk:
        correct_k = k.sum(k.cast(correct[:k_val], "float32"))
        res.append(correct_k * (100.0 / batch_size))
    return res
