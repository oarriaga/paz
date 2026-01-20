import os
import sys
import json
import time
import logging
import typing
from collections import Counter, defaultdict
from pathlib import Path
from functools import partial

import numpy as np
from numpy import prod
import tqdm

import keras

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Backend Utilities
# -------------------------------------------------------------------------


def get_shape(shape_val: typing.Any) -> typing.List[int]:
    """
    Standardize shape to list of integers, handling None (batch dim) as 1.
    """
    if shape_val is None:
        return []

    if hasattr(shape_val, "as_list"):
        return shape_val.as_list()

    if isinstance(shape_val, (int, float)):
        return [int(shape_val)]

    # Handle shape tuples
    clean_shape = []
    for s in shape_val:
        if s is None:
            clean_shape.append(1)
        else:
            clean_shape.append(int(s))
    return clean_shape


def sync_device(tensor=None):
    if tensor is None:
        return
    if hasattr(tensor, "value"):
        tensor = tensor.value
    if hasattr(tensor, "block_until_ready"):
        try:
            tensor.block_until_ready()
            return
        except Exception:
            pass
    if isinstance(tensor, (list, tuple)):
        for t in tensor:
            sync_device(t)
        return
    if hasattr(tensor, "numpy"):
        _ = tensor.numpy()
    elif hasattr(tensor, "cpu"):
        _ = tensor.cpu()


# -------------------------------------------------------------------------
# FLOP Handlers
# -------------------------------------------------------------------------


def dense_flop_handler(layer, inputs, outputs):
    input_shape = get_shape(inputs[0].shape)
    output_shape = get_shape(outputs.shape)

    batch_size = prod(input_shape[:-1])
    input_dim = input_shape[-1]
    output_dim = output_shape[-1]

    flop = batch_size * input_dim * output_dim
    return Counter({"addmm": flop})


def conv2d_flop_handler(layer, inputs, outputs):
    input_shape = get_shape(inputs[0].shape)
    output_shape = get_shape(outputs.shape)
    w_shape = layer.kernel.shape

    if layer.data_format == "channels_last":
        batch_size = input_shape[0]
        Cin = input_shape[-1]
        Cout = output_shape[-1]
        out_pixels = prod(output_shape[1:-1])
        kernel_ops = prod(w_shape[:2])
    else:
        batch_size = input_shape[0]
        Cin = input_shape[1]
        Cout = output_shape[1]
        out_pixels = prod(output_shape[2:])
        kernel_ops = prod(w_shape[:2])

    groups = getattr(layer, "groups", 1)
    flop = batch_size * out_pixels * (Cin // groups) * Cout * kernel_ops
    return Counter({"conv": flop})


def depthwise_conv2d_flop_handler(layer, inputs, outputs):
    input_shape = get_shape(inputs[0].shape)
    output_shape = get_shape(outputs.shape)
    w_shape = layer.kernel.shape

    if layer.data_format == "channels_last":
        batch_size = input_shape[0]
        Cin = input_shape[-1]
        out_pixels = prod(output_shape[1:-1])
    else:
        batch_size = input_shape[0]
        Cin = input_shape[1]
        out_pixels = prod(output_shape[2:])

    kernel_ops = prod(w_shape[:2])
    flop = batch_size * out_pixels * Cin * kernel_ops
    return Counter({"conv": flop})


def norm_flop_handler(layer, inputs, outputs):
    input_shape = get_shape(inputs[0].shape)
    if isinstance(layer, keras.layers.BatchNormalization):
        ops_per_element = 4
        name = "batchnorm"
    else:
        has_affine = getattr(layer, "center", False) or getattr(layer, "scale", False)
        ops_per_element = 5 if has_affine else 4
        name = "norm"

    flop = prod(input_shape) * ops_per_element
    return Counter({name: flop})


def pooling_flop_handler(layer, inputs, outputs):
    # PyTorch Benchmark 'elementwise_flop_counter' uses inputs[0] numel.
    # We match that logic here.
    input_shape = get_shape(inputs[0].shape)
    flop = prod(input_shape)
    return Counter({"pooling": flop})


def elementwise_handler(layer, inputs, outputs):
    input_shape = get_shape(inputs[0].shape)
    flop = prod(input_shape)
    name = "elementwise"
    if isinstance(layer, keras.layers.ReLU):
        name = "aten::relu"
    return Counter({name: flop})


def softmax_handler(layer, inputs, outputs):
    input_shape = get_shape(inputs[0].shape)
    flop = prod(input_shape) * 5
    return Counter({"softmax": flop})


def mha_flop_handler(layer, inputs, outputs):
    # Inputs[0] is Query.
    q_shape = get_shape(inputs[0].shape)

    B = q_shape[0]
    S = q_shape[1]
    D = q_shape[-1]

    num_heads = layer.num_heads
    key_dim = layer.key_dim
    value_dim = layer.value_dim if layer.value_dim else key_dim

    flops = 0.0
    # 1. Linear Projections (Q, K, V)
    # 3 Projections: (B, S, D) -> (B, S, num_heads * key_dim)
    flops += 3 * (B * S * D * (num_heads * key_dim))

    # 2. Attention Scores (MatMul): (B, num_heads, S, key_dim) * (B, num_heads, key_dim, S)
    flops += B * num_heads * S * S * key_dim

    # 3. Weighted Sum (MatMul): (B, num_heads, S, S) * (B, num_heads, S, value_dim)
    flops += B * num_heads * S * S * value_dim

    # 4. Output Projection (Linear): (B, S, num_heads * value_dim) -> (B, S, D)
    flops += B * S * (num_heads * value_dim) * D

    return Counter({"matmul": flops})


_LAYER_HANDLERS = {
    keras.layers.Dense: dense_flop_handler,
    keras.layers.EinsumDense: dense_flop_handler,
    keras.layers.Conv2D: conv2d_flop_handler,
    keras.layers.Conv1D: conv2d_flop_handler,
    keras.layers.Conv3D: conv2d_flop_handler,
    keras.layers.DepthwiseConv2D: depthwise_conv2d_flop_handler,
    keras.layers.Conv2DTranspose: conv2d_flop_handler,
    keras.layers.BatchNormalization: norm_flop_handler,
    keras.layers.LayerNormalization: norm_flop_handler,
    keras.layers.GroupNormalization: norm_flop_handler,
    keras.layers.MaxPooling2D: pooling_flop_handler,
    keras.layers.AveragePooling2D: pooling_flop_handler,
    keras.layers.GlobalAveragePooling2D: pooling_flop_handler,
    keras.layers.GlobalMaxPooling2D: pooling_flop_handler,
    keras.layers.ReLU: elementwise_handler,
    keras.layers.LeakyReLU: elementwise_handler,
    keras.layers.PReLU: elementwise_handler,
    keras.layers.ELU: elementwise_handler,
    keras.layers.Activation: elementwise_handler,
    keras.layers.Softmax: softmax_handler,
    keras.layers.Add: elementwise_handler,
    keras.layers.Subtract: elementwise_handler,
    keras.layers.Multiply: elementwise_handler,
    keras.layers.Dropout: lambda l, i, o: Counter(
        {"dropout": prod(get_shape(i[0].shape))}
    ),
    keras.layers.MultiHeadAttention: mha_flop_handler,
}


# -------------------------------------------------------------------------
# Shape inference helper
# -------------------------------------------------------------------------
def infer_input_shape_from_weights(layer, input_bs=1):
    """
    Try to infer input shape from layer weights for built layers.
    Works for Dense, Conv layers, etc.
    """
    if isinstance(layer, keras.layers.Dense):
        if hasattr(layer, "kernel") and layer.kernel is not None:
            # kernel shape is (input_dim, output_dim)
            input_dim = layer.kernel.shape[0]
            return [input_bs, input_dim]
    elif isinstance(layer, (keras.layers.Conv2D, keras.layers.Conv1D)):
        # For conv layers, we can't fully infer spatial dimensions
        # but we can get channel info
        pass
    return None


# -------------------------------------------------------------------------
# Core Logic
# -------------------------------------------------------------------------
def flop_count(
    model: keras.Model,
    inputs: typing.Tuple[object, ...],
    whitelist: typing.Union[typing.List[str], None] = None,
    customized_ops: typing.Union[
        typing.Dict[typing.Type, typing.Callable], None
    ] = None,
) -> typing.DefaultDict[str, float]:

    handlers = _LAYER_HANDLERS.copy()
    if customized_ops:
        handlers.update(customized_ops)

    # Resolve batch size and actual inputs
    input_bs = 1
    actual_inputs = inputs
    if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
        actual_inputs = inputs[0]

    try:
        if hasattr(actual_inputs[0], "shape"):
            input_bs = actual_inputs[0].shape[0]
    except:
        pass

    total_flop_counter = Counter()
    skipped_ops = Counter()

    try:
        if len(actual_inputs) == 1:
            _ = model(actual_inputs[0], training=False)
        elif len(actual_inputs) == 2 and isinstance(
            model, keras.layers.MultiHeadAttention
        ):
            # MHA expects [query, value] or [query, value, key]
            _ = model(actual_inputs[0], actual_inputs[1], training=False)
        else:
            _ = model(actual_inputs, training=False)
    except Exception as e:
        pass

    # Track parent input shapes during traversal
    layer_input_shapes = {}

    def get_all_layers_with_shapes(m, parent_input_shape=None):
        """
        Recursively extract layers and track their input shapes.
        """
        layers_found = []
        has_handler = type(m) in handlers

        if has_handler:
            if parent_input_shape is not None:
                layer_input_shapes[id(m)] = parent_input_shape
            return [m]

        # Try to get this layer/model's input shape
        current_shape = parent_input_shape
        if current_shape is None and hasattr(m, "_inbound_nodes") and m._inbound_nodes:
            try:
                node = m._inbound_nodes[0]
                if hasattr(node, "input_tensors"):
                    tensors = (
                        node.input_tensors
                        if isinstance(node.input_tensors, list)
                        else [node.input_tensors]
                    )
                    if tensors:
                        current_shape = get_shape(tensors[0].shape)
            except:
                pass

        # Method 1: Functional/Sequential models
        if hasattr(m, "layers") and m.layers:
            for layer in m.layers:
                layers_found.extend(get_all_layers_with_shapes(layer, None))
        else:
            # Method 2: Subclassed layers - introspect attributes
            found_sublayers = False
            intermediate_shape = current_shape

            for attr_name in sorted(dir(m)):  # Sort for deterministic order
                if attr_name.startswith("_"):
                    continue
                try:
                    attr = getattr(m, attr_name)
                    if isinstance(attr, keras.layers.Layer):
                        # Pass current shape to sublayer
                        sublayers = get_all_layers_with_shapes(attr, intermediate_shape)
                        layers_found.extend(sublayers)
                        found_sublayers = True

                        # Propagate shape: try to compute output for next layer
                        if intermediate_shape is not None and hasattr(
                            attr, "compute_output_shape"
                        ):
                            try:
                                out_shape = attr.compute_output_shape(
                                    intermediate_shape
                                )
                                intermediate_shape = get_shape(out_shape)
                            except:
                                pass

                    elif isinstance(attr, (list, tuple)):
                        for item in attr:
                            if isinstance(item, keras.layers.Layer):
                                sublayers = get_all_layers_with_shapes(
                                    item, intermediate_shape
                                )
                                layers_found.extend(sublayers)
                                found_sublayers = True
                except:
                    continue

            if not found_sublayers:
                if current_shape is not None:
                    layer_input_shapes[id(m)] = current_shape
                layers_found.append(m)

        return layers_found

    all_layers = get_all_layers_with_shapes(model, None)

    # Special case: if model itself has a handler, include it
    if type(model) in handlers and model not in all_layers:
        all_layers.insert(0, model)
    # --------------------------------

    for layer in all_layers:
        layer_type = type(layer)
        layer_name = layer_type.__name__

        if whitelist and layer_name not in whitelist:
            continue

        handler = handlers.get(layer_type, None)

        if handler:
            try:

                def resolve(s):
                    clean = get_shape(s)
                    if clean and (clean[0] is None or clean[0] == -1):
                        clean[0] = input_bs
                    return clean

                class ShapeWrapper:
                    def __init__(self, s):
                        self.shape = resolve(s)

                ins = None
                out = None

                # Try multiple methods to get input shape

                # Method 1: Standard input_shape attribute
                if hasattr(layer, "input_shape") and layer.input_shape is not None:
                    if isinstance(layer.input_shape, list):
                        ins = [ShapeWrapper(s) for s in layer.input_shape]
                    else:
                        ins = [ShapeWrapper(layer.input_shape)]

                # Method 2: From inbound nodes (Functional API)
                if (
                    ins is None
                    and hasattr(layer, "_inbound_nodes")
                    and layer._inbound_nodes
                ):
                    try:
                        node = layer._inbound_nodes[0]
                        if hasattr(node, "input_tensors"):
                            input_tensors = node.input_tensors
                            if not isinstance(input_tensors, list):
                                input_tensors = [input_tensors]
                            ins = [ShapeWrapper(t.shape) for t in input_tensors]
                    except Exception:
                        pass

                # Method 3: From traversal mapping
                if ins is None and id(layer) in layer_input_shapes:
                    ins = [ShapeWrapper(layer_input_shapes[id(layer)])]

                # Method 4: get_input_shape_at
                if ins is None:
                    try:
                        shape = layer.get_input_shape_at(0)
                        if isinstance(shape, list):
                            ins = [ShapeWrapper(s) for s in shape]
                        else:
                            ins = [ShapeWrapper(shape)]
                    except Exception:
                        pass

                # Method 5: Infer from weights (last resort for built layers)
                if ins is None and layer.built:
                    inferred = infer_input_shape_from_weights(layer, input_bs)
                    if inferred:
                        ins = [ShapeWrapper(inferred)]

                # Method 6: For top-level model
                if ins is None and layer is model:
                    # If model is the layer itself (e.g., standalone MHA)
                    if (
                        isinstance(model, keras.layers.MultiHeadAttention)
                        and len(actual_inputs) >= 2
                    ):
                        # MHA gets [query, value] or [query, value, key]
                        ins = [
                            ShapeWrapper(actual_inputs[0].shape),
                            ShapeWrapper(actual_inputs[1].shape),
                        ]
                    else:
                        ins = [ShapeWrapper(x.shape) for x in actual_inputs]

                if ins is None:
                    skipped_ops[layer_name] += 1
                    continue

                # Output shape resolution
                if hasattr(layer, "output_shape") and layer.output_shape is not None:
                    out = ShapeWrapper(layer.output_shape)

                if (
                    out is None
                    and hasattr(layer, "_inbound_nodes")
                    and layer._inbound_nodes
                ):
                    try:
                        node = layer._inbound_nodes[0]
                        if hasattr(node, "output_tensors"):
                            output_tensors = node.output_tensors
                            if not isinstance(output_tensors, list):
                                output_tensors = [output_tensors]
                            out = ShapeWrapper(output_tensors[0].shape)
                    except Exception:
                        pass

                if out is None:
                    try:
                        in_shapes_raw = [x.shape for x in ins]
                        if len(in_shapes_raw) == 1:
                            computed_shape = layer.compute_output_shape(
                                in_shapes_raw[0]
                            )
                        else:
                            computed_shape = layer.compute_output_shape(in_shapes_raw)
                        out = ShapeWrapper(computed_shape)
                    except Exception:
                        pass

                # MultiHeadAttention often fails compute_output_shape if inputs aren't standard.
                # Since the FLOP handler only uses inputs[0] (Query) to calculate FLOPs,
                # we can use the input shape as a proxy for the output shape to avoid skipping.
                if (
                    out is None
                    and isinstance(layer, keras.layers.MultiHeadAttention)
                    and ins
                ):
                    out = ShapeWrapper(ins[0].shape)
                # -------------------------

                if out is None:
                    skipped_ops[layer_name] += 1
                    continue

                flops = handler(layer, ins, out)
                total_flop_counter += flops

            except Exception as e:
                skipped_ops[layer_name] += 1
        else:
            if "Input" not in layer_name and "Flatten" not in layer_name:
                skipped_ops[layer_name] += 1

    final_count = defaultdict(float)
    for op in total_flop_counter:
        final_count[op] = total_flop_counter[op] / 1e9

    return final_count


def warmup(model, inputs, N=10):
    for i in range(N):
        out = model(inputs)
        sync_device(out)


def measure_time(model, inputs, N=10):
    sync_device(inputs)
    s = time.time()
    for i in range(N):
        out = model(inputs)
        sync_device(out)
    e = time.time()
    t = (e - s) / N
    return t


def fmt_res(data):
    if len(data) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(data.mean()),
        "std": float(data.std()),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def benchmark(model, dataset, output_dir):
    print("Get model size, FLOPs, and FPS")
    _outputs = {}

    n_parameters = model.count_params()
    _outputs.update({"nparam": n_parameters})

    images = []
    iter_limit = 20
    try:
        iterator = iter(dataset)
        for _ in range(iter_limit):
            batch = next(iterator)
            if isinstance(batch, (list, tuple)):
                images.append(batch[0])
            else:
                images.append(batch)
    except Exception:
        for idx in range(min(iter_limit, len(dataset))):
            batch = dataset[idx]
            if isinstance(batch, (list, tuple)):
                images.append(batch[0])
            else:
                images.append(batch)

    if not images:
        print("Warning: Dataset empty or format not recognized.")
        return {}

    warmup_step = 5
    first_input = images[0]
    if hasattr(first_input, "ndim") and first_input.ndim == 3:
        first_input = np.expand_dims(first_input, 0)
    elif hasattr(first_input, "shape") and len(first_input.shape) == 3:
        first_input = np.expand_dims(first_input, 0)

    warmup(model, first_input, N=5)

    flops_list = []
    times_list = []
    detailed_flops = {}

    for imgid, img in enumerate(tqdm.tqdm(images)):
        if hasattr(img, "ndim") and img.ndim == 3:
            inputs = np.expand_dims(img, axis=0)
        elif hasattr(img, "shape") and len(img.shape) == 3:
            inputs = np.expand_dims(img, axis=0)
        else:
            inputs = img

        inputs = keras.ops.convert_to_tensor(inputs)

        res = flop_count(model, (inputs,))
        total_flops = sum(res.values())
        flops_list.append(total_flops)

        t = measure_time(model, inputs, N=10)

        if imgid >= warmup_step:
            times_list.append(t)
            detailed_flops = res

    _outputs.update({"detailed_flops": dict(detailed_flops)})
    flops_arr = np.array(flops_list)
    times_arr = np.array(times_list)
    _outputs.update({"flops": fmt_res(flops_arr), "time": fmt_res(times_arr)})

    mean_infer_time = _outputs["time"]["mean"]
    fps = 1.0 / mean_infer_time if mean_infer_time > 0 else 0.0
    _outputs.update({"fps": fps})

    output_dir = Path(output_dir)
    log_dir = output_dir / "flops"
    log_dir.mkdir(parents=True, exist_ok=True)

    with (log_dir / "log.txt").open("a") as f:
        f.write("Test benchmark on Val Dataset" + "\n")
        f.write(json.dumps(_outputs, indent=2) + "\n")

    return _outputs
