import keras
from keras import ops, initializers
from keras.layers import Dense

NORM_EPSILON = 1e-8
TARGET_NAMES = {
    "q_proj", "v_proj", "k_proj",        # OWL-ViT style attention
    "qkv",                                # SigLIP2 style fused QKV
    "query", "key", "value",              # DINOv2 windowed attention
}


def column_norms(kernel):
    return ops.sqrt(ops.sum(ops.square(kernel), axis=0) + NORM_EPSILON)


# Kept as a Dense subclass: it owns trainable weights through add_weight and
# is hot-swapped for Dense by isinstance checks in replace_dense_layers, both
# of which a builder function cannot provide.
class LoRADense(Dense):
    def __init__(self, units, rank=16, lora_alpha=16, use_dora=False, original_layer=None, **kwargs):  # fmt: skip
        if original_layer is not None:
            kwargs = carry_original_settings(kwargs, original_layer)
        super().__init__(units, **kwargs)
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        self.use_dora = use_dora
        self._original_layer = original_layer

    def build(self, input_shape):
        super().build(input_shape)
        freeze_original_weights(self)
        copy_original_weights(self)
        add_low_rank_factors(self, int(input_shape[-1]))
        if self.use_dora:
            add_dora_magnitude(self)

    def call(self, inputs):
        merged = merge_lora_kernel(self)
        output = ops.matmul(inputs, merged)
        if self.bias is not None:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

    def merge_weights(self):
        self.kernel.assign(merge_lora_kernel(self))
        # Zero the factors so later forward passes reproduce the merge.
        self.lora_a.assign(ops.zeros_like(self.lora_a))
        self.lora_b.assign(ops.zeros_like(self.lora_b))
        if self.use_dora:
            self.magnitude.assign(column_norms(self.kernel))

    def get_config(self):
        config = super().get_config()
        keys = ("rank", "lora_alpha", "use_dora")
        values = (self.rank, self.lora_alpha, self.use_dora)
        config.update(dict(zip(keys, values)))
        return config


def carry_original_settings(kwargs, original_layer):
    kwargs.setdefault("use_bias", original_layer.use_bias)
    kwargs.setdefault("name", original_layer.name)
    if hasattr(original_layer, "kernel_initializer"):
        kwargs.setdefault("kernel_initializer", original_layer.kernel_initializer)  # fmt: skip
    return kwargs


def freeze_original_weights(layer):
    layer.kernel.trainable = False
    if layer.bias is not None:
        layer.bias.trainable = False


def copy_original_weights(layer):
    original = layer._original_layer
    if original is not None:
        layer.kernel.assign(original.kernel)
        if layer.bias is not None and original.bias is not None:
            layer.bias.assign(original.bias)


def add_low_rank_factors(layer, in_features):
    # Fan-in variance scaling matches standard LoRA init (kaiming_uniform
    # with a=sqrt(5)): uniform(-1/sqrt(fan_in), +1/sqrt(fan_in)).
    keys = ("scale", "mode", "distribution")
    values = (1.0, "fan_in", "uniform")
    scaling = initializers.VarianceScaling(**dict(zip(keys, values)))
    shape = (in_features, layer.rank)
    layer.lora_a = layer.add_weight(name="lora_a", shape=shape, initializer=scaling, trainable=True)  # fmt: skip
    shape = (layer.rank, layer.units)
    layer.lora_b = layer.add_weight(name="lora_b", shape=shape, initializer=initializers.Zeros(), trainable=True)  # fmt: skip


def add_dora_magnitude(layer):
    # Constant() takes the tensor directly, so the norms stay on device and
    # building a LoRA layer costs no host sync.
    initializer = initializers.Constant(column_norms(layer.kernel))
    shape = (layer.units,)
    layer.magnitude = layer.add_weight(name="magnitude", shape=shape, initializer=initializer, trainable=True)  # fmt: skip


def merge_lora_kernel(layer):
    delta = ops.matmul(layer.lora_a, layer.lora_b) * layer.scaling
    merged = layer.kernel + delta
    if layer.use_dora:
        # Normalise columns and rescale by the learned magnitude.
        merged = merged / column_norms(merged) * layer.magnitude
    return merged


def apply_lora_to_backbone(model, rank=16, lora_alpha=16, use_dora=True, target_names=None):  # fmt: skip
    target_names = target_names or TARGET_NAMES
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise ValueError("Model does not have a 'backbone' attribute.")
    encoder = resolve_encoder(backbone)
    if encoder is None:
        raise ValueError("Backbone does not have an 'encoder' attribute.")
    for weight in encoder.weights:
        weight._trainable = False
    replace_dense_layers(encoder, target_names, rank, lora_alpha, use_dora)
    return model


def resolve_encoder(backbone):
    encoder = None
    try:
        encoder = backbone.get_layer("backbone").get_layer("encoder")
    except (ValueError, AttributeError):
        encoder = None
    return encoder


def read_child(layer, attribute_name):
    child = None
    try:
        child = getattr(layer, attribute_name)
    except Exception:
        child = None
    return child


def build_lora_replacement(child, rank, lora_alpha, use_dora):
    keys = ("units", "rank", "lora_alpha", "use_dora", "original_layer", "use_bias", "name")  # fmt: skip
    values = (child.units, rank, lora_alpha, use_dora, child, child.use_bias, child.name)  # fmt: skip
    replacement = LoRADense(**dict(zip(keys, values)))
    if child.kernel is not None:
        replacement.build((None, child.kernel.shape[0]))
    return replacement


def replace_dense_child(layer, attribute_name, child, target_names, rank, lora_alpha, use_dora):  # fmt: skip
    if isinstance(child, Dense) and child.name in target_names:
        args = (child, rank, lora_alpha, use_dora)
        setattr(layer, attribute_name, build_lora_replacement(*args))
    elif isinstance(child, keras.layers.Layer):
        replace_dense_layers(child, target_names, rank, lora_alpha, use_dora)


def replace_dense_layers(layer, target_names, rank, lora_alpha, use_dora):
    for attribute_name in dir(layer):
        child = None
        if not attribute_name.startswith("_"):
            child = read_child(layer, attribute_name)
        if child is not None and not isinstance(child, LoRADense):
            args = (layer, attribute_name, child, target_names)
            replace_dense_child(*args, rank, lora_alpha, use_dora)


def merge_lora_weights(model):
    for layer in iter_all_layers(model):
        if isinstance(layer, LoRADense):
            layer.merge_weights()


def iter_child_layers(layer):
    if hasattr(layer, "_flatten_layers"):
        children = [c for c in layer._flatten_layers() if c is not layer]
    elif hasattr(layer, "layers"):
        children = layer.layers
    elif hasattr(layer, "_layers"):
        children = layer._layers
    else:
        children = []
    return children


def iter_all_layers(layer):
    yield layer
    flattened = hasattr(layer, "_flatten_layers")
    for child in iter_child_layers(layer):
        if flattened:
            yield child
        else:
            yield from iter_all_layers(child)
