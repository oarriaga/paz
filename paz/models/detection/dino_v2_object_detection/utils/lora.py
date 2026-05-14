"""Low-Rank Adaptation (LoRA/DoRA) for Dense layers.

This module provides a standalone Keras implementation of LoRA and DoRA
that can be applied to any ``Dense`` layer in a model without depending
on external libraries.

Reference:
    Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*
    Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation*
"""

import keras
from keras import ops, initializers
from keras.layers import Dense


class LoRADense(Dense):
    """Drop-in replacement for :class:`keras.layers.Dense` with LoRA.

    Keeps the original ``kernel`` frozen and learns two low-rank matrices
    ``lora_a`` (down-projection) and ``lora_b`` (up-projection) such that
    the effective weight becomes ``kernel + (lora_b @ lora_a) * scaling``.

    When ``use_dora=True``, the layer additionally normalises the merged
    weight column-wise and rescales it with a learned magnitude vector
    (DoRA).

    Args:
        units: Output dimensionality.
        rank: Rank of the low-rank decomposition.
        lora_alpha: Scaling factor (``scaling = lora_alpha / rank``).
        use_dora: Enable Weight-Decomposed Low-Rank Adaptation (DoRA).
        original_layer: An existing ``Dense`` layer whose weights are
            copied into this layer and frozen.
        **kwargs: Forwarded to :class:`keras.layers.Dense`.
    """

    def __init__(
        self,
        units,
        rank=16,
        lora_alpha=16,
        use_dora=False,
        original_layer=None,
        **kwargs,
    ):
        # Carry over settings from the original layer when provided
        if original_layer is not None:
            kwargs.setdefault("use_bias", original_layer.use_bias)
            kwargs.setdefault("name", original_layer.name)
            if hasattr(original_layer, "kernel_initializer"):
                kwargs.setdefault(
                    "kernel_initializer",
                    original_layer.kernel_initializer,
                )
        super().__init__(units, **kwargs)
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        self.use_dora = use_dora
        self._original_layer = original_layer

    def build(self, input_shape):
        super().build(input_shape)
        in_features = int(input_shape[-1])

        # Freeze original kernel (and bias)
        self.kernel.trainable = False
        if self.bias is not None:
            self.bias.trainable = False

        # Copy weights from the original layer if available
        if self._original_layer is not None:
            self.kernel.assign(self._original_layer.kernel)
            if self.bias is not None and self._original_layer.bias is not None:
                self.bias.assign(self._original_layer.bias)

        # Low-rank factors
        # Use fan-in variance scaling (equivalent to kaiming_uniform with
        # a=sqrt(5)) to match standard LoRA initialisation.  This gives
        # uniform(-1/sqrt(fan_in), +1/sqrt(fan_in)).
        self.lora_a = self.add_weight(
            name="lora_a",
            shape=(in_features, self.rank),
            initializer=initializers.VarianceScaling(
                scale=1.0, mode="fan_in", distribution="uniform",
            ),
            trainable=True,
        )
        self.lora_b = self.add_weight(
            name="lora_b",
            shape=(self.rank, self.units),
            initializer=initializers.Zeros(),
            trainable=True,
        )

        # DoRA magnitude vector
        if self.use_dora:
            col_norms = ops.sqrt(
                ops.sum(ops.square(self.kernel), axis=0, keepdims=False) + 1e-8
            )
            self.magnitude = self.add_weight(
                name="magnitude",
                shape=(self.units,),
                initializer=initializers.Constant(
                    ops.convert_to_numpy(col_norms)
                ),
                trainable=True,
            )

    def call(self, inputs):
        # Merged weight: W + B @ A * scaling
        delta = ops.matmul(self.lora_a, self.lora_b) * self.scaling
        merged = self.kernel + delta

        if self.use_dora:
            # Normalise columns and rescale by learned magnitude
            col_norms = ops.sqrt(
                ops.sum(ops.square(merged), axis=0, keepdims=False) + 1e-8
            )
            merged = merged / col_norms * self.magnitude

        output = ops.matmul(inputs, merged)
        if self.bias is not None:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

    def merge_weights(self):
        """Fold LoRA weights into the base kernel (in-place).

        After calling this method the layer behaves like a plain
        ``Dense`` layer with the adapted weights.
        """
        delta = ops.matmul(self.lora_a, self.lora_b) * self.scaling
        merged = self.kernel + delta
        if self.use_dora:
            col_norms = ops.sqrt(
                ops.sum(ops.square(merged), axis=0, keepdims=False) + 1e-8
            )
            merged = merged / col_norms * self.magnitude
        self.kernel.assign(merged)
        # Zero-out LoRA factors so subsequent forward passes are identity
        self.lora_a.assign(ops.zeros_like(self.lora_a))
        self.lora_b.assign(ops.zeros_like(self.lora_b))
        if self.use_dora:
            col_norms = ops.sqrt(
                ops.sum(ops.square(self.kernel), axis=0, keepdims=False) + 1e-8
            )
            self.magnitude.assign(col_norms)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rank": self.rank,
                "lora_alpha": self.lora_alpha,
                "use_dora": self.use_dora,
            }
        )
        return config


# -- Model surgery helpers ---------------------------------------------------

_TARGET_NAMES = {
    "q_proj", "v_proj", "k_proj",        # OWL-ViT style attention
    "qkv",                                # SigLIP2 style fused QKV
    "query", "key", "value",              # DINOv2 windowed attention
}


def apply_lora_to_backbone(model, rank=16, lora_alpha=16, use_dora=True,
                           target_names=None):
    """Replace Dense layers in the backbone encoder with LoRA variants.

    Walks the model's backbone encoder and swaps each Dense layer whose
    ``name`` matches one of *target_names* with a :class:`LoRADense`
    layer.  All non-LoRA parameters in the encoder are frozen.

    Args:
        model: The full detection model (must have a ``.backbone`` attribute).
        rank: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        use_dora: Whether to use DoRA.
        target_names: Set of Dense layer names to target.  Defaults to
            ``{"qkv", "proj", "query", "key", "value"}``.

    Returns:
        The model (modified in-place).
    """
    if target_names is None:
        target_names = _TARGET_NAMES

    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise ValueError("Model does not have a 'backbone' attribute.")

    encoder = getattr(backbone, "encoder", None)
    if encoder is None:
        raise ValueError("Backbone does not have an 'encoder' attribute.")

    # Freeze all encoder weights first
    for w in encoder.weights:
        w._trainable = False

    # Walk the encoder's layer tree and replace matching Dense layers
    _replace_dense_layers(encoder, target_names, rank, lora_alpha, use_dora)

    return model


def _replace_dense_layers(layer, target_names, rank, lora_alpha, use_dora):
    """Recursively replace matching Dense layers with LoRADense."""
    # Check direct attributes that hold sub-layers
    for attr_name in dir(layer):
        if attr_name.startswith("_"):
            continue
        try:
            child = getattr(layer, attr_name)
        except Exception:
            continue

        if isinstance(child, LoRADense):
            continue  # Already replaced

        if isinstance(child, Dense) and child.name in target_names:
            # Build the LoRADense replacement
            lora_layer = LoRADense(
                units=child.units,
                rank=rank,
                lora_alpha=lora_alpha,
                use_dora=use_dora,
                original_layer=child,
                use_bias=child.use_bias,
                name=child.name,
            )
            # Build with the same input shape
            if child.kernel is not None:
                input_shape = (None, child.kernel.shape[0])
                lora_layer.build(input_shape)
            setattr(layer, attr_name, lora_layer)

        elif hasattr(child, "weights") and hasattr(child, "name"):
            # Recurse into sub-layers (but not into non-layer objects)
            if isinstance(child, keras.layers.Layer):
                _replace_dense_layers(
                    child, target_names, rank, lora_alpha, use_dora
                )


def merge_lora_weights(model):
    """Fold all LoRA weights in the model back into their base kernels.

    After calling this function, the model behaves identically but no
    longer contains separate LoRA parameters.  This is typically done
    before exporting or saving the final model.

    Args:
        model: The detection model containing LoRA layers.
    """
    for layer in _iter_all_layers(model):
        if isinstance(layer, LoRADense):
            layer.merge_weights()


def _iter_all_layers(layer):
    """Yield all nested Keras layers (including *layer* itself)."""
    yield layer
    if hasattr(layer, "_flatten_layers"):
        for child in layer._flatten_layers():
            if child is not layer:
                yield child
    elif hasattr(layer, "layers"):
        for child in layer.layers:
            yield from _iter_all_layers(child)
    elif hasattr(layer, "_layers"):
        for child in layer._layers:
            yield from _iter_all_layers(child)
