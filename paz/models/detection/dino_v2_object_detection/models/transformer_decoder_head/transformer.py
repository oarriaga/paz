import math
import copy
import numpy as np
import keras
from keras import layers
from keras import ops

from paz.models.detection.dino_v2_object_detection.models.transformer_decoder_head.ms_deform_attn import (
    MSDeformAttn,
)


@keras.saving.register_keras_serializable(package="RFDETR")
class MLP(layers.Layer):
    """Multi-layer perceptron (feed-forward network).

    Applies a sequence of Dense layers with ReLU activations between
    all layers except the last.

    Attributes:
        num_layers (int): Total number of dense layers.
        layers_list (list): List of Dense layers.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        h = [hidden_dim] * (num_layers - 1)

        self.layers_list = []
        dims_in = [input_dim] + h
        dims_out = h + [output_dim]

        for i, (n, k) in enumerate(zip(dims_in, dims_out)):
            self.layers_list.append(layers.Dense(k, name=f"dense_{i}"))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self._input_dim,
                "hidden_dim": self._hidden_dim,
                "output_dim": self._output_dim,
                "num_layers": self.num_layers,
            }
        )
        return config

    def call(self, x):
        for i, layer in enumerate(self.layers_list):
            x = layer(x)
            if i < self.num_layers - 1:
                x = ops.relu(x)
        return x


def gen_sineembed_for_position(pos_tensor, dim=128):
    """Generate sinusoidal positional embeddings from coordinate tensors.

    Encodes each coordinate dimension (x, y and optionally w, h) into
    sin/cos pairs using a frequency spectrum, producing a fixed-length
    embedding vector for each query position.

    Args:
        pos_tensor (tensor): Position coordinates of shape
            (batch, num_queries, 2) for (x, y) or
            (batch, num_queries, 4) for (x, y, w, h).
            Coordinates should be normalized to [0, 1].
        dim (int): Half the output embedding dimension per coordinate
            pair. Total output dim is 2*dim for 2-D input, 4*dim for 4-D.

    Returns:
        tensor: Sinusoidal position embeddings of shape
            (batch, num_queries, 2*dim) or (batch, num_queries, 4*dim).
    """
    scale = 2 * math.pi

    # Build the frequency spectrum: 10000^(2i/dim) for dimension index i
    dim_t = ops.arange(dim, dtype=pos_tensor.dtype)
    exponent = 2 * ops.floor(dim_t / 2) / dim
    dim_t = 10000**exponent

    # Scale coordinates to [0, 2*pi] and divide by frequencies
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale

    pos_x = ops.expand_dims(x_embed, axis=-1) / dim_t
    pos_y = ops.expand_dims(y_embed, axis=-1) / dim_t

    # Interleave sin and cos: even indices get sin, odd indices get cos
    pos_x_sin = ops.sin(pos_x[:, :, 0::2])
    pos_x_cos = ops.cos(pos_x[:, :, 1::2])
    pos_x_stacked = ops.stack([pos_x_sin, pos_x_cos], axis=-1)
    pos_x_flattened = ops.reshape(
        pos_x_stacked, (ops.shape(pos_x)[0], ops.shape(pos_x)[1], -1)
    )

    pos_y_sin = ops.sin(pos_y[:, :, 0::2])
    pos_y_cos = ops.cos(pos_y[:, :, 1::2])
    pos_y_stacked = ops.stack([pos_y_sin, pos_y_cos], axis=-1)
    pos_y_flattened = ops.reshape(
        pos_y_stacked, (ops.shape(pos_y)[0], ops.shape(pos_y)[1], -1)
    )

    if ops.shape(pos_tensor)[-1] == 2:
        # 2-D positions: concatenate y and x embeddings
        pos = ops.concatenate([pos_y_flattened, pos_x_flattened], axis=2)
    elif ops.shape(pos_tensor)[-1] == 4:
        # 4-D positions: also encode width and height
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = ops.expand_dims(w_embed, axis=-1) / dim_t
        pos_w_sin = ops.sin(pos_w[:, :, 0::2])
        pos_w_cos = ops.cos(pos_w[:, :, 1::2])
        pos_w = ops.reshape(
            ops.stack([pos_w_sin, pos_w_cos], axis=-1),
            (ops.shape(pos_w)[0], ops.shape(pos_w)[1], -1),
        )

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = ops.expand_dims(h_embed, axis=-1) / dim_t
        pos_h_sin = ops.sin(pos_h[:, :, 0::2])
        pos_h_cos = ops.cos(pos_h[:, :, 1::2])
        pos_h = ops.reshape(
            ops.stack([pos_h_sin, pos_h_cos], axis=-1),
            (ops.shape(pos_h)[0], ops.shape(pos_h)[1], -1),
        )

        pos = ops.concatenate([pos_y_flattened, pos_x_flattened, pos_w, pos_h], axis=2)
    else:
        # Default to 2-D encoding for unrecognized coordinate dimensions
        pos = ops.concatenate([pos_y_flattened, pos_x_flattened], axis=2)

    return pos


def gen_encoder_output_proposals(
    memory, memory_padding_mask, spatial_shapes, unsigmoid=True
):
    """Generate initial object proposals from encoder memory features.

    Creates a uniform grid of anchor-like proposals for each spatial level,
    with center coordinates normalized by the valid (non-padded) region.
    Optionally applies inverse-sigmoid to convert proposals to logit space.

    Args:
        memory (tensor): Encoder output features, shape (N, S, C).
        memory_padding_mask (tensor): Boolean mask of shape (N, S) where
            True indicates padded positions, or None if no padding.
        spatial_shapes (list): List of (H, W) tuples for each feature level.
        unsigmoid (bool): If True, convert proposals to inverse-sigmoid
            (logit) space. If False, keep in normalized coordinate space.

    Returns:
        tuple: (output_memory, output_proposals) where:
            - output_memory: Filtered memory with padded/invalid positions
              zeroed, shape (N, S, C).
            - output_proposals: Proposal boxes of shape (N, S, 4), either
              in logit space or normalized coordinates depending on
              unsigmoid flag.
    """
    N_ = ops.shape(memory)[0]
    S_ = ops.shape(memory)[1]
    C_ = ops.shape(memory)[2]

    proposals = []
    _cur = 0

    num_levels = (
        len(spatial_shapes)
        if isinstance(spatial_shapes, list)
        else ops.shape(spatial_shapes)[0]
    )

    for lvl in range(num_levels):
        if isinstance(spatial_shapes, list):
            H_, W_ = spatial_shapes[lvl]
        else:
            H_ = spatial_shapes[lvl][0]
            W_ = spatial_shapes[lvl][1]

        # Compute the valid (non-padded) height and width for normalization
        if memory_padding_mask is not None:
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)]
            mask_flatten_ = ops.reshape(mask_flatten_, (N_, H_, W_, 1))
            not_mask = ops.logical_not(ops.cast(mask_flatten_, "bool"))

            valid_H = ops.sum(ops.cast(not_mask[:, :, 0, 0], "float32"), axis=1)
            valid_W = ops.sum(ops.cast(not_mask[:, 0, :, 0], "float32"), axis=1)
        else:
            valid_H = ops.full((N_,), ops.cast(H_, "float32"))
            valid_W = ops.full((N_,), ops.cast(W_, "float32"))

        # Create a uniform grid of proposal centers normalized to [0, 1]
        # by the valid spatial extent. The +0.5 centers each pixel.
        grid_y = ops.linspace(0.0, ops.cast(H_ - 1, "float32"), int(H_))
        grid_x = ops.linspace(0.0, ops.cast(W_ - 1, "float32"), int(W_))
        grid_y_mesh, grid_x_mesh = ops.meshgrid(grid_y, grid_x, indexing="ij")

        grid = ops.stack([grid_x_mesh, grid_y_mesh], axis=-1)
        scale = ops.stack([valid_W, valid_H], axis=1)
        scale = ops.reshape(scale, (N_, 1, 1, 2))

        grid_expanded = ops.expand_dims(grid, axis=0)
        grid_final = (grid_expanded + 0.5) / scale

        # Assign default width/height that grows with the feature level,
        # providing coarser anchors at lower-resolution levels
        wh = ops.ones_like(grid_final) * 0.05 * (2.0**lvl)
        proposal = ops.concatenate([grid_final, wh], axis=-1)
        proposal = ops.reshape(proposal, (N_, -1, 4))

        proposals.append(proposal)
        _cur += H_ * W_

    output_proposals = ops.concatenate(proposals, axis=1)

    # Mark proposals as valid only if all 4 coordinates lie in (0.01, 0.99),
    # filtering out proposals near image borders or in padded regions
    valid_mask = ops.logical_and(output_proposals > 0.01, output_proposals < 0.99)
    output_proposals_valid = ops.all(valid_mask, axis=-1, keepdims=True)

    if unsigmoid:
        # Convert to logit space via inverse-sigmoid; invalid positions get inf
        output_proposals_logit = ops.log(output_proposals / (1 - output_proposals))
        if memory_padding_mask is not None:
            mask_exp = ops.expand_dims(memory_padding_mask, axis=-1)
            output_proposals_logit = ops.where(
                mask_exp, float("inf"), output_proposals_logit
            )

        output_proposals_logit = ops.where(
            ops.logical_not(output_proposals_valid),
            float("inf"),
            output_proposals_logit,
        )
        output_proposals = output_proposals_logit
    else:
        # Keep in coordinate space; zero out invalid positions
        if memory_padding_mask is not None:
            mask_exp = ops.expand_dims(memory_padding_mask, axis=-1)
            output_proposals = ops.where(mask_exp, 0.0, output_proposals)
        output_proposals = ops.where(
            ops.logical_not(output_proposals_valid), 0.0, output_proposals
        )

    # Zero out memory features at padded or invalid proposal positions
    output_memory = memory
    if memory_padding_mask is not None:
        mask_exp = ops.expand_dims(memory_padding_mask, axis=-1)
        output_memory = ops.where(mask_exp, 0.0, output_memory)

    output_memory = ops.where(
        ops.logical_not(output_proposals_valid), 0.0, output_memory
    )

    output_memory = ops.cast(output_memory, memory.dtype)
    output_proposals = ops.cast(output_proposals, memory.dtype)

    return output_memory, output_proposals


@keras.saving.register_keras_serializable(package="RFDETR")
class TransformerDecoderLayer(layers.Layer):
    """Single transformer decoder layer with self-attention, deformable
    cross-attention, and a feed-forward network.

    Follows the standard DETR decoder layer pattern:
    1. Self-attention among object queries.
    2. Multi-scale deformable cross-attention over encoder features.
    3. Feed-forward network.
    Each sub-layer is followed by dropout and layer normalization.

    Attributes:
        d_model (int): Model embedding dimension.
        dropout_rate (float): Dropout probability.
        activation_name (str): Name of the activation function.
        normalize_before (bool): Whether to apply pre-norm instead of
            post-norm.
        group_detr (int): Number of groups for grouped DETR training.
            Queries are split into groups during training for efficiency.
        skip_self_attn (bool): If True, skip the self-attention sub-layer.
    """
    def __init__(
        self,
        d_model,
        sa_nhead,
        ca_nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        group_detr=1,
        num_feature_levels=4,
        dec_n_points=4,
        skip_self_attn=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout
        self.activation_name = activation
        self.normalize_before = normalize_before
        self.group_detr = group_detr
        self.skip_self_attn = skip_self_attn
        self._sa_nhead = sa_nhead
        self._ca_nhead = ca_nhead
        self._dim_feedforward = dim_feedforward
        self._num_feature_levels = num_feature_levels
        self._dec_n_points = dec_n_points

        # Self-attention: standard multi-head attention among queries
        self.self_attn = layers.MultiHeadAttention(
            num_heads=sa_nhead, key_dim=d_model // sa_nhead, dropout=dropout
        )
        self.dropout1 = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)

        # Cross-attention: multi-scale deformable attention over encoder features
        self.cross_attn = MSDeformAttn(
            d_model=d_model,
            n_levels=num_feature_levels,
            n_heads=ca_nhead,
            n_points=dec_n_points,
        )
        self.dropout2 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        # Feed-forward network: two-layer MLP with activation
        self.linear1 = layers.Dense(dim_feedforward)
        self.dropout = layers.Dropout(dropout)
        self.linear2 = layers.Dense(d_model)
        self.dropout3 = layers.Dropout(dropout)
        self.norm3 = layers.LayerNormalization(epsilon=1e-5)

        if activation == "relu":
            self.activation = keras.activations.relu
        elif activation == "gelu":
            self.activation = keras.activations.gelu
        elif activation == "glu":
            self.activation = keras.activations.glu
        else:
            self.activation = keras.activations.get(activation)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "sa_nhead": self._sa_nhead,
                "ca_nhead": self._ca_nhead,
                "dim_feedforward": self._dim_feedforward,
                "dropout": self.dropout_rate,
                "activation": self.activation_name,
                "normalize_before": self.normalize_before,
                "group_detr": self.group_detr,
                "num_feature_levels": self._num_feature_levels,
                "dec_n_points": self._dec_n_points,
                "skip_self_attn": self.skip_self_attn,
            }
        )
        return config

    def with_pos_embed(self, tensor, pos):
        """Add positional embedding to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def call(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
        query_sine_embed=None,
        is_first=False,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        training=None,
    ):
        """Forward pass for a single decoder layer.

        Args:
            tgt (tensor): Query features, shape (B, num_queries, d_model).
            memory (tensor): Encoder features, shape (B, Len_memory, d_model).
            tgt_mask (tensor): Attention mask for self-attention.
            memory_mask (tensor): Attention mask for cross-attention.
            tgt_key_padding_mask (tensor): Padding mask for queries.
            memory_key_padding_mask (tensor): Padding mask for encoder
                features (True = padded).
            pos (tensor): Positional embedding for encoder features.
            query_pos (tensor): Positional embedding for queries.
            query_sine_embed (tensor): Sinusoidal embedding of query
                reference points.
            is_first (bool): Whether this is the first decoder layer.
            reference_points (tensor): Reference point coordinates for
                deformable cross-attention.
            spatial_shapes (array): Spatial shapes of multi-scale features.
            level_start_index (tensor): Start indices per feature level.
            training (bool): Training mode flag.

        Returns:
            tensor: Updated query features, shape (B, num_queries, d_model).
        """
        # Self-attention: queries attend to each other
        q = k = self.with_pos_embed(tgt, query_pos)
        v = tgt

        bs = ops.shape(tgt)[0]
        num_queries = ops.shape(tgt)[1]

        # During grouped DETR training, split queries into groups so each
        # group performs self-attention independently. This reduces memory
        # and enables parallel group-wise training.
        if training and self.group_detr > 1:
            q = ops.reshape(
                q, (bs * self.group_detr, num_queries // self.group_detr, self.d_model)
            )
            k = ops.reshape(
                k, (bs * self.group_detr, num_queries // self.group_detr, self.d_model)
            )
            v = ops.reshape(
                v, (bs * self.group_detr, num_queries // self.group_detr, self.d_model)
            )

        tgt2 = self.self_attn(
            query=q, value=v, key=k, attention_mask=tgt_mask, training=training
        )

        # Merge groups back after self-attention
        if training and self.group_detr > 1:
            tgt2 = ops.reshape(tgt2, (bs, num_queries, self.d_model))

        # Residual connection + layer norm for self-attention
        tgt = tgt + self.dropout1(tgt2, training=training)
        tgt = self.norm1(tgt)

        # Cross-attention: queries attend to multi-scale encoder features
        # using deformable attention at learned sampling locations
        query_cross = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.cross_attn(
            query=query_cross,
            reference_points=reference_points,
            input_flatten=memory,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
            input_padding_mask=memory_key_padding_mask,
            training=training,
        )

        # Residual connection + layer norm for cross-attention
        tgt = tgt + self.dropout2(tgt2, training=training)
        tgt = self.norm2(tgt)

        # Feed-forward network with residual connection
        tgt2 = self.linear2(
            self.dropout(self.activation(self.linear1(tgt)), training=training)
        )
        tgt = tgt + self.dropout3(tgt2, training=training)
        tgt = self.norm3(tgt)

        return tgt


def _get_clones(module_class, N, **kwargs):
    """Create N independent instances of a layer class."""
    return [module_class(**kwargs) for _ in range(N)]


@keras.saving.register_keras_serializable(package="RFDETR")
class TransformerDecoder(layers.Layer):
    """Stack of transformer decoder layers with iterative reference point
    refinement.

    Runs the query features through multiple decoder layers, optionally
    refining the reference points (bounding box proposals) between layers
    using a bbox regression head. Supports both standard and
    lite (single-computation) reference point refinement modes.

    Attributes:
        d_model (int): Model embedding dimension.
        num_layers (int): Number of stacked decoder layers.
        return_intermediate (bool): If True, return outputs from all layers
            (used for auxiliary loss during training).
        lite_refpoint_refine (bool): If True, compute reference-derived
            query position embeddings once before the layer loop rather
            than recomputing at each layer.
        bbox_reparam (bool): If True, use reparameterized bbox encoding
            where center offsets are relative to reference box size and
            width/height use log-space deltas.
        bbox_embed (MLP): Optional bbox regression head for iterative
            reference point refinement between layers.
    """
    def __init__(
        self,
        d_model,
        num_layers,
        return_intermediate=False,
        lite_refpoint_refine=False,
        bbox_reparam=False,
        decoder_layer_kwargs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.lite_refpoint_refine = lite_refpoint_refine
        self.bbox_reparam = bbox_reparam
        self._decoder_layer_kwargs = decoder_layer_kwargs

        if decoder_layer_kwargs is None:
            decoder_layer_kwargs = {}

        self.layers_list = [
            TransformerDecoderLayer(**decoder_layer_kwargs, name=f"layer_{i}")
            for i in range(num_layers)
        ]

        self.norm = layers.LayerNormalization(epsilon=1e-5)

        # MLP that converts sinusoidal reference point embeddings to query
        # position embeddings. Input dim is 2*d_model because it receives
        # the concatenated sin/cos embeddings of all coordinates.
        self.ref_point_head = MLP(
            input_dim=2 * d_model, hidden_dim=d_model, output_dim=d_model, num_layers=2
        )

        # Optional bbox regression head, assigned externally when iterative
        # refinement is used
        self.bbox_embed = None

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_layers": self.num_layers,
                "return_intermediate": self.return_intermediate,
                "lite_refpoint_refine": self.lite_refpoint_refine,
                "bbox_reparam": self.bbox_reparam,
                "decoder_layer_kwargs": self._decoder_layer_kwargs,
            }
        )
        return config

    def refpoints_refine(self, refpoints_unsigmoid, new_refpoints_delta):
        """Apply predicted deltas to refine reference points.

        Two modes depending on bbox_reparam:
        - Reparameterized: center deltas are scaled by reference box size,
          width/height deltas are exponentiated and multiplied by reference
          size. This stabilizes training for large/small objects.
        - Standard: deltas are directly added to reference coordinates.

        Args:
            refpoints_unsigmoid (tensor): Current reference points in
                inverse-sigmoid (logit) or raw coordinate space.
            new_refpoints_delta (tensor): Predicted refinement deltas.

        Returns:
            tensor: Refined reference points.
        """
        if self.bbox_reparam:
            new_refpoints_cxcy = (
                new_refpoints_delta[..., :2] * refpoints_unsigmoid[..., 2:]
                + refpoints_unsigmoid[..., :2]
            )
            new_refpoints_wh = (
                ops.exp(new_refpoints_delta[..., 2:]) * refpoints_unsigmoid[..., 2:]
            )
            new_refpoints_unsigmoid = ops.concatenate(
                [new_refpoints_cxcy, new_refpoints_wh], axis=-1
            )
        else:
            new_refpoints_unsigmoid = refpoints_unsigmoid + new_refpoints_delta
        return new_refpoints_unsigmoid

    def call(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        refpoints_unsigmoid=None,
        level_start_index=None,
        spatial_shapes=None,
        valid_ratios=None,
        training=None,
    ):
        """Run query features through the decoder layer stack.

        Args:
            tgt (tensor): Initial query features, shape (B, N_q, d_model).
            memory (tensor): Encoder features, shape (B, S, d_model).
            tgt_mask (tensor): Self-attention mask for queries.
            memory_mask (tensor): Cross-attention mask.
            tgt_key_padding_mask (tensor): Query padding mask.
            memory_key_padding_mask (tensor): Encoder padding mask.
            pos (tensor): Encoder positional embeddings.
            refpoints_unsigmoid (tensor): Initial reference point
                coordinates in logit space, shape (B, N_q, 4).
            level_start_index (tensor): Start indices for each feature level.
            spatial_shapes (list): Spatial shapes per feature level.
            valid_ratios (tensor): Ratio of valid (non-padded) area per
                level, used to scale reference points.
            training (bool): Training mode flag.

        Returns:
            tuple: (hidden_states, reference_points) where:
                - hidden_states: stacked layer outputs if return_intermediate,
                  else single output with leading dim 1.
                - reference_points: stacked reference points history.
        """
        output = tgt
        intermediate = []
        hs_refpoints_unsigmoid = [refpoints_unsigmoid]

        def get_reference(refpoints):
            """Compute query positional embeddings from reference points.

            Generates sinusoidal embeddings from the reference coordinates
            and transforms them through the ref_point_head MLP to produce
            query position embeddings for the decoder layer.
            """
            obj_center = refpoints[..., :4]
            refpoints_input = ops.expand_dims(obj_center, axis=2)

            # Scale reference points by valid ratios to account for
            # padding in multi-scale features
            if valid_ratios is not None:
                vr = ops.concatenate([valid_ratios, valid_ratios], axis=-1)
                vr = ops.expand_dims(vr, axis=1)
                refpoints_input = refpoints_input * vr

            # Generate sinusoidal position embeddings and project to
            # d_model-dimensional query position embeddings
            query_sine_embed = gen_sineembed_for_position(
                refpoints_input[..., 0, :], self.d_model // 2
            )
            query_pos = self.ref_point_head(query_sine_embed)

            return obj_center, refpoints_input, query_pos, query_sine_embed

        # In lite mode, compute reference-derived embeddings once before
        # the layer loop (saves computation when refinement is not per-layer)
        if self.lite_refpoint_refine:
            if self.bbox_reparam:
                obj_center, refpoints_input, query_pos, query_sine_embed = (
                    get_reference(refpoints_unsigmoid)
                )
            else:
                obj_center, refpoints_input, query_pos, query_sine_embed = (
                    get_reference(ops.sigmoid(refpoints_unsigmoid))
                )

        for layer_id, layer in enumerate(self.layers_list):
            # In standard mode, recompute reference embeddings at each layer
            # using the latest (possibly refined) reference points
            if not self.lite_refpoint_refine:
                if self.bbox_reparam:
                    obj_center, refpoints_input, query_pos, query_sine_embed = (
                        get_reference(refpoints_unsigmoid)
                    )
                else:
                    obj_center, refpoints_input, query_pos, query_sine_embed = (
                        get_reference(ops.sigmoid(refpoints_unsigmoid))
                    )

            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                query_sine_embed=query_sine_embed,
                is_first=(layer_id == 0),
                reference_points=refpoints_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                training=training,
            )

            # Iterative reference point refinement: predict bbox deltas
            # from the layer output and update reference points for the
            # next layer. Detach gradients to prevent refinement from
            # back-propagating through earlier layers.
            if not self.lite_refpoint_refine:
                if self.bbox_embed is not None:
                    new_refpoints_delta = self.bbox_embed(output)
                    new_refpoints_unsigmoid = self.refpoints_refine(
                        refpoints_unsigmoid, new_refpoints_delta
                    )
                    if layer_id != self.num_layers - 1:
                        hs_refpoints_unsigmoid.append(new_refpoints_unsigmoid)
                    refpoints_unsigmoid = ops.stop_gradient(new_refpoints_unsigmoid)

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        # Final layer normalization
        if self.norm is not None:
            output = self.norm(output)
            # Replace the last intermediate with the fully normalized output
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return ops.stack(intermediate), ops.stack(hs_refpoints_unsigmoid)
            else:
                return ops.stack(intermediate), ops.stack(hs_refpoints_unsigmoid)

        return ops.expand_dims(output, axis=0), ops.expand_dims(
            refpoints_unsigmoid, axis=0
        )


@keras.saving.register_keras_serializable(package="RFDETR")
class Transformer(keras.Model):
    """Top-level transformer model for RF-DETR object detection.

    Flattens multi-scale feature maps from the backbone, optionally
    generates two-stage object proposals from encoder memory, and
    runs the decoder to produce detection outputs.

    Attributes:
        d_model (int): Model embedding dimension.
        num_queries (int): Number of object queries.
        dec_layers (int): Number of decoder layers.
        two_stage (bool): If True, generate initial proposals from
            encoder memory (two-stage detection).
        hidden_dim (int): Alias for d_model.
        bbox_reparam (bool): Use reparameterized bbox encoding.
        group_detr (int): Number of DETR groups for training.
        decoder (TransformerDecoder): The decoder layer stack.
        enc_output (list): Per-group projection layers for encoder
            memory (two-stage only).
        enc_output_norm (list): Per-group normalization layers for
            encoder memory (two-stage only).
        enc_out_class_embed (list): Per-group classification heads
            for proposal scoring (assigned externally, two-stage only).
        enc_out_bbox_embed (list): Per-group bbox regression heads
            for proposal generation (assigned externally, two-stage only).
    """
    def __init__(
        self,
        d_model=512,
        sa_nhead=8,
        ca_nhead=8,
        num_queries=300,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        group_detr=1,
        two_stage=False,
        num_feature_levels=4,
        dec_n_points=4,
        lite_refpoint_refine=False,
        decoder_norm_type="LN",
        bbox_reparam=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_queries = num_queries
        self.dec_layers = num_decoder_layers
        self.two_stage = two_stage
        self.hidden_dim = d_model
        self.bbox_reparam = bbox_reparam
        self.group_detr = group_detr
        self._sa_nhead = sa_nhead
        self._ca_nhead = ca_nhead
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._activation = activation
        self._normalize_before = normalize_before
        self._return_intermediate_dec = return_intermediate_dec
        self._num_feature_levels = num_feature_levels
        self._dec_n_points = dec_n_points
        self._lite_refpoint_refine = lite_refpoint_refine
        self._decoder_norm_type = decoder_norm_type

        # Build shared decoder layer configuration
        decoder_layer_kwargs = dict(
            d_model=d_model,
            sa_nhead=sa_nhead,
            ca_nhead=ca_nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
            group_detr=group_detr,
            num_feature_levels=num_feature_levels,
            dec_n_points=dec_n_points,
        )

        self.decoder = TransformerDecoder(
            d_model=d_model,
            num_layers=num_decoder_layers,
            return_intermediate=return_intermediate_dec,
            lite_refpoint_refine=lite_refpoint_refine,
            bbox_reparam=bbox_reparam,
            decoder_layer_kwargs=decoder_layer_kwargs,
        )

        # Two-stage components: per-group projection and normalization
        # for generating initial object proposals from encoder memory
        if two_stage:
            self.enc_output = [layers.Dense(d_model) for _ in range(group_detr)]
            self.enc_output_norm = [
                layers.LayerNormalization(epsilon=1e-5) for _ in range(group_detr)
            ]
            self.enc_out_class_embed = None
            self.enc_out_bbox_embed = None

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "sa_nhead": self._sa_nhead,
                "ca_nhead": self._ca_nhead,
                "num_queries": self.num_queries,
                "num_decoder_layers": self.dec_layers,
                "dim_feedforward": self._dim_feedforward,
                "dropout": self._dropout,
                "activation": self._activation,
                "normalize_before": self._normalize_before,
                "return_intermediate_dec": self._return_intermediate_dec,
                "group_detr": self.group_detr,
                "two_stage": self.two_stage,
                "num_feature_levels": self._num_feature_levels,
                "dec_n_points": self._dec_n_points,
                "lite_refpoint_refine": self._lite_refpoint_refine,
                "decoder_norm_type": self._decoder_norm_type,
                "bbox_reparam": self.bbox_reparam,
            }
        )
        return config

    def get_valid_ratio(self, mask):
        """Compute the ratio of valid (non-padded) spatial extent.

        Args:
            mask (tensor): Boolean padding mask of shape (B, H, W).

        Returns:
            tensor: Valid width and height ratios, shape (B, 2).
        """
        _, H, W = ops.shape(mask)[0], ops.shape(mask)[1], ops.shape(mask)[2]

        not_mask = ops.logical_not(ops.cast(mask, "bool"))
        valid_H = ops.sum(ops.cast(not_mask[:, :, 0], "float32"), axis=1)
        valid_W = ops.sum(ops.cast(not_mask[:, 0, :], "float32"), axis=1)

        valid_ratio_h = valid_H / ops.cast(H, "float32")
        valid_ratio_w = valid_W / ops.cast(W, "float32")

        valid_ratio = ops.stack([valid_ratio_w, valid_ratio_h], axis=-1)
        return valid_ratio

    def call(
        self,
        srcs,
        masks,
        pos_embeds,
        query_feat=None,
        refpoint_embed=None,
        training=None,
    ):
        """Forward pass of the full transformer.

        Args:
            srcs (list): Multi-scale feature maps, each of shape
                (B, C, H, W) or (B, H, W, C).
            masks (list): Padding masks per level, each (B, H, W).
            pos_embeds (list): Positional embeddings per level,
                same shape as srcs.
            query_feat (tensor): Learnable query features,
                shape (N_q, d_model).
            refpoint_embed (tensor): Initial reference point coordinates,
                shape (N_q, 4).
            training (bool): Training mode flag.

        Returns:
            tuple: (hs, references, memory_ts, boxes_ts) where:
                - hs: Decoder hidden states (stacked across layers if
                  return_intermediate).
                - references: Reference points history.
                - memory_ts: Two-stage selected memory features or None.
                - boxes_ts: Two-stage proposal boxes or None.
        """
        # Flatten multi-scale feature maps and positional embeddings
        # into a single sequence for the decoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        for lvl, key in enumerate(srcs):
            src = srcs[lvl]
            pos_embed = pos_embeds[lvl]

            # Auto-detect channel ordering: channels-last (B, H, W, C)
            # vs channels-first (B, C, H, W) and flatten accordingly
            s_shape = ops.shape(src)
            if s_shape[-1] == self.d_model or (
                len(s_shape) == 4 and s_shape[1] != self.d_model
            ):
                # Channels-last: (B, H, W, C) -> (B, H*W, C)
                bs, h, w, c = s_shape[0], s_shape[1], s_shape[2], s_shape[3]
                src = ops.reshape(src, (bs, -1, c))
                pos_embed = ops.reshape(pos_embed, (bs, -1, c))
            else:
                # Channels-first: (B, C, H, W) -> (B, H*W, C)
                bs, c, h, w = s_shape[0], s_shape[1], s_shape[2], s_shape[3]
                src = ops.reshape(src, (bs, c, -1))
                src = ops.transpose(src, (0, 2, 1))
                pos_embed = ops.reshape(pos_embed, (bs, c, -1))
                pos_embed = ops.transpose(pos_embed, (0, 2, 1))

            spatial_shapes.append((h, w))
            src_flatten.append(src)
            lvl_pos_embed_flatten.append(pos_embed)

            if masks is not None:
                mask = masks[lvl]
                mask = ops.reshape(mask, (bs, -1))
                mask_flatten.append(mask)

        # Concatenate all levels into one sequence
        memory = ops.concatenate(src_flatten, axis=1)
        lvl_pos_embed_flatten = ops.concatenate(lvl_pos_embed_flatten, axis=1)

        # Compute valid ratios and level start indices for deformable attention
        valid_ratios = None
        mask_flatten_concat = None
        if masks is not None:
            mask_flatten_concat = ops.concatenate(mask_flatten, axis=1)
            valid_ratios = ops.stack([self.get_valid_ratio(m) for m in masks], axis=1)

        # Compute cumulative start index for each feature level in the
        # flattened sequence, used by deformable attention to split levels
        spatial_shapes_tensor = ops.convert_to_tensor(spatial_shapes, dtype="int64")
        lens = spatial_shapes_tensor[:, 0] * spatial_shapes_tensor[:, 1]

        zero = ops.zeros((1,), dtype="int64")
        cumsum = ops.cumsum(lens, axis=0)[:-1]
        level_start_index = ops.concatenate([zero, cumsum], axis=0)

        # Two-stage proposal generation: select top-k proposals from
        # encoder memory using classification scores
        refpoint_embed_ts = None
        memory_ts = None
        boxes_ts = None

        if self.two_stage:
            # Generate initial proposals from the encoder memory
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory,
                mask_flatten_concat,
                spatial_shapes,
                unsigmoid=not self.bbox_reparam,
            )

            refpoint_embed_ts_list = []
            memory_ts_list = []
            boxes_ts_list = []

            # Process each DETR group: project memory, classify proposals,
            # regress box coordinates, and select top-k proposals
            group_detr = self.group_detr if training else 1

            for g_idx in range(group_detr):
                # Project and normalize encoder memory for this group
                output_memory_gidx = self.enc_output_norm[g_idx](
                    self.enc_output[g_idx](output_memory)
                )

                enc_outputs_class_unselected_gidx = self.enc_out_class_embed[g_idx](
                    output_memory_gidx
                )

                # Predict bbox coordinates for all spatial positions
                if self.bbox_reparam:
                    enc_outputs_coord_delta_gidx = self.enc_out_bbox_embed[g_idx](
                        output_memory_gidx
                    )
                    enc_outputs_coord_cxcy_gidx = (
                        enc_outputs_coord_delta_gidx[..., :2]
                        * output_proposals[..., 2:]
                        + output_proposals[..., :2]
                    )
                    enc_outputs_coord_wh_gidx = (
                        ops.exp(enc_outputs_coord_delta_gidx[..., 2:])
                        * output_proposals[..., 2:]
                    )
                    enc_outputs_coord_unselected_gidx = ops.concatenate(
                        [enc_outputs_coord_cxcy_gidx, enc_outputs_coord_wh_gidx],
                        axis=-1,
                    )
                else:
                    enc_outputs_coord_unselected_gidx = (
                        self.enc_out_bbox_embed[g_idx](output_memory_gidx)
                        + output_proposals
                    )

                # Select top-k proposals based on max classification score
                topk = min(
                    self.num_queries, ops.shape(enc_outputs_class_unselected_gidx)[-2]
                )
                topk_proposals_gidx = ops.top_k(
                    ops.max(enc_outputs_class_unselected_gidx, axis=-1), topk
                )[1]

                # Gather the selected proposals' coordinates and features.
                # Stop gradient on reference embeddings to prevent proposal
                # selection from affecting the encoder gradients.
                topk_indices_expanded = ops.expand_dims(topk_proposals_gidx, axis=-1)

                refpoint_embed_gidx_undetach = ops.take_along_axis(
                    enc_outputs_coord_unselected_gidx, topk_indices_expanded, axis=1
                )
                refpoint_embed_gidx = ops.stop_gradient(refpoint_embed_gidx_undetach)

                tgt_undetach_gidx = ops.take_along_axis(
                    output_memory_gidx, topk_indices_expanded, axis=1
                )

                refpoint_embed_ts_list.append(refpoint_embed_gidx)
                memory_ts_list.append(tgt_undetach_gidx)
                boxes_ts_list.append(refpoint_embed_gidx_undetach)

            # Concatenate proposals from all groups along the query dimension
            refpoint_embed_ts = ops.concatenate(refpoint_embed_ts_list, axis=1)
            memory_ts = ops.concatenate(memory_ts_list, axis=1)
            boxes_ts = ops.concatenate(boxes_ts_list, axis=1)

        # Prepare decoder input: broadcast query features and reference
        # points across the batch dimension
        tgt = None

        if self.dec_layers > 0:
            tgt = ops.expand_dims(query_feat, axis=0)
            tgt = ops.repeat(tgt, bs, axis=0)

            refpoint_embed_dec = ops.expand_dims(refpoint_embed, axis=0)
            refpoint_embed_dec = ops.repeat(refpoint_embed_dec, bs, axis=0)

            # In two-stage mode, combine the two-stage proposals with
            # the learned reference point embeddings
            if self.two_stage:
                ts_len = ops.shape(refpoint_embed_ts)[-2]
                refpoint_embed_ts_subset = refpoint_embed_dec[..., :ts_len, :]
                refpoint_embed_subset = refpoint_embed_dec[..., ts_len:, :]

                if self.bbox_reparam:
                    refpoint_embed_cxcy = (
                        refpoint_embed_ts_subset[..., :2] * refpoint_embed_ts[..., 2:]
                        + refpoint_embed_ts[..., :2]
                    )
                    refpoint_embed_wh = (
                        ops.exp(refpoint_embed_ts_subset[..., 2:])
                        * refpoint_embed_ts[..., 2:]
                    )
                    refpoint_embed_ts_subset = ops.concatenate(
                        [refpoint_embed_cxcy, refpoint_embed_wh], axis=-1
                    )
                else:
                    refpoint_embed_ts_subset = (
                        refpoint_embed_ts_subset + refpoint_embed_ts
                    )

                refpoint_embed_dec = ops.concatenate(
                    [refpoint_embed_ts_subset, refpoint_embed_subset], axis=-2
                )

            hs, references = self.decoder(
                tgt,
                memory,
                memory_key_padding_mask=mask_flatten_concat,
                pos=lvl_pos_embed_flatten,
                refpoints_unsigmoid=refpoint_embed_dec,
                level_start_index=level_start_index,
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
                training=training,
            )
        else:
            hs = None
            references = None

        if self.two_stage:
            if self.bbox_reparam:
                return hs, references, memory_ts, boxes_ts
            else:
                return hs, references, memory_ts, ops.sigmoid(boxes_ts)

        return hs, references, None, None
