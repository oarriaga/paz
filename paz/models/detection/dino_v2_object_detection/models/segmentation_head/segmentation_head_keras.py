import keras
from keras import ops
import numpy as np


def grid_sample(input_tensor, grid, align_corners=False):
    """Bilinear sampling of a 4-D tensor at continuous grid locations.

    Implements bilinear interpolation to sample values from
    ``input_tensor`` at the (x, y) positions given by ``grid``.

    Args:
        input_tensor: Tensor of shape (N, C, H, W) in NCHW layout.
        grid: Tensor of shape (N, H_out, W_out, 2) with coordinates
            normalised to [-1, 1].
        align_corners (bool): When True the extreme grid values map
            exactly to corner pixels; otherwise they map to pixel
            centres.

    Returns:
        Tensor of shape (N, C, H_out, W_out) with bilinearly
        interpolated values.
    """
    N, C, H, W = ops.shape(input_tensor)
    _, H_out, W_out, _ = ops.shape(grid)

    # Separate normalised x (horizontal) and y (vertical) coordinates
    x = grid[..., 0]
    y = grid[..., 1]

    # Map normalised [-1, 1] coordinates to pixel space
    if align_corners:
        x = ((x + 1) / 2) * (W - 1)
        y = ((y + 1) / 2) * (H - 1)
    else:
        x = ((x + 1) * W - 1) / 2
        y = ((y + 1) * H - 1) / 2

    # Floor and ceil pixel neighbours for bilinear interpolation
    x0 = ops.floor(x)
    x1 = x0 + 1
    y0 = ops.floor(y)
    y1 = y0 + 1

    # Clamp to valid image bounds (border replication)
    x0_safe = ops.clip(x0, 0, W - 1)
    x1_safe = ops.clip(x1, 0, W - 1)
    y0_safe = ops.clip(y0, 0, H - 1)
    y1_safe = ops.clip(y1, 0, H - 1)

    x0_int = ops.cast(x0_safe, "int32")
    x1_int = ops.cast(x1_safe, "int32")
    y0_int = ops.cast(y0_safe, "int32")
    y1_int = ops.cast(y1_safe, "int32")

    # Permute to NHWC so the spatial dimensions can be flattened for
    # index-based gathering of the four corner pixel values.
    input_perm = ops.transpose(input_tensor, (0, 2, 3, 1))  # (N, H, W, C)

    def get_pixel_values(y_idx, x_idx):
        """Gather pixel values at integer (y, x) positions across batches."""
        # Flatten all spatial positions across the batch into one axis
        super_flat_input = ops.reshape(input_perm, (-1, C))  # (N*H*W, C)

        # Compute linear indices: y * W + x for each output position
        flat_indices = y_idx * W + x_idx  # (N, H_out, W_out)

        # Offset each batch element so indices are globally unique
        batch_offset = ops.arange(N) * (H * W)
        batch_offset = ops.reshape(batch_offset, (N, 1, 1))
        global_indices = flat_indices + batch_offset

        global_indices_flat = ops.reshape(global_indices, (-1,))
        values = ops.take(super_flat_input, global_indices_flat, axis=0)
        return ops.reshape(values, (N, H_out, W_out, C))

    # Gather pixel values at the four surrounding integer positions
    Ia = get_pixel_values(y0_int, x0_int)  # top-left
    Ib = get_pixel_values(y1_int, x0_int)  # bottom-left
    Ic = get_pixel_values(y0_int, x1_int)  # top-right
    Id = get_pixel_values(y1_int, x1_int)  # bottom-right

    # Bilinear interpolation weights based on sub-pixel distances
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # Expand weights to broadcast over the channel dimension
    wa = ops.expand_dims(wa, -1)
    wb = ops.expand_dims(wb, -1)
    wc = ops.expand_dims(wc, -1)
    wd = ops.expand_dims(wd, -1)

    # Weighted sum of the four corner values
    out = wa * Ia + wb * Ib + wc * Ic + wd * Id  # (N, H_out, W_out, C)

    # Convert back to NCHW layout
    return ops.transpose(out, (0, 3, 1, 2))


def point_sample(input_tensor, point_coords, **kwargs):
    """Sample feature values at a set of point coordinates.

    Converts point coordinates from [0, 1] range to the [-1, 1]
    grid expected by ``grid_sample`` and returns interpolated values.

    Args:
        input_tensor: Tensor of shape (N, C, H, W) in NCHW layout.
        point_coords: Tensor of shape (N, P, 2) or
            (N, H_out, W_out, 2) with coordinates in [0, 1].
        **kwargs: Forwarded to ``grid_sample`` (e.g. ``align_corners``).

    Returns:
        Tensor of shape (N, C, P) when ``point_coords`` is 3-D, or
        (N, C, H_out, W_out) when 4-D.
    """
    # When given (N, P, 2) points, add a dummy spatial dimension so the
    # grid becomes (N, P, 1, 2) which grid_sample can process.
    add_dim = False
    if ops.ndim(point_coords) == 3:
        add_dim = True
        point_coords = ops.expand_dims(point_coords, 2)

    # Rescale from [0, 1] to the [-1, 1] range expected by grid_sample
    grid = 2.0 * point_coords - 1.0

    align_corners = kwargs.get("align_corners", False)
    output = grid_sample(input_tensor, grid, align_corners=align_corners)

    if add_dim:
        output = ops.squeeze(output, 3)  # remove dummy W dim -> (N, C, P)

    return output


def calculate_uncertainty(logits):
    """Compute per-point uncertainty from binary mask logits.

    Points whose logits are close to zero are least certain;
    returning the negated absolute value makes higher values
    correspond to higher uncertainty (suitable for top-k selection).

    Args:
        logits: Tensor of shape (R, 1, ...) with raw mask logits.

    Returns:
        Tensor of the same shape with uncertainty scores.
    """
    return -ops.abs(logits)


def get_uncertain_point_coords_with_randomness(
    coarse_logits,
    uncertainty_func,
    num_points,
    oversample_ratio=3,
    importance_sample_ratio=0.75,
):
    """Select point coordinates biased toward high-uncertainty regions.

    Oversamples random candidate points, evaluates their uncertainty,
    keeps the top-k most uncertain ones, and fills the remainder with
    uniformly random points.

    Args:
        coarse_logits: Tensor of shape (N, 1, H, W) with mask logits.
        uncertainty_func: Callable mapping logits to uncertainty scores.
        num_points (int): Total number of points to return.
        oversample_ratio (float): Factor by which to oversample
            candidates (must be >= 1).
        importance_sample_ratio (float): Fraction of ``num_points``
            selected by uncertainty (rest are random).

    Returns:
        Tensor of shape (N, num_points, 2) with coordinates in [0, 1].
    """
    N = ops.shape(coarse_logits)[0]
    num_sampled = int(num_points * oversample_ratio)

    # Sample an oversampled set of random candidate points
    point_coords = keras.random.uniform((N, num_sampled, 2), minval=0.0, maxval=1.0)

    # Evaluate logits and uncertainty at each candidate
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    point_uncertainties = uncertainty_func(point_logits)

    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points

    # Select the top-k most uncertain candidate indices
    uncertainties_sub = point_uncertainties[:, 0, :]  # (N, P)
    _, idx = ops.top_k(uncertainties_sub, k=num_uncertain_points)

    # Gather the coordinates of the most uncertain candidates.
    # Flatten across batches so a single ops.take can be used, then
    # add per-batch offsets so each sample indexes its own candidates.
    shift = ops.arange(N) * num_sampled
    shift = ops.expand_dims(shift, -1)  # (N, 1)
    idx_flat = ops.reshape(idx + ops.cast(shift, idx.dtype), (-1,))

    point_coords_flat = ops.reshape(point_coords, (-1, 2))
    selected_points = ops.take(point_coords_flat, idx_flat, axis=0)
    selected_points = ops.reshape(selected_points, (N, num_uncertain_points, 2))

    # Fill remaining slots with uniformly random points
    if num_random_points > 0:
        random_points = keras.random.uniform(
            (N, num_random_points, 2), minval=0.0, maxval=1.0
        )
        selected_points = ops.concatenate([selected_points, random_points], axis=1)

    return selected_points


class DepthwiseConvBlock(keras.Layer):
    """Depthwise-separable convolution block with optional layer scaling.

    Applies a 3x3 depthwise convolution followed by layer-normalisation,
    a pointwise (1x1) dense projection, GELU activation, and an optional
    learnable per-channel scaling factor (gamma).  A residual connection
    adds the input back to the output.

    All convolutions use NCHW layout; internal normalisation and dense
    layers operate in NHWC after an explicit transpose.

    Attributes:
        dim (int): Number of input and output channels.
        layer_scale_init_value (float): Initial value for the per-channel
            gamma parameter.  Set to 0 to disable layer scaling.
    """

    def __init__(self, dim, layer_scale_init_value=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.layer_scale_init_value = layer_scale_init_value

        # 3x3 depthwise convolution (NCHW, same padding)
        self.dwconv = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            padding="same",
            data_format="channels_first",
            depth_multiplier=1,
            use_bias=True,
        )
        # LayerNorm applied on the channel axis after NHWC transpose
        self.norm = keras.layers.LayerNormalization(
            epsilon=1e-6, axis=-1
        )
        self.pwconv1 = keras.layers.Dense(dim)
        self.act = keras.layers.Activation("gelu")

        # Gamma
        if layer_scale_init_value > 0:
            self.gamma = self.add_weight(
                name="gamma",
                shape=(dim,),
                initializer=keras.initializers.Constant(layer_scale_init_value),
                trainable=True,
            )
        else:
            self.gamma = None

    def call(self, x):
        """Forward pass.  Input and output are NCHW tensors."""
        input_tensor = x
        x = self.dwconv(x)
        x = ops.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC for norm/dense
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = ops.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
        return x + input_tensor


class MLPBlock(keras.Layer):
    """Two-layer MLP with pre-normalisation and optional layer scaling.

    Applies LayerNorm, a hidden dense layer (4x expansion), GELU, a
    projection back to ``dim``, optional gamma scaling, and a residual
    connection.

    Attributes:
        dim (int): Input and output feature dimensionality.
        layer_scale_init_value (float): Initial gamma value; 0 disables.
    """
    def __init__(self, dim, layer_scale_init_value=0, **kwargs):
        super().__init__(**kwargs)
        self.norm_in = keras.layers.LayerNormalization(
            epsilon=1e-5, axis=-1
        )

        self.linear1 = keras.layers.Dense(dim * 4)
        self.act = keras.layers.Activation("gelu")
        self.linear2 = keras.layers.Dense(dim)

        if layer_scale_init_value > 0:
            self.gamma = self.add_weight(
                name="gamma",
                shape=(dim,),
                initializer=keras.initializers.Constant(layer_scale_init_value),
                trainable=True,
            )
        else:
            self.gamma = None

    def call(self, x):
        """Forward pass with residual connection."""
        input_tensor = x
        x = self.norm_in(x)

        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)

        if self.gamma is not None:
            x = self.gamma * x

        return x + input_tensor


class SegmentationHead(keras.Layer):
    """Instance segmentation head producing per-query mask logits.

    Iterates over decoder layers: each spatial feature map is refined
    by a ``DepthwiseConvBlock``, projected to an interaction
    dimensionality, then combined with the corresponding query features
    via an einsum dot product to produce mask logits.

    Attributes:
        in_dim (int): Channel dimensionality of input spatial features.
        num_blocks (int): Number of decoder-layer interaction blocks.
        bottleneck_ratio (int | None): Reduction factor for the
            interaction dimension.  ``None`` keeps the original ``in_dim``.
        downsample_ratio (int): Factor by which the image resolution is
            divided to obtain the target feature map size.
    """
    def __init__(
        self, in_dim, num_blocks, bottleneck_ratio=1, downsample_ratio=4, **kwargs
    ):
        super().__init__(**kwargs)
        self.downsample_ratio = downsample_ratio

        self.interaction_dim = (
            in_dim // bottleneck_ratio if bottleneck_ratio is not None else in_dim
        )

        # One depthwise conv block per decoder layer
        self.blocks = [DepthwiseConvBlock(in_dim) for _ in range(num_blocks)]

        # Optional 1x1 projection to reduce spatial feature channels
        if bottleneck_ratio is None:
            self.spatial_features_proj = keras.layers.Identity()
        else:
            self.spatial_features_proj = keras.layers.Conv2D(
                self.interaction_dim,
                kernel_size=1,
                data_format="channels_first",
                use_bias=True,
            )

        # MLP that refines query features before dot-product interaction
        self.query_features_block = MLPBlock(in_dim)

        # Optional projection to match query features to interaction dim
        if bottleneck_ratio is None:
            self.query_features_proj = keras.layers.Identity()
        else:
            self.query_features_proj = keras.layers.Dense(self.interaction_dim)

        # Learnable scalar bias added to every mask logit
        self.bias = self.add_weight(
            name="bias", shape=(1,), initializer="zeros", trainable=True
        )

        self._export_mode = False

    def export(self):
        """Switch to export mode (single query features only)."""
        self._export_mode = True

    def call(
        self, spatial_features, query_features, image_size=None, skip_blocks=False
    ):
        """Produce per-layer mask logits for each query.

        Args:
            spatial_features: (B, C, H, W) NCHW feature map.
            query_features: List of (B, N, C) tensors, one per
                decoder layer.
            image_size: Optional (H, W) of the original image used to
                resize features to the target mask resolution.
            skip_blocks: When True, skips the depthwise conv blocks and
                directly projects the spatial features.  Only valid when
                ``query_features`` has exactly one element.

        Returns:
            List of (B, N, H', W') mask logit tensors.
        """
        # Delegate to the streamlined export path when active
        if self._export_mode:
            return self.call_export(
                spatial_features, query_features, image_size, skip_blocks
            )

        # spatial_features stays NCHW throughout

        # Resize spatial features to the target mask resolution
        if image_size is not None:
            target_h = image_size[0] // self.downsample_ratio
            target_w = image_size[1] // self.downsample_ratio
            target_size = (target_h, target_w)

            # Resize requires NHWC, so transpose around it
            sf = ops.transpose(spatial_features, (0, 2, 3, 1))
            sf = ops.image.resize(sf, target_size, interpolation="bilinear")
            spatial_features = ops.transpose(sf, (0, 3, 1, 2))

        mask_logits = []
        if not skip_blocks:
            # Each block refines spatial features, then mask logits are
            # computed by dotting projected spatial features with the
            # corresponding decoder-layer query features.
            for i in range(len(self.blocks)):
                if i >= len(query_features):
                    break

                block = self.blocks[i]
                qf = query_features[i]

                spatial_features = block(spatial_features)

                # Project spatial and query features to the interaction
                # dimension, then compute mask logits via einsum.
                s_proj = self.spatial_features_proj(spatial_features)
                qf_out = self.query_features_block(qf)
                qf_out = self.query_features_proj(qf_out)

                logit = ops.einsum("bchw,bnc->bnhw", s_proj, qf_out)
                mask_logits.append(logit + self.bias)
        else:
            if len(query_features) != 1:
                raise ValueError(
                    "skip_blocks is only supported for length 1 query features"
                )

            qf = query_features[0]
            qf_out = self.query_features_block(qf)
            qf_out = self.query_features_proj(qf_out)

            logit = ops.einsum("bchw,bnc->bnhw", spatial_features, qf_out)
            mask_logits.append(logit + self.bias)

        return mask_logits

    def call_export(
        self, spatial_features, query_features, image_size, skip_blocks=False
    ):
        """Simplified forward for export with a single query tensor.

        All decoder-layer blocks are applied sequentially to the spatial
        features before computing a single set of mask logits.

        Raises:
            ValueError: If ``query_features`` has more than one element.
        """
        if len(query_features) != 1:
            raise ValueError(
                "at export time, segmentation head expects exactly one query feature"
            )

        # spatial_features is NCHW

        if image_size is not None:
            target_h = image_size[0] // self.downsample_ratio
            target_w = image_size[1] // self.downsample_ratio
            target_size = (target_h, target_w)

            sf = ops.transpose(spatial_features, (0, 2, 3, 1))
            sf = ops.image.resize(sf, target_size, interpolation="bilinear")
            spatial_features = ops.transpose(sf, (0, 3, 1, 2))

        if not skip_blocks:
            for block in self.blocks:
                spatial_features = block(spatial_features)

        spatial_features_proj = self.spatial_features_proj(spatial_features)

        qf = query_features[0]
        qf_out = self.query_features_block(qf)
        qf_out = self.query_features_proj(qf_out)

        logit = ops.einsum("bchw,bnc->bnhw", spatial_features_proj, qf_out)
        return [logit + self.bias]

    def sparse_call(
        self, spatial_features, query_features, image_size, skip_blocks=False
    ):
        """Return intermediate components instead of final logits.

        Instead of computing the full einsum, returns a list of dicts
        each containing ``spatial_features``, ``query_features``, and
        ``bias`` so the caller can perform the final dot product
        externally (useful for sparse / selective evaluation).

        Args:
            spatial_features: (B, C, H, W) NCHW feature map.
            query_features: List of (B, N, C) tensors.
            image_size: (H, W) original image size.
            skip_blocks: When True, bypasses the conv blocks.

        Returns:
            List of dicts with keys ``spatial_features``,
            ``query_features``, and ``bias``.
        """

        if image_size is not None:
            target_h = image_size[0] // self.downsample_ratio
            target_w = image_size[1] // self.downsample_ratio
            target_size = (target_h, target_w)

            sf = ops.transpose(spatial_features, (0, 2, 3, 1))
            sf = ops.image.resize(sf, target_size, interpolation="bilinear")
            spatial_features = ops.transpose(sf, (0, 3, 1, 2))

        output_dicts = []

        if not skip_blocks:
            for i in range(len(self.blocks)):
                if i >= len(query_features):
                    break
                block = self.blocks[i]
                qf = query_features[i]

                spatial_features = block(spatial_features)
                s_proj = self.spatial_features_proj(spatial_features)

                qf_out = self.query_features_block(qf)
                qf_out = self.query_features_proj(qf_out)

                output_dicts.append(
                    {
                        "spatial_features": s_proj,
                        "query_features": qf_out,
                        "bias": self.bias,
                    }
                )
        else:
            if len(query_features) != 1:
                raise ValueError("skip_blocks...")
            qf = query_features[0]
            qf_out = self.query_features_block(qf)
            qf_out = self.query_features_proj(qf_out)

            output_dicts.append(
                {
                    "spatial_features": spatial_features,
                    "query_features": qf_out,
                    "bias": self.bias,
                }
            )

        return output_dicts
