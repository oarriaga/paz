import keras
from keras import ops


def point_sample(input, point_coords, align_corners=False, **kwargs):
    """
    Bilinear point sampling of 2D feature maps.
    Args:
        input: (N, C, H, W) tensor of feature maps
        point_coords: (N, P, 2) tensor of normalized point coordinates
                      in the range [0, 1], or (N, Hpoint, Wpoint, 2)
        align_corners: whether to align corners when scaling coords
        **kwargs: for compatibility with other implementations
    Returns:
        output: (N, C, P) or (N, C, Hpoint, Wpoint) tensor of sampled features at
                the point coordinates.
    """
    # Safety check for unimplemented features
    if kwargs.get("mode", "bilinear") != "bilinear":
        raise NotImplementedError("Keras point_sample only supports mode='bilinear'")
    if kwargs.get("padding_mode", "zeros") != "zeros":
        raise NotImplementedError(
            "Keras point_sample only supports padding_mode='zeros'"
        )

    # 1. Handle Input Shapes
    # Transpose (N, C, H, W) -> (N, H, W, C) for easier gathering
    input_transposed = ops.transpose(input, axes=(0, 2, 3, 1))

    input_shape = ops.shape(input_transposed)
    N = input_shape[0]
    H = input_shape[1]
    W = input_shape[2]
    C = input_shape[3]

    # Handle points shape
    points_shape = ops.shape(point_coords)
    if len(points_shape) == 4:
        P = points_shape[1] * points_shape[2]
        points = ops.reshape(point_coords, (N, P, 2))
    else:
        P = points_shape[1]
        points = point_coords

    # 2. Map Coordinates to Pixel Space
    x = points[:, :, 0]
    y = points[:, :, 1]

    # Ensure float32/float16 stability
    dtype = x.dtype

    if align_corners:
        x = x * ops.cast(W - 1, dtype)
        y = y * ops.cast(H - 1, dtype)
    else:
        x = x * ops.cast(W, dtype) - 0.5
        y = y * ops.cast(H, dtype) - 0.5

    # 3. Bilinear Interpolation
    x0 = ops.floor(x)
    x1 = x0 + 1
    y0 = ops.floor(y)
    y1 = y0 + 1

    # Check bounds (Padding mode = zeros)
    W_cast = ops.cast(W, dtype)
    H_cast = ops.cast(H, dtype)

    # Validity masks
    mask_a = (x0 >= 0) & (x0 < W_cast) & (y0 >= 0) & (y0 < H_cast)
    mask_b = (x0 >= 0) & (x0 < W_cast) & (y1 >= 0) & (y1 < H_cast)
    mask_c = (x1 >= 0) & (x1 < W_cast) & (y0 >= 0) & (y0 < H_cast)
    mask_d = (x1 >= 0) & (x1 < W_cast) & (y1 >= 0) & (y1 < H_cast)

    # Clamp indices to avoid out-of-bounds access errors
    # (The values gathered here might be wrong, but they are masked out by 0 later)
    x0_safe = ops.clip(ops.cast(x0, "int32"), 0, W - 1)
    x1_safe = ops.clip(ops.cast(x1, "int32"), 0, W - 1)
    y0_safe = ops.clip(ops.cast(y0, "int32"), 0, H - 1)
    y1_safe = ops.clip(ops.cast(y1, "int32"), 0, H - 1)

    # Helper to gather and mask
    batch_indices = ops.expand_dims(ops.arange(N, dtype="int32"), axis=1)  # (N, 1)

    def get_pixel_values(y_idx, x_idx, mask):
        # Calculate flat indices: b * (H*W) + y * W + x
        # Flatten input to (N*H*W, C)
        flat_input = ops.reshape(input_transposed, (-1, C))

        # Calculate gather indices (N, P)
        flat_indices = (batch_indices * (H * W)) + (y_idx * W) + x_idx
        flat_indices = ops.reshape(flat_indices, (-1,))

        # Gather (N*P, C) -> (N, P, C)
        gathered = ops.take(flat_input, flat_indices, axis=0)
        gathered = ops.reshape(gathered, (N, P, C))

        # Apply Zero Padding Mask
        mask = ops.expand_dims(mask, -1)
        mask = ops.cast(mask, gathered.dtype)
        return gathered * mask

    Ia = get_pixel_values(y0_safe, x0_safe, mask_a)
    Ib = get_pixel_values(y1_safe, x0_safe, mask_b)
    Ic = get_pixel_values(y0_safe, x1_safe, mask_c)
    Id = get_pixel_values(y1_safe, x1_safe, mask_d)

    # Calculate Weights
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    wa = ops.expand_dims(wa, -1)
    wb = ops.expand_dims(wb, -1)
    wc = ops.expand_dims(wc, -1)
    wd = ops.expand_dims(wd, -1)

    out = (Ia * wa) + (Ib * wb) + (Ic * wc) + (Id * wd)

    # 4. Final Reshape to (N, C, ...)
    out = ops.transpose(out, axes=(0, 2, 1))
    if len(points_shape) == 4:
        out = ops.reshape(out, (N, C, points_shape[1], points_shape[2]))

    return out


def get_uncertain_point_coords_with_randomness(
    coarse_logits,
    uncertainty_func,
    num_points,
    oversample_ratio=3,
    importance_sample_ratio=0.75,
):
    """
    Get point coordinates corresponding to the most uncertain points,
    with some randomness.
    Args:
        coarse_logits: Tensor of shape (N, C, Hmask, Wmask)
        uncertainty_func: Function taking (N, C, P) -> (N, 1, P)
        num_points: int
        oversample_ratio: int
        importance_sample_ratio: float

    Returns:
        point_coords: (N, P, 2)
    """
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1

    input_shape = ops.shape(coarse_logits)
    num_boxes = input_shape[0]
    num_sampled = int(num_points * oversample_ratio)

    # 1. Generate Random Points
    # Shape: (N, num_sampled, 2)
    point_coords = keras.random.uniform(
        shape=(num_boxes, num_sampled, 2), minval=0.0, maxval=1.0, dtype="float32"
    )

    # 2. Sample Logits at these points
    # (N, C, num_sampled)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)

    # 3. Calculate Uncertainty
    # (N, 1, num_sampled)
    point_uncertainties = uncertainty_func(point_logits)

    # 4. Select Top-K Uncertain Points
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    _, idx = ops.top_k(point_uncertainties[:, 0, :], k=num_uncertain_points)

    # 5. Gather the coordinates corresponding to high uncertainty
    idx_expanded = ops.expand_dims(idx, -1)

    # Gather selected coordinates
    selected_point_coords = ops.take_along_axis(point_coords, idx_expanded, axis=1)

    # 6. Add fresh random points if needed
    if num_random_points > 0:
        random_point_coords = keras.random.uniform(
            shape=(num_boxes, num_random_points, 2),
            minval=0.0,
            maxval=1.0,
            dtype="float32",
        )
        point_coords = ops.concatenate(
            [selected_point_coords, random_point_coords], axis=1
        )
    else:
        point_coords = selected_point_coords

    return point_coords
