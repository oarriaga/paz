import numpy as np
import torch


def to_numpy(t):
    """Convert a tensor from any framework to a NumPy array.

    Args:
        t: A PyTorch tensor, Keras/JAX tensor, or array-like.

    Returns:
        numpy.ndarray: The data as a NumPy array.
    """
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    elif hasattr(t, "numpy"):
        return t.numpy()
    return np.array(t)


def assert_allclose(a, b, atol=1e-5, rtol=1e-5):
    """Assert element-wise near-equality between two tensors.

    Args:
        a: First tensor (any framework).
        b: Second tensor (any framework).
        atol (float): Absolute tolerance.
        rtol (float): Relative tolerance.
    """
    a_np = to_numpy(a)
    b_np = to_numpy(b)
    np.testing.assert_allclose(a_np, b_np, atol=atol, rtol=rtol)


def copy_depthwise_conv_block(pt_block, keras_block):
    """Transfer weights for a single DepthwiseConvBlock.

    Copies the depthwise convolution kernel and bias (transposing from
    OIHW to HWIO layout), layer-norm parameters, pointwise dense
    weights + bias, and the optional gamma scaling factor.

    Args:
        pt_block: Reference DepthwiseConvBlock with trained weights.
        keras_block: Keras DepthwiseConvBlock to receive the weights.
    """
    # Depthwise convolution: transpose (C,1,kH,kW) -> (kH,kW,C,1)
    pt_w = pt_block.dwconv.weight.data.cpu().numpy()
    pt_b = pt_block.dwconv.bias.data.cpu().numpy()
    keras_w = np.transpose(pt_w, (2, 3, 0, 1))
    keras_block.dwconv.set_weights([keras_w, pt_b])

    # Layer normalisation
    keras_block.norm.set_weights(
        [pt_block.norm.weight.data.cpu().numpy(), pt_block.norm.bias.data.cpu().numpy()]
    )

    # Pointwise dense: transpose weight matrix for Dense convention
    keras_block.pwconv1.set_weights(
        [
            pt_block.pwconv1.weight.data.cpu().numpy().T,
            pt_block.pwconv1.bias.data.cpu().numpy(),
        ]
    )

    # Optional per-channel gamma
    if keras_block.gamma is not None and pt_block.gamma is not None:
        keras_block.gamma.assign(pt_block.gamma.data.cpu().numpy())


def copy_mlp_block(pt_block, keras_block):
    """Transfer weights for a single MLPBlock.

    Copies layer-norm parameters, two dense layers (with transposed
    weight matrices), and the optional gamma scaling factor.

    Args:
        pt_block: Reference MLPBlock with trained weights.
        keras_block: Keras MLPBlock to receive the weights.
    """
    # Layer normalisation
    keras_block.norm_in.set_weights(
        [
            pt_block.norm_in.weight.data.cpu().numpy(),
            pt_block.norm_in.bias.data.cpu().numpy(),
        ]
    )

    # First dense layer (reference layers[0] -> Keras linear1)
    keras_block.linear1.set_weights(
        [
            pt_block.layers[0].weight.data.cpu().numpy().T,
            pt_block.layers[0].bias.data.cpu().numpy(),
        ]
    )

    # Second dense layer (reference layers[2] -> Keras linear2)
    keras_block.linear2.set_weights(
        [
            pt_block.layers[2].weight.data.cpu().numpy().T,
            pt_block.layers[2].bias.data.cpu().numpy(),
        ]
    )

    # Optional per-channel gamma
    if keras_block.gamma is not None and pt_block.gamma is not None:
        keras_block.gamma.assign(pt_block.gamma.data.cpu().numpy())


def copy_segmentation_head(pt_head, keras_head):
    """Transfer all weights from a reference SegmentationHead to Keras.

    Copies depthwise conv blocks, the spatial-features 1x1 projection,
    the query-features MLP block + projection, and the scalar bias.

    Args:
        pt_head: Reference SegmentationHead with trained weights.
        keras_head: Keras SegmentationHead to receive the weights.
    """
    # Copy each depthwise convolution block
    for pt_b, keras_b in zip(pt_head.blocks, keras_head.blocks):
        copy_depthwise_conv_block(pt_b, keras_b)

    # Spatial features 1x1 conv projection (skip if Identity)
    if isinstance(pt_head.spatial_features_proj, torch.nn.Conv2d):
        pt_w = pt_head.spatial_features_proj.weight.data.cpu().numpy()
        pt_b = pt_head.spatial_features_proj.bias.data.cpu().numpy()
        # Transpose (out,in,kH,kW) -> (kH,kW,in,out)
        keras_w = np.transpose(pt_w, (2, 3, 1, 0))
        keras_head.spatial_features_proj.set_weights([keras_w, pt_b])

    # Query features MLP block
    copy_mlp_block(pt_head.query_features_block, keras_head.query_features_block)

    # Query features dense projection (skip if Identity)
    if isinstance(pt_head.query_features_proj, torch.nn.Linear):
        keras_head.query_features_proj.set_weights(
            [
                pt_head.query_features_proj.weight.data.cpu().numpy().T,
                pt_head.query_features_proj.bias.data.cpu().numpy(),
            ]
        )

    # Scalar bias
    keras_head.bias.assign(pt_head.bias.data.cpu().numpy())
