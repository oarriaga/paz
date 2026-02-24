import numpy as np
import torch


def to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    elif hasattr(t, "numpy"):  # Keras tensor
        return t.numpy()
    return np.array(t)


def assert_allclose(a, b, atol=1e-5, rtol=1e-5):
    a_np = to_numpy(a)
    b_np = to_numpy(b)
    np.testing.assert_allclose(a_np, b_np, atol=atol, rtol=rtol)


# --- Weight Transfer Helpers ---


def copy_depthwise_conv_block(pt_block, keras_block):
    # dwconv
    pt_w = pt_block.dwconv.weight.data.cpu().numpy()  # (C, 1, 3, 3)
    pt_b = pt_block.dwconv.bias.data.cpu().numpy()

    # Transpose PT weights to Keras format: (kH, kW, C, 1)
    keras_w = np.transpose(pt_w, (2, 3, 0, 1))

    keras_block.dwconv.set_weights([keras_w, pt_b])

    # norm (LayerNorm)
    keras_block.norm.set_weights(
        [pt_block.norm.weight.data.cpu().numpy(), pt_block.norm.bias.data.cpu().numpy()]
    )

    # pwconv1 (Linear/Dense)
    keras_block.pwconv1.set_weights(
        [
            pt_block.pwconv1.weight.data.cpu().numpy().T,
            pt_block.pwconv1.bias.data.cpu().numpy(),
        ]
    )

    # gamma parameter
    if keras_block.gamma is not None and pt_block.gamma is not None:
        keras_block.gamma.assign(pt_block.gamma.data.cpu().numpy())


def copy_mlp_block(pt_block, keras_block):
    # norm_in
    keras_block.norm_in.set_weights(
        [
            pt_block.norm_in.weight.data.cpu().numpy(),
            pt_block.norm_in.bias.data.cpu().numpy(),
        ]
    )

    # layers[0] -> linear1
    keras_block.linear1.set_weights(
        [
            pt_block.layers[0].weight.data.cpu().numpy().T,
            pt_block.layers[0].bias.data.cpu().numpy(),
        ]
    )

    # layers[2] -> linear2
    keras_block.linear2.set_weights(
        [
            pt_block.layers[2].weight.data.cpu().numpy().T,
            pt_block.layers[2].bias.data.cpu().numpy(),
        ]
    )

    # gamma
    if keras_block.gamma is not None and pt_block.gamma is not None:
        keras_block.gamma.assign(pt_block.gamma.data.cpu().numpy())


def copy_segmentation_head(pt_head, keras_head):
    # blocks
    for pt_b, keras_b in zip(pt_head.blocks, keras_head.blocks):
        copy_depthwise_conv_block(pt_b, keras_b)

    # spatial_features_proj
    if isinstance(pt_head.spatial_features_proj, torch.nn.Conv2d):
        pt_w = pt_head.spatial_features_proj.weight.data.cpu().numpy()
        pt_b = pt_head.spatial_features_proj.bias.data.cpu().numpy()

        # Transpose (out, in, k, k) -> (k, k, in, out)
        keras_w = np.transpose(pt_w, (2, 3, 1, 0))
        keras_head.spatial_features_proj.set_weights([keras_w, pt_b])

    # query_features_block
    copy_mlp_block(pt_head.query_features_block, keras_head.query_features_block)

    # query_features_proj
    if isinstance(pt_head.query_features_proj, torch.nn.Linear):
        keras_head.query_features_proj.set_weights(
            [
                pt_head.query_features_proj.weight.data.cpu().numpy().T,
                pt_head.query_features_proj.bias.data.cpu().numpy(),
            ]
        )

    # bias
    keras_head.bias.assign(pt_head.bias.data.cpu().numpy())
