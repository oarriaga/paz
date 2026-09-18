import numpy as np
import torch


def to_numpy(t):
    if isinstance(t, torch.Tensor):
        result = t.detach().cpu().numpy()
    elif hasattr(t, "numpy"):
        result = t.numpy()
    else:
        result = np.array(t)
    return result


def assert_allclose(a, b, atol=1e-5, rtol=1e-5):
    np.testing.assert_allclose(to_numpy(a), to_numpy(b), atol=atol, rtol=rtol)


def dense_weights(module):
    kernel = module.weight.data.cpu().numpy().T
    return [kernel, module.bias.data.cpu().numpy()]


def norm_weights(module):
    gamma = module.weight.data.cpu().numpy()
    return [gamma, module.bias.data.cpu().numpy()]


def copy_depthwise_conv_block(pt_block, keras_model, name):
    # Depthwise convolution: transpose (C,1,kH,kW) -> (kH,kW,C,1)
    kernel = pt_block.dwconv.weight.data.cpu().numpy()
    bias = pt_block.dwconv.bias.data.cpu().numpy()
    transposed = np.transpose(kernel, (2, 3, 0, 1))
    keras_model.get_layer(f"{name}_dwconv").set_weights([transposed, bias])
    norm = norm_weights(pt_block.norm)
    keras_model.get_layer(f"{name}_norm").set_weights(norm)
    pointwise = dense_weights(pt_block.pwconv1)
    keras_model.get_layer(f"{name}_pwconv1").set_weights(pointwise)
    copy_gamma(pt_block, keras_model, f"{name}_gamma")


def copy_mlp_block(pt_block, keras_model, name):
    norm = norm_weights(pt_block.norm_in)
    keras_model.get_layer(f"{name}_norm_in").set_weights(norm)
    # Reference layers[0] -> linear1 and layers[2] -> linear2.
    first = dense_weights(pt_block.layers[0])
    keras_model.get_layer(f"{name}_linear1").set_weights(first)
    second = dense_weights(pt_block.layers[2])
    keras_model.get_layer(f"{name}_linear2").set_weights(second)
    copy_gamma(pt_block, keras_model, f"{name}_gamma")


def copy_gamma(pt_block, keras_model, name):
    # gamma lives in an EinsumDense (its .kernel) when layer scaling is on,
    # else an Identity with no kernel; assign un-transposed when both exist.
    gamma_layer = keras_model.get_layer(name)
    if pt_block.gamma is not None and hasattr(gamma_layer, "kernel"):
        gamma_layer.kernel.assign(pt_block.gamma.data.cpu().numpy())


def copy_spatial_projection(pt_head, keras_model):
    # Skipped when the reference head uses an Identity projection.
    if isinstance(pt_head.spatial_features_proj, torch.nn.Conv2d):
        projection = pt_head.spatial_features_proj
        kernel = projection.weight.data.cpu().numpy()
        bias = projection.bias.data.cpu().numpy()
        # Transpose (out,in,kH,kW) -> (kH,kW,in,out)
        transposed = np.transpose(kernel, (2, 3, 1, 0))
        layer = keras_model.get_layer("spatial_features_proj")
        layer.set_weights([transposed, bias])


def copy_query_projection(pt_head, keras_model):
    if isinstance(pt_head.query_features_proj, torch.nn.Linear):
        weights = dense_weights(pt_head.query_features_proj)
        keras_model.get_layer("query_features_proj").set_weights(weights)


def copy_segmentation_head(pt_head, keras_model):
    for index, pt_block in enumerate(pt_head.blocks):
        copy_depthwise_conv_block(pt_block, keras_model, f"block_{index}")
    copy_spatial_projection(pt_head, keras_model)
    query_block = pt_head.query_features_block
    copy_mlp_block(query_block, keras_model, "query_features_block")
    copy_query_projection(pt_head, keras_model)
    # Scalar bias lives in the EinsumDense (1,) kernel; assign un-transposed
    keras_model.get_layer("bias").kernel.assign(pt_head.bias.data.cpu().numpy())
