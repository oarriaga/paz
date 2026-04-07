import jax
import jax.numpy as jp
import paz


def legacy_forward_differences(image):
    H, W, C = image.shape
    dy = image[1:, :, :] - image[:-1, :, :]
    dx = image[:, 1:, :] - image[:, :-1, :]
    dy = jp.concatenate([dy, jp.zeros((1, W, C))], axis=0)
    dx = jp.concatenate([dx, jp.zeros((H, 1, C))], axis=1)
    return dy, dx


def legacy_guided_smoothing(true_depth, pred_depth):
    dy_true, dx_true = legacy_forward_differences(true_depth)
    dy_pred, dx_pred = legacy_forward_differences(pred_depth)
    x_weight = jp.exp(jp.mean(jp.abs(dx_true)))
    y_weight = jp.exp(jp.mean(jp.abs(dy_true)))
    x_smooth = jp.mean(jp.abs(dx_pred * x_weight))
    y_smooth = jp.mean(jp.abs(dy_pred * y_weight))
    return x_smooth + y_smooth


def test_guided_smoothing_matches_example_formula():
    true_depth = jp.array([[[1.0], [2.0]], [[4.0], [8.0]]])
    pred_depth = jp.array([[[0.5], [1.5]], [[2.0], [4.0]]])
    result = paz.losses.depth.guided_smoothing(true_depth, pred_depth)
    expected = legacy_guided_smoothing(true_depth, pred_depth)
    assert jp.allclose(result, expected)


def test_guided_smoothing_batches_inputs():
    true_depth = jp.array([[[1.0], [2.0]], [[4.0], [8.0]]])
    pred_depth = jp.array([[[0.5], [1.5]], [[2.0], [4.0]]])
    true_batch = jp.stack((true_depth, true_depth * 2.0))
    pred_batch = jp.stack((pred_depth, pred_depth * 0.5))
    result = paz.losses.depth.guided_smoothing(
        true_batch, pred_batch, reduction="none")
    expected = jax.vmap(legacy_guided_smoothing)(true_batch, pred_batch)
    mean_value = paz.losses.depth.guided_smoothing(true_batch, pred_batch)
    total = paz.losses.depth.guided_smoothing(
        true_batch, pred_batch, reduction="sum")
    assert jp.allclose(result, expected)
    assert jp.allclose(mean_value, expected.mean())
    assert jp.allclose(total, expected.sum())
