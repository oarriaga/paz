import jax.numpy as jp
import paz


def test_masked_mae_single_mask():
    y_true = jp.array([[[1.0], [0.0]], [[0.0], [0.0]]])
    y_pred = jp.zeros_like(y_true)
    mask = jp.array([[1.0, 0.0], [0.0, 0.0]])
    result = paz.losses.masked_mae(y_true, y_pred, mask)
    assert jp.allclose(result, 0.25)


def test_masked_mae_sums_over_mask_stack():
    y_true = jp.array([[[1.0], [2.0]], [[0.0], [0.0]]])
    y_pred = jp.zeros_like(y_true)
    masks = jp.array(
        [
            [[[1.0], [0.0]], [[0.0], [0.0]]],
            [[[0.0], [1.0]], [[0.0], [0.0]]],
        ]
    )
    result = paz.losses.masked_mae(y_true, y_pred, masks)
    assert jp.allclose(result, 0.75)


def test_masked_mse():
    y_true = jp.array([[[2.0], [0.0]], [[0.0], [0.0]]])
    y_pred = jp.zeros_like(y_true)
    mask = jp.array([[1.0, 0.0], [0.0, 0.0]])
    result = paz.losses.masked_mse(y_true, y_pred, mask)
    assert jp.allclose(result, 1.0)


def test_masked_bce():
    y_true = jp.array([[[1.0], [0.0]], [[0.0], [0.0]]])
    y_pred = jp.array([[[0.5], [0.5]], [[0.5], [0.5]]])
    mask = jp.array([[1.0, 0.0], [0.0, 0.0]])
    result = paz.losses.masked_bce(y_true, y_pred, mask)
    expected = -jp.log(0.5) / 4.0
    assert jp.allclose(result, expected)


def test_soft_box_barrier_matches_example_formula():
    values = jp.array([-1.0, 0.5, 2.0])
    result = paz.losses.soft_box_barrier(values, 0.0, 1.0, 2.0)
    negative = 2.0 * (values - 0.0)
    positive = 2.0 * (values - 1.0)
    expected = (jp.exp(-negative) + jp.exp(positive)).sum()
    assert jp.allclose(result, expected)


def test_soft_half_quadratic_is_zero_for_satisfied_inequalities():
    values = jp.array([-2.0, 0.0, -0.1])
    result = paz.losses.soft_half_quadratic(values)
    assert jp.allclose(result, 0.0)


def test_soft_half_quadratic_penalizes_positive_violations():
    values = jp.array([-1.0, 2.0, 3.0])
    result = paz.losses.soft_half_quadratic(values)
    assert jp.allclose(result, 13.0)


def test_soft_half_quadratic_supports_axis_reduction():
    values = jp.array([[-1.0, 2.0], [3.0, -4.0]])
    result = paz.losses.soft_half_quadratic(values, axis=1, reduction="mean")
    expected = jp.array([2.0, 4.5])
    assert jp.allclose(result, expected)


def test_soft_half_quadratic_supports_none_reduction():
    values = jp.array([[-1.0, 2.0], [3.0, -4.0]])
    result = paz.losses.soft_half_quadratic(values, reduction="none")
    expected = jp.array([[0.0, 4.0], [9.0, 0.0]])
    assert jp.allclose(result, expected)


def test_soft_half_quadratic_matches_minimum_distance_violation():
    minimum_distance = 0.1
    distances = jp.array([0.3, 0.1, 0.0, -0.2])
    violations = minimum_distance - distances
    result = paz.losses.soft_half_quadratic(violations, reduction="none")
    expected = jp.array([0.0, 0.0, 0.01, 0.09])
    assert jp.allclose(result, expected)
