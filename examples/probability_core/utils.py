import jax
import jax.numpy as jp
from jax.scipy import special as jsp_special


LOG_PI = jp.log(jp.pi)
LOG_TWO_PI = jp.log(2.0 * jp.pi)
SQRT_TWO = jp.sqrt(2.0)


def common_dtype(*values):
    arrays = [jp.asarray(value) for value in values if value is not None]
    if len(arrays) == 0:
        return jp.float32
    dtype = jp.result_type(*arrays)
    if jp.issubdtype(dtype, jp.inexact):
        return dtype
    return jp.float32


def to_float(values, dtype=None):
    values = jp.asarray(values)
    if dtype is not None:
        return values.astype(dtype)
    if jp.issubdtype(values.dtype, jp.inexact):
        return values
    return values.astype(jp.float32)


def build_sample_shape(num_samples):
    if isinstance(num_samples, tuple):
        return num_samples
    if isinstance(num_samples, list):
        return tuple(num_samples)
    return (num_samples,)


def broadcast_shape(*values):
    values = [jp.asarray(value) for value in values if value is not None]
    if len(values) == 0:
        return ()
    return jp.broadcast_arrays(*values)[0].shape


def sum_rightmost(values, num_dims):
    if num_dims == 0:
        return values
    axes = tuple(range(values.ndim - num_dims, values.ndim))
    return values.sum(axis=axes)


def sum_event_dims(values, event_shape):
    return sum_rightmost(values, len(event_shape))


def event_size(event_shape):
    if len(event_shape) == 0:
        return 1
    return int(jp.prod(jp.asarray(event_shape)))


def multiply_no_nan(x, y):
    x = jp.asarray(x)
    y = jp.asarray(y, dtype=x.dtype)
    return jp.where(y == 0, jp.zeros_like(x), x * y)


def logit(values):
    values = to_float(values)
    return jp.log(values) - jp.log1p(-values)


def sigmoid(values):
    return jax.nn.sigmoid(values)


def is_integer(values):
    values = jp.asarray(values)
    return values == jp.floor(values)


def logsum_expbig_minus_expsmall(big, small):
    big = jp.asarray(big)
    small = jp.asarray(small, dtype=big.dtype)
    return big + jp.log1p(-jp.exp(small - big))


def normal_cdf(values):
    return jsp_special.ndtr(values)


def normal_log_cdf(values):
    return jsp_special.log_ndtr(values)


def normal_icdf(values):
    values = to_float(values)
    return SQRT_TWO * jsp_special.erfinv(2.0 * values - 1.0)


def normal_cdf_difference(high, low):
    is_low_positive = low >= 0
    high_hat = jp.where(is_low_positive, -low, high)
    low_hat = jp.where(is_low_positive, -high, low)
    return normal_cdf(high_hat) - normal_cdf(low_hat)


def normal_log_cdf_difference(high, low):
    is_low_positive = low >= 0
    high_hat = jp.where(is_low_positive, -low, high)
    low_hat = jp.where(is_low_positive, -high, low)
    return logsum_expbig_minus_expsmall(
        normal_log_cdf(high_hat), normal_log_cdf(low_hat)
    )


def softplus_inverse(values):
    values = to_float(values)
    threshold = jp.log(jp.finfo(values.dtype).eps) + 2.0
    use_small = values < jp.exp(threshold)
    use_large = values > -threshold
    small_values = jp.log(values)
    large_values = values
    safe_values = jp.where(use_small | use_large, 1.0, values)
    middle_values = safe_values + jp.log(-jp.expm1(-safe_values))
    return jp.where(
        use_small, small_values, jp.where(use_large, large_values, middle_values)
    )


def diag_matrix(diagonal):
    diagonal = jp.asarray(diagonal)
    size = diagonal.shape[-1]
    eye = jp.eye(size, dtype=diagonal.dtype)
    return diagonal[..., :, None] * eye


def wrap_angle(values):
    values = jp.asarray(values)
    return values - 2.0 * jp.pi * jp.round(values / (2.0 * jp.pi))


def log_prob_inverse(distribution, bijector, inverse_values):
    forward_values = bijector(inverse_values)
    event_ndims = len(distribution.event_shape)
    log_prob = distribution.log_prob(forward_values)
    log_det = bijector.forward_log_det_jacobian(
        inverse_values, event_ndims
    )
    return log_prob + log_det
