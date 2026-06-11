from collections import namedtuple

import jax
import jax.numpy as jp

from .utils import event_size, logit, sigmoid, softplus_inverse, to_float


class Bijector:
    def __call__(self, values):
        raise NotImplementedError

    def inverse(self, values):
        raise NotImplementedError

    def forward_log_det_jacobian(self, values, event_ndims=0):
        raise NotImplementedError

    def inverse_log_det_jacobian(self, values, event_ndims=0):
        inverse_values = self.inverse(values)
        return -self.forward_log_det_jacobian(inverse_values, event_ndims)

    def forward_event_shape(self, event_shape):
        return event_shape

    def inverse_event_shape(self, event_shape):
        return event_shape

    def is_increasing(self):
        return True


_IdentityBase = namedtuple("Identity", [])


class Identity(_IdentityBase, Bijector):
    __slots__ = ()

    def __call__(self, values):
        return values

    def inverse(self, values):
        return values

    def forward_log_det_jacobian(self, values, event_ndims=0):
        dtype = to_float(values).dtype
        return jp.array(0.0, dtype=dtype)


_ShiftBase = namedtuple("Shift", ["shift"])


class Shift(_ShiftBase, Bijector):
    __slots__ = ()

    def __call__(self, values):
        return values + self.shift

    def inverse(self, values):
        return values - self.shift

    def forward_log_det_jacobian(self, values, event_ndims=0):
        dtype = to_float(values).dtype
        return jp.array(0.0, dtype=dtype)


_ScaleBase = namedtuple("Scale", ["scale"])


class Scale(_ScaleBase, Bijector):
    __slots__ = ()

    def __call__(self, values):
        return values * self.scale

    def inverse(self, values):
        return values / self.scale

    def forward_log_det_jacobian(self, values, event_ndims=0):
        dtype = to_float(values).dtype
        scale = jp.asarray(self.scale)
        log_scale = jp.log(jp.abs(scale))
        if event_ndims == 0:
            zeros = jp.zeros(scale.shape, dtype=dtype)
            return zeros + log_scale
        if scale.ndim >= event_ndims:
            zeros = jp.zeros(scale.shape, dtype=dtype)
            return zeros[..., 0] + log_scale.sum(
                axis=tuple(range(scale.ndim - event_ndims, scale.ndim))
            )
        multiplier = event_size(to_float(values).shape[-event_ndims:])
        return multiplier * log_scale.sum()

    def is_increasing(self):
        return bool(jp.all(jp.asarray(self.scale) > 0))


_SigmoidBase = namedtuple("Sigmoid", ["low", "high"])


class Sigmoid(_SigmoidBase, Bijector):
    __slots__ = ()

    def __new__(cls, low=None, high=None):
        if (low is None) != (high is None):
            raise ValueError("low and high must be both set or both None.")
        return super().__new__(cls, low, high)

    def __call__(self, values):
        values = to_float(values)
        outputs = sigmoid(values)
        if self.low is None:
            return outputs
        return self.low + (self.high - self.low) * outputs

    def inverse(self, values):
        values = to_float(values)
        if self.low is None:
            return logit(values)
        unit_values = (values - self.low) / (self.high - self.low)
        return logit(unit_values)

    def forward_log_det_jacobian(self, values, event_ndims=0):
        values = to_float(values)
        log_det = -jax.nn.softplus(-values) - jax.nn.softplus(values)
        if self.low is None:
            return log_det if event_ndims == 0 else log_det.sum(
                axis=tuple(range(log_det.ndim - event_ndims, log_det.ndim))
            )
        log_det = log_det + jp.log(jp.abs(self.high - self.low))
        if event_ndims == 0:
            return log_det
        return log_det.sum(
            axis=tuple(range(log_det.ndim - event_ndims, log_det.ndim))
        )

    def is_increasing(self):
        if self.low is None:
            return True
        return bool(jp.all(jp.asarray(self.high) > jp.asarray(self.low)))


_ExpBase = namedtuple("Exp", [])


class Exp(_ExpBase, Bijector):
    __slots__ = ()

    def __call__(self, values):
        return jp.exp(values)

    def inverse(self, values):
        return jp.log(values)

    def forward_log_det_jacobian(self, values, event_ndims=0):
        values = to_float(values)
        if event_ndims == 0:
            return values
        return values.sum(
            axis=tuple(range(values.ndim - event_ndims, values.ndim))
        )

    def inverse_log_det_jacobian(self, values, event_ndims=0):
        values = to_float(values)
        log_det = -jp.log(values)
        if event_ndims == 0:
            return log_det
        return log_det.sum(
            axis=tuple(range(log_det.ndim - event_ndims, log_det.ndim))
        )


_SoftplusBase = namedtuple("Softplus", ["hinge_softness", "low"])


class Softplus(_SoftplusBase, Bijector):
    __slots__ = ()

    def __new__(cls, hinge_softness=None, low=None):
        return super().__new__(cls, hinge_softness, low)

    def __call__(self, values):
        values = to_float(values)
        if self.hinge_softness is None:
            outputs = jax.nn.softplus(values)
        else:
            softness = to_float(self.hinge_softness, values.dtype)
            outputs = softness * jax.nn.softplus(values / softness)
        if self.low is None:
            return outputs
        return outputs + self.low

    def inverse(self, values):
        values = to_float(values)
        if self.low is not None:
            values = values - self.low
        if self.hinge_softness is None:
            return softplus_inverse(values)
        softness = to_float(self.hinge_softness, values.dtype)
        return softness * softplus_inverse(values / softness)

    def forward_log_det_jacobian(self, values, event_ndims=0):
        values = to_float(values)
        if self.hinge_softness is not None:
            softness = to_float(self.hinge_softness, values.dtype)
            values = values / softness
        log_det = -jax.nn.softplus(-values)
        if event_ndims == 0:
            return log_det
        return log_det.sum(
            axis=tuple(range(log_det.ndim - event_ndims, log_det.ndim))
        )

    def inverse_log_det_jacobian(self, values, event_ndims=0):
        values = to_float(values)
        if self.low is not None:
            values = values - self.low
        if self.hinge_softness is not None:
            softness = to_float(self.hinge_softness, values.dtype)
            values = values / softness
        log_det = -jp.log(-jp.expm1(-values))
        if event_ndims == 0:
            return log_det
        return log_det.sum(
            axis=tuple(range(log_det.ndim - event_ndims, log_det.ndim))
        )


_SoftmaxCenteredBase = namedtuple("SoftmaxCentered", [])


class SoftmaxCentered(_SoftmaxCenteredBase, Bijector):
    __slots__ = ()

    def __call__(self, values):
        values = to_float(values)
        padded = jp.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, 1)])
        return jax.nn.softmax(padded, axis=-1)

    def inverse(self, values):
        values = to_float(values)
        log_values = jp.log(values)
        return log_values[..., :-1] - log_values[..., -1:]

    def forward_log_det_jacobian(self, values, event_ndims=1):
        values = to_float(values)
        size = values.shape[-1] + 1
        log_norm = jax.scipy.special.logsumexp(values, axis=-1)
        return (
            0.5 * jp.log(jp.asarray(size, dtype=values.dtype))
            + values.sum(axis=-1)
            - size * jax.nn.softplus(log_norm)
        )

    def inverse_log_det_jacobian(self, values, event_ndims=1):
        values = to_float(values)
        size = values.shape[-1]
        return -0.5 * jp.log(jp.asarray(size, dtype=values.dtype)) - jp.log(
            values
        ).sum(axis=-1)

    def forward_event_shape(self, event_shape):
        if len(event_shape) == 0:
            return (1,)
        return event_shape[:-1] + (event_shape[-1] + 1,)

    def inverse_event_shape(self, event_shape):
        if len(event_shape) == 0:
            return ()
        return event_shape[:-1] + (event_shape[-1] - 1,)


_InvertBase = namedtuple("Invert", ["bijector"])


class Invert(_InvertBase, Bijector):
    __slots__ = ()

    def __call__(self, values):
        return self.bijector.inverse(values)

    def inverse(self, values):
        return self.bijector(values)

    def forward_log_det_jacobian(self, values, event_ndims=0):
        return self.bijector.inverse_log_det_jacobian(values, event_ndims)

    def inverse_log_det_jacobian(self, values, event_ndims=0):
        return self.bijector.forward_log_det_jacobian(values, event_ndims)

    def forward_event_shape(self, event_shape):
        return self.bijector.inverse_event_shape(event_shape)

    def inverse_event_shape(self, event_shape):
        return self.bijector.forward_event_shape(event_shape)

    def is_increasing(self):
        return self.bijector.is_increasing()


_ChainBase = namedtuple("Chain", ["bijectors"])


class Chain(_ChainBase, Bijector):
    __slots__ = ()

    def __new__(cls, bijectors):
        return super().__new__(cls, tuple(bijectors))

    def __call__(self, values):
        for bijector in reversed(self.bijectors):
            values = bijector(values)
        return values

    def inverse(self, values):
        for bijector in self.bijectors:
            values = bijector.inverse(values)
        return values

    def forward_log_det_jacobian(self, values, event_ndims=0):
        values = to_float(values)
        log_det = jp.array(0.0, dtype=values.dtype)
        for bijector in reversed(self.bijectors):
            log_det = log_det + bijector.forward_log_det_jacobian(
                values, event_ndims
            )
            values = bijector(values)
        return log_det

    def inverse_log_det_jacobian(self, values, event_ndims=0):
        values = to_float(values)
        log_det = jp.array(0.0, dtype=values.dtype)
        for bijector in self.bijectors:
            log_det = log_det + bijector.inverse_log_det_jacobian(
                values, event_ndims
            )
            values = bijector.inverse(values)
        return log_det

    def forward_event_shape(self, event_shape):
        for bijector in reversed(self.bijectors):
            event_shape = bijector.forward_event_shape(event_shape)
        return event_shape

    def inverse_event_shape(self, event_shape):
        for bijector in self.bijectors:
            event_shape = bijector.inverse_event_shape(event_shape)
        return event_shape

    def is_increasing(self):
        direction = True
        for bijector in self.bijectors:
            direction = direction == bijector.is_increasing()
        return direction
