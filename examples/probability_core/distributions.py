from collections import namedtuple

import jax
import jax.numpy as jp
import numpy as np
from jax.scipy import linalg as jsp_linalg
from jax.scipy import special as jsp_special

from .utils import (
    LOG_PI,
    LOG_TWO_PI,
    broadcast_shape,
    build_sample_shape,
    common_dtype,
    diag_matrix,
    is_integer,
    logsum_expbig_minus_expsmall,
    multiply_no_nan,
    normal_cdf,
    normal_cdf_difference,
    normal_icdf,
    normal_log_cdf,
    normal_log_cdf_difference,
    sum_rightmost,
    to_float,
    wrap_angle,
)


class Distribution:
    @property
    def batch_shape(self):
        raise NotImplementedError

    @property
    def event_shape(self):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    def log_prob(self, values):
        raise NotImplementedError

    def prob(self, values):
        return jp.exp(self.log_prob(values))

    def cdf(self, values):
        raise NotImplementedError

    def log_cdf(self, values):
        return jp.log(self.cdf(values))

    def survival_function(self, values):
        return 1.0 - self.cdf(values)

    def log_survival_function(self, values):
        return jp.log(self.survival_function(values))

    def sample(self, num_samples=1, seed=None):
        raise NotImplementedError


_NormalBase = namedtuple("Normal", ["loc", "scale"])


class Normal(_NormalBase, Distribution):
    __slots__ = ()

    @property
    def dtype(self):
        return common_dtype(self.loc, self.scale)

    @property
    def batch_shape(self):
        return broadcast_shape(self.loc, self.scale)

    @property
    def event_shape(self):
        return ()

    def log_prob(self, values):
        values = to_float(values, self.dtype)
        half_log_two_pi = _half_log_two_pi(self.dtype)
        z = values / self.scale - self.loc / self.scale
        return -0.5 * z**2 - half_log_two_pi - jp.log(self.scale)

    def cdf(self, values):
        values = to_float(values, self.dtype)
        return normal_cdf((values - self.loc) / self.scale)

    def log_cdf(self, values):
        values = to_float(values, self.dtype)
        return normal_log_cdf((values - self.loc) / self.scale)

    def survival_function(self, values):
        return 1.0 - self.cdf(values)

    def log_survival_function(self, values):
        values = to_float(values, self.dtype)
        return normal_log_cdf(-(values - self.loc) / self.scale)

    def quantile(self, values):
        values = to_float(values, self.dtype)
        return self.loc + self.scale * normal_icdf(values)

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        shape = build_sample_shape(num_samples) + self.batch_shape
        noise = jax.random.normal(seed, shape, dtype=self.dtype)
        return self.loc + self.scale * noise


_UniformBase = namedtuple("Uniform", ["low", "high"])


class Uniform(_UniformBase, Distribution):
    __slots__ = ()

    @property
    def dtype(self):
        return common_dtype(self.low, self.high)

    @property
    def batch_shape(self):
        return broadcast_shape(self.low, self.high)

    @property
    def event_shape(self):
        return ()

    def log_prob(self, values):
        values = to_float(values, self.dtype)
        log_prob = -jp.log(self.high - self.low)
        is_inside = (values >= self.low) & (values <= self.high)
        return jp.where(is_inside, log_prob, -jp.inf)

    def cdf(self, values):
        values = to_float(values, self.dtype)
        unit_values = (values - self.low) / (self.high - self.low)
        return jp.clip(unit_values, 0.0, 1.0)

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        shape = build_sample_shape(num_samples) + self.batch_shape
        unit = jax.random.uniform(seed, shape, dtype=self.dtype)
        return self.low + (self.high - self.low) * unit


_DeterministicBase = namedtuple("Deterministic", ["loc"])


class Deterministic(_DeterministicBase, Distribution):
    __slots__ = ()

    @property
    def dtype(self):
        return jp.asarray(self.loc).dtype

    @property
    def batch_shape(self):
        return jp.asarray(self.loc).shape

    @property
    def event_shape(self):
        return ()

    def log_prob(self, values):
        values = jp.asarray(values)
        return jp.where(values == self.loc, 0.0, -jp.inf)

    def cdf(self, values):
        values = jp.asarray(values)
        dtype = common_dtype(values, self.loc)
        return jp.where(values < self.loc, 0.0, 1.0).astype(dtype)

    def sample(self, num_samples=1, seed=None):
        del seed
        shape = build_sample_shape(num_samples) + self.batch_shape
        return jp.broadcast_to(self.loc, shape)


_LaplaceBase = namedtuple("Laplace", ["loc", "scale"])


class Laplace(_LaplaceBase, Distribution):
    __slots__ = ()

    @property
    def dtype(self):
        return common_dtype(self.loc, self.scale)

    @property
    def batch_shape(self):
        return broadcast_shape(self.loc, self.scale)

    @property
    def event_shape(self):
        return ()

    def log_prob(self, values):
        values = to_float(values, self.dtype)
        z = (values - self.loc) / self.scale
        return -jp.abs(z) - jp.log(2.0) - jp.log(self.scale)

    def cdf(self, values):
        values = to_float(values, self.dtype)
        z = (values - self.loc) / self.scale
        lower = 0.5 * jp.exp(z)
        upper = 1.0 - 0.5 * jp.exp(-z)
        return jp.where(values < self.loc, lower, upper)

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        shape = build_sample_shape(num_samples) + self.batch_shape
        noise = _sample_laplace(shape, seed, self.dtype)
        return self.loc + self.scale * noise


_StudentTBase = namedtuple("StudentT", ["df", "loc", "scale"])


class StudentT(_StudentTBase, Distribution):
    __slots__ = ()

    @property
    def dtype(self):
        return common_dtype(self.df, self.loc, self.scale)

    @property
    def batch_shape(self):
        return broadcast_shape(self.df, self.loc, self.scale)

    @property
    def event_shape(self):
        return ()

    def log_prob(self, values):
        values = to_float(values, self.dtype)
        df = to_float(self.df, self.dtype)
        half = jp.asarray(0.5, dtype=self.dtype)
        y = (values - self.loc) * (jax.lax.rsqrt(df) / self.scale)
        log_unnorm = -half * (df + 1.0) * _log1p_square(y)
        log_norm = jp.log(jp.abs(self.scale)) + half * jp.log(df)
        log_norm = log_norm + jsp_special.betaln(half, half * df)
        return log_unnorm - log_norm

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        shape = build_sample_shape(num_samples) + self.batch_shape
        noise = jax.random.t(seed, self.df, shape=shape, dtype=self.dtype)
        return self.loc + self.scale * noise


_LogNormalBase = namedtuple("LogNormal", ["loc", "scale"])


class LogNormal(_LogNormalBase, Distribution):
    __slots__ = ()

    @property
    def dtype(self):
        return common_dtype(self.loc, self.scale)

    @property
    def batch_shape(self):
        return broadcast_shape(self.loc, self.scale)

    @property
    def event_shape(self):
        return ()

    def log_prob(self, values):
        values = to_float(values, self.dtype)
        safe_values = jp.where(values == 0.0, 1.0, values)
        base = Normal(self.loc, self.scale)
        log_prob = base.log_prob(jp.log(safe_values)) - jp.log(safe_values)
        return jp.where(values == 0.0, -jp.inf, log_prob)

    def cdf(self, values):
        values = to_float(values, self.dtype)
        safe_values = jp.where(values == 0.0, 1.0, values)
        base = Normal(self.loc, self.scale)
        cdf = base.cdf(jp.log(safe_values))
        return jp.where(values == 0.0, 0.0, cdf)

    def log_cdf(self, values):
        values = to_float(values, self.dtype)
        safe_values = jp.where(values == 0.0, 1.0, values)
        base = Normal(self.loc, self.scale)
        log_cdf = base.log_cdf(jp.log(safe_values))
        return jp.where(values == 0.0, -jp.inf, log_cdf)

    def sample(self, num_samples=1, seed=None):
        base = Normal(self.loc, self.scale)
        return jp.exp(base.sample(num_samples, seed=seed))


_TruncatedNormalBase = namedtuple(
    "TruncatedNormal", ["loc", "scale", "low", "high"]
)


class TruncatedNormal(_TruncatedNormalBase, Distribution):
    __slots__ = ()

    @property
    def dtype(self):
        return common_dtype(self.loc, self.scale, self.low, self.high)

    @property
    def batch_shape(self):
        return broadcast_shape(self.loc, self.scale, self.low, self.high)

    @property
    def event_shape(self):
        return ()

    def log_prob(self, values):
        values = to_float(values, self.dtype)
        std_low, std_high = _standardized_bounds(
            self.loc, self.scale, self.low, self.high
        )
        log_norm = normal_log_cdf_difference(std_high, std_low)
        half_log_two_pi = _half_log_two_pi(self.dtype)
        z = values / self.scale - self.loc / self.scale
        log_prob = -0.5 * z**2 - half_log_two_pi
        log_prob = log_prob - jp.log(self.scale) - log_norm
        is_outside = (values > self.high) | (values < self.low)
        return jp.where(is_outside, -jp.inf, log_prob)

    def cdf(self, values):
        values = to_float(values, self.dtype)
        std_low, std_high = _standardized_bounds(
            self.loc, self.scale, self.low, self.high
        )
        z = (values - self.loc) / self.scale
        numerator = normal_cdf_difference(jp.minimum(z, std_high), std_low)
        denominator = normal_cdf_difference(std_high, std_low)
        cdf = numerator / denominator
        cdf = jp.where(values < self.low, 0.0, cdf)
        return jp.where(values >= self.high, 1.0, cdf)

    def log_cdf(self, values):
        values = to_float(values, self.dtype)
        std_low, std_high = _standardized_bounds(
            self.loc, self.scale, self.low, self.high
        )
        z = (values - self.loc) / self.scale
        log_num = normal_log_cdf_difference(jp.minimum(z, std_high), std_low)
        log_den = normal_log_cdf_difference(std_high, std_low)
        log_cdf = log_num - log_den
        log_cdf = jp.where(values < self.low, -jp.inf, log_cdf)
        return jp.where(values >= self.high, 0.0, log_cdf)

    def survival_function(self, values):
        values = to_float(values, self.dtype)
        std_low, std_high = _standardized_bounds(
            self.loc, self.scale, self.low, self.high
        )
        z = (values - self.loc) / self.scale
        numerator = normal_cdf_difference(std_high, jp.maximum(z, std_low))
        denominator = normal_cdf_difference(std_high, std_low)
        sf = numerator / denominator
        sf = jp.where(values < self.low, 1.0, sf)
        return jp.where(values >= self.high, 0.0, sf)

    def log_survival_function(self, values):
        return jp.log(self.survival_function(values))

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        std_low, std_high = _standardized_bounds(
            self.loc, self.scale, self.low, self.high
        )
        low_cdf = normal_cdf(std_low)
        high_cdf = normal_cdf(std_high)
        shape = build_sample_shape(num_samples) + self.batch_shape
        uniforms = jax.random.uniform(
            seed, shape, dtype=self.dtype, minval=low_cdf, maxval=high_cdf
        )
        return self.loc + self.scale * normal_icdf(uniforms)


_BetaBase = namedtuple(
    "Beta", ["concentration1", "concentration0", "force_probs"]
)


class Beta(_BetaBase, Distribution):
    __slots__ = ()

    def __new__(
        cls,
        concentration1,
        concentration0,
        force_probs_to_zero_outside_support=False,
    ):
        return super().__new__(
            cls,
            concentration1,
            concentration0,
            bool(force_probs_to_zero_outside_support),
        )

    @property
    def dtype(self):
        return common_dtype(self.concentration1, self.concentration0)

    @property
    def batch_shape(self):
        return broadcast_shape(self.concentration1, self.concentration0)

    @property
    def event_shape(self):
        return ()

    def log_prob(self, values):
        values = to_float(values, self.dtype)
        alpha = to_float(self.concentration1, self.dtype)
        beta = to_float(self.concentration0, self.dtype)
        log_prob = (
            (alpha - 1.0) * jp.log(values)
            + (beta - 1.0) * jp.log1p(-values)
            - jsp_special.betaln(alpha, beta)
        )
        if self.force_probs:
            is_inside = (values >= 0.0) & (values <= 1.0)
            return jp.where(is_inside, log_prob, -jp.inf)
        return log_prob

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        shape = build_sample_shape(num_samples) + self.batch_shape
        return jax.random.beta(
            seed,
            a=self.concentration1,
            b=self.concentration0,
            shape=shape,
            dtype=self.dtype,
        )


_VonMisesBase = namedtuple("VonMises", ["loc", "concentration"])


class VonMises(_VonMisesBase, Distribution):
    __slots__ = ()

    @property
    def dtype(self):
        return common_dtype(self.loc, self.concentration)

    @property
    def batch_shape(self):
        return broadcast_shape(self.loc, self.concentration)

    @property
    def event_shape(self):
        return ()

    def log_prob(self, values):
        values = to_float(values, self.dtype)
        concentration = to_float(self.concentration, self.dtype)
        z = values - self.loc
        log_norm = jp.log(2.0 * jp.pi) + jp.log(jsp_special.i0e(concentration))
        log_prob = concentration * _cos_minus_one(z) - log_norm
        return log_prob

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        tiny = jp.finfo(self.dtype).tiny
        concentration = jp.maximum(to_float(self.concentration, self.dtype), tiny)
        shape = build_sample_shape(num_samples) + self.batch_shape
        samples = _sample_von_mises(shape, concentration, seed)
        return wrap_angle(samples + self.loc)


_BernoulliBase = namedtuple("Bernoulli", ["logits", "probs", "dtype"])


class Bernoulli(_BernoulliBase, Distribution):
    __slots__ = ()

    def __new__(cls, logits=None, probs=None, dtype=jp.int32):
        if (probs is None) == (logits is None):
            raise ValueError("Must pass probs or logits, but not both.")
        return super().__new__(cls, logits, probs, dtype)

    @property
    def batch_shape(self):
        params = self.probs if self.logits is None else self.logits
        return jp.asarray(params).shape

    @property
    def event_shape(self):
        return ()

    @property
    def parameter_dtype(self):
        return common_dtype(self.logits, self.probs)

    def logits_parameter(self):
        if self.logits is None:
            probs = to_float(self.probs, self.parameter_dtype)
            return jp.log(probs) - jp.log1p(-probs)
        return to_float(self.logits, self.parameter_dtype)

    def probs_parameter(self):
        if self.logits is None:
            return to_float(self.probs, self.parameter_dtype)
        return jax.nn.sigmoid(self.logits)

    def log_prob(self, values):
        values = to_float(values, self.parameter_dtype)
        log_probs0, log_probs1 = _bernoulli_log_probs(self.logits, self.probs)
        return multiply_no_nan(log_probs0, 1.0 - values) + multiply_no_nan(
            log_probs1, values
        )

    def cdf(self, values):
        values = jp.asarray(values)
        probs = self.probs_parameter()
        dtype = probs.dtype
        return jp.where(values < 0, 0.0, jp.where(values < 1, 1.0 - probs, 1.0))

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        shape = build_sample_shape(num_samples) + self.batch_shape
        draws = jax.random.bernoulli(
            seed, self.probs_parameter(), shape=shape
        )
        return draws.astype(self.dtype)


_CategoricalBase = namedtuple(
    "Categorical", ["logits", "probs", "dtype", "force_probs"]
)


class Categorical(_CategoricalBase, Distribution):
    __slots__ = ()

    def __new__(
        cls,
        logits=None,
        probs=None,
        dtype=jp.int32,
        force_probs_to_zero_outside_support=False,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("Must pass probs or logits, but not both.")
        return super().__new__(
            cls, logits, probs, dtype, bool(force_probs_to_zero_outside_support)
        )

    @property
    def batch_shape(self):
        params = self.probs if self.logits is None else self.logits
        return jp.asarray(params).shape[:-1]

    @property
    def event_shape(self):
        return ()

    @property
    def parameter_dtype(self):
        return common_dtype(self.logits, self.probs)

    def logits_parameter(self):
        if self.logits is None:
            return jp.log(to_float(self.probs, self.parameter_dtype))
        return to_float(self.logits, self.parameter_dtype)

    def probs_parameter(self):
        if self.logits is None:
            return to_float(self.probs, self.parameter_dtype)
        return jax.nn.softmax(self.logits, axis=-1)

    def log_prob(self, values):
        logits = self.logits_parameter()
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        values = jp.asarray(values)
        safe_values = values.astype(jp.int32)
        batch_shape = jp.broadcast_shapes(values.shape, logits.shape[:-1])
        safe_values = jp.broadcast_to(safe_values, batch_shape)
        values = jp.broadcast_to(values, batch_shape)
        log_probs = jp.broadcast_to(log_probs, batch_shape + log_probs.shape[-1:])
        num_categories = logits.shape[-1]
        clipped = jp.clip(safe_values, 0, num_categories - 1)
        gathered = jp.take_along_axis(
            log_probs, clipped[..., None], axis=-1
        )[..., 0]
        if not self.force_probs:
            return gathered
        in_support = (safe_values >= 0) & (safe_values < num_categories)
        if jp.issubdtype(values.dtype, jp.inexact):
            in_support = in_support & (values == safe_values.astype(values.dtype))
        return jp.where(in_support, gathered, -jp.inf)

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        logits = self.logits_parameter()
        shape = build_sample_shape(num_samples) + self.batch_shape
        samples = jax.random.categorical(seed, logits, axis=-1, shape=shape)
        return samples.astype(self.dtype)


_PoissonBase = namedtuple("Poisson", ["rate", "log_rate", "force_probs"])


class Poisson(_PoissonBase, Distribution):
    __slots__ = ()

    def __new__(
        cls, rate=None, log_rate=None, force_probs_to_zero_outside_support=False
    ):
        if (rate is None) == (log_rate is None):
            raise ValueError("Must specify exactly one of rate and log_rate.")
        return super().__new__(
            cls, rate, log_rate, bool(force_probs_to_zero_outside_support)
        )

    @property
    def dtype(self):
        return common_dtype(self.rate, self.log_rate)

    @property
    def batch_shape(self):
        return broadcast_shape(self.rate, self.log_rate)

    @property
    def event_shape(self):
        return ()

    def log_rate_parameter(self):
        if self.log_rate is None:
            return jp.log(to_float(self.rate, self.dtype))
        return to_float(self.log_rate, self.dtype)

    def rate_parameter(self):
        if self.rate is None:
            return jp.exp(to_float(self.log_rate, self.dtype))
        return to_float(self.rate, self.dtype)

    def log_prob(self, values):
        values = to_float(values, self.dtype)
        log_rate = self.log_rate_parameter()
        safe_values = jp.maximum(
            jp.floor(values) if self.force_probs else values, 0.0
        )
        log_prob = multiply_no_nan(log_rate, safe_values)
        log_prob = log_prob - jsp_special.gammaln(1.0 + safe_values)
        log_prob = jp.where(values == safe_values, log_prob, -jp.inf)
        log_prob = log_prob - jp.exp(log_rate)
        if self.force_probs:
            log_prob = jp.where(jp.isinf(log_prob), -jp.inf, log_prob)
        return log_prob

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        shape = build_sample_shape(num_samples) + self.batch_shape
        samples = jax.random.poisson(seed, self.rate_parameter(), shape=shape)
        return samples.astype(self.dtype)


_IndependentBase = namedtuple(
    "Independent", ["distribution", "reinterpreted_batch_ndims"]
)


class Independent(_IndependentBase, Distribution):
    __slots__ = ()

    def __new__(cls, distribution, reinterpreted_batch_ndims):
        return super().__new__(cls, distribution, int(reinterpreted_batch_ndims))

    @property
    def dtype(self):
        return self.distribution.dtype

    @property
    def batch_shape(self):
        if self.reinterpreted_batch_ndims == 0:
            return self.distribution.batch_shape
        return self.distribution.batch_shape[: -self.reinterpreted_batch_ndims]

    @property
    def event_shape(self):
        if self.reinterpreted_batch_ndims == 0:
            return self.distribution.event_shape
        event_batch = self.distribution.batch_shape[-self.reinterpreted_batch_ndims :]
        return event_batch + self.distribution.event_shape

    def log_prob(self, values):
        log_prob = self.distribution.log_prob(values)
        return sum_rightmost(log_prob, self.reinterpreted_batch_ndims)

    def sample(self, num_samples=1, seed=None):
        return self.distribution.sample(num_samples, seed=seed)


_TransformedBase = namedtuple(
    "TransformedDistribution", ["distribution", "bijector"]
)


class TransformedDistribution(_TransformedBase, Distribution):
    __slots__ = ()

    @property
    def dtype(self):
        return self.distribution.dtype

    @property
    def batch_shape(self):
        return self.distribution.batch_shape

    @property
    def event_shape(self):
        return self.bijector.forward_event_shape(self.distribution.event_shape)

    def log_prob(self, values):
        inverse_values = self.bijector.inverse(values)
        event_ndims = len(self.event_shape)
        log_prob = self.distribution.log_prob(inverse_values)
        log_det = self.bijector.forward_log_det_jacobian(
            inverse_values, event_ndims
        )
        return log_prob - log_det

    def cdf(self, values):
        if self.event_shape != ():
            raise NotImplementedError("cdf is only implemented for scalars.")
        inverse_values = self.bijector.inverse(values)
        if self.bijector.is_increasing():
            return self.distribution.cdf(inverse_values)
        return self.distribution.survival_function(inverse_values)

    def log_cdf(self, values):
        if self.event_shape != ():
            raise NotImplementedError("log_cdf is only implemented for scalars.")
        inverse_values = self.bijector.inverse(values)
        if self.bijector.is_increasing():
            return self.distribution.log_cdf(inverse_values)
        return self.distribution.log_survival_function(inverse_values)

    def survival_function(self, values):
        if self.event_shape != ():
            raise NotImplementedError("sf is only implemented for scalars.")
        inverse_values = self.bijector.inverse(values)
        if self.bijector.is_increasing():
            return self.distribution.survival_function(inverse_values)
        return self.distribution.cdf(inverse_values)

    def log_survival_function(self, values):
        if self.event_shape != ():
            raise NotImplementedError("log_sf is only implemented for scalars.")
        inverse_values = self.bijector.inverse(values)
        if self.bijector.is_increasing():
            return self.distribution.log_survival_function(inverse_values)
        return self.distribution.log_cdf(inverse_values)

    def sample(self, num_samples=1, seed=None):
        values = self.distribution.sample(num_samples, seed=seed)
        return self.bijector(values)


_RelaxedBase = namedtuple(
    "RelaxedOneHotCategorical", ["temperature", "logits", "probs"]
)


class RelaxedOneHotCategorical(_RelaxedBase, Distribution):
    __slots__ = ()

    def __new__(cls, temperature, logits=None, probs=None):
        if (probs is None) == (logits is None):
            raise ValueError("Must pass probs or logits, but not both.")
        return super().__new__(cls, temperature, logits, probs)

    @property
    def dtype(self):
        return common_dtype(self.temperature, self.logits, self.probs)

    @property
    def batch_shape(self):
        params = self.logits if self.logits is not None else self.probs
        return jp.broadcast_shapes(
            jp.asarray(self.temperature).shape, jp.asarray(params).shape[:-1]
        )

    @property
    def event_shape(self):
        params = self.logits if self.logits is not None else self.probs
        return (jp.asarray(params).shape[-1],)

    def logits_parameter(self):
        if self.logits is None:
            return jp.log(to_float(self.probs, self.dtype))
        return to_float(self.logits, self.dtype)

    def probs_parameter(self):
        if self.logits is None:
            return to_float(self.probs, self.dtype)
        return jax.nn.softmax(self.logits, axis=-1)

    def log_prob(self, values):
        values = to_float(values, self.dtype)
        logits = self.logits_parameter()
        temperature = to_float(self.temperature, self.dtype)
        log_values = jp.log(values)
        logits = jp.broadcast_to(logits, values.shape)
        log_values = jp.broadcast_to(log_values, logits.shape)
        event_size = logits.shape[-1]
        log_norm = jsp_special.gammaln(event_size)
        log_norm = log_norm + (event_size - 1.0) * jp.log(temperature)
        log_softmax = jax.nn.log_softmax(
            logits - log_values * temperature[..., None], axis=-1
        )
        return log_norm + log_softmax.sum(axis=-1) - log_values.sum(axis=-1)

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        logits = self.logits_parameter()
        temperature = to_float(self.temperature, self.dtype)
        shape = (
            build_sample_shape(num_samples) + self.batch_shape + self.event_shape
        )
        tiny = jp.finfo(self.dtype).tiny
        uniforms = jax.random.uniform(
            seed, shape, dtype=self.dtype, minval=tiny, maxval=1.0
        )
        gumbels = -jp.log(-jp.log(uniforms))
        noisy_logits = (gumbels + logits) / temperature[..., None]
        return jax.nn.softmax(noisy_logits, axis=-1)


_MVNDiagBase = namedtuple("MultivariateNormalDiag", ["loc", "scale_diag"])


class MultivariateNormalDiag(_MVNDiagBase, Distribution):
    __slots__ = ()

    @property
    def dtype(self):
        return common_dtype(self.loc, self.scale_diag)

    @property
    def batch_shape(self):
        return broadcast_shape(self.loc, self.scale_diag)[:-1]

    @property
    def event_shape(self):
        return broadcast_shape(self.loc, self.scale_diag)[-1:]

    def log_prob(self, values):
        values = to_float(values, self.dtype)
        normal = Normal(self.loc, self.scale_diag)
        return normal.log_prob(values).sum(axis=-1)

    def variance(self):
        return jp.square(self.scale_diag)

    def covariance(self):
        return diag_matrix(self.variance())

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        shape = build_sample_shape(num_samples) + self.batch_shape + self.event_shape
        noise = jax.random.normal(seed, shape, dtype=self.dtype)
        return self.loc + self.scale_diag * noise


_MVNFullBase = namedtuple(
    "MultivariateNormalFullCovariance", ["loc", "covariance_matrix"]
)


class MultivariateNormalFullCovariance(_MVNFullBase, Distribution):
    __slots__ = ()

    @property
    def dtype(self):
        return common_dtype(self.loc, self.covariance_matrix)

    @property
    def batch_shape(self):
        loc_shape = jp.asarray(self.loc).shape[:-1]
        cov_shape = jp.asarray(self.covariance_matrix).shape[:-2]
        return jp.broadcast_shapes(loc_shape, cov_shape)

    @property
    def event_shape(self):
        return (jp.asarray(self.covariance_matrix).shape[-1],)

    def covariance(self):
        return jp.asarray(self.covariance_matrix)

    def log_prob(self, values):
        values = to_float(values, self.dtype)
        covariance = to_float(self.covariance_matrix, self.dtype)
        chol = jp.linalg.cholesky(covariance)
        diff = values - self.loc
        solved = jsp_linalg.solve_triangular(
            chol, diff[..., None], lower=True
        )[..., 0]
        standard = Normal(
            jp.asarray(0.0, dtype=self.dtype),
            jp.asarray(1.0, dtype=self.dtype),
        )
        log_prob = standard.log_prob(solved).sum(axis=-1)
        log_det = jp.log(jp.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=-1)
        return log_prob - log_det

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        covariance = to_float(self.covariance_matrix, self.dtype)
        chol = jp.linalg.cholesky(covariance)
        shape = build_sample_shape(num_samples) + self.batch_shape + self.event_shape
        noise = jax.random.normal(seed, shape, dtype=self.dtype)
        transformed = jp.einsum("...ij,...j->...i", chol, noise)
        return self.loc + transformed


_MixtureBase = namedtuple(
    "MixtureSameFamily",
    ["mixture_distribution", "components_distribution"],
)


class MixtureSameFamily(_MixtureBase, Distribution):
    __slots__ = ()

    @property
    def dtype(self):
        return self.components_distribution.dtype

    @property
    def batch_shape(self):
        component_shape = self.components_distribution.batch_shape[:-1]
        mixture_shape = self.mixture_distribution.batch_shape
        return jp.broadcast_shapes(component_shape, mixture_shape)

    @property
    def event_shape(self):
        return self.components_distribution.event_shape

    def log_prob(self, values):
        values = to_float(values, self.dtype)
        event_ndims = len(self.event_shape)
        axis = values.ndim - event_ndims if event_ndims else values.ndim
        expanded = jp.expand_dims(values, axis=axis)
        component_log_prob = self.components_distribution.log_prob(expanded)
        mixture_log_prob = jax.nn.log_softmax(
            self.mixture_distribution.logits_parameter(), axis=-1
        )
        return jsp_special.logsumexp(
            component_log_prob + mixture_log_prob, axis=-1
        )

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        component_seed, mixture_seed = jax.random.split(seed)
        component_samples = self.components_distribution.sample(
            num_samples, seed=component_seed
        )
        mixture_samples = self.mixture_distribution.sample(
            num_samples, seed=mixture_seed
        )
        num_components = self.components_distribution.batch_shape[-1]
        mask = jax.nn.one_hot(
            mixture_samples.astype(jp.int32),
            num_components,
            dtype=component_samples.dtype,
        )
        for _ in self.event_shape:
            mask = mask[..., None]
        axis = -(len(self.event_shape) + 1) if self.event_shape else -1
        return (component_samples * mask).sum(axis=axis)


_QuantizedBase = namedtuple(
    "QuantizedDistribution", ["distribution", "low", "high"]
)


class QuantizedDistribution(_QuantizedBase, Distribution):
    __slots__ = ()

    def __new__(cls, distribution, low=None, high=None):
        return super().__new__(cls, distribution, low, high)

    @property
    def dtype(self):
        return self.distribution.dtype

    @property
    def batch_shape(self):
        return self.distribution.batch_shape

    @property
    def event_shape(self):
        return self.distribution.event_shape

    def sample(self, num_samples=1, seed=None):
        values = self.distribution.sample(num_samples, seed=seed)
        values = jp.ceil(values)
        if self.low is not None:
            values = jp.where(values < self.low, self.low, values)
        if self.high is not None:
            values = jp.where(values > self.high, self.high, values)
        return values

    def log_prob(self, values):
        values = jp.floor(to_float(values, self.dtype))
        return logsum_expbig_minus_expsmall(
            self.log_cdf(values), self.log_cdf(values - 1.0)
        )

    def prob(self, values):
        values = jp.floor(to_float(values, self.dtype))
        return self.cdf(values) - self.cdf(values - 1.0)

    def cdf(self, values):
        values = jp.floor(to_float(values, self.dtype))
        cdf = self.distribution.cdf(values)
        if self.low is not None:
            cdf = jp.where(values < self.low, 0.0, cdf)
        if self.high is not None:
            cdf = jp.where(values < self.high, cdf, 1.0)
        return cdf

    def log_cdf(self, values):
        values = jp.floor(to_float(values, self.dtype))
        log_cdf = self.distribution.log_cdf(values)
        if self.low is not None:
            log_cdf = jp.where(values < self.low, -jp.inf, log_cdf)
        if self.high is not None:
            log_cdf = jp.where(values < self.high, log_cdf, 0.0)
        return log_cdf


def _require_seed(seed):
    if seed is None:
        raise ValueError("seed is required for sampling.")
    return seed


def _standardized_bounds(loc, scale, low, high):
    return (low - loc) / scale, (high - loc) / scale


def _bernoulli_log_probs(logits, probs):
    if logits is None:
        probs = to_float(probs)
        return jp.log1p(-probs), jp.log(probs)
    logits = to_float(logits)
    return -jax.nn.softplus(logits), -jax.nn.softplus(-logits)


def _cos_minus_one(values):
    return -2.0 * jp.square(jp.sin(values / 2.0))


def _sample_von_mises(shape, concentration, seed):
    concentration = jp.broadcast_to(concentration, shape)
    r = 1.0 + jp.sqrt(1.0 + 4.0 * concentration**2)
    rho = (r - jp.sqrt(2.0 * r)) / (2.0 * concentration)
    s_exact = (1.0 + rho**2) / (2.0 * rho)
    if concentration.dtype == jp.float16:
        cutoff = 1.8e-1
    elif concentration.dtype == jp.float32:
        cutoff = 2.0e-2
    else:
        cutoff = 1.2e-4
    s = jp.where(concentration > cutoff, s_exact, 1.0 / concentration)

    def cond(state):
        done, _, _, _, step = state
        return (~jp.all(done)) & (step < 100)

    def body(state):
        done, old_u, old_w, key, step = state
        u_key, v_key, next_key = jax.random.split(key, 3)
        u = jax.random.uniform(
            u_key, shape, dtype=concentration.dtype, minval=-1.0, maxval=1.0
        )
        z = jp.cos(jp.pi * u)
        w = jp.where(done, old_w, (1.0 + s * z) / (s + z))
        y = concentration * (s - w)
        v = jax.random.uniform(v_key, shape, dtype=concentration.dtype)
        accept = (y * (2.0 - y) >= v) | (jp.log(y / v) + 1.0 >= y)
        return done | accept, jp.where(done, old_u, u), w, next_key, step + 1

    init = (
        jp.zeros(shape, dtype=bool),
        jp.zeros(shape, dtype=concentration.dtype),
        jp.zeros(shape, dtype=concentration.dtype),
        seed,
        0,
    )
    _, u, w, _, _ = jax.lax.while_loop(cond, body, init)
    return jp.sign(u) * jp.arccos(w)


def _sample_laplace(shape, seed, dtype):
    uniforms = jax.random.uniform(
        seed, shape, dtype=dtype, minval=-1.0, maxval=1.0
    )
    return -jp.sign(uniforms) * jp.log1p(-jp.abs(uniforms))


def _half_log_two_pi(dtype):
    return jp.asarray(0.5 * np.log(2.0 * np.pi), dtype=dtype)


def _log1p_square(values):
    values = jp.asarray(values)
    eps = jp.asarray(np.finfo(values.dtype).eps, dtype=values.dtype)
    is_large = jp.abs(values) > jp.power(eps, -0.5)
    safe_values = jp.where(is_large, jp.abs(values), 1.0)
    large_values = 2.0 * jp.log(safe_values)
    small_values = jp.log1p(values**2)
    return jp.where(is_large, large_values, small_values)
