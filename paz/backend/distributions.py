from collections import namedtuple

import jax
import jax.numpy as jp
import numpy as np
from jax.scipy import linalg as jsp_linalg
from jax.scipy import special as jsp_special

from paz.backend.standard import (
    broadcast_shape,
    build_sample_shape,
    common_dtype,
    diag_matrix,
    multiply_no_nan,
    normal_cdf,
    normal_icdf,
    normal_log_cdf,
    sum_rightmost,
    to_float,
)


LOG_PI = jp.log(jp.pi)
LOG_TWO = jp.log(2.0)
SQRT_TWO = jp.sqrt(2.0)

# Minimax coefficients for the Stirling correction, matching TFP's
# log_gamma_correction (DiDonato and Morris 1988, eq. 32).
_LOG_GAMMA_CORRECTION = [
    0.833333333333333e-01,
    -0.277777777760991e-02,
    0.793650666825390e-03,
    -0.595202931351870e-03,
    0.837308034031215e-03,
    -0.165322962780713e-02,
]


def logsum_expbig_minus_expsmall(big, small):
    big = jp.asarray(big)
    small = jp.asarray(small, dtype=big.dtype)
    return big + jp.log1p(-jp.exp(small - big))


def normal_log_cdf_difference(high, low):
    is_low_positive = low >= 0
    high_hat = jp.where(is_low_positive, -low, high)
    low_hat = jp.where(is_low_positive, -high, low)
    return logsum_expbig_minus_expsmall(
        normal_log_cdf(high_hat), normal_log_cdf(low_hat)
    )


def wrap_angle(values):
    values = jp.asarray(values)
    return values - 2.0 * jp.pi * jp.round(values / (2.0 * jp.pi))


def log_gamma_correction(values):
    inverse = 1.0 / values
    inverse_squared = inverse * inverse
    accum = jp.asarray(_LOG_GAMMA_CORRECTION[5], values.dtype)
    for index in reversed(range(5)):
        coefficient = jp.asarray(_LOG_GAMMA_CORRECTION[index], values.dtype)
        accum = accum * inverse_squared + coefficient
    return accum * inverse


def log_gamma_difference(x, y):
    half = jp.asarray(0.5, y.dtype)
    cancelled = -(x + y - half) * jp.log1p(x / y) - x * jp.log(y) + x
    correction = log_gamma_correction(y) - log_gamma_correction(x + y)
    return correction + cancelled


@jax.custom_jvp
def lbeta(x, y):
    return _lbeta_forward(x, y)


def _lbeta_forward(x, y):
    low = jp.minimum(x, y)
    high = jp.maximum(x, y)
    half = jp.asarray(0.5, low.dtype)
    log_two_pi = jp.asarray(jp.log(2.0 * jp.pi), low.dtype)
    two_large = (
        half * log_two_pi - half * jp.log(high)
        + log_gamma_correction(low) + log_gamma_correction(high)
        - log_gamma_correction(low + high)
        + (low - half) * jp.log(low / (low + high))
        - high * jp.log1p(low / high)
    )
    one_large = jsp_special.gammaln(low) + log_gamma_difference(low, high)
    small = (
        jsp_special.gammaln(low) + jsp_special.gammaln(high)
        - jsp_special.gammaln(low + high)
    )
    return jp.where(low >= 8, two_large, jp.where(high >= 8, one_large, small))


@lbeta.defjvp
def _lbeta_jvp(primals, tangents):
    x, y = primals
    x_dot, y_dot = tangents
    total = jsp_special.digamma(x + y)
    grad = (jsp_special.digamma(x) - total) * x_dot
    grad = grad + (jsp_special.digamma(y) - total) * y_dot
    return _lbeta_forward(x, y), grad


def log1mexp(values):
    values = jp.abs(values)
    return jp.where(
        values < LOG_TWO,
        jp.log(-jp.expm1(-values)),
        jp.log1p(-jp.exp(-values)),
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

    def mean(self):
        return _fill(self.loc, self.batch_shape, self.dtype)

    def mode(self):
        return self.mean()

    def variance(self):
        return _fill(jp.square(self.scale), self.batch_shape, self.dtype)

    def stddev(self):
        return _fill(jp.abs(self.scale), self.batch_shape, self.dtype)

    def entropy(self):
        entropy = 0.5 * jp.log(2.0 * jp.pi * jp.e) + jp.log(self.scale)
        return _fill(entropy, self.batch_shape, self.dtype)

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

    def mean(self):
        mean = (self.low + self.high) / 2.0
        return _fill(mean, self.batch_shape, self.dtype)

    def variance(self):
        variance = jp.square(self.high - self.low) / 12.0
        return _fill(variance, self.batch_shape, self.dtype)

    def stddev(self):
        return jp.sqrt(self.variance())

    def entropy(self):
        return _fill(jp.log(self.high - self.low), self.batch_shape, self.dtype)

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

    def mean(self):
        return _fill(self.loc, self.batch_shape, self.dtype)

    def mode(self):
        return self.mean()

    def variance(self):
        return jp.zeros(self.batch_shape, self.dtype)

    def stddev(self):
        return self.variance()

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

    def mean(self):
        return _fill(self.loc, self.batch_shape, self.dtype)

    def mode(self):
        return self.mean()

    def variance(self):
        return _fill(2.0 * jp.square(self.scale), self.batch_shape, self.dtype)

    def stddev(self):
        return _fill(SQRT_TWO * self.scale, self.batch_shape, self.dtype)

    def entropy(self):
        return _fill(jp.log(2.0 * self.scale) + 1.0, self.batch_shape,
                     self.dtype)

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
        log_norm = log_norm + lbeta(half, half * df)
        return log_unnorm - log_norm

    def mean(self):
        df = to_float(self.df, self.dtype)
        return _fill(jp.where(df > 1.0, self.loc, jp.nan), self.batch_shape,
                     self.dtype)

    def mode(self):
        return _fill(self.loc, self.batch_shape, self.dtype)

    def variance(self):
        df = to_float(self.df, self.dtype)
        scale = to_float(self.scale, self.dtype)
        denom = jp.where(df > 2.0, df - 2.0, 1.0)
        variance = jp.square(scale) * df / denom
        variance = jp.where(df > 2.0, variance, jp.inf)
        variance = jp.where(df > 1.0, variance, jp.nan)
        return _fill(variance, self.batch_shape, self.dtype)

    def stddev(self):
        return jp.sqrt(self.variance())

    def entropy(self):
        df = to_float(self.df, self.dtype)
        half = jp.asarray(0.5, self.dtype)
        entropy = jp.log(jp.abs(self.scale)) + 0.5 * jp.log(df)
        entropy = entropy + lbeta(half, half * df)
        digamma_diff = jsp_special.digamma(0.5 * (df + 1.0))
        digamma_diff = digamma_diff - jsp_special.digamma(0.5 * df)
        entropy = entropy + 0.5 * (df + 1.0) * digamma_diff
        return _fill(entropy, self.batch_shape, self.dtype)

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

    def mean(self):
        mean = jp.exp(self.loc + 0.5 * jp.square(self.scale))
        return _fill(mean, self.batch_shape, self.dtype)

    def mode(self):
        mode = jp.exp(self.loc - jp.square(self.scale))
        return _fill(mode, self.batch_shape, self.dtype)

    def variance(self):
        variance = jp.expm1(jp.square(self.scale))
        variance = variance * jp.exp(2.0 * self.loc + jp.square(self.scale))
        return _fill(variance, self.batch_shape, self.dtype)

    def stddev(self):
        return jp.sqrt(self.variance())

    def entropy(self):
        entropy = self.loc + 0.5 + 0.5 * jp.log(2.0 * jp.pi)
        entropy = entropy + jp.log(self.scale)
        return _fill(entropy, self.batch_shape, self.dtype)

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
        return jp.exp(self.log_cdf(values))

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
        return -jp.expm1(self.log_cdf(values))

    def log_survival_function(self, values):
        return log1mexp(self.log_cdf(values))

    def mean(self):
        std_low, std_high = _standardized_bounds(
            self.loc, self.scale, self.low, self.high
        )
        log_pdf_low = _normal_log_pdf(std_low)
        log_pdf_high = _normal_log_pdf(std_high)
        log_diff = _log_abs_sub_exp(log_pdf_low, log_pdf_high)
        sign = jp.where(log_pdf_low >= log_pdf_high, 1.0, -1.0)
        log_norm = normal_log_cdf_difference(std_high, std_low)
        mean = self.loc + self.scale * sign * jp.exp(log_diff - log_norm)
        return _fill(mean, self.batch_shape, self.dtype)

    def mode(self):
        loc = jp.broadcast_to(self.loc, self.batch_shape)
        return _fill(jp.clip(loc, self.low, self.high), self.batch_shape,
                     self.dtype)

    def variance(self):
        std_low, std_high = _standardized_bounds(
            self.loc, self.scale, self.low, self.high
        )
        log_norm = normal_log_cdf_difference(std_high, std_low)
        norm = jp.exp(log_norm)
        weighted = std_low * _normal_pdf(std_low)
        weighted = weighted - std_high * _normal_pdf(std_high)
        log_diff = _log_abs_sub_exp(
            _normal_log_pdf(std_low), _normal_log_pdf(std_high)
        )
        variance = jp.square(self.scale) * (
            1.0 + weighted / norm - jp.exp(2.0 * (log_diff - log_norm))
        )
        return _fill(variance, self.batch_shape, self.dtype)

    def stddev(self):
        return jp.sqrt(self.variance())

    def entropy(self):
        std_low, std_high = _standardized_bounds(
            self.loc, self.scale, self.low, self.high
        )
        log_norm = normal_log_cdf_difference(std_high, std_low)
        weighted = std_low * _normal_pdf(std_low)
        weighted = weighted - std_high * _normal_pdf(std_high)
        entropy = 0.5 * (1.0 + jp.log(2.0) + LOG_PI) + jp.log(self.scale)
        entropy = entropy + log_norm + 0.5 * weighted / jp.exp(log_norm)
        return _fill(entropy, self.batch_shape, self.dtype)

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
            - lbeta(alpha, beta)
        )
        if self.force_probs:
            is_inside = (values >= 0.0) & (values <= 1.0)
            return jp.where(is_inside, log_prob, -jp.inf)
        return log_prob

    def mean(self):
        alpha = to_float(self.concentration1, self.dtype)
        beta = to_float(self.concentration0, self.dtype)
        return _fill(alpha / (alpha + beta), self.batch_shape, self.dtype)

    def mode(self):
        alpha = to_float(self.concentration1, self.dtype)
        beta = to_float(self.concentration0, self.dtype)
        mode = (alpha - 1.0) / (alpha + beta - 2.0)
        is_peaked = (alpha > 1.0) & (beta > 1.0)
        return _fill(jp.where(is_peaked, mode, jp.nan), self.batch_shape,
                     self.dtype)

    def variance(self):
        alpha = to_float(self.concentration1, self.dtype)
        beta = to_float(self.concentration0, self.dtype)
        total = alpha + beta
        variance = alpha * beta / (jp.square(total) * (total + 1.0))
        return _fill(variance, self.batch_shape, self.dtype)

    def stddev(self):
        return jp.sqrt(self.variance())

    def entropy(self):
        alpha = to_float(self.concentration1, self.dtype)
        beta = to_float(self.concentration0, self.dtype)
        total = alpha + beta
        entropy = lbeta(alpha, beta)
        entropy = entropy - (alpha - 1.0) * jsp_special.digamma(alpha)
        entropy = entropy - (beta - 1.0) * jsp_special.digamma(beta)
        entropy = entropy + (total - 2.0) * jsp_special.digamma(total)
        return _fill(entropy, self.batch_shape, self.dtype)

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

    def mean(self):
        return _fill(self.loc, self.batch_shape, self.dtype)

    def mode(self):
        return self.mean()

    def variance(self):
        concentration = to_float(self.concentration, self.dtype)
        variance = 1.0 - jsp_special.i1e(concentration) / jsp_special.i0e(
            concentration
        )
        return _fill(variance, self.batch_shape, self.dtype)

    def entropy(self):
        concentration = to_float(self.concentration, self.dtype)
        i0e = jsp_special.i0e(concentration)
        i1e = jsp_special.i1e(concentration)
        entropy = concentration * (1.0 - i1e / i0e) + jp.log(i0e)
        entropy = entropy + jp.log(2.0 * jp.pi)
        return _fill(entropy, self.batch_shape, self.dtype)

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        tiny = jp.finfo(self.dtype).tiny
        concentration = to_float(self.concentration, self.dtype)
        concentration = jp.maximum(concentration, tiny)
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
        return jp.where(values < 0, 0.0, jp.where(values < 1, 1.0 - probs, 1.0))

    def mean(self):
        return self.probs_parameter()

    def mode(self):
        return (self.probs_parameter() > 0.5).astype(self.dtype)

    def variance(self):
        probs = self.probs_parameter()
        return probs * (1.0 - probs)

    def stddev(self):
        return jp.sqrt(self.variance())

    def entropy(self):
        probs = self.probs_parameter()
        log_probs0, log_probs1 = _bernoulli_log_probs(self.logits, self.probs)
        entropy = -multiply_no_nan(log_probs0, 1.0 - probs)
        return entropy - multiply_no_nan(log_probs1, probs)

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
        log_probs = jp.broadcast_to(
            log_probs, batch_shape + log_probs.shape[-1:]
        )
        num_categories = logits.shape[-1]
        mask = jax.nn.one_hot(
            safe_values, num_categories, dtype=log_probs.dtype
        )
        gathered = jp.sum(multiply_no_nan(log_probs, mask), axis=-1)
        if not self.force_probs:
            return gathered
        in_support = (safe_values >= 0) & (safe_values < num_categories)
        if jp.issubdtype(values.dtype, jp.inexact):
            same = values == safe_values.astype(values.dtype)
            in_support = in_support & same
        return jp.where(in_support, gathered, -jp.inf)

    def mode(self):
        return jp.argmax(self.logits_parameter(), axis=-1).astype(self.dtype)

    def entropy(self):
        logits = self.logits_parameter()
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        probs = jax.nn.softmax(logits, axis=-1)
        return -jp.sum(multiply_no_nan(log_probs, probs), axis=-1)

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

    def mean(self):
        return _fill(self.rate_parameter(), self.batch_shape, self.dtype)

    def variance(self):
        return self.mean()

    def stddev(self):
        return jp.sqrt(self.variance())

    def mode(self):
        return _fill(jp.floor(self.rate_parameter()), self.batch_shape,
                     self.dtype)

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
        ndims = int(reinterpreted_batch_ndims)
        return super().__new__(cls, distribution, ndims)

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
        ndims = self.reinterpreted_batch_ndims
        event_batch = self.distribution.batch_shape[-ndims:]
        return event_batch + self.distribution.event_shape

    def log_prob(self, values):
        log_prob = self.distribution.log_prob(values)
        return sum_rightmost(log_prob, self.reinterpreted_batch_ndims)

    def mean(self):
        return self.distribution.mean()

    def mode(self):
        return self.distribution.mode()

    def variance(self):
        return self.distribution.variance()

    def stddev(self):
        return self.distribution.stddev()

    def entropy(self):
        entropy = self.distribution.entropy()
        return sum_rightmost(entropy, self.reinterpreted_batch_ndims)

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
            raise NotImplementedError("log_cdf is only defined for scalars.")
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
        shape = build_sample_shape(num_samples) + self.batch_shape
        shape = shape + self.event_shape
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

    def mean(self):
        shape = self.batch_shape + self.event_shape
        return _fill(self.loc, shape, self.dtype)

    def mode(self):
        return self.mean()

    def stddev(self):
        shape = self.batch_shape + self.event_shape
        return _fill(jp.abs(self.scale_diag), shape, self.dtype)

    def entropy(self):
        num_dims = self.event_shape[-1]
        log_det = jp.sum(jp.log(jp.abs(self.scale_diag)), axis=-1)
        entropy = 0.5 * num_dims * jp.log(2.0 * jp.pi * jp.e) + log_det
        return _fill(entropy, self.batch_shape, self.dtype)

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        shape = build_sample_shape(num_samples) + self.batch_shape
        shape = shape + self.event_shape
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

    def mean(self):
        shape = self.batch_shape + self.event_shape
        return _fill(self.loc, shape, self.dtype)

    def mode(self):
        return self.mean()

    def variance(self):
        variance = jp.diagonal(self.covariance(), axis1=-2, axis2=-1)
        shape = self.batch_shape + self.event_shape
        return _fill(variance, shape, self.dtype)

    def stddev(self):
        return jp.sqrt(self.variance())

    def entropy(self):
        covariance = to_float(self.covariance_matrix, self.dtype)
        chol = jp.linalg.cholesky(covariance)
        log_det = jp.log(jp.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=-1)
        num_dims = self.event_shape[-1]
        entropy = 0.5 * num_dims * jp.log(2.0 * jp.pi * jp.e) + log_det
        return _fill(entropy, self.batch_shape, self.dtype)

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
        shape = build_sample_shape(num_samples) + self.batch_shape
        shape = shape + self.event_shape
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

    def mean(self):
        mixture_logits = self.mixture_distribution.logits_parameter()
        weights = jax.nn.softmax(mixture_logits, axis=-1)
        component_mean = self.components_distribution.mean()
        for _ in self.event_shape:
            weights = weights[..., None]
        axis = -(len(self.event_shape) + 1)
        return (weights * component_mean).sum(axis=axis)

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


_GammaBase = namedtuple("Gamma", ["concentration", "rate"])


class Gamma(_GammaBase, Distribution):
    __slots__ = ()

    @property
    def dtype(self):
        return common_dtype(self.concentration, self.rate)

    @property
    def batch_shape(self):
        return broadcast_shape(self.concentration, self.rate)

    @property
    def event_shape(self):
        return ()

    def log_prob(self, values):
        values = to_float(values, self.dtype)
        concentration = to_float(self.concentration, self.dtype)
        rate = to_float(self.rate, self.dtype)
        log_unnorm = jsp_special.xlogy(concentration - 1.0, values)
        log_unnorm = log_unnorm - rate * values
        log_norm = jsp_special.gammaln(concentration)
        log_norm = log_norm - concentration * jp.log(rate)
        return log_unnorm - log_norm

    def cdf(self, values):
        values = to_float(values, self.dtype)
        concentration = to_float(self.concentration, self.dtype)
        rate = to_float(self.rate, self.dtype)
        cdf = jsp_special.gammainc(concentration, rate * values)
        return jp.where(values < 0.0, 0.0, cdf)

    def mean(self):
        mean = to_float(self.concentration, self.dtype) / self.rate
        return _fill(mean, self.batch_shape, self.dtype)

    def variance(self):
        concentration = to_float(self.concentration, self.dtype)
        variance = concentration / jp.square(self.rate)
        return _fill(variance, self.batch_shape, self.dtype)

    def stddev(self):
        stddev = jp.sqrt(to_float(self.concentration, self.dtype)) / self.rate
        return _fill(stddev, self.batch_shape, self.dtype)

    def mode(self):
        concentration = to_float(self.concentration, self.dtype)
        mode = (concentration - 1.0) / self.rate
        return _fill(jp.where(concentration > 1.0, mode, jp.nan),
                     self.batch_shape, self.dtype)

    def entropy(self):
        concentration = to_float(self.concentration, self.dtype)
        entropy = concentration - jp.log(self.rate)
        entropy = entropy + jsp_special.gammaln(concentration)
        entropy = entropy + (1.0 - concentration) * jsp_special.digamma(
            concentration
        )
        return _fill(entropy, self.batch_shape, self.dtype)

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        shape = build_sample_shape(num_samples) + self.batch_shape
        draws = jax.random.gamma(seed, self.concentration, shape=shape,
                                 dtype=self.dtype)
        return draws / self.rate


_ExponentialBase = namedtuple("Exponential", ["rate"])


class Exponential(_ExponentialBase, Distribution):
    __slots__ = ()

    @property
    def dtype(self):
        return common_dtype(self.rate)

    @property
    def batch_shape(self):
        return broadcast_shape(self.rate)

    @property
    def event_shape(self):
        return ()

    def log_prob(self, values):
        values = to_float(values, self.dtype)
        rate = to_float(self.rate, self.dtype)
        return jp.log(rate) - rate * values

    def cdf(self, values):
        values = to_float(values, self.dtype)
        rate = to_float(self.rate, self.dtype)
        return jp.where(values < 0.0, 0.0, -jp.expm1(-rate * values))

    def survival_function(self, values):
        values = to_float(values, self.dtype)
        return jp.exp(-to_float(self.rate, self.dtype) * values)

    def log_survival_function(self, values):
        values = to_float(values, self.dtype)
        return -to_float(self.rate, self.dtype) * values

    def mean(self):
        return _fill(1.0 / to_float(self.rate, self.dtype), self.batch_shape,
                     self.dtype)

    def variance(self):
        return _fill(1.0 / jp.square(self.rate), self.batch_shape, self.dtype)

    def stddev(self):
        return jp.sqrt(self.variance())

    def mode(self):
        return _fill(jp.nan, self.batch_shape, self.dtype)

    def entropy(self):
        return _fill(1.0 - jp.log(self.rate), self.batch_shape, self.dtype)

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        shape = build_sample_shape(num_samples) + self.batch_shape
        draws = jax.random.exponential(seed, shape, dtype=self.dtype)
        return draws / self.rate


_InverseGammaBase = namedtuple("InverseGamma", ["concentration", "scale"])


class InverseGamma(_InverseGammaBase, Distribution):
    __slots__ = ()

    @property
    def dtype(self):
        return common_dtype(self.concentration, self.scale)

    @property
    def batch_shape(self):
        return broadcast_shape(self.concentration, self.scale)

    @property
    def event_shape(self):
        return ()

    def log_prob(self, values):
        values = to_float(values, self.dtype)
        concentration = to_float(self.concentration, self.dtype)
        scale = to_float(self.scale, self.dtype)
        log_unnorm = -(1.0 + concentration) * jp.log(values) - scale / values
        log_norm = jsp_special.gammaln(concentration)
        log_norm = log_norm - concentration * jp.log(scale)
        return log_unnorm - log_norm

    def cdf(self, values):
        values = to_float(values, self.dtype)
        concentration = to_float(self.concentration, self.dtype)
        scale = to_float(self.scale, self.dtype)
        return jsp_special.gammaincc(concentration, scale / values)

    def mean(self):
        concentration = to_float(self.concentration, self.dtype)
        scale = to_float(self.scale, self.dtype)
        mean = scale / (concentration - 1.0)
        return _fill(jp.where(concentration > 1.0, mean, jp.nan),
                     self.batch_shape, self.dtype)

    def variance(self):
        concentration = to_float(self.concentration, self.dtype)
        scale = to_float(self.scale, self.dtype)
        variance = jp.square(scale) / jp.square(concentration - 1.0)
        variance = variance / (concentration - 2.0)
        return _fill(jp.where(concentration > 2.0, variance, jp.nan),
                     self.batch_shape, self.dtype)

    def stddev(self):
        return jp.sqrt(self.variance())

    def mode(self):
        concentration = to_float(self.concentration, self.dtype)
        scale = to_float(self.scale, self.dtype)
        mode = scale / (1.0 + concentration)
        return _fill(mode, self.batch_shape, self.dtype)

    def entropy(self):
        concentration = to_float(self.concentration, self.dtype)
        scale = to_float(self.scale, self.dtype)
        entropy = concentration + jp.log(scale) + jsp_special.gammaln(
            concentration
        )
        entropy = entropy - (1.0 + concentration) * jsp_special.digamma(
            concentration
        )
        return _fill(entropy, self.batch_shape, self.dtype)

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        shape = build_sample_shape(num_samples) + self.batch_shape
        draws = jax.random.gamma(seed, self.concentration, shape=shape,
                                 dtype=self.dtype)
        return self.scale / draws


_Chi2Base = namedtuple("Chi2", ["df"])


class Chi2(_Chi2Base, Distribution):
    __slots__ = ()

    @property
    def dtype(self):
        return common_dtype(self.df)

    @property
    def batch_shape(self):
        return broadcast_shape(self.df)

    @property
    def event_shape(self):
        return ()

    def _gamma(self):
        half = jp.asarray(0.5, self.dtype)
        return Gamma(to_float(self.df, self.dtype) * half, half)

    def log_prob(self, values):
        return self._gamma().log_prob(values)

    def cdf(self, values):
        return self._gamma().cdf(values)

    def mean(self):
        return self._gamma().mean()

    def variance(self):
        return self._gamma().variance()

    def stddev(self):
        return self._gamma().stddev()

    def mode(self):
        return self._gamma().mode()

    def entropy(self):
        return self._gamma().entropy()

    def sample(self, num_samples=1, seed=None):
        return self._gamma().sample(num_samples, seed=seed)


_HalfNormalBase = namedtuple("HalfNormal", ["scale"])


class HalfNormal(_HalfNormalBase, Distribution):
    __slots__ = ()

    @property
    def dtype(self):
        return common_dtype(self.scale)

    @property
    def batch_shape(self):
        return broadcast_shape(self.scale)

    @property
    def event_shape(self):
        return ()

    def log_prob(self, values):
        values = to_float(values, self.dtype)
        scale = to_float(self.scale, self.dtype)
        log_unnorm = -0.5 * jp.square(values / scale)
        half_log = jp.asarray(0.5 * np.log(np.pi / 2.0), self.dtype)
        log_norm = jp.log(scale) + half_log
        return jp.where(values >= 0.0, log_unnorm - log_norm, -jp.inf)

    def cdf(self, values):
        values = to_float(values, self.dtype)
        scale = to_float(self.scale, self.dtype)
        return jsp_special.erf(jax.nn.relu(values) / scale / SQRT_TWO)

    def survival_function(self, values):
        values = to_float(values, self.dtype)
        scale = to_float(self.scale, self.dtype)
        return jsp_special.erfc(jax.nn.relu(values) / scale / SQRT_TWO)

    def log_survival_function(self, values):
        return jp.log(self.survival_function(values))

    def mean(self):
        mean = to_float(self.scale, self.dtype) * SQRT_TWO / jp.sqrt(jp.pi)
        return _fill(mean, self.batch_shape, self.dtype)

    def variance(self):
        variance = jp.square(self.scale) * (1.0 - 2.0 / jp.pi)
        return _fill(variance, self.batch_shape, self.dtype)

    def stddev(self):
        return jp.sqrt(self.variance())

    def mode(self):
        return jp.zeros(self.batch_shape, self.dtype)

    def entropy(self):
        entropy = 0.5 * jp.log(jp.pi * jp.square(self.scale) / 2.0) + 0.5
        return _fill(entropy, self.batch_shape, self.dtype)

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        shape = build_sample_shape(num_samples) + self.batch_shape
        noise = jax.random.normal(seed, shape, dtype=self.dtype)
        return jp.abs(noise) * self.scale


_CauchyBase = namedtuple("Cauchy", ["loc", "scale"])


class Cauchy(_CauchyBase, Distribution):
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
        log_norm = LOG_PI + jp.log(self.scale)
        return -jp.log1p(jp.square(z)) - log_norm

    def cdf(self, values):
        values = to_float(values, self.dtype)
        z = (values - self.loc) / self.scale
        return jp.arctan(z) / jp.pi + 0.5

    def quantile(self, values):
        values = to_float(values, self.dtype)
        return self.loc + self.scale * jp.tan(jp.pi * (values - 0.5))

    def mean(self):
        return _fill(jp.nan, self.batch_shape, self.dtype)

    def variance(self):
        return _fill(jp.nan, self.batch_shape, self.dtype)

    def stddev(self):
        return _fill(jp.nan, self.batch_shape, self.dtype)

    def mode(self):
        return _fill(self.loc, self.batch_shape, self.dtype)

    def entropy(self):
        entropy = jp.log(4.0 * jp.pi) + jp.log(self.scale)
        return _fill(entropy, self.batch_shape, self.dtype)

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        shape = build_sample_shape(num_samples) + self.batch_shape
        uniforms = jax.random.uniform(seed, shape, dtype=self.dtype)
        return self.quantile(uniforms)


_HalfCauchyBase = namedtuple("HalfCauchy", ["loc", "scale"])


class HalfCauchy(_HalfCauchyBase, Distribution):
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
        log_prob = jp.asarray(np.log(2.0 / np.pi), self.dtype)
        log_prob = log_prob - jp.log(self.scale) - jp.log1p(jp.square(z))
        return jp.where(values < self.loc, -jp.inf, log_prob)

    def cdf(self, values):
        values = to_float(values, self.dtype)
        z = (values - self.loc) / self.scale
        cdf = 2.0 / jp.pi * jp.arctan(z)
        return jp.where(values < self.loc, 0.0, cdf)

    def mean(self):
        return _fill(jp.nan, self.batch_shape, self.dtype)

    def variance(self):
        return _fill(jp.nan, self.batch_shape, self.dtype)

    def stddev(self):
        return _fill(jp.nan, self.batch_shape, self.dtype)

    def mode(self):
        return _fill(self.loc, self.batch_shape, self.dtype)

    def entropy(self):
        entropy = jp.log(2.0 * jp.pi) + jp.log(self.scale)
        return _fill(entropy, self.batch_shape, self.dtype)

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        shape = build_sample_shape(num_samples) + self.batch_shape
        uniforms = jax.random.uniform(seed, shape, dtype=self.dtype)
        return self.loc + self.scale * jp.tan(0.5 * jp.pi * uniforms)


_LogisticBase = namedtuple("Logistic", ["loc", "scale"])


class Logistic(_LogisticBase, Distribution):
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
        return -z - 2.0 * jax.nn.softplus(-z) - jp.log(self.scale)

    def cdf(self, values):
        values = to_float(values, self.dtype)
        return jax.nn.sigmoid((values - self.loc) / self.scale)

    def log_cdf(self, values):
        values = to_float(values, self.dtype)
        return -jax.nn.softplus(-(values - self.loc) / self.scale)

    def survival_function(self, values):
        values = to_float(values, self.dtype)
        return jax.nn.sigmoid(-(values - self.loc) / self.scale)

    def log_survival_function(self, values):
        values = to_float(values, self.dtype)
        return -jax.nn.softplus((values - self.loc) / self.scale)

    def mean(self):
        return _fill(self.loc, self.batch_shape, self.dtype)

    def mode(self):
        return self.mean()

    def stddev(self):
        stddev = to_float(self.scale, self.dtype) * jp.pi / jp.sqrt(3.0)
        return _fill(stddev, self.batch_shape, self.dtype)

    def variance(self):
        return jp.square(self.stddev())

    def entropy(self):
        return _fill(2.0 + jp.log(self.scale), self.batch_shape, self.dtype)

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        shape = build_sample_shape(num_samples) + self.batch_shape
        uniforms = jax.random.uniform(seed, shape, dtype=self.dtype)
        return self.loc + self.scale * (jp.log(uniforms) - jp.log1p(-uniforms))


_GumbelBase = namedtuple("Gumbel", ["loc", "scale"])


class Gumbel(_GumbelBase, Distribution):
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
        return -(z + jp.exp(-z)) - jp.log(self.scale)

    def cdf(self, values):
        values = to_float(values, self.dtype)
        z = (values - self.loc) / self.scale
        return jp.exp(-jp.exp(-z))

    def mean(self):
        mean = self.loc + self.scale * np.euler_gamma
        return _fill(mean, self.batch_shape, self.dtype)

    def mode(self):
        return _fill(self.loc, self.batch_shape, self.dtype)

    def stddev(self):
        stddev = to_float(self.scale, self.dtype) * jp.pi / jp.sqrt(6.0)
        return _fill(stddev, self.batch_shape, self.dtype)

    def variance(self):
        return jp.square(self.stddev())

    def entropy(self):
        entropy = 1.0 + jp.log(self.scale) + np.euler_gamma
        return _fill(entropy, self.batch_shape, self.dtype)

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        shape = build_sample_shape(num_samples) + self.batch_shape
        uniforms = jax.random.uniform(seed, shape, dtype=self.dtype)
        return self.loc - self.scale * jp.log(-jp.log(uniforms))


_DirichletBase = namedtuple("Dirichlet", ["concentration"])


class Dirichlet(_DirichletBase, Distribution):
    __slots__ = ()

    @property
    def dtype(self):
        return common_dtype(self.concentration)

    @property
    def batch_shape(self):
        return jp.asarray(self.concentration).shape[:-1]

    @property
    def event_shape(self):
        return jp.asarray(self.concentration).shape[-1:]

    def log_prob(self, values):
        values = to_float(values, self.dtype)
        concentration = to_float(self.concentration, self.dtype)
        log_unnorm = jsp_special.xlogy(concentration - 1.0, values).sum(axis=-1)
        return log_unnorm - _log_multi_beta(concentration)

    def mean(self):
        concentration = to_float(self.concentration, self.dtype)
        total = concentration.sum(axis=-1, keepdims=True)
        return concentration / total

    def variance(self):
        concentration = to_float(self.concentration, self.dtype)
        total = concentration.sum(axis=-1, keepdims=True)
        mean = concentration / total
        return mean * (1.0 - mean) / (total + 1.0)

    def entropy(self):
        concentration = to_float(self.concentration, self.dtype)
        num_classes = self.event_shape[-1]
        total = concentration.sum(axis=-1)
        entropy = _log_multi_beta(concentration)
        entropy = entropy + (total - num_classes) * jsp_special.digamma(total)
        weighted = (concentration - 1.0) * jsp_special.digamma(concentration)
        return entropy - weighted.sum(axis=-1)

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        shape = build_sample_shape(num_samples) + self.batch_shape
        shape = shape + self.event_shape
        draws = jax.random.gamma(seed, self.concentration, shape=shape,
                                 dtype=self.dtype)
        return draws / draws.sum(axis=-1, keepdims=True)


_MVNTriLBase = namedtuple("MultivariateNormalTriL", ["loc", "scale_tril"])


class MultivariateNormalTriL(_MVNTriLBase, Distribution):
    __slots__ = ()

    @property
    def dtype(self):
        return common_dtype(self.loc, self.scale_tril)

    @property
    def batch_shape(self):
        loc_shape = jp.asarray(self.loc).shape[:-1]
        tril_shape = jp.asarray(self.scale_tril).shape[:-2]
        return jp.broadcast_shapes(loc_shape, tril_shape)

    @property
    def event_shape(self):
        return (jp.asarray(self.scale_tril).shape[-1],)

    def _diag(self):
        return jp.diagonal(to_float(self.scale_tril, self.dtype),
                           axis1=-2, axis2=-1)

    def log_prob(self, values):
        values = to_float(values, self.dtype)
        chol = to_float(self.scale_tril, self.dtype)
        diff = values - self.loc
        solved = jsp_linalg.solve_triangular(
            chol, diff[..., None], lower=True
        )[..., 0]
        standard = Normal(
            jp.asarray(0.0, self.dtype), jp.asarray(1.0, self.dtype)
        )
        log_prob = standard.log_prob(solved).sum(axis=-1)
        log_det = jp.log(jp.abs(self._diag())).sum(axis=-1)
        return log_prob - log_det

    def covariance(self):
        chol = to_float(self.scale_tril, self.dtype)
        return jp.einsum("...ij,...kj->...ik", chol, chol)

    def variance(self):
        chol = to_float(self.scale_tril, self.dtype)
        variance = jp.square(chol).sum(axis=-1)
        shape = self.batch_shape + self.event_shape
        return _fill(variance, shape, self.dtype)

    def stddev(self):
        return jp.sqrt(self.variance())

    def mean(self):
        shape = self.batch_shape + self.event_shape
        return _fill(self.loc, shape, self.dtype)

    def mode(self):
        return self.mean()

    def entropy(self):
        log_det = jp.log(jp.abs(self._diag())).sum(axis=-1)
        num_dims = self.event_shape[-1]
        entropy = 0.5 * num_dims * jp.log(2.0 * jp.pi * jp.e) + log_det
        return _fill(entropy, self.batch_shape, self.dtype)

    def sample(self, num_samples=1, seed=None):
        seed = _require_seed(seed)
        chol = to_float(self.scale_tril, self.dtype)
        shape = build_sample_shape(num_samples) + self.batch_shape
        shape = shape + self.event_shape
        noise = jax.random.normal(seed, shape, dtype=self.dtype)
        transformed = jp.einsum("...ij,...j->...i", chol, noise)
        return self.loc + transformed


def _log_multi_beta(concentration):
    log_gamma_sum = jsp_special.gammaln(concentration).sum(axis=-1)
    return log_gamma_sum - jsp_special.gammaln(concentration.sum(axis=-1))


def kl_divergence(distribution_a, distribution_b):
    compute = _KL_DIVERGENCES.get((type(distribution_a), type(distribution_b)))
    if compute is None:
        raise NotImplementedError(
            f"kl_divergence not registered for "
            f"{type(distribution_a).__name__} and "
            f"{type(distribution_b).__name__}."
        )
    return compute(distribution_a, distribution_b)


def _kl_normal_normal(a, b):
    diff_log_scale = jp.log(a.scale) - jp.log(b.scale)
    squared = jp.square(a.loc / b.scale - b.loc / b.scale)
    return 0.5 * squared + 0.5 * jp.expm1(2.0 * diff_log_scale) - diff_log_scale


def _kl_mvn_diag(a, b):
    per_dim = _kl_normal_normal(Normal(a.loc, a.scale_diag),
                                Normal(b.loc, b.scale_diag))
    return per_dim.sum(axis=-1)


def _kl_independent(a, b):
    inner = kl_divergence(a.distribution, b.distribution)
    return sum_rightmost(inner, a.reinterpreted_batch_ndims)


def _kl_categorical(a, b):
    log_probs_a = jax.nn.log_softmax(a.logits_parameter(), axis=-1)
    log_probs_b = jax.nn.log_softmax(b.logits_parameter(), axis=-1)
    probs_a = jax.nn.softmax(a.logits_parameter(), axis=-1)
    return jp.sum(probs_a * (log_probs_a - log_probs_b), axis=-1)


def _kl_bernoulli(a, b):
    log0_a, log1_a = _bernoulli_log_probs(a.logits, a.probs)
    log0_b, log1_b = _bernoulli_log_probs(b.logits, b.probs)
    probs_a = a.probs_parameter()
    kl = multiply_no_nan(log1_a, probs_a) - multiply_no_nan(log1_b, probs_a)
    kl = kl + multiply_no_nan(log0_a, 1.0 - probs_a)
    return kl - multiply_no_nan(log0_b, 1.0 - probs_a)


def _kl_beta_beta(a, b):
    alpha_a = to_float(a.concentration1, a.dtype)
    beta_a = to_float(a.concentration0, a.dtype)
    alpha_b = to_float(b.concentration1, b.dtype)
    beta_b = to_float(b.concentration0, b.dtype)
    kl = lbeta(alpha_b, beta_b) - lbeta(alpha_a, beta_a)
    kl = kl + (alpha_a - alpha_b) * jsp_special.digamma(alpha_a)
    kl = kl + (beta_a - beta_b) * jsp_special.digamma(beta_a)
    total_diff = (alpha_b - alpha_a) + (beta_b - beta_a)
    return kl + total_diff * jsp_special.digamma(alpha_a + beta_a)


def _fill(value, shape, dtype):
    return jp.broadcast_to(jp.asarray(value, dtype), shape)


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


def _normal_log_pdf(values):
    return -0.5 * jp.square(values) - _half_log_two_pi(values.dtype)


def _normal_pdf(values):
    return jp.exp(_normal_log_pdf(values))


def _log_abs_sub_exp(big, small):
    high = jp.maximum(big, small)
    low = jp.minimum(big, small)
    return logsum_expbig_minus_expsmall(high, low)


def _log1p_square(values):
    values = jp.asarray(values)
    eps = jp.asarray(np.finfo(values.dtype).eps, dtype=values.dtype)
    is_large = jp.abs(values) > jp.power(eps, -0.5)
    safe_values = jp.where(is_large, jp.abs(values), 1.0)
    large_values = 2.0 * jp.log(safe_values)
    small_values = jp.log1p(values**2)
    return jp.where(is_large, large_values, small_values)


_KL_DIVERGENCES = {
    (Normal, Normal): _kl_normal_normal,
    (MultivariateNormalDiag, MultivariateNormalDiag): _kl_mvn_diag,
    (Independent, Independent): _kl_independent,
    (Categorical, Categorical): _kl_categorical,
    (Bernoulli, Bernoulli): _kl_bernoulli,
    (Beta, Beta): _kl_beta_beta,
}
