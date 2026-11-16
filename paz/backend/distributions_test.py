from pathlib import Path

import jax
import jax.numpy as jp
import pytest

import paz
from paz.backend import bijectors as B
from paz.backend import distributions as D

ROUNDS = 25
KEY = jax.random.PRNGKey(0)


@pytest.fixture(autouse=True)
def double_precision():
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def sample(key, shape, low, high):
    return jax.random.uniform(key, shape, jp.float64, low, high)


# Snapshots hold TensorFlow Probability outputs for the same inputs. NaN
# positions are stored as a second row because NaN never compares close.
def assert_snapshot(outputs, filename, atol=1e-9, rtol=1e-7):
    values = jp.concatenate([jp.ravel(output) for output in outputs])
    is_nan = jp.isnan(values)
    array = jp.stack([jp.where(is_nan, 0.0, values), is_nan])
    path = str(Path(__file__).parent / "snapshots" / filename)
    paz.assert_snapshot(array, path, rtol=rtol, atol=atol)


def compute_methods(distribution, values, methods):
    return [getattr(distribution, method)(values) for method in methods]


def compute_moments(distribution, methods):
    return [getattr(distribution, method)() for method in methods]


SCALAR = ["log_prob", "prob", "cdf", "log_cdf", "survival_function",
          "log_survival_function"]
MOMENTS = ["mean", "variance", "stddev", "mode", "entropy"]
DENSITY = ["log_prob", "prob"]


def build_continuous(key):
    loc = sample(key[0], (4,), -5, 5)
    scale = sample(key[1], (4,), 0.05, 6)
    positive = sample(key[2], (4,), 0.05, 12)
    real = sample(key[3], (4,), -12, 12)
    df = sample(key[4], (4,), 0.5, 40)
    low = sample(key[5], (4,), -6, -0.4)
    high = sample(key[6], (4,), 0.4, 6)
    return [
        (D.Normal(loc, scale), real, SCALAR),
        (D.Laplace(loc, scale), real, SCALAR),
        (D.StudentT(df, loc, scale), real, ["log_prob", "prob"]),
        (D.LogNormal(loc, scale), positive, SCALAR),
        (D.Logistic(loc, scale), real, SCALAR),
        (D.Gumbel(loc, scale), real, ["log_prob", "prob", "cdf", "log_cdf"]),
        (D.Cauchy(loc, scale), real, ["log_prob", "prob", "cdf"]),
        (D.Gamma(positive, scale), positive, ["log_prob", "prob", "cdf"]),
        (D.Exponential(scale), positive,
         ["log_prob", "prob", "cdf", "survival_function",
          "log_survival_function"]),
        (D.InverseGamma(positive, scale), positive,
         ["log_prob", "prob", "cdf"]),
        (D.Chi2(df), positive, ["log_prob", "prob", "cdf"]),
        (D.HalfNormal(scale), positive, SCALAR),
        (D.TruncatedNormal(loc, scale, low, high), real, SCALAR),
    ]


def test_continuous_value_fuzz():
    outputs = []
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, round_index), 8)
        for distribution, values, methods in build_continuous(key):
            outputs += compute_methods(distribution, values, methods)
    assert_snapshot(outputs, "distributions_continuous_values.npy")


def test_continuous_moment_fuzz():
    # Cancellation-prone statistics such as entropy are held to a looser bar.
    outputs = []
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, round_index), 8)
        for distribution, _, _ in build_continuous(key):
            outputs += compute_moments(distribution, MOMENTS)
    filename = "distributions_continuous_moments.npy"
    assert_snapshot(outputs, filename, atol=1e-6, rtol=1e-6)


def build_positive_family(key):
    scale = sample(key[1], (4,), 0.1, 5)
    beta_values = sample(key[2], (4,), 1e-3, 1 - 1e-3)
    alpha = sample(key[3], (4,), 0.1, 10)
    beta = sample(key[4], (4,), 0.1, 10)
    angles = sample(key[5], (4,), -3.14159, 3.14159)
    concentration = sample(key[0], (4,), 0.05, 60)
    von_mises_moments = ["mean", "variance", "mode", "entropy"]
    return [
        (D.Beta(alpha, beta), beta_values, MOMENTS),
        (D.VonMises(scale, concentration), angles, von_mises_moments),
    ]


def test_positive_family_fuzz():
    values, moments = [], []
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, round_index), 6)
        for distribution, points, methods in build_positive_family(key):
            values += compute_methods(distribution, points, DENSITY)
            moments += compute_moments(distribution, methods)
    assert_snapshot(values, "distributions_positive_values.npy")
    filename = "distributions_positive_moments.npy"
    assert_snapshot(moments, filename, atol=1e-6, rtol=1e-6)


def build_discrete(key):
    logits = sample(key[0], (4, 5), -4, 4)
    integers = jax.random.randint(key[1], (4,), -1, 6)
    bernoulli_logits = sample(key[2], (4,), -5, 5)
    binary = jp.round(sample(key[3], (4,), 0, 1))
    rate = sample(key[4], (4,), 0.05, 30)
    counts = jp.floor(sample(key[5], (4,), 0, 35))
    poisson_moments = ["mean", "variance", "stddev", "mode"]
    return [
        (D.Categorical(logits=logits), integers, ["mode", "entropy"]),
        (D.Bernoulli(logits=bernoulli_logits), binary, MOMENTS),
        (D.Poisson(rate=rate), counts, poisson_moments),
    ]


def test_discrete_value_fuzz():
    values, moments = [], []
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, round_index), 6)
        for distribution, points, methods in build_discrete(key):
            values += compute_methods(distribution, points, DENSITY)
            moments += compute_moments(distribution, methods)
    assert_snapshot(values, "distributions_discrete_values.npy")
    filename = "distributions_discrete_moments.npy"
    assert_snapshot(moments, filename, atol=1e-6, rtol=1e-6)


def build_multivariate(key):
    loc = sample(key[0], (3, 4), -3, 3)
    scale_diag = sample(key[1], (3, 4), 0.05, 4)
    values = sample(key[2], (3, 4), -6, 6)
    concentration = sample(key[3], (3, 4), 0.3, 6)
    simplex = jax.random.dirichlet(key[4], concentration)
    normal_moments = MOMENTS + ["covariance"]
    return [
        (D.MultivariateNormalDiag(loc, scale_diag), values, normal_moments),
        (D.Dirichlet(concentration), simplex, ["mean", "variance", "entropy"]),
    ]


def test_multivariate_value_fuzz():
    values, moments = [], []
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, round_index), 6)
        for distribution, points, methods in build_multivariate(key):
            values += compute_methods(distribution, points, DENSITY)
            moments += compute_moments(distribution, methods)
    assert_snapshot(values, "distributions_multivariate_values.npy")
    filename = "distributions_multivariate_moments.npy"
    assert_snapshot(moments, filename, atol=1e-6, rtol=1e-6)


def build_kl_pairs(key):
    loc_a = sample(key[0], (4,), -3, 3)
    loc_b = sample(key[1], (4,), -3, 3)
    scale_a = sample(key[2], (4,), 0.1, 3)
    scale_b = sample(key[3], (4,), 0.1, 3)
    logits_a = sample(key[4], (4, 5), -3, 3)
    logits_b = sample(key[5], (4, 5), -3, 3)
    return [
        (D.Normal(loc_a, scale_a), D.Normal(loc_b, scale_b)),
        (D.Bernoulli(logits=loc_a), D.Bernoulli(logits=loc_b)),
        (D.Beta(scale_a, scale_b), D.Beta(scale_b, scale_a)),
        (D.Categorical(logits=logits_a), D.Categorical(logits=logits_b)),
    ]


def test_kl_divergence_fuzz():
    outputs = []
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, 2000 + round_index), 8)
        for distribution_a, distribution_b in build_kl_pairs(key):
            outputs.append(D.kl_divergence(distribution_a, distribution_b))
    assert_snapshot(outputs, "distributions_kl_divergence.npy")


def build_covariance(key, shape):
    factor = sample(key, shape + shape[-1:], -1, 1)
    return factor @ jp.swapaxes(factor, -1, -2) + 0.5 * jp.eye(shape[-1])


def build_composites(key):
    loc = sample(key[0], (3, 4), -3, 3)
    scale = sample(key[1], (3, 4), 0.05, 4)
    values = sample(key[2], (3, 4), -6, 6)
    low = sample(key[3], (3, 4), -7, -6)
    high = sample(key[4], (3, 4), 6, 7)
    probs = jax.nn.softmax(sample(key[5], (3,), -1, 1))
    covariance = build_covariance(key[6], (3, 4))
    simplex = jax.random.dirichlet(key[7], jp.ones((3, 4)))
    temperature = sample(key[8], (), 0.3, 2)
    mixture = D.Categorical(probs=probs)
    normal_full = D.MultivariateNormalFullCovariance(loc, covariance)
    normal_diag = D.MultivariateNormalDiag(loc, scale)
    independent = D.Independent(D.Normal(loc, scale), 1)
    scalar_mixture = D.MixtureSameFamily(mixture, D.Normal(loc[0, :3], 0.5))
    relaxed = D.RelaxedOneHotCategorical(temperature, logits=loc)
    full_moments = ["mean", "entropy", "covariance"]
    return [
        (D.Uniform(low, high), values, ["log_prob", "prob", "cdf"], []),
        (D.Uniform(low, high), values + 20.0, ["log_prob"], []),
        (D.Deterministic(jp.round(loc)), jp.round(values), ["log_prob"], []),
        (independent, values, ["log_prob"], ["mean", "entropy"]),
        (normal_full, values, ["log_prob", "prob"], full_moments),
        (scalar_mixture, values[0], ["log_prob"], ["mean"]),
        (D.MixtureSameFamily(mixture, normal_diag), values, ["log_prob"], []),
        (D.MixtureSameFamily(mixture, normal_full), values, ["log_prob"], []),
        (relaxed, simplex, ["log_prob"], []),
    ]


def test_composite_value_fuzz():
    outputs = []
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, 5000 + round_index), 9)
        for distribution, values, methods, moments in build_composites(key):
            outputs += compute_methods(distribution, values, methods)
            outputs += compute_moments(distribution, moments)
    assert_snapshot(outputs, "distributions_composite_values.npy")


def build_transformed(key):
    loc = sample(key[0], (), -1, 1)
    scale = sample(key[1], (), 0.3, 2)
    low = sample(key[2], (), -4, -1)
    width = sample(key[3], (), 1, 5)
    positive = sample(key[4], (6,), 0.05, 8)
    bounded = low + width * sample(key[5], (6,), 0.01, 0.99)
    bounding = B.Chain([B.Shift(low), B.Scale(width), B.Sigmoid()])
    base = D.Normal(loc, scale)
    return [
        (D.TransformedDistribution(base, B.Exp()), positive, SCALAR),
        (D.TransformedDistribution(base, bounding), bounded, SCALAR),
    ]


def test_transformed_value_fuzz():
    outputs = []
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, 6000 + round_index), 6)
        for distribution, values, methods in build_transformed(key):
            outputs += compute_methods(distribution, values, methods)
    assert_snapshot(outputs, "distributions_transformed_values.npy")


def build_quantized(key):
    f = lambda value: jp.asarray(value, jp.float64)
    loc = sample(key[0], (), -1, 1)
    scale = sample(key[1], (), 0.3, 2)
    # The grids reach far into the tails and outside the (low, high) bounds.
    # TensorFlow Probability gives NaN below zero for LogNormal, where paz
    # gives zero mass, so that grid starts at 1.
    cases = [
        (D.Normal(loc, scale), jp.arange(-40.0, 60.0, 3.0)),
        (D.LogNormal(loc, scale), jp.arange(1.0, 60.0, 3.0)),
    ]
    quantized = []
    for base, grid in cases:
        quantized.append((D.QuantizedDistribution(base), grid))
        bounded = D.QuantizedDistribution(base, f(-3.0), f(4.0))
        quantized.append((bounded, grid))
    return quantized


def test_quantized_value_fuzz():
    outputs = []
    for round_index in range(5):
        key = jax.random.split(jax.random.fold_in(KEY, 7000 + round_index), 2)
        for distribution, grid in build_quantized(key):
            outputs += compute_methods(distribution, grid, SCALAR)
    assert_snapshot(outputs, "distributions_quantized_values.npy")


def compute_gradients(build, params, values):
    def loss(*raw):
        return jp.sum(build(*raw).log_prob(values))

    return list(jax.grad(loss, tuple(range(len(params))))(*params))


def test_log_prob_gradient_parity():
    f = lambda value: jp.asarray(value, jp.float64)
    cases = [
        (D.Normal, (f(0.2), f(1.3)), f([-0.4, 0.3, 1.2])),
        (D.Gamma, (f(2.3), f(1.4)), f([0.5, 1.2, 3.0])),
        (D.Beta, (f(2.5), f(3.5)), f([0.2, 0.5, 0.8])),
        (D.StudentT, (f(12.0), f(0.2), f(1.1)), f([-0.8, 0.3, 1.1])),
        (D.Dirichlet, (f([2.0, 3.0, 4.0]),), f([0.2, 0.3, 0.5])),
    ]
    outputs = []
    for build, params, values in cases:
        outputs += compute_gradients(build, params, values)
    filename = "distributions_log_prob_gradients.npy"
    assert_snapshot(outputs, filename, atol=1e-9, rtol=0.0)


def test_extreme_masks_match_snapshot():
    f = lambda value: jp.asarray(value, jp.float64)
    values = f([-jp.inf, -1e300, 0.0, 1e300, jp.inf, jp.nan])
    tiny = jp.finfo(jp.float64).tiny
    bounds = f([0.0, tiny, 1.0, jp.inf, jp.nan])
    normal_log_prob = D.Normal(f(0.0), f(1.0)).log_prob(values)
    log_normal_log_prob = D.LogNormal(f(0.0), f(1.0)).log_prob(bounds)
    outputs = [normal_log_prob, log_normal_log_prob]
    assert_snapshot(outputs, "distributions_extreme_masks.npy")


def test_log_normal_has_no_mass_below_zero():
    log_normal = D.LogNormal(0.0, 1.0)
    values = jp.array([-1.0, 0.0])
    assert jp.array_equal(log_normal.cdf(values), jp.zeros(2))
    assert jp.array_equal(log_normal.survival_function(values), jp.ones(2))
    assert jp.array_equal(log_normal.log_prob(values), jp.full(2, -jp.inf))
    assert jp.array_equal(log_normal.log_cdf(values), jp.full(2, -jp.inf))
    log_survival = log_normal.log_survival_function(values)
    assert jp.array_equal(log_survival, jp.zeros(2))


def build_depth_mixture(key):
    probs = sample(key[0], (50,), 0.05, 0.95)
    probs = jp.stack([probs, 1.0 - probs], axis=-1)
    loc = sample(key[1], (50,), 0.5, 4.5)
    scale = sample(key[2], (50,), 0.01, 0.5)
    values = sample(key[3], (50,), -1.0, 6.0)
    return probs, loc, scale, values


def test_mixture_log_prob_matches_snapshot():
    key = jax.random.split(jax.random.fold_in(KEY, 3000), 4)
    probs, loc, scale, values = build_depth_mixture(key)
    uniform = D.Uniform(jp.zeros(50), 5.0 * jp.ones(50))
    mixture = D.Mixture(probs, [D.Laplace(loc, scale), uniform])
    outputs = [mixture.log_prob(values)]
    assert_snapshot(outputs, "distributions_mixture_log_prob.npy")


def test_mixture_three_families_match_snapshot():
    key = jax.random.split(jax.random.fold_in(KEY, 3001), 3)
    logits = sample(key[0], (50, 3), -2, 2)
    probs = jax.nn.softmax(logits, axis=-1)
    values = sample(key[1], (50,), -4.0, 4.0)
    loc = sample(key[2], (50,), -1.0, 1.0)
    ones = jp.ones(50)
    normal = D.Normal(loc, 0.7 * ones)
    laplace = D.Laplace(-loc, 1.3 * ones)
    uniform = D.Uniform(-2.0 * ones, 3.0 * ones)
    mixture = D.Mixture(probs, [normal, laplace, uniform])
    outputs = [mixture.log_prob(values)]
    assert_snapshot(outputs, "distributions_mixture_three_families.npy")


def test_mixture_log_prob_by_hand():
    # Laplace density at its location is 1 / (2 * scale) = 1 / 4.
    # Uniform(0, 5) density is 1 / 5.
    mixture = D.Mixture(jp.array([0.3, 0.7]),
                        [D.Laplace(2.0, 2.0), D.Uniform(0.0, 5.0)])
    expected = jp.log((0.3 / 4.0) + (0.7 / 5.0))
    assert jp.allclose(mixture.log_prob(2.0), expected, rtol=1e-12)


def test_mixture_outside_uniform_support_is_finite():
    def log_prob(probability, loc, scale):
        probs = jp.stack([probability, 1.0 - probability])
        components = [D.Laplace(loc, scale), D.Uniform(0.0, 5.0)]
        return D.Mixture(probs, components).log_prob(7.0)

    value = log_prob(0.4, 6.0, 0.5)
    expected = jp.log(0.4) - jp.abs((7.0 - 6.0) / 0.5) - jp.log(2.0 * 0.5)
    assert jp.isfinite(value)
    assert jp.allclose(value, expected, rtol=1e-12)
    gradients = jax.grad(log_prob, (0, 1, 2))(0.4, 6.0, 0.5)
    expected_gradients = (1.0 / 0.4, 1.0 / 0.5, (1.0 / 0.25) - (1.0 / 0.5))
    for gradient, expected_gradient in zip(gradients, expected_gradients):
        assert jp.isfinite(gradient)
        assert jp.allclose(gradient, expected_gradient, rtol=1e-12)


def test_mixture_gradient_is_finite_at_zero_weight():
    def log_prob(gate, value):
        probs = jp.stack([1.0 - gate, gate])
        components = [D.Deterministic(0.0), D.Normal(0.5, 1.2)]
        return D.Mixture(probs, components).log_prob(value)

    density = jp.exp(D.Normal(0.5, 1.2).log_prob(0.0))
    gate_gradient = jax.jit(jax.grad(log_prob))
    on_at_zero = (density - 1.0) / density
    assert jp.allclose(gate_gradient(1.0, 0.7), 1.0, rtol=1e-12)
    assert jp.allclose(gate_gradient(0.0, 0.0), density - 1.0, rtol=1e-12)
    assert jp.allclose(gate_gradient(1.0, 0.0), on_at_zero, rtol=1e-12)
    value_gradient = jax.grad(log_prob, 1)(1.0, 0.0)
    assert jp.allclose(value_gradient, 0.5 / 1.2**2, rtol=1e-12)


def test_mixture_jit_and_grad_match_snapshot():
    key = jax.random.split(jax.random.fold_in(KEY, 3002), 4)
    probs, loc, scale, values = build_depth_mixture(key)

    def loss(probability, loc, scale):
        probs = jp.stack([probability, 1.0 - probability], axis=-1)
        uniform = D.Uniform(jp.zeros(50), 5.0 * jp.ones(50))
        components = [D.Laplace(loc, scale), uniform]
        return jp.sum(D.Mixture(probs, components).log_prob(values))

    params = (probs[:, 0], loc, scale)
    gradients = jax.jit(jax.grad(loss, (0, 1, 2)))(*params)
    outputs = list(gradients) + [jax.jit(loss)(*params)]
    assert_snapshot(outputs, "distributions_mixture_jit_and_grad.npy")


def test_mixture_sample_frequencies_match_probs():
    probs = jp.array([[0.2, 0.8], [0.7, 0.3]])
    mixture = D.Mixture(probs, [D.Normal(-10.0, 1.0), D.Normal(10.0, 1.0)])
    samples = mixture.sample(20000, seed=KEY)
    assert samples.shape == (20000, 2)
    assert mixture.batch_shape == (2,)
    negative_frequency = jp.mean(samples < 0.0, axis=0)
    assert jp.allclose(negative_frequency, probs[:, 0], atol=0.015)


def test_masked_log_prob_matches_snapshot():
    key = jax.random.split(jax.random.fold_in(KEY, 4000), 3)
    loc = sample(key[0], (50,), -1.0, 1.0)
    values = sample(key[1], (50,), -3.0, 3.0)
    is_valid = jax.random.bernoulli(key[2], 0.6, (50,))
    masked = D.Masked(D.Normal(loc, 0.8), is_valid)
    outputs = [masked.log_prob(values)]
    assert_snapshot(outputs, "distributions_masked_log_prob.npy")


def test_masked_invalid_entries_are_zero_and_valid_unchanged():
    distribution = D.Bernoulli(probs=jp.array([0.2, 0.7, 0.9]))
    is_valid = jp.array([True, False, True])
    log_prob = D.Masked(distribution, is_valid).log_prob(jp.ones(3))
    expected = jp.array([jp.log(0.2), 0.0, jp.log(0.9)])
    assert jp.allclose(log_prob, expected, rtol=1e-12)


def test_masked_gradient_ignores_invalid_entries():
    is_valid = jp.array([True, False])

    def total_log_prob(loc):
        masked = D.Masked(D.Normal(loc, 1.0), is_valid)
        return masked.log_prob(jp.array([0.5, 3.0])).sum()

    gradient = jax.grad(total_log_prob)(jp.zeros(2))
    # d/dloc of log N(x; loc, 1) at loc 0 is x
    assert jp.allclose(gradient, jp.array([0.5, 0.0]), rtol=1e-12)


def test_masked_shapes_and_sample():
    masked = D.Masked(D.Normal(jp.zeros(4), 1.0), jp.ones(4, bool))
    assert masked.batch_shape == (4,)
    assert masked.event_shape == ()
    draws = masked.sample(10, seed=jax.random.PRNGKey(0))
    assert draws.shape == (10, 4)
