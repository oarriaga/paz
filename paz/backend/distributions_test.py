from collections import namedtuple

import jax
import jax.numpy as jp
import pytest
from scipy import special
from scipy import stats

from paz.backend import bijectors as B
from paz.backend import distributions as D

ROUNDS = 25
Quantized = namedtuple("Quantized", [
    "log_prob", "prob", "cdf", "log_cdf", "survival_function",
    "log_survival_function"])
KEY = jax.random.PRNGKey(0)


@pytest.fixture(autouse=True)
def double_precision():
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def sample(key, shape, low, high):
    return jax.random.uniform(key, shape, jp.float64, low, high)


def assert_close(actual, expected, rtol=1e-7, atol=1e-9):
    expected = jp.asarray(expected)
    assert jp.shape(actual) == expected.shape
    assert jp.allclose(actual, expected, rtol, atol, equal_nan=True)


# Undefined moments differ by convention (NaN here, inf or NaN in scipy), so
# only the defined ones are compared. Cancellation-prone statistics such as
# entropy are held to a looser bar.
def assert_moment(actual, expected):
    expected = jp.asarray(expected)
    assert jp.shape(actual) == expected.shape
    is_finite = jp.isfinite(expected)
    actual = jp.where(is_finite, actual, 0.0)
    expected = jp.where(is_finite, expected, 0.0)
    assert jp.allclose(actual, expected, rtol=1e-6, atol=1e-6)


SCIPY_NAMES = {
    "log_prob": "logpdf", "prob": "pdf", "cdf": "cdf", "log_cdf": "logcdf",
    "survival_function": "sf", "log_survival_function": "logsf",
    "mean": "mean", "variance": "var", "stddev": "std", "entropy": "entropy"}
DISCRETE_NAMES = {**SCIPY_NAMES, "log_prob": "logpmf", "prob": "pmf"}
DENSITY = ["log_prob", "prob"]
CUMULATIVE = DENSITY + ["cdf"]
SURVIVAL = ["survival_function", "log_survival_function"]
SCALAR = CUMULATIVE + ["log_cdf"] + SURVIVAL
SPREAD = ["mean", "variance", "stddev"]
MOMENTS = SPREAD + ["entropy"]


def assert_methods(distribution, reference, values, methods):
    assert_named_methods(distribution, reference, values, methods, SCIPY_NAMES)


def assert_named_methods(distribution, reference, values, methods, names):
    for method in methods:
        actual = getattr(distribution, method)(values)
        expected = getattr(reference, names[method])(values)
        assert_close(actual, expected)


def assert_moments(distribution, reference, methods):
    for method in methods:
        actual = getattr(distribution, method)()
        expected = getattr(reference, SCIPY_NAMES[method])()
        assert_moment(actual, expected)


def build_continuous(key):
    loc = sample(key[0], (4,), -5, 5)
    scale = sample(key[1], (4,), 0.05, 6)
    positive = sample(key[2], (4,), 0.05, 12)
    real = sample(key[3], (4,), -12, 12)
    df = sample(key[4], (4,), 0.5, 40)
    low = sample(key[5], (4,), -6, -0.4)
    high = sample(key[6], (4,), 0.4, 6)
    log_normal = stats.lognorm(scale, scale=jp.exp(loc))
    gamma = stats.gamma(positive, scale=1.0 / scale)
    inverse_gamma = stats.invgamma(positive, scale=scale)
    bounds = ((low - loc) / scale, (high - loc) / scale)
    truncated = stats.truncnorm(*bounds, loc, scale)
    truncated_normal = D.TruncatedNormal(loc, scale, low, high)
    with_log_cdf = CUMULATIVE + ["log_cdf"]
    # TruncatedNormal log_survival_function is the log of a probability
    # that rounds to zero in the far tail, so it is left out.
    without_log_survival = with_log_cdf + ["survival_function"]
    return [
        (D.Normal(loc, scale), stats.norm(loc, scale), real, SCALAR),
        (D.Laplace(loc, scale), stats.laplace(loc, scale), real, SCALAR),
        (D.StudentT(df, loc, scale), stats.t(df, loc, scale), real, DENSITY),
        (D.LogNormal(loc, scale), log_normal, positive, SCALAR),
        (D.Logistic(loc, scale), stats.logistic(loc, scale), real, SCALAR),
        (D.Gumbel(loc, scale), stats.gumbel_r(loc, scale), real, with_log_cdf),
        (D.Cauchy(loc, scale), stats.cauchy(loc, scale), real, CUMULATIVE),
        (D.Gamma(positive, scale), gamma, positive, CUMULATIVE),
        (D.Exponential(scale), stats.expon(0.0, 1.0 / scale), positive,
         CUMULATIVE + SURVIVAL),
        (D.InverseGamma(positive, scale), inverse_gamma, positive, CUMULATIVE),
        (D.Chi2(df), stats.chi2(df), positive, CUMULATIVE),
        (D.HalfNormal(scale), stats.halfnorm(0.0, scale), positive, SCALAR),
        (truncated_normal, truncated, real, without_log_survival),
    ]


def test_continuous_value_fuzz():
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, round_index), 8)
        for case in build_continuous(key):
            assert_methods(*case)


def test_continuous_moment_fuzz():
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, round_index), 8)
        for distribution, reference, _, _ in build_continuous(key):
            assert_moments(distribution, reference, MOMENTS)


def test_modes_by_hand():
    f = lambda value: jp.asarray(value, jp.float64)
    loc, scale = f(1.0), f(0.5)
    truncated = D.TruncatedNormal(f([-9.0, 0.0, 9.0]), scale, f(-2.0), f(3.0))
    diagonal = D.MultivariateNormalDiag(f([1.0, 2.0]), f([0.5, 0.5]))
    cases = [
        (D.Normal(loc, scale), loc),
        (D.Laplace(loc, scale), loc),
        (D.StudentT(f(3.0), loc, scale), loc),
        (D.LogNormal(loc, scale), jp.exp(loc - scale**2)),
        (D.Logistic(loc, scale), loc),
        (D.Gumbel(loc, scale), loc),
        (D.Cauchy(loc, scale), loc),
        (D.Gamma(f(3.0), f(2.0)), 1.0),
        (D.InverseGamma(f(3.0), f(2.0)), 0.5),
        (D.Chi2(f(5.0)), 3.0),
        (D.HalfNormal(scale), 0.0),
        (truncated, f([-2.0, 0.0, 3.0])),
        (D.Beta(f(2.0), f(3.0)), 1.0 / 3.0),
        (D.VonMises(loc, f(2.0)), loc),
        (D.Bernoulli(logits=f([-1.0, 1.0])), jp.array([0, 1])),
        (D.Poisson(rate=f([0.5, 3.7])), f([0.0, 3.0])),
        (D.Categorical(logits=f([0.1, 0.5, 0.2])), 1),
        (diagonal, f([1.0, 2.0])),
    ]
    for distribution, expected in cases:
        assert_close(distribution.mode(), expected, rtol=1e-12)


def test_positive_family_fuzz():
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, round_index), 6)
        loc = sample(key[1], (4,), 0.1, 5)
        beta_values = sample(key[2], (4,), 1e-3, 1 - 1e-3)
        alpha = sample(key[3], (4,), 0.1, 10)
        beta = sample(key[4], (4,), 0.1, 10)
        angles = sample(key[5], (4,), -3.14159, 3.14159)
        concentration = sample(key[0], (4,), 0.05, 60)
        reference = stats.beta(alpha, beta)
        assert_methods(D.Beta(alpha, beta), reference, beta_values, DENSITY)
        assert_moments(D.Beta(alpha, beta), reference, MOMENTS)
        von_mises = D.VonMises(loc, concentration)
        reference = stats.vonmises(concentration, loc)
        assert_methods(von_mises, reference, angles, DENSITY)
        bessel_ratio = special.i1e(concentration) / special.i0e(concentration)
        assert_moment(von_mises.mean(), loc)
        assert_moment(von_mises.variance(), 1.0 - bessel_ratio)
        assert_moment(von_mises.entropy(), reference.entropy())


def test_categorical_fuzz():
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, round_index), 2)
        logits = sample(key[0], (4, 5), -4, 4)
        integers = jax.random.randint(key[1], (4,), 0, 5)
        categorical = D.Categorical(logits=logits)
        log_probs = jax.nn.log_softmax(logits)
        expected = jp.take_along_axis(log_probs, integers[:, None], -1)[:, 0]
        assert_close(categorical.log_prob(integers), expected)
        assert_close(categorical.prob(integers), jp.exp(expected))
        entropy = -jp.sum(jp.exp(log_probs) * log_probs, axis=-1)
        assert_moment(categorical.entropy(), entropy)
        assert jp.array_equal(categorical.mode(), jp.argmax(logits, axis=-1))


def test_bernoulli_and_poisson_fuzz():
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, round_index), 4)
        logits = sample(key[0], (4,), -5, 5)
        binary = jp.round(sample(key[1], (4,), 0, 1))
        rate = sample(key[2], (4,), 0.05, 30)
        counts = jp.floor(sample(key[3], (4,), 0, 35))
        bernoulli = stats.bernoulli(jax.nn.sigmoid(logits))
        poisson = stats.poisson(rate)
        cases = [(D.Bernoulli(logits=logits), bernoulli, binary, MOMENTS),
                 (D.Poisson(rate=rate), poisson, counts, SPREAD)]
        for distribution, reference, values, moments in cases:
            args = (distribution, reference, values, DENSITY, DISCRETE_NAMES)
            assert_named_methods(*args)
            assert_moments(distribution, reference, moments)


def test_multivariate_normal_diag_fuzz():
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, round_index), 3)
        loc = sample(key[0], (3, 4), -3, 3)
        scale = sample(key[1], (3, 4), 0.05, 4)
        values = sample(key[2], (3, 4), -6, 6)
        normal = D.MultivariateNormalDiag(loc, scale)
        log_prob = stats.norm(loc, scale).logpdf(values).sum(axis=-1)
        assert_close(normal.log_prob(values), log_prob)
        assert_close(normal.prob(values), jp.exp(log_prob))
        entropy = stats.norm(loc, scale).entropy().sum(axis=-1)
        assert_moment(normal.entropy(), entropy)
        assert_moment(normal.mean(), loc)
        assert_moment(normal.variance(), scale**2)
        assert_moment(normal.stddev(), scale)
        assert_moment(normal.covariance(), stack_rows(jp.diag, scale**2))


def stack_rows(function, *arrays):
    return jp.stack([function(*rows) for rows in zip(*arrays)])


def test_dirichlet_fuzz():
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, round_index), 2)
        concentration = sample(key[0], (3, 4), 0.3, 6)
        simplex = jax.random.dirichlet(key[1], concentration)
        dirichlet = D.Dirichlet(concentration)
        log_pdf = lambda alpha, point: stats.dirichlet(alpha).logpdf(point)
        log_prob = stack_rows(log_pdf, concentration, simplex)
        assert_close(dirichlet.log_prob(simplex), log_prob)
        assert_close(dirichlet.prob(simplex), jp.exp(log_prob))
        for method in ["mean", "var", "entropy"]:
            compute = lambda alpha: getattr(stats.dirichlet(alpha), method)()
            expected = stack_rows(compute, concentration)
            name = "variance" if method == "var" else method
            assert_moment(getattr(dirichlet, name)(), expected)


def test_kl_divergence_fuzz():
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, 2000 + round_index), 6)
        loc_a = sample(key[0], (4,), -3, 3)
        loc_b = sample(key[1], (4,), -3, 3)
        scale_a = sample(key[2], (4,), 0.1, 3)
        scale_b = sample(key[3], (4,), 0.1, 3)
        logits_a = sample(key[4], (4, 5), -3, 3)
        logits_b = sample(key[5], (4, 5), -3, 3)
        pairs = [
            (D.Normal(loc_a, scale_a), D.Normal(loc_b, scale_b)),
            (D.Bernoulli(logits=loc_a), D.Bernoulli(logits=loc_b)),
            (D.Beta(scale_a, scale_b), D.Beta(scale_b, scale_a)),
            (D.Categorical(logits=logits_a), D.Categorical(logits=logits_b)),
        ]
        args = (loc_a, loc_b, scale_a, scale_b, logits_a, logits_b)
        for pair, expected in zip(pairs, compute_kl_references(*args)):
            assert_close(D.kl_divergence(*pair), expected)


def compute_kl_references(loc_a, loc_b, scale_a, scale_b, logits_a, logits_b):
    normal = jp.log(scale_b / scale_a) - 0.5
    normal += (scale_a**2 + (loc_a - loc_b)**2) / (2.0 * scale_b**2)
    probs = jax.nn.sigmoid(loc_a)
    log_ratio = jax.nn.log_sigmoid(loc_a) - jax.nn.log_sigmoid(loc_b)
    log_ratio_off = jax.nn.log_sigmoid(-loc_a) - jax.nn.log_sigmoid(-loc_b)
    bernoulli = probs * log_ratio + (1.0 - probs) * log_ratio_off
    # Beta(a, b) against Beta(b, a).
    difference = scale_a - scale_b
    digammas = special.digamma(scale_a) - special.digamma(scale_b)
    beta = difference * digammas
    log_a, log_b = jax.nn.log_softmax(logits_a), jax.nn.log_softmax(logits_b)
    categorical = jp.sum(jp.exp(log_a) * (log_a - log_b), axis=-1)
    return [normal, bernoulli, beta, categorical]


def test_uniform_and_deterministic_fuzz():
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, 5000 + round_index), 4)
        loc = sample(key[0], (3, 4), -3, 3)
        values = sample(key[1], (3, 4), -6, 6)
        low = sample(key[2], (3, 4), -7, -6)
        high = sample(key[3], (3, 4), 6, 7)
        uniform = D.Uniform(low, high)
        reference = stats.uniform(low, high - low)
        assert_methods(uniform, reference, values, CUMULATIVE)
        assert_methods(uniform, reference, values + 20.0, ["log_prob"])
        deterministic = D.Deterministic(jp.round(loc))
        is_equal = jp.round(values) == jp.round(loc)
        expected = jp.where(is_equal, 0.0, -jp.inf)
        assert_close(deterministic.log_prob(jp.round(values)), expected)


def test_independent_and_full_covariance_fuzz():
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, 5000 + round_index), 4)
        loc = sample(key[0], (3, 4), -3, 3)
        scale = sample(key[1], (3, 4), 0.05, 4)
        values = sample(key[2], (3, 4), -6, 6)
        covariance = build_covariance(key[3], (3, 4))
        independent = D.Independent(D.Normal(loc, scale), 1)
        log_prob = stats.norm(loc, scale).logpdf(values).sum(axis=-1)
        assert_close(independent.log_prob(values), log_prob)
        entropy = stats.norm(loc, scale).entropy().sum(axis=-1)
        assert_moment(independent.entropy(), entropy)
        assert_moment(independent.mean(), loc)
        normal = D.MultivariateNormalFullCovariance(loc, covariance)
        args = (compute_full_log_pdf, loc, covariance, values)
        log_prob = stack_rows(*args)
        assert_close(normal.log_prob(values), log_prob)
        assert_close(normal.prob(values), jp.exp(log_prob))
        entropy = stack_rows(compute_full_entropy, loc, covariance)
        assert_moment(normal.entropy(), entropy)
        assert_moment(normal.mean(), loc)
        assert_moment(normal.covariance(), covariance)


def build_covariance(key, shape):
    factor = sample(key, shape + shape[-1:], -1, 1)
    return factor @ jp.swapaxes(factor, -1, -2) + 0.5 * jp.eye(shape[-1])


def compute_full_log_pdf(mean, covariance, values):
    return stats.multivariate_normal(mean, covariance).logpdf(values)


def compute_full_entropy(mean, covariance):
    return stats.multivariate_normal(mean, covariance).entropy()


def test_mixture_same_family_fuzz():
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, 5000 + round_index), 5)
        loc = sample(key[0], (3, 4), -3, 3)
        scale = sample(key[1], (3, 4), 0.05, 4)
        values = sample(key[2], (3, 4), -6, 6)
        probs = jax.nn.softmax(sample(key[3], (3,), -1, 1))
        covariance = build_covariance(key[4], (3, 4))
        mixture = D.Categorical(probs=probs)
        scalar = D.MixtureSameFamily(mixture, D.Normal(loc[0, :3], 0.5))
        log_probs = stats.norm(loc[0, :3], 0.5).logpdf(values[0][:, None])
        expected = special.logsumexp(log_probs, axis=-1, b=probs)
        assert_close(scalar.log_prob(values[0]), expected)
        assert_moment(scalar.mean(), jp.sum(probs * loc[0, :3]))
        diagonal = stack_rows(jp.diag, scale**2)
        normal_diag = D.MultivariateNormalDiag(loc, scale)
        normal_full = D.MultivariateNormalFullCovariance(loc, covariance)
        cases = [(normal_diag, diagonal), (normal_full, covariance)]
        for components, covariances in cases:
            log_probs = []
            for mean, component_covariance in zip(loc, covariances):
                args = (mean, component_covariance)
                component = stats.multivariate_normal(*args)
                log_probs.append(component.logpdf(values))
            log_probs = jp.stack(log_probs, axis=-1)
            expected = special.logsumexp(log_probs, axis=-1, b=probs)
            actual = D.MixtureSameFamily(mixture, components).log_prob(values)
            assert_close(actual, expected)


def test_relaxed_one_hot_categorical_fuzz():
    # Concrete density: (K - 1)! t^(K - 1) prod(p_i y_i^(-t - 1)) divided by
    # sum(p_i y_i^(-t))^K, for temperature t and K classes.
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, 5000 + round_index), 3)
        logits = sample(key[0], (3, 4), -3, 3)
        simplex = jax.random.dirichlet(key[1], jp.ones((3, 4)))
        temperature = sample(key[2], (), 0.3, 2)
        relaxed = D.RelaxedOneHotCategorical(temperature, logits=logits)
        log_simplex = jp.log(simplex)
        scaled = logits - temperature * log_simplex
        expected = special.gammaln(4.0) + 3.0 * jp.log(temperature)
        expected += jp.sum(scaled - log_simplex, axis=-1)
        expected -= 4.0 * special.logsumexp(scaled, axis=-1)
        assert_close(relaxed.log_prob(simplex), expected)


def test_transformed_value_fuzz():
    for round_index in range(ROUNDS):
        key = jax.random.split(jax.random.fold_in(KEY, 6000 + round_index), 6)
        loc = sample(key[0], (), -1, 1)
        scale = sample(key[1], (), 0.3, 2)
        low = sample(key[2], (), -4, -1)
        width = sample(key[3], (), 1, 5)
        positive = sample(key[4], (6,), 0.05, 8)
        bounded = low + width * sample(key[5], (6,), 0.01, 0.99)
        bounding = B.Chain([B.Shift(low), B.Scale(width), B.Sigmoid()])
        base = D.Normal(loc, scale)
        log_normal = stats.lognorm(scale, scale=jp.exp(loc))
        transformed = D.TransformedDistribution(base, B.Exp())
        assert_methods(transformed, log_normal, positive, SCALAR)
        # A Normal pushed through low + width * sigmoid(x), by hand.
        transformed = D.TransformedDistribution(base, bounding)
        normal = stats.norm(loc, scale)
        unit = (bounded - low) / width
        points = special.logit(unit)
        log_det = jp.log(width) + jp.log(unit) + jp.log1p(-unit)
        log_prob = normal.logpdf(points) - log_det
        assert_close(transformed.log_prob(bounded), log_prob)
        assert_close(transformed.prob(bounded), jp.exp(log_prob))
        for method in ["cdf", "log_cdf"] + SURVIVAL:
            expected = getattr(normal, SCIPY_NAMES[method])(points)
            assert_close(getattr(transformed, method)(bounded), expected)


def test_quantized_value_fuzz():
    f = lambda value: jp.asarray(value, jp.float64)
    low, high = f(-3.0), f(4.0)
    for round_index in range(5):
        key = jax.random.split(jax.random.fold_in(KEY, 7000 + round_index), 2)
        loc = sample(key[0], (), -1, 1)
        scale = sample(key[1], (), 0.3, 2)
        log_normal = stats.lognorm(scale, scale=jp.exp(loc))
        # The grids reach far into the tails and outside the (low, high)
        # bounds. LogNormal is NaN below zero in TensorFlow Probability and
        # has zero mass here, so that grid starts at 1.
        cases = [
            (D.Normal(loc, scale), stats.norm(loc, scale),
             jp.arange(-40.0, 60.0, 3.0)),
            (D.LogNormal(loc, scale), log_normal, jp.arange(1.0, 60.0, 3.0)),
        ]
        for base, reference, grid in cases:
            expected = compute_quantized_reference(reference, grid)
            quantized = D.QuantizedDistribution(base)
            assert_quantized(quantized, grid, expected)
            args = (expected, reference, grid, low, high)
            expected = bound_quantized_reference(*args)
            quantized = D.QuantizedDistribution(base, low, high)
            assert_quantized(quantized, grid, expected)


# Mass of the integer bin (grid - 1, grid], taken from the survival function
# in the upper tail, where the cdf rounds to 1.
def compute_quantized_reference(base, grid):
    log_cdf, log_survival = base.logcdf(grid), base.logsf(grid)
    lower = jp.stack([log_cdf, base.logcdf(grid - 1.0)])
    upper = jp.stack([base.logsf(grid - 1.0), log_survival])
    pair = jp.where(log_survival < log_cdf, upper, lower)
    signs = jp.array([[1.0], [-1.0]])
    log_prob = special.logsumexp(pair, axis=0, b=signs)
    values = (log_prob, jp.exp(log_prob), base.cdf(grid), log_cdf)
    return Quantized(*values, base.sf(grid), log_survival)


def assert_quantized(quantized, grid, expected):
    for method, values in expected._asdict().items():
        assert_close(getattr(quantized, method)(grid), values)


# Outside (low, high) the log-probability is NaN, as in TensorFlow
# Probability. The bin at high holds all the mass above high - 1.
def bound_quantized_reference(reference, base, grid, low, high):
    log_prob, prob, cdf, log_cdf, survival, log_survival = reference
    below, above, at_high = grid < low, grid >= high, grid == high
    log_prob = jp.where(below | above, jp.nan, log_prob)
    log_prob = jp.where(at_high, base.logsf(high - 1.0), log_prob)
    prob = jp.where(below | above, 0.0, prob)
    prob = jp.where(at_high, base.sf(high - 1.0), prob)
    cdf = jp.where(below, 0.0, jp.where(above, 1.0, cdf))
    log_cdf = jp.where(below, -jp.inf, jp.where(above, 0.0, log_cdf))
    survival = jp.where(below, 1.0, jp.where(above, 0.0, survival))
    log_survival = jp.where(above, -jp.inf, log_survival)
    log_survival = jp.where(below, 0.0, log_survival)
    return Quantized(log_prob, prob, cdf, log_cdf, survival, log_survival)


def test_log_prob_gradient_matches_finite_differences():
    f = lambda value: jp.asarray(value, jp.float64)
    cases = [
        (D.Normal, stats.norm, (f(0.2), f(1.3)), f([-0.4, 0.3, 1.2])),
        (D.Gamma, build_scipy_gamma, (f(2.3), f(1.4)), f([0.5, 1.2, 3.0])),
        (D.Beta, stats.beta, (f(2.5), f(3.5)), f([0.2, 0.5, 0.8])),
        (D.StudentT, stats.t, (f(12.0), f(0.2), f(1.1)), f([-0.8, 0.3, 1.1])),
        (D.Dirichlet, stats.dirichlet, (f([2.0, 3.0, 4.0]),),
         f([0.2, 0.3, 0.5])),
    ]
    for build, reference, params, values in cases:
        loss = lambda *raw: jp.sum(reference(*raw).logpdf(values))
        gradients = compute_gradients(build, params, values)
        for index, gradient in enumerate(gradients):
            expected = compute_numeric_gradient(loss, params, index)
            assert_close(gradient, expected, rtol=0.0, atol=1e-8)


def build_scipy_gamma(concentration, rate):
    return stats.gamma(concentration, scale=1.0 / rate)


def compute_gradients(build, params, values):
    def loss(*raw):
        return jp.sum(build(*raw).log_prob(values))

    return jax.grad(loss, tuple(range(len(params))))(*params)


def compute_numeric_gradient(loss, params, index, step=1e-5):
    offsets = jp.eye(params[index].size).reshape((-1,) + params[index].shape)
    gradient = []
    for offset in offsets:
        upper, lower = list(params), list(params)
        upper[index] = params[index] + step * offset
        lower[index] = params[index] - step * offset
        gradient.append((loss(*upper) - loss(*lower)) / (2.0 * step))
    return jp.reshape(jp.array(gradient), params[index].shape)


def test_extreme_values_match_scipy():
    f = lambda value: jp.asarray(value, jp.float64)
    values = f([-jp.inf, -1e300, 0.0, 1e300, jp.inf, jp.nan])
    tiny = jp.finfo(jp.float64).tiny
    bounds = f([0.0, tiny, 1.0, jp.inf, jp.nan])
    normal_log_prob = D.Normal(f(0.0), f(1.0)).log_prob(values)
    assert_close(normal_log_prob, stats.norm(0.0, 1.0).logpdf(values))
    log_normal_log_prob = D.LogNormal(f(0.0), f(1.0)).log_prob(bounds)
    assert_close(log_normal_log_prob, stats.lognorm(1.0).logpdf(bounds))


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


def test_mixture_log_prob_matches_scipy():
    key = jax.random.split(jax.random.fold_in(KEY, 3000), 4)
    probs, loc, scale, values = build_depth_mixture(key)
    uniform = D.Uniform(jp.zeros(50), 5.0 * jp.ones(50))
    mixture = D.Mixture(probs, [D.Laplace(loc, scale), uniform])
    densities = [stats.laplace(loc, scale).pdf(values)]
    densities.append(stats.uniform(0.0, 5.0).pdf(values))
    expected = jp.log(jp.sum(probs * jp.stack(densities, axis=-1), axis=-1))
    assert_close(mixture.log_prob(values), expected)


def test_mixture_three_families_match_scipy():
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
    densities = [stats.norm(loc, 0.7).pdf(values)]
    densities.append(stats.laplace(-loc, 1.3).pdf(values))
    densities.append(stats.uniform(-2.0, 5.0).pdf(values))
    expected = jp.log(jp.sum(probs * jp.stack(densities, axis=-1), axis=-1))
    assert_close(mixture.log_prob(values), expected)


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


def test_mixture_jit_and_grad_match_closed_form():
    key = jax.random.split(jax.random.fold_in(KEY, 3002), 4)
    probs, loc, scale, values = build_depth_mixture(key)
    probability = probs[:, 0]

    def loss(probability, loc, scale):
        probs = jp.stack([probability, 1.0 - probability], axis=-1)
        uniform = D.Uniform(jp.zeros(50), 5.0 * jp.ones(50))
        components = [D.Laplace(loc, scale), uniform]
        return jp.sum(D.Mixture(probs, components).log_prob(values))

    params = (probability, loc, scale)
    gradients = jax.jit(jax.grad(loss, (0, 1, 2)))(*params)
    laplace = stats.laplace(loc, scale).pdf(values)
    uniform = stats.uniform(0.0, 5.0).pdf(values)
    density = probability * laplace + (1.0 - probability) * uniform
    assert_close(jax.jit(loss)(*params), jp.sum(jp.log(density)))
    # The Laplace log-density has slope sign(x - loc) / scale in loc and
    # |x - loc| / scale^2 - 1 / scale in scale.
    share = probability * laplace / density
    distance = jp.abs(values - loc)
    assert_close(gradients[0], (laplace - uniform) / density)
    assert_close(gradients[1], share * jp.sign(values - loc) / scale)
    assert_close(gradients[2], share * (distance / scale**2 - 1.0 / scale))


def test_mixture_sample_frequencies_match_probs():
    probs = jp.array([[0.2, 0.8], [0.7, 0.3]])
    mixture = D.Mixture(probs, [D.Normal(-10.0, 1.0), D.Normal(10.0, 1.0)])
    samples = mixture.sample(20000, seed=KEY)
    assert samples.shape == (20000, 2)
    assert mixture.batch_shape == (2,)
    negative_frequency = jp.mean(samples < 0.0, axis=0)
    assert jp.allclose(negative_frequency, probs[:, 0], atol=0.015)


def test_masked_log_prob_matches_scipy():
    key = jax.random.split(jax.random.fold_in(KEY, 4000), 3)
    loc = sample(key[0], (50,), -1.0, 1.0)
    values = sample(key[1], (50,), -3.0, 3.0)
    is_valid = jax.random.bernoulli(key[2], 0.6, (50,))
    masked = D.Masked(D.Normal(loc, 0.8), is_valid)
    log_prob = stats.norm(loc, 0.8).logpdf(values)
    assert_close(masked.log_prob(values), jp.where(is_valid, log_prob, 0.0))


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


def test_multivariate_normal_log_prob_broadcasts_values_and_parameters():
    key = jax.random.split(KEY, 3)
    loc, values = sample(key[0], (3,), -3, 3), sample(key[1], (4, 3), -6, 6)
    covariance = build_covariance(key[2], (3,))
    scale_tril = jp.linalg.cholesky(covariance)
    expected = compute_full_log_pdf(loc, covariance, values)
    full = D.MultivariateNormalFullCovariance(loc, covariance)
    assert_close(full.log_prob(values), expected)
    tril = D.MultivariateNormalTriL(loc, scale_tril)
    assert_close(tril.log_prob(values), expected)
    scales = jp.array([1.0, 2.0])[:, None, None]
    expected_0 = compute_full_log_pdf(loc, covariance, values[0])
    expected_1 = compute_full_log_pdf(loc, 2.0 * covariance, values[0])
    expected = jp.stack([expected_0, expected_1])
    full = D.MultivariateNormalFullCovariance(loc, scales * covariance)
    assert_close(full.log_prob(values[0]), expected)
    tril = D.MultivariateNormalTriL(loc, jp.sqrt(scales) * scale_tril)
    assert_close(tril.log_prob(values[0]), expected)


def test_transformed_cumulative_selects_direction_per_element():
    base = D.Normal(jp.array([0.5, 1.0]), 0.7)
    bijector = B.Scale(jp.array([2.0, -1.0]))
    transformed = D.TransformedDistribution(base, bijector)
    reference = stats.norm([1.0, -1.0], [1.4, 0.7])
    assert_methods(transformed, reference, jp.array([0.3, 0.3]), SCALAR)


def test_transformed_cumulative_is_jittable_over_the_bijector():
    def compute_cdf(scale, values):
        bijector = B.Chain([B.Shift(1.0), B.Scale(scale)])
        transformed = D.TransformedDistribution(D.Normal(0.5, 0.7), bijector)
        return transformed.cdf(values)

    expected = stats.norm(0.0, 1.4).cdf(0.3)
    assert_close(jax.jit(compute_cdf)(-2.0, 0.3), expected, rtol=1e-6)


def test_truncated_normal_moments_accept_python_floats():
    truncated = D.TruncatedNormal(0.5, 1.3, -1.0, 2.0)
    reference = stats.truncnorm(-1.5 / 1.3, 1.5 / 1.3, 0.5, 1.3)
    assert_moments(truncated, reference, MOMENTS)


def test_truncated_normal_samples_the_upper_tail_in_single_precision():
    f = lambda value: jp.asarray(value, jp.float32)
    truncated = D.TruncatedNormal(f(0.0), f(1.0), f(6.0), f(9.0))
    samples = truncated.sample(2000, seed=KEY)
    expected_mean = stats.truncnorm(6.0, 9.0).mean()
    assert jp.all((samples >= 6.0) & (samples <= 9.0))
    assert jp.isclose(samples.mean(), expected_mean, atol=0.02)


def test_exponential_has_no_mass_below_zero():
    exponential = D.Exponential(1.7)
    values = jp.array([-1.0, -0.1])
    assert jp.array_equal(exponential.log_prob(values), jp.full(2, -jp.inf))
    assert jp.array_equal(exponential.cdf(values), jp.zeros(2))
    assert jp.array_equal(exponential.survival_function(values), jp.ones(2))
    log_survival = exponential.log_survival_function(values)
    assert jp.array_equal(log_survival, jp.zeros(2))
    assert exponential.mode() == 0.0


def test_beta_log_prob_on_the_boundary_with_unit_concentration():
    f = lambda value: jp.asarray(value, jp.float64)
    alpha, beta = f([1.0, 1.0, 2.0]), f([1.0, 2.0, 1.0])
    values = f([0.0, 0.0, 1.0])
    expected = stats.beta(alpha, beta).logpdf(values)
    assert_close(D.Beta(alpha, beta).log_prob(values), expected)
