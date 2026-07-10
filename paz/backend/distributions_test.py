import jax
import jax.numpy as jp
import pytest

pytest.importorskip("tensorflow_probability")
from tensorflow_probability.substrates import jax as tfp

from paz.backend import distributions as D

tfd = tfp.distributions
tfb = tfp.bijectors

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


def assert_parity(name, ours, theirs, atol=1e-9, rtol=1e-7):
    ours = jp.asarray(ours)
    theirs = jp.asarray(theirs)
    assert ours.shape == theirs.shape, f"{name}: shape"
    assert jp.array_equal(jp.isnan(ours), jp.isnan(theirs)), f"{name}: nan"
    assert jp.array_equal(jp.isposinf(ours), jp.isposinf(theirs)), name
    assert jp.array_equal(jp.isneginf(ours), jp.isneginf(theirs)), name
    finite = jp.isfinite(ours) & jp.isfinite(theirs)
    if int(finite.sum()):
        gap = jp.abs(ours[finite] - theirs[finite])
        assert bool(jp.all(gap <= atol + rtol * jp.abs(theirs[finite]))), name


def check_methods(name, ours, theirs, values, methods):
    for method in methods:
        assert_parity(f"{name}.{method}", getattr(ours, method)(values),
                      getattr(theirs, method)(values))


def check_moments(name, ours, theirs, methods):
    # Cancellation-prone statistics such as entropy are held to a looser bar.
    for method in methods:
        try:
            their_value = getattr(theirs, method)()
        except NotImplementedError:
            continue
        assert_parity(f"{name}.{method}", getattr(ours, method)(),
                      their_value, atol=1e-6, rtol=1e-6)


SCALAR = ["log_prob", "prob", "cdf", "log_cdf", "survival_function",
          "log_survival_function"]
MOMENTS = ["mean", "variance", "stddev", "mode", "entropy"]


def build_continuous(key):
    loc = sample(key[0], (4,), -5, 5)
    scale = sample(key[1], (4,), 0.05, 6)
    positive = sample(key[2], (4,), 0.05, 12)
    real = sample(key[3], (4,), -12, 12)
    df = sample(key[4], (4,), 0.5, 40)
    low = sample(key[5], (4,), -6, -0.4)
    high = sample(key[6], (4,), 0.4, 6)
    return [
        ("Normal", D.Normal(loc, scale), tfd.Normal(loc, scale), real, SCALAR),
        ("Laplace", D.Laplace(loc, scale), tfd.Laplace(loc, scale), real,
         ["log_prob", "prob", "cdf"]),
        ("StudentT", D.StudentT(df, loc, scale), tfd.StudentT(df, loc, scale),
         real, ["log_prob", "prob"]),
        ("LogNormal", D.LogNormal(loc, scale), tfd.LogNormal(loc, scale),
         positive, ["log_prob", "prob", "cdf", "log_cdf"]),
        ("Logistic", D.Logistic(loc, scale), tfd.Logistic(loc, scale), real,
         SCALAR),
        ("Gumbel", D.Gumbel(loc, scale), tfd.Gumbel(loc, scale), real,
         ["log_prob", "prob", "cdf", "log_cdf"]),
        ("Cauchy", D.Cauchy(loc, scale), tfd.Cauchy(loc, scale), real,
         ["log_prob", "prob", "cdf"]),
        ("Gamma", D.Gamma(positive, scale), tfd.Gamma(positive, scale),
         positive, ["log_prob", "prob", "cdf"]),
        ("Exponential", D.Exponential(scale), tfd.Exponential(scale), positive,
         ["log_prob", "prob", "cdf", "survival_function",
          "log_survival_function"]),
        ("InverseGamma", D.InverseGamma(positive, scale),
         tfd.InverseGamma(positive, scale), positive,
         ["log_prob", "prob", "cdf"]),
        ("Chi2", D.Chi2(df), tfd.Chi2(df), positive,
         ["log_prob", "prob", "cdf"]),
        ("HalfNormal", D.HalfNormal(scale), tfd.HalfNormal(scale), positive,
         SCALAR),
        ("TruncatedNormal", D.TruncatedNormal(loc, scale, low, high),
         tfd.TruncatedNormal(loc, scale, low, high), real, SCALAR),
    ]


@pytest.mark.parametrize("round_index", range(ROUNDS))
def test_continuous_value_fuzz(round_index):
    key = jax.random.split(jax.random.fold_in(KEY, round_index), 8)
    for name, ours, theirs, values, methods in build_continuous(key):
        check_methods(name, ours, theirs, values, methods)


@pytest.mark.parametrize("round_index", range(ROUNDS))
def test_continuous_moment_fuzz(round_index):
    key = jax.random.split(jax.random.fold_in(KEY, round_index), 8)
    for name, ours, theirs, _, _ in build_continuous(key):
        check_moments(name, ours, theirs, MOMENTS)


@pytest.mark.parametrize("round_index", range(ROUNDS))
def test_positive_family_fuzz(round_index):
    key = jax.random.split(jax.random.fold_in(KEY, round_index), 6)
    scale = sample(key[1], (4,), 0.1, 5)
    beta_values = sample(key[2], (4,), 1e-3, 1 - 1e-3)
    alpha = sample(key[3], (4,), 0.1, 10)
    beta = sample(key[4], (4,), 0.1, 10)
    angles = sample(key[5], (4,), -3.14159, 3.14159)
    check_methods("Beta", D.Beta(alpha, beta), tfd.Beta(alpha, beta),
                  beta_values, ["log_prob", "prob"])
    check_moments("Beta", D.Beta(alpha, beta), tfd.Beta(alpha, beta), MOMENTS)
    concentration = sample(key[0], (4,), 0.05, 60)
    check_methods("VonMises", D.VonMises(scale, concentration),
                  tfd.VonMises(scale, concentration), angles,
                  ["log_prob", "prob"])
    check_moments("VonMises", D.VonMises(scale, concentration),
                  tfd.VonMises(scale, concentration),
                  ["mean", "variance", "mode", "entropy"])


@pytest.mark.parametrize("round_index", range(ROUNDS))
def test_discrete_value_fuzz(round_index):
    key = jax.random.split(jax.random.fold_in(KEY, round_index), 6)
    logits = sample(key[0], (4, 5), -4, 4)
    integers = jax.random.randint(key[1], (4,), -1, 6)
    bernoulli_logits = sample(key[2], (4,), -5, 5)
    binary = jp.round(sample(key[3], (4,), 0, 1))
    rate = sample(key[4], (4,), 0.05, 30)
    counts = jp.floor(sample(key[5], (4,), 0, 35))
    check_methods("Categorical", D.Categorical(logits=logits),
                  tfd.Categorical(logits=logits), integers,
                  ["log_prob", "prob"])
    check_moments("Categorical", D.Categorical(logits=logits),
                  tfd.Categorical(logits=logits), ["mode", "entropy"])
    check_methods("Bernoulli", D.Bernoulli(logits=bernoulli_logits),
                  tfd.Bernoulli(logits=bernoulli_logits), binary,
                  ["log_prob", "prob"])
    check_moments("Bernoulli", D.Bernoulli(logits=bernoulli_logits),
                  tfd.Bernoulli(logits=bernoulli_logits), MOMENTS)
    check_methods("Poisson", D.Poisson(rate=rate), tfd.Poisson(rate=rate),
                  counts, ["log_prob", "prob"])
    check_moments("Poisson", D.Poisson(rate=rate), tfd.Poisson(rate=rate),
                  ["mean", "variance", "stddev", "mode"])


@pytest.mark.parametrize("round_index", range(ROUNDS))
def test_multivariate_value_fuzz(round_index):
    key = jax.random.split(jax.random.fold_in(KEY, round_index), 6)
    loc = sample(key[0], (3, 4), -3, 3)
    scale_diag = sample(key[1], (3, 4), 0.05, 4)
    values = sample(key[2], (3, 4), -6, 6)
    concentration = sample(key[3], (3, 4), 0.3, 6)
    simplex = jax.random.dirichlet(key[4], concentration)
    check_methods("MVNDiag", D.MultivariateNormalDiag(loc, scale_diag),
                  tfd.MultivariateNormalDiag(loc, scale_diag), values,
                  ["log_prob", "prob"])
    check_moments("MVNDiag", D.MultivariateNormalDiag(loc, scale_diag),
                  tfd.MultivariateNormalDiag(loc, scale_diag),
                  ["mean", "variance", "stddev", "mode", "entropy",
                   "covariance"])
    check_methods("Dirichlet", D.Dirichlet(concentration),
                  tfd.Dirichlet(concentration), simplex, ["log_prob", "prob"])
    check_moments("Dirichlet", D.Dirichlet(concentration),
                  tfd.Dirichlet(concentration), ["mean", "variance", "entropy"])


@pytest.mark.parametrize("round_index", range(ROUNDS))
def test_kl_divergence_fuzz(round_index):
    key = jax.random.split(jax.random.fold_in(KEY, 2000 + round_index), 8)
    loc_a = sample(key[0], (4,), -3, 3)
    loc_b = sample(key[1], (4,), -3, 3)
    scale_a = sample(key[2], (4,), 0.1, 3)
    scale_b = sample(key[3], (4,), 0.1, 3)
    logits_a = sample(key[4], (4, 5), -3, 3)
    logits_b = sample(key[5], (4, 5), -3, 3)
    pairs = [
        ("Normal", D.Normal(loc_a, scale_a), D.Normal(loc_b, scale_b),
         tfd.Normal(loc_a, scale_a), tfd.Normal(loc_b, scale_b)),
        ("Bernoulli", D.Bernoulli(logits=loc_a), D.Bernoulli(logits=loc_b),
         tfd.Bernoulli(logits=loc_a), tfd.Bernoulli(logits=loc_b)),
        ("Beta", D.Beta(scale_a, scale_b), D.Beta(scale_b, scale_a),
         tfd.Beta(scale_a, scale_b), tfd.Beta(scale_b, scale_a)),
        ("Categorical", D.Categorical(logits=logits_a),
         D.Categorical(logits=logits_b), tfd.Categorical(logits=logits_a),
         tfd.Categorical(logits=logits_b)),
    ]
    for name, our_a, our_b, their_a, their_b in pairs:
        assert_parity(f"KL[{name}]", D.kl_divergence(our_a, our_b),
                      tfd.kl_divergence(their_a, their_b))


def gradient_gap(build_ours, build_theirs, params, values):
    def our_loss(*raw):
        return jp.sum(build_ours(*raw).log_prob(values))

    def their_loss(*raw):
        return jp.sum(build_theirs(*raw).log_prob(values))

    argnums = tuple(range(len(params)))
    our_grad = jax.grad(our_loss, argnums)(*params)
    their_grad = jax.grad(their_loss, argnums)(*params)
    gaps = [float(jp.max(jp.abs(a - b)))
            for a, b in zip(our_grad, their_grad)]
    return max(gaps)


def test_log_prob_gradient_parity():
    f = lambda value: jp.asarray(value, jp.float64)
    cases = [
        (lambda a, b: D.Normal(a, b), lambda a, b: tfd.Normal(a, b),
         (f(0.2), f(1.3)), f([-0.4, 0.3, 1.2])),
        (lambda a, b: D.Gamma(a, b), lambda a, b: tfd.Gamma(a, b),
         (f(2.3), f(1.4)), f([0.5, 1.2, 3.0])),
        (lambda a, b: D.Beta(a, b), lambda a, b: tfd.Beta(a, b),
         (f(2.5), f(3.5)), f([0.2, 0.5, 0.8])),
        (lambda d, a, b: D.StudentT(d, a, b),
         lambda d, a, b: tfd.StudentT(d, a, b),
         (f(12.0), f(0.2), f(1.1)), f([-0.8, 0.3, 1.1])),
        (lambda c: D.Dirichlet(c), lambda c: tfd.Dirichlet(c),
         (f([2.0, 3.0, 4.0]),), f([0.2, 0.3, 0.5])),
    ]
    for build_ours, build_theirs, params, values in cases:
        assert gradient_gap(build_ours, build_theirs, params, values) < 1e-9


def test_extreme_masks_match_tfp():
    f = lambda value: jp.asarray(value, jp.float64)
    values = f([-jp.inf, -1e300, 0.0, 1e300, jp.inf, jp.nan])
    assert_parity("Normal.log_prob", D.Normal(0.0, 1.0).log_prob(values),
                  tfd.Normal(f(0.0), f(1.0)).log_prob(values))
    tiny = jp.finfo(jp.float64).tiny
    bounds = f([0.0, tiny, 1.0, jp.inf, jp.nan])
    assert_parity("LogNormal.log_prob", D.LogNormal(0.0, 1.0).log_prob(bounds),
                  tfd.LogNormal(f(0.0), f(1.0)).log_prob(bounds))
