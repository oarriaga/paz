# Probability Demos

The distribution and bijector layer that replaces TensorFlow Probability now
lives in the library as `paz.distributions` and `paz.bijectors`
(`paz/backend/distributions.py` and `paz/backend/bijectors.py`, with the shared
numeric helpers in `paz/backend/standard.py`). This directory keeps the
runnable demos that exercise that layer.

## Implemented Surface
- Distributions:
  `Deterministic`, `Normal`, `Laplace`, `StudentT`, `Uniform`,
  `LogNormal`, `TruncatedNormal`, `Beta`, `VonMises`,
  `Bernoulli`, `Categorical`, `Poisson`,
  `Gamma`, `Exponential`, `InverseGamma`, `Chi2`,
  `HalfNormal`, `Cauchy`, `HalfCauchy`, `Logistic`, `Gumbel`,
  `Dirichlet`,
  `Independent`, `TransformedDistribution`,
  `RelaxedOneHotCategorical`, `QuantizedDistribution`,
  `MultivariateNormalDiag`, `MultivariateNormalFullCovariance`,
  `MultivariateNormalTriL`, `MixtureSameFamily`
- Distribution statistics:
  `mean`, `variance`, `stddev`, `mode`, `entropy`, and `kl_divergence`
- Bijectors:
  `Identity`, `Shift`, `Scale`, `Sigmoid`,
  `Exp`, `Log`, `Softplus`, `SoftmaxCentered`,
  `Tanh`, `Square`, `Reciprocal`, `Power`, `NormalCDF`, `Cumsum`,
  `Invert`, `Chain`

## Verification
Parity against the TensorFlow Probability JAX substrate is checked by
`paz/backend/distributions_test.py` and `paz/backend/bijectors_test.py`. Those
tests skip automatically when TFP is not installed and match TFP at machine
precision for value methods (`1e-6` for cancellation-prone statistics).

## Commands
```bash
pytest paz/backend/distributions_test.py
pytest paz/backend/bijectors_test.py
JAX_PLATFORMS=cpu python3 -m examples.probability_core.demo_fit_bijector
JAX_PLATFORMS=cpu python3 -m examples.probability_core.demo_linear_regression
JAX_PLATFORMS=cpu \
python3 -m examples.probability_core.demo_hierarchical_regression
```
