# Probability Core Prototype

This directory starts the TensorFlow Probability removal work outside
`paz/`. It now covers the full distribution and bijector surface that the
current repo uses, with a TFP-shaped API and deep parity tests against the
installed TFP JAX substrate.

## Ranked Extraction Backlog
1. Base protocols: `Distribution`, `Bijector`, pytrees, `sample`,
   `log_prob`, `prob`, `inverse`, `forward_log_det_jacobian`
2. Core bijectors: `Identity`, `Chain`, `Shift`, `Scale`, `Sigmoid`
3. Core continuous distributions: `Normal`, `Uniform`
4. Wrapper distributions: `Independent`, `TransformedDistribution`
5. Discrete scalar distributions: `Categorical`, `Bernoulli`
6. Grid wrapper: `QuantizedDistribution`
7. Multivariate Gaussians: `MultivariateNormalDiag`,
   `MultivariateNormalFullCovariance`
8. Mixtures: `MixtureSameFamily`
9. Support-constrained priors: `LogNormal`, `TruncatedNormal`, `Laplace`
10. Specialized priors: `Beta`, `VonMises`, `RelaxedOneHotCategorical`
11. Serialization tail: `Deterministic`, `StudentT`, `Poisson`
12. Late bijectors: `Softplus`, `Exp`, `SoftmaxCentered`, `Invert`

## Implemented Surface
- Distributions:
  `Deterministic`, `Normal`, `Laplace`, `StudentT`, `Uniform`,
  `LogNormal`, `TruncatedNormal`, `Beta`, `VonMises`,
  `Bernoulli`, `Categorical`, `Poisson`,
  `Independent`, `TransformedDistribution`,
  `RelaxedOneHotCategorical`, `QuantizedDistribution`,
  `MultivariateNormalDiag`, `MultivariateNormalFullCovariance`,
  `MixtureSameFamily`
- Bijectors:
  `Identity`, `Shift`, `Scale`, `Sigmoid`,
  `Exp`, `Softplus`, `SoftmaxCentered`, `Invert`, `Chain`
- Fitting:
  `fit_bijector`

## Test Scope
- Shape, dtype, `log_prob`, and `prob` parity for all implemented classes
- Value-method parity for repo-used APIs such as
  `quantile`, `cdf`, `log_cdf`, `probs_parameter`,
  `logits_parameter`, `variance`, and `covariance`
- Bijector `forward`, `inverse`, and `forward_log_det_jacobian` parity,
  including vector-event cases such as `SoftmaxCentered`
- Sampling parity for the core frequently used families

## Commands
```bash
pytest examples/probability_core/core_test.py
pytest examples/probability_core/parity_test.py
pytest examples/probability_core
JAX_PLATFORMS=cpu python3 -m examples.probability_core.demo_fit_bijector
JAX_PLATFORMS=cpu python3 -m examples.probability_core.demo_linear_regression
JAX_PLATFORMS=cpu \
python3 -m examples.probability_core.demo_hierarchical_regression
```

## Next Integration Step
- Move these modules into `paz/probability/`
- Switch `paz/inference/prior.py`, `latent.py`, `latent_space.py`, and
  `bijector_fitting.py` to the paz-native layer
- Then replace the TFP imports in the remaining probabilistic examples,
  GMM code, discretizer, and serializer
