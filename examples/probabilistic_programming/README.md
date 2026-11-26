# Probabilistic programming

Small models written with `paz.inference`: priors, likelihoods, MCMC
sampling, marginalized discrete variables and model serialization.

| Script | What it does |
| --- | --- |
| `change_point_switch.py` | Change-point linear regression; marginalizes the discrete switch index. |
| `discretize_distribution.py` | Discretizes a normal and a skewed distribution into a fixed number of steps. |
| `fit_bijector.py` | Fits a bijector so a base distribution matches a LogNormal target. |
| `fit_ellipsoid.py` | Fits an ellipsoid to the pointcloud of a FewSOL object. |
| `gaussian_mixture_model.py` | Fits Gaussian mixtures with EM on 1D, 2D and ring data. |
| `gaussian_mixture_model_inference_comparison.py` | Compares inference methods on a two-component mixture. |
| `hierarchical_linear_regression.py` | Grouped regression in centered and non-centered form. |
| `ising_model.py` | Infers the coupling of an Ising lattice with a pseudolikelihood. |
| `linear_regression.py` | Bayesian linear regression sampled with several chains. |
| `lotka_volterra_ode.py` | Fits a predator-prey ODE to the lynx and hare data. |
| `multi_change_point_switch.py` | Regression with several change points indexed by one discrete variable. |
| `neals_funnel.py` | Samples Neal's funnel in centered and non-centered form. |
| `normal_normal.py` | Compares a sampled posterior with the analytical normal-normal posterior. |
| `posterior_sampling.py` | Samples a log density and prints acceptance rate, mean and stdv. |
| `robust_outlier_switch.py` | Robust regression with a marginalized inlier/outlier switch. |
| `serialization_overview.py` | Saves a model with `paz.inference.save`, reloads it and compares log probabilities. |
| `tune_sigma.py` | Compares sampling with a tuned proposal sigma against the initial sigma. |

## Run

```bash
python3 examples/probabilistic_programming/posterior_sampling.py \
    --seed 777 --sigma 1.0 --num_samples 10000 --num_chains 5
```

The script prints the chain acceptance rate and the sample mean and stdv
next to the true values. Most other scripts open matplotlib figures.

## Requirements

`fit_ellipsoid.py` needs the FewSOL dataset and `pyrender`.
