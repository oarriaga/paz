"""
Normal-Normal Bayesian Model with Analytical Posterior Verification

Model:
    Prior:      μ ~ Normal(μ₀, σ₀)
    Likelihood: X₁, ..., Xₙ | μ ~ Normal(μ, σ)  [σ known]

Analytical Posterior:
    μ | X ~ Normal(μ_post, σ_post)

    where:
        σ²_post = 1 / (1/σ₀² + n/σ²)
        μ_post  = σ²_post * (μ₀/σ₀² + n*X̄/σ²)

This example verifies that MCMC samples converge to the analytical posterior.
"""
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
from collections import namedtuple
from tensorflow_probability.substrates import jax as tfp

import paz

tfd = tfp.distributions


AnalyticalPosterior = namedtuple("AnalyticalPosterior", ["mean", "stdv"])


def compute_analytical_posterior(prior_mean, prior_stdv, likelihood_stdv, observations):
    """Compute the analytical posterior for the Normal-Normal model."""
    n = len(observations)
    obs_mean = observations.mean()

    prior_precision = 1.0 / prior_stdv**2
    likelihood_precision = n / likelihood_stdv**2

    posterior_precision = prior_precision + likelihood_precision
    posterior_variance = 1.0 / posterior_precision
    posterior_stdv = jp.sqrt(posterior_variance)

    posterior_mean = posterior_variance * (
        prior_mean * prior_precision + obs_mean * likelihood_precision
    )

    return AnalyticalPosterior(posterior_mean, posterior_stdv)


def build_normal_normal_model(prior_mean, prior_stdv, likelihood_stdv, observations):
    """Build the Normal-Normal PGM."""

    def Likelihood(likelihood_stdv, observations):
        n = len(observations)

        def apply(mu):
            return tfd.Normal(loc=jp.full(n, mu), scale=likelihood_stdv)

        return apply

    mu = paz.Prior(tfd.Normal(prior_mean, prior_stdv), name="mu")
    x = paz.Observable(
        Likelihood(likelihood_stdv, observations), observations, name="x"
    )(mu)
    return paz.PGM([mu], [x], "normal_normal")


def run_mcmc(model, key, num_samples, num_chains, sigma):
    """Run MCMC sampling."""
    log_density_fn = lambda params: model.apply(params).log_prob_sum

    key, init_key = jax.random.split(key)
    positions = model.sample_inverse(init_key, num_chains)

    samples, infos = paz.metropolis_hastings.sample(
        key, log_density_fn, positions, sigma, num_samples, num_chains
    )
    return samples, infos


def compute_mcmc_statistics(samples, burn_in):
    """Compute mean and std from MCMC samples after burn-in."""
    posterior_samples = samples.position.mu[burn_in:].flatten()
    return posterior_samples.mean(), posterior_samples.std()


def verify_posterior(analytical, mcmc_mean, mcmc_stdv, tolerance_mean=0.05, tolerance_stdv=0.1):
    """Verify MCMC samples match analytical posterior within tolerance."""
    mean_error = abs(mcmc_mean - analytical.mean) / abs(analytical.mean)
    stdv_error = abs(mcmc_stdv - analytical.stdv) / analytical.stdv

    mean_ok = mean_error < tolerance_mean
    stdv_ok = stdv_error < tolerance_stdv

    return mean_ok, stdv_ok, mean_error, stdv_error


def plot_results(samples, burn_in, analytical, prior_mean, prior_stdv):
    """Plot MCMC samples against analytical posterior."""
    posterior_samples = samples.position.mu[burn_in:].flatten()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Trace plot
    axes[0].plot(samples.position.mu[:, 0], alpha=0.7, linewidth=0.5)
    axes[0].axhline(analytical.mean, color="red", linestyle="--", label="Analytical mean")
    axes[0].axvline(burn_in, color="gray", linestyle=":", label="Burn-in")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("μ")
    axes[0].set_title("Trace plot (chain 0)")
    axes[0].legend()

    # Histogram vs analytical posterior
    x_range = jp.linspace(
        analytical.mean - 4 * analytical.stdv,
        analytical.mean + 4 * analytical.stdv,
        200,
    )
    analytical_pdf = tfd.Normal(analytical.mean, analytical.stdv).prob(x_range)
    prior_pdf = tfd.Normal(prior_mean, prior_stdv).prob(x_range)

    axes[1].hist(posterior_samples, bins=50, density=True, alpha=0.7, label="MCMC samples")
    axes[1].plot(x_range, analytical_pdf, "r-", linewidth=2, label="Analytical posterior")
    axes[1].plot(x_range, prior_pdf, "g--", linewidth=2, label="Prior")
    axes[1].set_xlabel("μ")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Posterior distribution")
    axes[1].legend()

    # QQ plot
    sorted_samples = jp.sort(posterior_samples)
    n = len(sorted_samples)
    theoretical_quantiles = tfd.Normal(analytical.mean, analytical.stdv).quantile(
        jp.linspace(0.001, 0.999, n)
    )
    axes[2].scatter(theoretical_quantiles, sorted_samples, alpha=0.3, s=1)
    lims = [
        min(theoretical_quantiles.min(), sorted_samples.min()),
        max(theoretical_quantiles.max(), sorted_samples.max()),
    ]
    axes[2].plot(lims, lims, "r--", linewidth=2)
    axes[2].set_xlabel("Theoretical quantiles")
    axes[2].set_ylabel("Sample quantiles")
    axes[2].set_title("Q-Q plot")

    plt.tight_layout()
    return fig


def main():
    # Model parameters
    prior_mean = 0.0
    prior_stdv = 2.0
    likelihood_stdv = 1.0
    true_mu = 3.0

    # Generate synthetic observations
    key = jax.random.PRNGKey(42)
    key, data_key = jax.random.split(key)
    n_observations = 20
    observations = true_mu + likelihood_stdv * jax.random.normal(data_key, (n_observations,))

    print("=" * 60)
    print("Normal-Normal Bayesian Model")
    print("=" * 60)
    print(f"\nPrior:      μ ~ Normal({prior_mean}, {prior_stdv})")
    print(f"Likelihood: X | μ ~ Normal(μ, {likelihood_stdv})")
    print(f"\nTrue μ:           {true_mu}")
    print(f"Observations:     n={n_observations}, mean={observations.mean():.4f}")

    # Compute analytical posterior
    analytical = compute_analytical_posterior(
        prior_mean, prior_stdv, likelihood_stdv, observations
    )
    print(f"\nAnalytical Posterior:")
    print(f"  μ | X ~ Normal({analytical.mean:.4f}, {analytical.stdv:.4f})")

    # Build model and run MCMC
    model = build_normal_normal_model(prior_mean, prior_stdv, likelihood_stdv, observations)

    num_samples = 5000
    num_chains = 4
    sigma = 0.2
    burn_in = 1000

    print(f"\nRunning MCMC...")
    print(f"  Samples: {num_samples}, Chains: {num_chains}, Burn-in: {burn_in}")

    samples, infos = run_mcmc(model, key, num_samples, num_chains, sigma)

    # Compute MCMC statistics
    mcmc_mean, mcmc_stdv = compute_mcmc_statistics(samples, burn_in)
    acceptance_rate = infos.acceptance_rate[burn_in:].mean()

    print(f"\nMCMC Results:")
    print(f"  Acceptance rate: {acceptance_rate:.3f}")
    print(f"  Posterior mean:  {mcmc_mean:.4f} (analytical: {analytical.mean:.4f})")
    print(f"  Posterior stdv:  {mcmc_stdv:.4f} (analytical: {analytical.stdv:.4f})")

    # Verify convergence
    mean_ok, stdv_ok, mean_error, stdv_error = verify_posterior(
        analytical, mcmc_mean, mcmc_stdv
    )

    print(f"\nVerification:")
    print(f"  Mean error: {mean_error:.2%} {'✓' if mean_ok else '✗'}")
    print(f"  Stdv error: {stdv_error:.2%} {'✓' if stdv_ok else '✗'}")

    if mean_ok and stdv_ok:
        print("\n✓ MCMC converged to analytical posterior!")
    else:
        print("\n✗ MCMC did not converge - try more samples or tuning")

    # Plot results
    fig = plot_results(samples, burn_in, analytical, prior_mean, prior_stdv)
    plt.show()

    return analytical, mcmc_mean, mcmc_stdv


if __name__ == "__main__":
    main()
