"""
Normal-Normal Bayesian Model with Analytical Posterior Verification

Model:
    Prior:      mu ~ Normal(mu_0, sigma_0)
    Likelihood: X_1, ..., X_n | mu ~ Normal(mu, sigma)  [sigma known]

Analytical Posterior:
    mu | X ~ Normal(mu_post, sigma_post)

    where:
        sigma^2_post = 1 / (1/sigma_0^2 + n/sigma^2)
        mu_post  = sigma^2_post * (mu_0/sigma_0^2 + n*X_bar/sigma^2)

This example verifies that MCMC samples converge to the analytical posterior.
"""

import jax
import jax.numpy as jp
import numpy as np
from collections import namedtuple
from tensorflow_probability.substrates import jax as tfp

import paz
import paz.plot as plot

tfd = tfp.distributions


AnalyticalPosterior = namedtuple("AnalyticalPosterior", ["mean", "stdv"])


def compute_analytical_posterior(
    prior_mean, prior_stdv, likelihood_stdv, observations
):
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


def build_normal_normal_model(
    prior_mean, prior_stdv, likelihood_stdv, num_observations
):
    """Build the Normal-Normal PGM."""

    def Likelihood(likelihood_stdv, num_observations):
        def apply(mu):
            return tfd.Normal(
                loc=jp.full(num_observations, mu), scale=likelihood_stdv
            )

        return apply

    mu = paz.Prior(tfd.Normal(prior_mean, prior_stdv), name="mu")
    x = paz.Observable(Likelihood(likelihood_stdv, num_observations), name="x")(
        mu
    )
    return paz.PGM([mu], [x], "normal_normal")


def run_mcmc(model, key, data, num_samples, num_chains, sigma, warmup):
    """Run MCMC sampling."""
    model.configure(
        num_chains=num_chains,
        warmup=warmup,
        sigma=sigma,
        tuner=paz.AdaptiveStepTuner(sigma=sigma),
    )
    return model.infer(key, data, num_samples=num_samples)


def compute_mcmc_statistics(samples):
    """Compute mean and std from MCMC samples."""
    posterior_samples = samples.mu.flatten()
    return posterior_samples.mean(), posterior_samples.std()


def verify_posterior(
    analytical, mcmc_mean, mcmc_stdv, tolerance_mean=0.05, tolerance_stdv=0.1
):
    """Verify MCMC samples match analytical posterior within tolerance."""
    mean_error = abs(mcmc_mean - analytical.mean) / abs(analytical.mean)
    stdv_error = abs(mcmc_stdv - analytical.stdv) / analytical.stdv

    mean_ok = mean_error < tolerance_mean
    stdv_ok = stdv_error < tolerance_stdv

    return mean_ok, stdv_ok, mean_error, stdv_error


def main():
    # Configure plotting
    plot.configure(fontsize=14, latex=False)

    # Model parameters
    prior_mean = 0.0
    prior_stdv = 2.0
    likelihood_stdv = 1.0
    true_mu = 3.0

    # Generate synthetic observations
    key = jax.random.PRNGKey(42)
    key, data_key = jax.random.split(key)
    n_observations = 20
    observations = true_mu + likelihood_stdv * jax.random.normal(
        data_key, (n_observations,)
    )

    print("=" * 60)
    print("Normal-Normal Bayesian Model")
    print("=" * 60)
    print(f"\nPrior:      mu ~ Normal({prior_mean}, {prior_stdv})")
    print(f"Likelihood: X | mu ~ Normal(mu, {likelihood_stdv})")
    print(f"\nTrue mu:          {true_mu}")
    print(
        f"Observations:     n={n_observations}, mean={observations.mean():.4f}"
    )

    # Compute analytical posterior
    analytical = compute_analytical_posterior(
        prior_mean, prior_stdv, likelihood_stdv, observations
    )
    print(f"\nAnalytical Posterior:")
    print(f"  mu | X ~ Normal({analytical.mean:.4f}, {analytical.stdv:.4f})")

    # Build model and run MCMC
    model = build_normal_normal_model(
        prior_mean, prior_stdv, likelihood_stdv, n_observations
    )
    data = {"x": observations}

    num_samples = 5000
    num_chains = 4
    sigma = 0.2
    warmup = 1000

    print(f"\nRunning MCMC...")
    print(f"  Samples: {num_samples}, Chains: {num_chains}, Warmup: {warmup}")

    posterior = run_mcmc(
        model, key, data, num_samples, num_chains, sigma, warmup
    )
    samples, infos = posterior.inverse_samples, posterior.infos
    gaussian_density = posterior.as_density(method="gaussian")

    # Compute MCMC statistics
    mcmc_mean, mcmc_stdv = compute_mcmc_statistics(samples)
    acceptance_rate = infos.acceptance_rate.mean()

    print(f"\nMCMC Results:")
    print(f"  Acceptance rate: {acceptance_rate:.3f}")
    print(
        f"  Posterior mean:  {mcmc_mean:.4f} (analytical: {analytical.mean:.4f})"
    )
    print(
        f"  Posterior stdv:  {mcmc_stdv:.4f} (analytical: {analytical.stdv:.4f})"
    )

    # Verify convergence
    mean_ok, stdv_ok, mean_error, stdv_error = verify_posterior(
        analytical, mcmc_mean, mcmc_stdv
    )

    print(f"\nVerification:")
    print(f"  Mean error: {mean_error:.2%} {'[OK]' if mean_ok else '[FAIL]'}")
    print(f"  Stdv error: {stdv_error:.2%} {'[OK]' if stdv_ok else '[FAIL]'}")

    if mean_ok and stdv_ok:
        print("\n[OK] MCMC converged to analytical posterior!")
    else:
        print("\n[FAIL] MCMC did not converge - try more samples or tuning")

    posterior_samples = samples.mu.flatten()

    # Trace panel for all chains
    plot.trace_panel({"mu": samples.mu}, title="MCMC Traces")
    plot.show()

    # Trace plot with analytical reference
    fig, ax = plot.subplots()
    plot.trace_lines(samples.mu, ax)
    plot.hline(
        float(analytical.mean),
        ax,
        color=plot.EARTH.primary,
        linestyle="--",
        label="Analytical mean",
    )
    plot.hline(
        float(true_mu),
        ax,
        color=plot.EARTH.secondary,
        linestyle=":",
        label="True mu",
    )
    plot.set_labels(ax, x="Iteration", y="mu")
    plot.clean(ax)
    ax.legend()
    ax.set_title("Trace plot (all chains)")
    plot.show()

    # Distribution comparison: MCMC histogram + analytical + prior + Gaussian fit
    x_range = np.linspace(
        float(analytical.mean - 4 * analytical.stdv),
        float(analytical.mean + 4 * analytical.stdv),
        200,
    )
    analytical_pdf = np.array(
        tfd.Normal(analytical.mean, analytical.stdv).prob(x_range)
    )
    prior_pdf = np.array(tfd.Normal(prior_mean, prior_stdv).prob(x_range))
    gaussian_fit_pdf = np.array(gaussian_density.prob({"mu": x_range}))

    fig, ax = plot.subplots()
    plot.histogram(
        np.array(posterior_samples),
        ax,
        bins=50,
        alpha=0.5,
        color=plot.BLUE_GREY.primary,
    )
    plot.compare_densities(
        x_range,
        [analytical_pdf, prior_pdf, gaussian_fit_pdf],
        ["Analytical posterior", "Prior", "Gaussian fit"],
        ax,
        colors=[plot.EARTH.primary, plot.EARTH.secondary, "black"],
    )
    plot.set_labels(ax, x="mu", y="Density")
    plot.clean(ax)
    ax.legend()
    ax.set_title("Posterior distribution comparison")
    plot.show()

    # Prior vs posterior comparison
    fig, axes = plot.prior_posterior_comparison(
        prior_samples=np.random.normal(
            prior_mean, prior_stdv, size=len(posterior_samples)
        ),
        posterior_samples=np.array(posterior_samples),
        name="mu",
    )
    plot.show()

    # Q-Q plot against analytical posterior
    fig, ax = plot.subplots()
    sorted_samples = np.sort(np.array(posterior_samples))
    n = len(sorted_samples)
    theoretical_quantiles = np.array(
        tfd.Normal(analytical.mean, analytical.stdv).quantile(
            np.linspace(0.001, 0.999, n)
        )
    )
    # TODO this is not working
    plot.qq_plot(
        sorted_samples, theoretical_quantiles, ax, color=plot.BLUE_GREY.primary
    )
    plot.set_labels(ax, x="Theoretical quantiles", y="Sample quantiles")
    plot.clean(ax)
    ax.set_title("Q-Q plot (vs analytical posterior)")
    plot.show()

    # Diagnostics
    fig, ax = plot.subplots()
    plot.diagnostics(infos.acceptance_rate, ax, color=plot.DANDELION.primary)
    plot.clean(ax)
    ax.set_title("Acceptance rates per chain")
    plot.show()

    return analytical, mcmc_mean, mcmc_stdv


if __name__ == "__main__":
    main()
