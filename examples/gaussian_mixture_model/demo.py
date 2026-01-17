import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

import paz
from paz.inference.types import SampleType

tfd = tfp.distributions


def build_gaussian_mixture_model(mu0_value, mu1_value, sigma_value, p_value):
    # Names are optional; omit name= to auto-generate names like prior_normal_0.
    mu0 = paz.Prior(tfd.Deterministic(mu0_value), name="mu0")
    mu1 = paz.Prior(tfd.Deterministic(mu1_value), name="mu1")
    sigma = paz.Prior(tfd.Deterministic(sigma_value), name="sigma")
    p = paz.Prior(tfd.Deterministic(p_value), name="p")

    def z_distribution(p):
        return tfd.Bernoulli(probs=p)

    z = paz.Latent(z_distribution, name="z")(p)

    def y_distribution(z, mu0, mu1, sigma):
        mean = jp.where(z == 1, mu1, mu0)
        return tfd.Normal(mean, sigma)

    y_obs = paz.Observable(y_distribution, name="y")(z, mu0, mu1, sigma)
    return paz.PGM([mu0, mu1, sigma, p], [y_obs], "gaussian_mixture")


def build_full_inverse_samples(
    mu0_value, mu1_value, sigma_value, p_value, z_value
):
    Sample = SampleType(["mu0", "mu1", "sigma", "p", "z"])
    return Sample(mu0_value, mu1_value, sigma_value, p_value, z_value)


def build_theta_inverse_samples(mu0_value, mu1_value, sigma_value, p_value):
    Sample = SampleType(["mu0", "mu1", "sigma", "p"])
    return Sample(mu0_value, mu1_value, sigma_value, p_value)


def main():
    mu0_value = jp.array(0.0)
    mu1_value = jp.array(2.0)
    sigma_value = jp.array(1.0)
    p_value = jp.array(0.3)
    y_value = jp.array(1.5)

    model = build_gaussian_mixture_model(
        mu0_value, mu1_value, sigma_value, p_value
    )
    data = {"y": y_value}

    full_samples_z0 = build_full_inverse_samples(
        mu0_value, mu1_value, sigma_value, p_value, jp.array(0.0)
    )
    full_samples_z1 = build_full_inverse_samples(
        mu0_value, mu1_value, sigma_value, p_value, jp.array(1.0)
    )

    log_joint_z0 = model.prior.log_prob(full_samples_z0, space="inv")
    log_joint_z0 = log_joint_z0 + model.likelihood.log_prob(
        full_samples_z0, data, space="inv"
    )
    log_joint_z1 = model.prior.log_prob(full_samples_z1, space="inv")
    log_joint_z1 = log_joint_z1 + model.likelihood.log_prob(
        full_samples_z1, data, space="inv"
    )
    log_marginal_enum = jp.logaddexp(log_joint_z0, log_joint_z1)

    print("=" * 60)
    print("Gaussian mixture model (enumeration vs marginalization)")
    print("=" * 60)
    print(
        f"mu0={mu0_value}, mu1={mu1_value}, sigma={sigma_value}, p={p_value}"
    )
    print(f"y={y_value}")
    print(f"log p(y, z=0): {log_joint_z0:.6f}")
    print(f"log p(y, z=1): {log_joint_z1:.6f}")
    print(f"log p(y) via enumeration: {log_marginal_enum:.6f}")

    model_marg = paz.marginalize(model, ["z"])
    theta_samples = build_theta_inverse_samples(
        mu0_value, mu1_value, sigma_value, p_value
    )
    log_marginal = model_marg.prior.log_prob(theta_samples, space="inv")
    log_marginal = log_marginal + model_marg.likelihood.log_prob(
        theta_samples, data, space="inv"
    )
    print(f"log p(y) via marginalize(): {log_marginal:.6f}")

    posterior = paz.recover_discrete_posterior(
        model_marg, "z", theta_samples, data
    ).posterior
    print(f"p(z=0 | y): {posterior[0]:.6f}")
    print(f"p(z=1 | y): {posterior[1]:.6f}")


if __name__ == "__main__":
    main()
