# Infer the coupling of an Ising lattice with a pseudolikelihood: each spin
# is Bernoulli given the sum of its four neighbors.

import argparse

import jax
import jax.numpy as jp

import paz
import paz.utils.plot as plot


def sample_lattice(key, size, coupling, num_steps):
    spins = jp.array([-1.0, 1.0])
    lattice = jax.random.choice(key, spins, shape=(size, size))
    parity = (jp.arange(size)[:, None] + jp.arange(size)) % 2

    def step(step_arg, lattice):
        neighbor_sums = compute_neighbor_sums(lattice)
        energy_change = 2 * lattice * coupling * neighbor_sums
        step_key = jax.random.fold_in(key, step_arg)
        uniform = jax.random.uniform(step_key, lattice.shape)
        is_accepted = uniform < jp.exp(-energy_change)
        # Checkerboard update: neighbors never flip in the same step.
        is_flipped = is_accepted & (parity == step_arg % 2)
        return jp.where(is_flipped, -lattice, lattice)

    return jax.lax.fori_loop(0, num_steps, step, lattice)


def compute_neighbor_sums(lattice):
    vertical = jp.roll(lattice, 1, axis=0) + jp.roll(lattice, -1, axis=0)
    horizontal = jp.roll(lattice, 1, axis=1) + jp.roll(lattice, -1, axis=1)
    return vertical + horizontal


def Likelihood(neighbor_sums):
    def apply(coupling):
        # The logit of a spin being +1 is 2 * coupling * sum(neighbors).
        logits = 2.0 * coupling * neighbor_sums
        return paz.distributions.Bernoulli(logits=logits)

    return apply


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ising coupling inference")
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--coupling", type=float, default=0.4)
    parser.add_argument("--num_lattice_steps", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=100_000)
    parser.add_argument("--num_chains", type=int, default=5)
    parser.add_argument("--warmup", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=0.01)
    args = parser.parse_args()

    data_key = jax.random.PRNGKey(args.data_seed)
    lattice_args = (args.size, args.coupling, args.num_lattice_steps)
    lattice = sample_lattice(data_key, *lattice_args)
    neighbor_sums = compute_neighbor_sums(lattice).flatten()
    is_spin_up = (lattice.flatten() + 1) / 2

    normal = paz.distributions.Normal(0.5, 0.5)
    coupling = paz.Prior(normal, name="coupling")
    spins = paz.Observable(Likelihood(neighbor_sums), name="spins")(coupling)
    model = paz.PGM([coupling], [spins], "ising_pseudolikelihood")

    tuner = paz.AdaptiveStepTuner(args.sigma)
    kwargs = {"warmup": args.warmup, "tuner": tuner}
    model.configure(num_chains=args.num_chains, **kwargs)
    key = jax.random.PRNGKey(args.seed)
    posterior = model.infer(key, is_spin_up, num_samples=args.num_samples)
    samples = posterior.samples.coupling
    print(f"True coupling: {args.coupling}")
    print(f"Inferred coupling (mean): {samples.mean():.4f}")
    print(f"Acceptance rate: {posterior.infos.acceptance_rate.mean():.3f}")

    plot.configure(fontsize=12)
    figure, axes = plot.subplots(ncols=2, figsize=(12, 5))
    plot.imshow(lattice, axes[0], cmap="coolwarm", colorbar=False)
    axes[0].axis("off")
    axes[0].set_title(f"Observed Ising lattice (true J={args.coupling})")

    for chain in range(args.num_chains):
        kwargs = {"bins": 30, "alpha": 0.3, "label": f"Chain {chain}"}
        plot.histogram(samples[:, chain], axes[1], **kwargs)
    plot.vline(args.coupling, axes[1], color="red", label="True J")
    plot.set_labels(axes[1], x="Coupling strength J", y="Density")
    plot.legend(axes[1])
    plot.clean(axes[1])
    axes[1].set_title("Posterior distribution of J")
    figure.tight_layout()
    plot.show()
