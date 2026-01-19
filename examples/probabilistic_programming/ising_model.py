import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp

import paz

tfd = tfp.distributions
tfb = tfp.bijectors


def compute_neighbor_sums(grid):
    return (
        jp.roll(grid, 1, axis=0)
        + jp.roll(grid, -1, axis=0)
        + jp.roll(grid, 1, axis=1)
        + jp.roll(grid, -1, axis=1)
    )


# --- 1. Helper: Generate Synthetic Ising Data (Metropolis Step) ---
def generate_ising_grid(key, size, J_true, steps=1000):
    # Initialize random spins {-1, 1}
    grid = jax.random.choice(key, jp.array([-1.0, 1.0]), shape=(size, size))

    def step_fn(i, grid):
        key_step = jax.random.fold_in(key, i)
        # Calculate energy change dE = 2 * s_i * local_field
        neighbor_sum = compute_neighbor_sums(grid)
        dE = 2 * grid * J_true * neighbor_sum

        # Metropolis acceptance probability
        prob = jp.exp(-dE)
        accept = jax.random.uniform(key_step, grid.shape) < prob

        # Flip spins where accepted
        new_grid = jp.where(accept, -grid, grid)
        # Checkerboard update (simplified for demonstration)
        mask = (jp.arange(size)[:, None] + jp.arange(size)) % 2 == (i % 2)
        return jp.where(mask, new_grid, grid)

    return jax.lax.fori_loop(0, steps, step_fn, grid)


# --- 2. Prepare Data ---
SIZE = 30
TRUE_J = 0.4  # Near critical temp

key_data = jax.random.PRNGKey(42)
ising_grid = generate_ising_grid(key_data, SIZE, TRUE_J)

# Calculate features (Sum of neighbors) for every pixel
neighbor_sums = compute_neighbor_sums(ising_grid)

# Flatten for regression
# X: Sum of neighbors (can be -4, -2, 0, 2, 4)
# y: Target spin converted to {0, 1} for Bernoulli
X_flat = neighbor_sums.flatten()
y_flat = (ising_grid.flatten() + 1) / 2

# --- 3. Define the Model (Pseudolikelihood) ---


def IsingConditional(neighbor_sums):
    def apply(coupling):
        # Physics: Logit = 2 * J * sum(neighbors)
        # We model the probability of spin being +1
        return tfd.Bernoulli(logits=2.0 * coupling * neighbor_sums)

    return apply


# Prior for Coupling Strength J
# We expect ferromagnetic behavior (positive J), roughly around 0.3-0.6
J_prior = paz.Prior(tfd.Normal(0.5, 0.5), name="coupling")

# Likelihood connects J and Neighbor Sums to the actual Spin values
spins = paz.Observable(IsingConditional(X_flat), name="spins")(J_prior)

# Build PGM
model = paz.PGM([J_prior], [spins], "ising_pseudolikelihood")

# --- 4. Inference ---
num_chains = 5
num_samples = 100_000
burn_in = 0.2

# Compile and Sample
tuner = paz.AdaptiveStepTuner(0.01)
model.compile(num_chains=num_chains, warmup=burn_in, tuner=tuner)

# Note: We pass y_flat (0s and 1s) as the observed data
key_infer = jax.random.PRNGKey(101)
posterior = model.infer(key_infer, y_flat, num_samples=num_samples)
samples = posterior.samples

# --- 5. Visualization ---
print(f"True J: {TRUE_J}")
print(f"Inferred J (Mean): {samples.position.coupling.mean():.4f}")
print(f"Acceptance Rate: {posterior.infos.acceptance_rate.mean():.3f}")

plt.figure(figsize=(12, 5))

# Plot 1: The Observed Ising Grid
plt.subplot(1, 2, 1)
plt.imshow(ising_grid, cmap="coolwarm")
plt.title(f"Observed Ising Lattice (True J={TRUE_J})")
plt.axis("off")

# Plot 2: Posterior of Coupling Constant J
plt.subplot(1, 2, 2)
# Using histogram since it's 1D
for chain in range(num_chains):
    plt.hist(
        samples.position.coupling[:, chain],
        bins=30,
        alpha=0.3,
        density=True,
        label=f"Chain {chain}",
    )
plt.axvline(TRUE_J, color="red", linestyle="--", label="True J")
plt.xlabel("Coupling Strength J")
plt.ylabel("Density")
plt.title("Posterior Distribution of Parameter J")
plt.legend()

plt.tight_layout()
plt.show()
