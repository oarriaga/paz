import paz
import jax

import optax
import jax.numpy as jp
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


def Likelihood(X):
    def apply(mean, bias, stdv):
        return tfd.Normal(jax.vmap(lambda x: mean * x + bias)(X), stdv)

    return apply


X = jp.linspace(0, 1, 200)
observations = 0.5 * X + 0.1 + 0.05 * jp.sin(50 * X)

mean = paz.Prior("mean", tfd.Normal(0.0, 1.0))
bias = paz.Prior("bias", tfd.Normal(0.0, 1.0))
low, high = 0.001, 0.3
bijector = tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.Sigmoid()])
stdv = paz.Prior("stdv", tfd.Uniform(low, high), bijector=bijector)
y = paz.Observable("y_pred", Likelihood(X), observations)(mean, bias, stdv)
line = paz.PGM([mean, bias, stdv], [y], "line")

key = jax.random.PRNGKey(888)
for key in jax.random.split(key, num_samples := 100):
    sample = line.sample(key)
    plt.plot(X, sample.y_pred, color="blue", alpha=0.2)
plt.plot(X, observations, color="red", alpha=0.8)
plt.xlim(0, 1)
plt.ylim(-4, 4)
plt.show()

parameters = line.sample_inverse(key)
states = line.apply(parameters)


def compute_loss(parameters):
    return -line.apply(parameters).log_prob_sum


learning_rate = 0.001
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(parameters)


@jax.jit
def update_step(parameters, opt_state):
    loss, grads = jax.value_and_grad(compute_loss)(parameters)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(parameters, updates)
    return new_params, new_opt_state, loss


num_steps = 20_000
for step in range(num_steps):
    parameters, opt_state, loss = update_step(parameters, opt_state)
    print(f"Step {step:3d}, Loss: {loss:.4f}")

print(f"Final optimized parameters: {parameters}")
state = line.apply(parameters)

key = jax.random.PRNGKey(888)
for key in jax.random.split(key, num_samples := 100):
    plt.plot(X, state.sample.y_pred.sample(seed=key), color="blue", alpha=0.2)
plt.plot(X, observations, color="red", alpha=0.8)
plt.xlim(0, 1)
plt.ylim(-4, 4)
plt.show()


paz.inference.metropolis_hastings
