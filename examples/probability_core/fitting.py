import jax
import jax.numpy as jp
import optax

from .distributions import TransformedDistribution


def fit_bijector(
    source_distribution,
    target_distribution,
    initial_bijector,
    key,
    num_samples=10_000,
    optimizer=None,
    num_steps=1000,
):
    target_samples = target_distribution.sample(num_samples, seed=key)
    parameters, treedef = jax.tree_util.tree_flatten(initial_bijector)
    optimizer = optax.adam(5e-2) if optimizer is None else optimizer
    state = optimizer.init(parameters)

    def loss_fn(parameters):
        bijector = jax.tree_util.tree_unflatten(treedef, parameters)
        transformed = TransformedDistribution(source_distribution, bijector)
        return -jp.mean(transformed.log_prob(target_samples))

    @jax.jit
    def update_step(parameters, state):
        loss, grads = jax.value_and_grad(loss_fn)(parameters)
        updates, state = optimizer.update(grads, state, parameters)
        parameters = optax.apply_updates(parameters, updates)
        return parameters, state, loss

    losses = []
    for _ in range(num_steps):
        parameters, state, loss = update_step(parameters, state)
        losses.append(float(loss))
    bijector = jax.tree_util.tree_unflatten(treedef, parameters)
    return bijector, losses
