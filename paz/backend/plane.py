import jax
import paz
import jax.numpy as jp

import optax
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


def student_t_loss(normal, centroid, scale, DOF, points):
    normal = normal / jp.linalg.norm(normal)
    distances = jp.dot(points - centroid, normal)
    return -tfd.StudentT(DOF, 0.0, scale).log_prob(distances).sum()


def fit(key, points, optimizer=None, loss=None, num_steps=200):
    optimizer = optax.adam(0.01) if optimizer is None else optimizer
    loss = paz.lock(student_t_loss, 0.2, 2.0, points) if loss is None else loss
    compute_loss_and_gradients = jax.value_and_grad(loss, [0, 1])

    def gradient_step(state, step_arg):
        parameters, optimizer_state = state
        loss, gradients = compute_loss_and_gradients(*parameters)
        optimizer_args = (gradients, optimizer_state, parameters)
        updates, optimizer_state = optimizer.update(*optimizer_args)
        parameters = optax.apply_updates(parameters, updates)
        return (parameters, optimizer_state), loss

    parameters = (jax.random.normal(key, (3,)), jp.mean(points, axis=0))
    optimizer_state = optimizer.init(parameters)
    state = (parameters, optimizer_state)
    results = jax.lax.scan(gradient_step, state, jp.arange(num_steps))
    ((normal, centroid), _), losses = results
    normal = normal / jp.linalg.norm(normal)
    return (normal, centroid), losses
