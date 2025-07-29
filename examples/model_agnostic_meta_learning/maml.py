from functools import partial
import jax
import jax.numpy as jp


def build_state(model, optimizer):
    parameters = (model.trainable_variables, model.non_trainable_variables)
    state = (parameters, optimizer.variables)
    return state


def call(model, parameters, x):
    y_pred, _ = model.stateless_call(*parameters, x)
    return y_pred


def compute_task_loss(model, loss_fn, variables, static_parameters, x, y):
    parameters = (variables, static_parameters)
    y_pred = call(model, parameters, x)
    return loss_fn(y, y_pred)


def gradient_step(step_size, parameters, gradients):

    def _gradient_step(parameters, gradients):
        return parameters - step_size * gradients

    variables, static_parameters = parameters
    variables = jax.tree.map(_gradient_step, variables, gradients)
    return variables, static_parameters


def adapt(model, loss_fn, step_size, num_steps, parameters, support_data):
    compute_gradients = jax.grad(partial(compute_task_loss, model, loss_fn))

    def step(step, parameters):
        gradients = compute_gradients(*parameters, *support_data)
        parameters = gradient_step(step_size, parameters, gradients)
        return parameters

    return jax.lax.fori_loop(0, num_steps, step, parameters)


def train_step(model, loss_fn, optimizer, step_size, num_steps, state, data):
    _compute_task_loss = partial(compute_task_loss, model, loss_fn)
    _adapt = partial(adapt, model, loss_fn, step_size, num_steps)

    def compute_meta_loss(
        variables, static_parameters, x_support, y_support, x_queries, y_queries
    ):
        parameters = (variables, static_parameters)
        params = _adapt(parameters, (x_support, y_support))
        meta_loss = _compute_task_loss(*params, x_queries, y_queries)
        return meta_loss

    compute_meta_gradients = jax.vmap(
        jax.value_and_grad(compute_meta_loss),
        in_axes=(None, None, 0, 0, 0, 0),
        out_axes=0,
    )

    parameters, optimizer_state = state
    variables, static_parameters = parameters
    loss, gradients = compute_meta_gradients(*parameters, *data)
    gradients = jax.tree.map(lambda gradient: jp.mean(gradient, 0), gradients)
    optimizer_args = (optimizer_state, gradients, variables)
    variables, optimizer_state = optimizer.stateless_apply(*optimizer_args)
    return jp.mean(loss), ((variables, static_parameters), optimizer_state)


def predict(model, loss_fn, step_size, num_steps, parameters, support_data, x):
    adapt_args = (model, loss_fn, step_size, num_steps)
    parameters = adapt(*adapt_args, parameters, support_data)
    return call(model, parameters, x)
