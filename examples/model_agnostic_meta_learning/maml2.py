import keras
import jax
import jax.numpy as jp
from jax.tree_util import tree_map


def update_metrics(metrics, metrics_theta, loss, y=None, y_pred=None):
    """
    A helper function to update metrics in a stateless way, following the
    provided functional structure.
    """
    new_metrics_theta, logs = [], {}
    for metric in metrics:
        first_arg = len(new_metrics_theta)
        # Correctly access the number of variables for the metric
        final_arg = len(new_metrics_theta) + len(metric.variables)
        theta = metrics_theta[first_arg:final_arg]

        # Update logic that handles both loss and other metrics
        if metric.name == "loss":
            theta = metric.stateless_update_state(theta, loss)
        # For MAML, we update other metrics based on query set performance
        elif y is not None and y_pred is not None:
            theta = metric.stateless_update_state(theta, y, y_pred)

        logs[metric.name] = metric.stateless_result(theta)
        new_metrics_theta = new_metrics_theta + theta
    return logs, new_metrics_theta


class MAML(keras.Model):
    def __init__(self, model, inner_learning_rate=0.01, **kwargs):
        super().__init__(**kwargs)
        self.inner_learning_rate = inner_learning_rate
        self.model = model

    def build(self, input_shape):

    def _compute_loss_and_predictions(
        self, theta, theta_static, x, y, training=False
    ):
        """
        Computes loss for a task and returns predictions and updated non-trainable state.
        This is a helper for both inner and outer loop calculations.
        """
        y_pred, theta_static = self.stateless_call(
            theta, theta_static, x, training=training
        )
        loss = self.compute_loss(x, y, y_pred)
        return loss, (y_pred, theta_static)

    def train_step(self, state, data):
        """
        Performs a single MAML training step using a functional, state-passing style.
        """
        theta, theta_static, theta_optimizer, metrics_theta = state
        # x_support_batch, x_queries_batch, y_support_batch, y_queries_batch = (
        #     data
        # )
        # x_support_batch, y_support_batch = data["support"]
        # x_queries_batch, y_queries_batch = data["queries"]
        # x_support_batch, y_support_batch, x_queries_batch, y_queries_batch = ( data)
        # x_support_batch, x_queries_batch, y_support_batch, y_queries_batch = (
        #     data
        # )
        (x_support_batch, x_queries_batch) = data[0]
        (y_support_batch, y_queries_batch) = data[1]

        def compute_meta_loss(
            theta, theta_static, x_support, y_support, x_queries, y_queries
        ):
            """
            Computes the meta-loss for a single task. This involves:
            1. An inner update on the support set.
            2. Calculating the loss of the updated model on the query set.
            """
            # 1. Inner Update: Get gradients for the support set.
            # `has_aux=True` is needed because our loss function returns (loss, (aux_data)).
            (_, _), inner_gradients = jax.value_and_grad(
                self._compute_loss_and_predictions, has_aux=True
            )(theta, theta_static, x_support, y_support, training=True)

            # 2. Compute "fast weights" by applying a single gradient descent step.
            updated_theta = tree_map(
                lambda t, g: t - self.inner_learning_rate * g,
                theta,
                inner_gradients,
            )

            # 3. Meta-Loss: Evaluate the fast weights on the query set.
            # This loss is what we will differentiate with respect to the original `theta`.
            meta_loss, (y_pred_queries, _) = self._compute_loss_and_predictions(
                updated_theta, theta_static, x_queries, y_queries, training=True
            )
            return meta_loss, y_pred_queries

        # Vectorize the meta-loss computation across the batch of tasks.
        # We need both the value (for loss) and the grad (for the meta-update).
        # `has_aux=True` is used again because `compute_meta_loss` returns (loss, y_pred).
        compute_batch_meta_gradients = jax.vmap(
            jax.value_and_grad(compute_meta_loss, has_aux=True),
            in_axes=(None, None, 0, 0, 0, 0),
            out_axes=0,
        )

        # Execute the vectorized computation.
        (losses, _), gradients = compute_batch_meta_gradients(
            theta,
            theta_static,
            x_support_batch,
            y_support_batch,
            x_queries_batch,
            y_queries_batch,
        )

        # Average the losses and gradients across the batch for the final meta-update.
        loss = jp.mean(losses)
        gradients = tree_map(lambda g: jp.mean(g, axis=0), gradients)

        # Apply the meta-gradients to the original model parameters.
        apply_args = (theta_optimizer, gradients, theta)
        updated_theta, updated_theta_optimizer = self.optimizer.stateless_apply(
            *apply_args
        )

        # Update the metrics using the mean loss of the batch.
        # Note: For simplicity, we are not updating metrics that require y_pred here,
        # but this structure allows for it if you were to pass the averaged predictions.
        metrics_args = (self.metrics, metrics_theta, loss)
        logs, updated_metrics_theta = update_metrics(*metrics_args)

        # Return the logs and the new state for the next step.
        new_state = (
            updated_theta,
            theta_static,
            updated_theta_optimizer,
            updated_metrics_theta,
        )
        return logs, new_state
