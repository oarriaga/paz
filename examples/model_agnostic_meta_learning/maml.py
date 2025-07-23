import jax
import keras


def update_metrics(metrics, metrics_theta, loss, y, y_pred):
    new_metrics_theta, logs = [], {}
    for metric in metrics:
        first_arg = len(new_metrics_theta)
        final_arg = len(new_metrics_theta) + len(metric.theta)
        theta = metrics_theta[first_arg:final_arg]
        if metric.name == "loss":
            theta = metric.stateless_update_state(theta, loss)
        else:
            theta = metric.stateless_update_state(theta, y, y_pred)
        logs[metric.name] = metric.stateless_result(theta)
        new_metrics_theta = new_metrics_theta + theta
    return logs, new_metrics_theta


class MAML(keras.Model):

    def compute_loss(self, theta, theta_static, x, y, training=False):
        args = (theta, theta_static, x, training)
        y_pred, theta_static = self.stateless_call(*args, training=training)
        loss = self.compute_loss(x, y, y_pred)
        return loss, (y_pred, theta_static)

    def train_step(self, state, data):
        theta, theta_static, theta_optimizer, metrics_theta = state
        x, y = data

        compute_gradients = jax.value_and_grad(self.compute_loss, has_aux=True)

        args = (theta, theta_static, x, y)
        values, gradients = compute_gradients(*args, training=True)
        loss, (y_pred, theta_static) = values

        args = (theta_optimizer, gradients, theta)
        theta, theta_optimizer = self.optimizer.stateless_apply(*args)

        args = (self.metrics, metrics_theta, loss, y, y_pred)
        logs, metrics_theta = update_metrics(*args)

        return logs, (theta, theta_static, theta_optimizer, metrics_theta)
