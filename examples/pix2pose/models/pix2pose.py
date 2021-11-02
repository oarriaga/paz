from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean
import tensorflow as tf


class Pix2Pose(Model):
    def __init__(self, image_shape, discriminator, generator, latent_dim):
        super(Pix2Pose, self).__init__()
        self.image_shape = image_shape
        self.D = discriminator
        self.G = generator
        self.latent_dim = latent_dim

    @property
    def metrics(self):
        return [self.G_loss_metric, self.D_loss_metric]

    def compile(self, optimizer_D, optimizer_G, loss):
        super(Pix2Pose, self).compile()
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.loss = loss
        self.G_loss_metric = Mean(name='generator_loss')
        self.D_loss_metric = Mean(name='discriminator_loss')

    def _build_discriminator_labels(self, batch_size):
        return tf.concat([tf.ones(batch_size, 1), tf.zeros(batch_size, 1)], 0)

    def _add_noise_to_labels(self, labels):
        noise = tf.random.uniform(tf.shape(labels))
        labels = labels + 0.05 * noise
        return labels

    def _train_D(self, y_true, x_combined):
        with tf.GradientTape() as tape:
            y_pred = self.D(x_combined)
            D_loss = self.loss(y_true, y_pred)
        grads = tape.gradient(D_loss, self.D.trainable_weights)
        self.optimizer_D.apply_gradients(zip(grads, self.D.trainable_weights))
        return D_loss

    def _train_G(self, RGB_inputs):
        batch_size = tf.shape(RGB_inputs)[0]
        y_misleading = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            y_pred = self.D(self.G(RGB_inputs)[:, :, :, 0:3])
            G_loss = self.loss(y_misleading, y_pred)
        grads = tape.gradient(G_loss, self.G.trainable_weights)
        self.optimizer_G.apply_gradients(zip(grads, self.G.trainable_weights))
        return G_loss

    def _update_metrics(self, D_loss, G_loss):
        self.D_loss_metric.update_state(D_loss)
        self.G_loss_metric.update_state(G_loss)

    def train_step(self, data):
        RGB_inputs, RGB_labels = data
        RGB_inputs = RGB_inputs['RGB_input'][:, :, :, 0:3]
        RGB_labels = RGB_labels['RGB_with_error'][:, :, :, 0:3]
        RGB_generated = self.G(RGB_inputs)[:, :, :, 0:3]

        combined_images = tf.concat([RGB_generated, RGB_labels], axis=0)
        batch_size = tf.shape(RGB_inputs)[0]
        y_true = self._build_discriminator_labels(batch_size)
        y_true = self._add_noise_to_labels(y_true)

        D_loss = self._train_D(y_true, combined_images)
        G_loss = self._train_G(RGB_inputs)
        self._update_metrics(D_loss, G_loss)
        return {"discriminator_loss": self.D_loss_metric.result(),
                "generator_loss": self.G_loss_metric.result()}
    """
    def call(self, data):
        generated = self.G(data)
        predictions = self.D(generated)
        return generated , predictions
    """
