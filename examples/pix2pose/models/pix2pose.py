from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean
import tensorflow as tf
from loss import compute_weighted_reconstruction_loss_with_error
from loss import compute_error_prediction_loss


class Pix2Pose(Model):
    def __init__(self, image_shape, discriminator, generator, latent_dim):
        super(Pix2Pose, self).__init__()
        self.image_shape = image_shape
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    @property
    def metrics(self):
        return [self.generator_loss, self.discriminator_loss]

    def compile(self, optimizer_D, optimizer_G, gan_loss):
        super(Pix2Pose, self).compile()
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.gan_loss = gan_loss
        # self.reconstruction = reconstruction
        # self.error_prediction = error_prediction
        self.generator_loss = Mean(name='generator_loss')
        self.discriminator_loss = Mean(name='discriminator_loss')
        self.reconstruction_loss = Mean(name='weighted_reconstruction')
        self.error_prediction_loss = Mean(name='error_prediction')

    def _build_discriminator_labels(self, batch_size):
        return tf.concat([tf.ones(batch_size, 1), tf.zeros(batch_size, 1)], 0)

    def _add_noise_to_labels(self, labels):
        noise = tf.random.uniform(tf.shape(labels))
        labels = labels + 0.05 * noise
        return labels

    def _train_D(self, y_true, x_combined):
        with tf.GradientTape() as tape:
            y_pred = self.discriminator(x_combined)
            discriminator_loss = self.gan_loss(y_true, y_pred)
        grads = tape.gradient(
            discriminator_loss, self.discriminator.trainable_weights)
        self.optimizer_D.apply_gradients(
            zip(grads, self.discriminator.trainable_weights))
        return discriminator_loss

    def _train_G(self, RGB_inputs):
        batch_size = tf.shape(RGB_inputs)[0]
        y_misleading = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            y_pred = self.discriminator(
                self.generator(RGB_inputs)[:, :, :, 0:3])
            generator_loss = self.gan_loss(y_misleading, y_pred)
        grads = tape.gradient(generator_loss, self.generator.trainable_weights)
        self.optimizer_G.apply_gradients(
            zip(grads, self.generator.trainable_weights))
        return generator_loss

    def _train_G_reconstruction(self, RGB_inputs, RGBA_true):
        with tf.GradientTape() as tape:
            RGBE_pred = self.generator(RGB_inputs)
            loss = compute_weighted_reconstruction_loss_with_error(
                RGBA_true, RGBE_pred, beta=3.0)
        grads = tape.gradient(loss, self.generator.trainable_weights)
        self.optimizer_G.apply_gradients(
            zip(grads, self.generator.trainable_weights))
        return loss

    def _train_G_error_prediction(self, RGB_inputs, RGBA_true):
        with tf.GradientTape() as tape:
            RGBE_pred = self.generator(RGB_inputs)
            loss = compute_error_prediction_loss(RGBA_true, RGBE_pred)
        grads = tape.gradient(loss, self.generator.trainable_weights)
        self.optimizer_G.apply_gradients(
            zip(grads, self.generator.trainable_weights))
        return loss

    def _update_metrics(self, discriminator_loss, generator_loss):
        self.discriminator_loss.update_state(discriminator_loss)
        self.generator_loss.update_state(generator_loss)

    def train_step(self, data):
        inputs, labels = data
        RGB_inputs, RGBA_true = inputs['RGB_input'], labels['RGB_with_error']

        reconstruction_loss = self._train_G_reconstruction(RGB_inputs, RGBA_true)
        self.reconstruction_loss.update_state(reconstruction_loss)

        error_prediction_loss = self._train_G_error_prediction(RGB_inputs, RGBA_true)
        self.error_prediction_loss.update_state(error_prediction_loss)
        # reconstruction_loss = self.error_prediction(RGBA_true, RGBE_pred, beta)

        RGB_labels = RGBA_true[:, :, :, 0:3]
        RGB_generated = self.generator(RGB_inputs)[:, :, :, 0:3]

        combined_images = tf.concat([RGB_generated, RGB_labels], axis=0)
        batch_size = tf.shape(RGB_inputs)[0]
        y_true = self._build_discriminator_labels(batch_size)
        y_true = self._add_noise_to_labels(y_true)

        discriminator_loss = self._train_D(y_true, combined_images)
        generator_loss = self._train_G(RGB_inputs)
        self._update_metrics(discriminator_loss, generator_loss)
        return {'discriminator_loss': self.discriminator_loss.result(),
                'generator_loss': self.generator_loss.result(),
                'reconstruction_loss': self.reconstruction_loss.result(),
                'error_prediction_loss': self.error_prediction_loss.result()}

    """
    def call(self, data):
        generated = self.generator(data)
        predictions = self.discriminator(generated)
        return generated , predictions
    """
