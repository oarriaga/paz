from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean
import tensorflow as tf
# from loss import compute_weighted_reconstruction_loss_with_error
# from loss import compute_error_prediction_loss


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

    def compile(self, optimizers, losses, loss_weights):
        super(Pix2Pose, self).compile()
        self.optimizer_generator = optimizers['generator']
        self.optimizer_discriminator = optimizers['discriminator']
        self.compute_reconstruction_loss = losses['weighted_reconstruction']
        self.compute_error_prediction_loss = losses['error_prediction']
        self.compute_discriminator_loss = losses['discriminator']

        self.generator_loss = Mean(name='generator_loss')
        self.discriminator_loss = Mean(name='discriminator_loss')
        self.reconstruction_loss = Mean(name='weighted_reconstruction')
        self.error_prediction_loss = Mean(name='error_prediction')
        self.reconstruction_weight = loss_weights['weighted_reconstruction']
        self.error_prediction_weight = loss_weights['error_prediction']

    def _build_discriminator_labels(self, batch_size):
        return tf.concat([tf.ones(batch_size, 1), tf.zeros(batch_size, 1)], 0)

    def _add_noise_to_labels(self, labels):
        noise = tf.random.uniform(tf.shape(labels))
        labels = labels + 0.05 * noise
        return labels

    def _get_batch_size(self, values):
        return tf.shape(values)[0]

    def _train_discriminator(self, RGB_inputs, RGBA_true):
        RGB_true = RGBA_true[:, :, :, 0:3]
        RGB_fake = self.generator(RGB_inputs)[:, :, :, 0:3]
        RGB_fake_true = tf.concat([RGB_fake, RGB_true], axis=0)

        batch_size = self._get_batch_size(RGB_inputs)
        y_true = self._build_discriminator_labels(batch_size)
        y_true = self._add_noise_to_labels(y_true)

        with tf.GradientTape() as tape:
            y_pred = self.discriminator(RGB_fake_true)
            discriminator_loss = self.compute_discriminator_loss(
                y_true, y_pred)
        gradients = tape.gradient(discriminator_loss,
                                  self.discriminator.trainable_weights)
        self.optimizer_discriminator.apply_gradients(
            zip(gradients, self.discriminator.trainable_weights))
        return discriminator_loss

    def _train_generator(self, RGB_inputs):
        batch_size = tf.shape(RGB_inputs)[0]
        y_misleading = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            RGBE_preds = self.generator(RGB_inputs)
            y_pred = self.discriminator(RGBE_preds[..., 0:3])
            generator_loss = self.compute_discriminator_loss(
                y_misleading, y_pred)
        gradients = tape.gradient(generator_loss,
                                  self.generator.trainable_weights)
        self.optimizer_generator.apply_gradients(
            zip(gradients, self.generator.trainable_weights))
        return generator_loss

    def _train_reconstruction(self, RGB_inputs, RGBA_true):
        with tf.GradientTape() as tape:
            RGBE_pred = self.generator(RGB_inputs)
            reconstruction_loss = self.compute_reconstruction_loss(
                RGBA_true, RGBE_pred)
            reconstruction_loss = (
                self.reconstruction_weight * reconstruction_loss)
        gradients = tape.gradient(reconstruction_loss,
                                  self.generator.trainable_weights)
        self.optimizer_generator.apply_gradients(
            zip(gradients, self.generator.trainable_weights))
        return reconstruction_loss

    def _train_error_prediction(self, RGB_inputs, RGBA_true):
        with tf.GradientTape() as tape:
            RGBE_pred = self.generator(RGB_inputs)
            error_prediction_loss = self.compute_error_prediction_loss(
                RGBA_true, RGBE_pred)
            error_prediction_loss = (
                self.error_prediction_weight * error_prediction_loss)
        gradients = tape.gradient(
            error_prediction_loss, self.generator.trainable_weights)
        self.optimizer_generator.apply_gradients(
            zip(gradients, self.generator.trainable_weights))
        return error_prediction_loss

    def train_step(self, data):
        RGB_inputs, RGBA_true = data[0]['RGB_input'], data[1]['RGB_with_error']

        reconstruction_loss = self._train_reconstruction(RGB_inputs, RGBA_true)
        self.reconstruction_loss.update_state(reconstruction_loss)

        error_loss = self._train_error_prediction(RGB_inputs, RGBA_true)
        self.error_prediction_loss.update_state(error_loss)

        discriminator_loss = self._train_discriminator(RGB_inputs, RGBA_true)
        self.discriminator_loss.update_state(discriminator_loss)

        generator_loss = self._train_generator(RGB_inputs)
        self.generator_loss.update_state(generator_loss)

        return {'discriminator_loss': self.discriminator_loss.result(),
                'generator_loss': self.generator_loss.result(),
                'reconstruction_loss': self.reconstruction_loss.result(),
                'error_prediction_loss': self.error_prediction_loss.result()}
