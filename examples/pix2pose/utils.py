import tensorflow as tf


def loss_color_wrapped(rotation_matrices):
    def loss_color_unwrapped(color_image, predicted_color_image):
        min_loss = tf.float32.max

        # Bring the image in the range between 0 and 1
        color_image = (color_image + 1) * 0.5

        # Calculate masks for the object and the background (they are independent of the rotation)
        mask_object = tf.repeat(tf.expand_dims(tf.math.reduce_max(tf.math.ceil(color_image), axis=-1), axis=-1),
                                repeats=3, axis=-1)
        mask_background = tf.ones(tf.shape(mask_object)) - mask_object

        # Bring the image again in the range between -1 and 1
        color_image = (color_image * 2) - 1

        # Iterate over all possible rotations
        for rotation_matrix in rotation_matrices:

            real_color_image = tf.identity(color_image)

            # Add a small epsilon value to avoid the discontinuity problem
            real_color_image = real_color_image + tf.ones_like(real_color_image) * 0.0001

            # Rotate the object
            real_color_image = tf.einsum('ij,mklj->mkli', tf.convert_to_tensor(np.array(rotation_matrix), dtype=tf.float32), real_color_image)
            #real_color_image = tf.where(tf.math.less(real_color_image, 0), tf.ones_like(real_color_image) + real_color_image, real_color_image)

            # Set the background to be all -1
            real_color_image *= mask_object
            real_color_image += (mask_background*tf.constant(-1.))

            # Get the number of pixels
            num_pixels = tf.math.reduce_prod(tf.shape(real_color_image)[1:3])
            beta = 3

            # Calculate the difference between the real and predicted images including the mask
            diff_object = tf.math.abs(predicted_color_image*mask_object - real_color_image*mask_object)
            diff_background = tf.math.abs(predicted_color_image*mask_background - real_color_image*mask_background)

            # Calculate the total loss
            loss_colors = tf.cast((1/num_pixels), dtype=tf.float32)*(beta*tf.math.reduce_sum(diff_object, axis=[1, 2, 3]) + tf.math.reduce_sum(diff_background, axis=[1, 2, 3]))
            min_loss = tf.math.minimum(loss_colors, min_loss)
        return min_loss

    return loss_color_unwrapped


def loss_error(real_error_image, predicted_error_image):
    # Get the number of pixels
    num_pixels = tf.math.reduce_prod(tf.shape(real_error_image)[1:3])
    loss_error = tf.cast((1/num_pixels), dtype=tf.float32)*(tf.math.reduce_sum(tf.math.square(predicted_error_image - tf.clip_by_value(tf.math.abs(real_error_image), tf.float32.min, 1.)), axis=[1, 2, 3]))

    return loss_error


