import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from paz.backend.quaternion import quarternion_to_rotation_matrix, quaternion_multiply

import tensorflow as tf

from scenes import SingleView

def loss_color(real_color_image, predicted_color_image, rotation_matrix):
    # Calculate masks for the object and the background
    mask_object = tf.repeat(tf.expand_dims(tf.math.reduce_max(tf.math.ceil(real_color_image), axis=-1), axis=-1), repeats=3, axis=-1)
    mask_background = tf.ones(tf.shape(mask_object)) - mask_object

    plt.imshow(real_color_image.numpy()[0].astype(np.uint8))
    plt.show()

    #real_color_image_rotated = np.zeros(shape=tf.shape(real_color_image))

    #for num_sample in range(tf.shape(real_color_image)[0]):
    #    for i in range(tf.shape(real_color_image)[1]):
    #        for j in range(tf.shape(real_color_image)[2]):
    #            real_color_image_rotated[num_sample, i, j] = tf.linalg.matvec(rotation_matrix, real_color_image[num_sample, i, j])

    print(tf.shape(real_color_image[0]))
    real_color_image_rotated = tf.einsum('ij,mklj->mkli', rotation_matrix, real_color_image)
    #real_color_image_rotated += 255
    #real_color_image_rotated = tf.math.floormod(real_color_image_rotated, 256)
    #print(type(real_color_image_rotated))
    #real_color_image_rotated = tf.cond(tf.less(real_color_image_rotated, 0), lambda: tf.add(real_color_image_rotated, 255), lambda: real_color_image_rotated)
    real_color_image_rotated = tf.where(tf.math.less_equal(real_color_image_rotated, 0), 255 * tf.ones_like(real_color_image_rotated) + real_color_image_rotated, real_color_image_rotated)

    # Turn 255 back to 0 in the symmetry plane
    tensor01 = tf.einsum('ij,mklj->mkli', tf.convert_to_tensor(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), dtype=tf.float32), real_color_image_rotated)
    tensor02 = tf.where(tf.math.equal(tensor01, 255),  tf.zeros_like(tensor01), tensor01)
    tensor03 = tf.einsum('ij,mklj->mkli', tf.convert_to_tensor(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]), dtype=tf.float32), real_color_image_rotated)

    real_color_image_rotated = tensor02 + tensor03

    #real_color_image_rotated = tf.convert_to_tensor(real_color_image_rotated, dtype=tf.float32)
    plt.imshow(real_color_image_rotated.numpy().astype(np.uint8)[0])
    plt.show()

    # Get the number of pixels
    num_pixels = tf.math.reduce_prod(tf.shape(real_color_image)[1:3])
    beta = 3

    # Calculate the difference between the real and predicted images including the mask
    diff_object = tf.math.abs(predicted_color_image*mask_object - real_color_image*mask_object)
    diff_background = tf.math.abs(predicted_color_image*mask_background - real_color_image*mask_background)

    # Calculate the total loss
    loss_colors = tf.cast((1/num_pixels), dtype=tf.float32)*(beta*tf.math.reduce_sum(diff_object, axis=[1, 2, 3]) + tf.math.reduce_sum(diff_background, axis=[1, 2, 3]))

    return loss_colors

num_samples = 5
list_images = list()
# view = SingleView(filepath="/home/fabian/.keras/datasets/036_wood_block/textured_edited.obj")
#view = SingleView(filepath="/home/fabian/.keras/datasets/001_chips_can/tsdf/textured_edited.obj")
view = SingleView(filepath="/home/fabian/.keras/datasets/custom_objects/symmetry_z_2_object.obj")

image_original, image_colors, alpha_original = view.render()
image_colors = image_colors.copy()


angle = np.pi
rotation_matrix = quarternion_to_rotation_matrix(np.array([0, np.sin(angle / 2), 0, np.cos(angle / 2)]))
rotation_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
print(rotation_matrix)

image_colors_copy = np.expand_dims(image_colors, axis=0)
loss_color(tf.convert_to_tensor(image_colors_copy, dtype=tf.float32), tf.convert_to_tensor(image_colors_copy, dtype=tf.float32), tf.convert_to_tensor(rotation_matrix, dtype=tf.float32))

plt.imshow(image_colors)
plt.show()
# Applying the rotation to the image (easy way)
image_colors_rotated = np.einsum('ij,klj->kli', rotation_matrix, image_colors)
image_colors_rotated = np.mod(image_colors_rotated, 256).astype(np.uint8)
plt.imshow(image_colors_rotated)
plt.show()

# Applying the rotation to the image (complicated way)
for i in range(image_colors.shape[0]):
    for j in range(image_colors.shape[1]):
        if not np.array_equal(image_colors[i, j], np.array([0., 0., 0.])):
            a = np.dot(rotation_matrix, image_colors[i, j])
            image_colors[i, j] = np.dot(rotation_matrix, image_colors[i, j])
