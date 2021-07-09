import matplotlib.pyplot as plt
import numpy as np
from paz.backend.quaternion import quarternion_to_rotation_matrix

import tensorflow as tf

from scenes import SingleView

def loss_color(real_color_image, predicted_color_image, rotation_matrix):
    # Calculate masks for the object and the background
    mask_object = tf.repeat(tf.expand_dims(tf.math.reduce_max(tf.math.ceil(real_color_image), axis=-1), axis=-1), repeats=3, axis=-1)
    mask_background = tf.ones(tf.shape(mask_object)) - mask_object

    # Show the original image
    plt.imshow(real_color_image.numpy()[0].astype(np.float))
    plt.show()

    # Apply the rotation matrix to all points
    real_color_image = real_color_image + tf.ones_like(real_color_image)*0.0001
    real_color_image_rotated = tf.einsum('ij,mklj->mkli', rotation_matrix, real_color_image)

    # Solve the discontinuity problem:
    # When there is a 0 in the rotation plane it needs to be converted to a 1.
    # Because tensors cannot be modified this gets a little bit complicated
    #tensor01 = tf.einsum('ij,mklj->mkli', tf.convert_to_tensor(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), dtype=tf.float32), real_color_image_rotated)
    #tensor02 = tf.einsum('ij,mklj->mkli', tf.convert_to_tensor(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]), dtype=tf.float32), real_color_image_rotated)
    #tensor03 = tensor02 + tf.einsum('ij,mklj->mkli', tf.convert_to_tensor(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), dtype=tf.float32), tf.ones_like(tensor02))
    #tensor04 = tf.where(tf.math.equal(tensor03, 0),  1*tf.ones_like(tensor03), tensor03)
    #tensor05 = tf.einsum('ij,mklj->mkli', tf.convert_to_tensor(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]), dtype=tf.float32), tensor04)

    #tensor06 = tensor01 + tensor05

    tensor06 = real_color_image_rotated

    # Add 1 when the rotation values are below 0
    real_color_image_rotated = tf.where(tf.math.less(tensor06, 0), 1 * tf.ones_like(tensor06) + tensor06, tensor06)

    # Exclude the background
    real_color_image_rotated = real_color_image_rotated*mask_object

    plt.imshow(mask_object.numpy().astype(np.float)[0])
    plt.show()

    #real_color_image_rotated = tf.convert_to_tensor(real_color_image_rotated, dtype=tf.float32)
    plt.imshow(real_color_image_rotated.numpy().astype(np.float)[0])
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
view = SingleView(filepath="/home/fabian/.keras/datasets/custom_objects/simple_symmetry_object.obj")

image_original, image_colors, alpha_original = view.render()
image_colors = image_colors.copy()
image_colors = image_colors/255.

# Apply the rotation with Tensorflow
epsilon = 0.0001
#rotation_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
#rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
rotation_matrices = np.load("./rotation_matrices/simple_symmetry_object_two_matrices.npy")

image_colors_copy = np.expand_dims(image_colors, axis=0)
#loss_color(tf.convert_to_tensor(image_colors_copy/255., dtype=tf.float32), tf.convert_to_tensor(image_colors_copy/255., dtype=tf.float32), tf.convert_to_tensor(rotation_matrices[0], dtype=tf.float32))

# Applying the rotation to the image (easy way)
plt.imshow(image_colors)
plt.show()
mask_image = (np.sum(image_colors, axis=-1) != 0).astype(float)
mask_image = np.repeat(mask_image[...,np.newaxis], 3, axis=-1)
image_colors_rotated = image_colors + np.ones_like(image_colors)*0.0001
image_colors_rotated = np.einsum('ij,klj->kli', rotation_matrices[0], image_colors_rotated)
image_colors_rotated = np.where(np.less(image_colors_rotated, 0), 1 * np.ones_like(image_colors_rotated) + image_colors_rotated, image_colors_rotated)
image_colors_rotated = np.clip(image_colors_rotated, a_min=0.0, a_max=1.0)
image_colors_rotated = image_colors_rotated*mask_image

plt.imshow(mask_image)
plt.show()

plt.imshow(image_colors_rotated)
plt.show()

# Applying the rotation to the image (complicated way)
for i in range(image_colors.shape[0]):
    for j in range(image_colors.shape[1]):
        if not np.array_equal(image_colors[i, j], np.array([0., 0., 0.])):
            a = np.dot(rotation_matrices[0], image_colors[i, j])
            image_colors[i, j] = np.dot(rotation_matrices[0], image_colors[i, j])
