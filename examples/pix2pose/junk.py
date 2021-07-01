import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from paz.backend.quaternion import quarternion_to_rotation_matrix, quaternion_multiply

from scenes import SingleView

num_samples = 5
list_images = list()
# view = SingleView(filepath="/home/fabian/.keras/datasets/036_wood_block/textured_edited.obj")
view = SingleView(filepath="/home/fabian/.keras/datasets/001_chips_can/tsdf/textured_edited.obj")

image_original, image_colors, alpha_original, _ = view.render()
#image_colors = image_colors/255.
image_colors = image_colors.copy()

plt.imshow(image_colors)
plt.show()

angle = np.pi
#final_rotation = quaternion_multiply(rotation, np.array([0, np.sin(angle / 2), 0, np.cos(angle / 2)]))
#final_rotation /= np.linalg.norm(final_rotation)
rotation_matrix = quarternion_to_rotation_matrix(np.array([0, 0, np.sin(angle / 2), np.cos(angle / 2)]))

print(rotation_matrix)
for i in range(image_colors.shape[0]):
    for j in range(image_colors.shape[1]):
        image_colors[i, j] = rotation_matrix@image_colors[i, j]

image_colors = np.interp(image_colors, (image_colors.min(), image_colors.max()), (0, 1.0))
plt.imshow(image_colors)
plt.show()