import jax
import jax.numpy as jp
import paz
import matplotlib.pyplot as plt


def build_checkerboard(color_a, color_b, grid_size=5, block_size=50):
    image_size = grid_size * block_size
    image = jp.zeros((image_size, image_size, 3))

    for row in range(grid_size):
        for col in range(grid_size):
            if (row + col) % 2 == 0:
                color = color_a
            else:
                color = color_b

            start_row = row * block_size
            end_row = start_row + block_size
            start_col = col * block_size
            end_col = start_col + block_size
            image = image.at[start_row:end_row, start_col:end_col].set(color)
    return image


material = paz.graphics.Material(paz.graphics.WHITE)
color_A = jp.array([154, 213, 135]) / 255.0
color_B = jp.array([0, 98, 52]) / 255.0
image = build_checkerboard(color_A, color_B)
# plt.imshow(image)
# plt.show()
pattern = paz.graphics.Pattern(jp.eye(4), paz.graphics.PLANAR_PATTERN, image)
empty_pattern = paz.graphics.Pattern(
    jp.eye(4), paz.graphics.PLANAR_PATTERN, jp.zeros_like(image)
)
transform = paz.SE3.scaling(jp.array([5, 0.1, 5]))
floor = paz.graphics.Cube(transform, material, pattern)
axes = paz.graphics.load("axes.json")
scene = paz.graphics.Scene([floor, paz.graphics.Sphere(), axes])
# scene = paz.graphics.Scene([floor, paz.graphics.Sphere(pattern=empty_pattern)])

light = paz.graphics.PointLight(jp.ones(3), jp.array([-4.0, 5.0, 6.0]))
# scene, _, _ = paz.graphics.scene.compile(scene, light, None)

camera_pose = paz.SE3.view_transform(
    jp.array([0, 10, 10]), jp.array([0, 0, 0]), jp.array([0, 1, 0])
)
paz.graphics.viewer(scene, camera_pose)

H, W = 480, 640
y_FOV = jp.pi / 4.0
rays = paz.graphics.camera.build_rays((H, W), y_FOV, camera_pose)
render = jax.jit(paz.graphics.render, static_argnums=(0,))
image, depth = render((H, W), camera_pose, rays, scene, light)
paz.image.show(paz.image.denormalize(image))
