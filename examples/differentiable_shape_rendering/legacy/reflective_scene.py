from functools import partial
import jax.numpy as jp
import jax
import paz

camera_origin = jp.array([0, 3.0, 4.0])
camera_target = jp.array([0.0, 0.0, 0.0])
world_up = jp.array([0.0, 1.0, 0.0])
lights = paz.graphics.PointLight(jp.full(3, 1.2), jp.array([3.0, 3.0, 5.0]))
camera_pose = paz.SE3.view_transform(camera_origin, camera_target, world_up)
H, W, y_FOV = image_size = 1024, 1024, jp.pi / 4.0
render_kwargs = {"lights": lights, "mask": None, "shadows": False, "num_bounces": 3}  # fmt: skip
rays = paz.graphics.camera.build_rays(image_size, y_FOV, camera_pose)
render_args = ((H, W), camera_pose, rays)

material = paz.graphics.Material(paz.graphics.RED, 0.3, 0.5, 0.8, 16)
pose = paz.SE3.translation(jp.array([0.0, 1.0, 0.0]))


def make_grid(box_size, line_width, rows, cols, line_color, fill_color):
    step_size = line_width + box_size
    H_image = (rows * step_size) + line_width
    W_image = (cols * step_size) + line_width
    y, x = jp.indices((H_image, W_image))
    position_in_step_x = x % step_size
    position_in_step_y = y % step_size
    is_line_pixel_x = position_in_step_x < line_width
    is_line_pixel_y = position_in_step_y < line_width
    is_grid_line = jp.expand_dims(is_line_pixel_x | is_line_pixel_y, axis=-1)
    return (is_grid_line * line_color) + ((1 - is_grid_line) * fill_color)


# floor ------------------------------------------------
GREEN = jp.array([85 / 255, 181 / 255, 103 / 255])
GRAY = jp.array([0.662, 0.647, 0.576])
pattern = make_grid(50, 5, 10, 10, GRAY, GRAY * 0.2)
pattern = paz.graphics.PlanarPattern(
    pattern,
    paz.SE3.translation(jp.array([1.0, 0.0, 1.0]))
    @ paz.SE3.scaling(jp.full(3, 1.0)),
)
floor_scale = paz.SE3.scaling(jp.array([15, 0.1, 15]))
floor_angle = paz.SE3.rotation_y(jp.pi / 4)
floor_pose = floor_angle @ floor_scale
floor_material = paz.graphics.Material(
    jp.zeros(3), 0.4, 0.1, 0.0, 100, 0.2, 0.0, 0.1
)
# zero_material = Material(jp.zeros(3), 0.85, 0.1, 0.0, 100)
floor = paz.graphics.Cube(floor_pose, floor_material, pattern)
#  --------------------------------------------------------

shape = paz.graphics.Sphere(pose, material)
scene = paz.graphics.Scene([shape, floor])

render = jax.jit(partial(paz.graphics.render, *render_args, **render_kwargs))
image, depth = render(scene=scene)
paz.image.show(paz.image.denormalize(image))
