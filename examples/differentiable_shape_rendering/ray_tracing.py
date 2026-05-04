from functools import partial
import jax.numpy as jp
import jax
import paz

camera_origin = 0.78 * jp.array([0, 5.0, 5.0])
camera_target = jp.array([0.0, 0.0, 0.0])

world_up = jp.array([0.0, 1.0, 0.0])
light_position = 0.5 * jp.array([3.0, 5.0, -3.65])
lights = paz.graphics.PointLight(jp.full(3, 1.1), light_position)
GREEN = jp.array([85 / 255, 181 / 255, 103 / 255])  # YlGnL
GRAY = jp.array([0.662, 0.647, 0.576])
WHITE = jp.ones(3)

world_to_camera = paz.SE3.view_transform(camera_origin, camera_target, world_up)
camera_to_world = jp.linalg.inv(world_to_camera)
camera_shift = paz.SE3.translation(jp.array([0.0, 0.0, -1.3]))
camera_to_world = camera_shift @ camera_to_world
world_to_camera = jp.linalg.inv(camera_to_world)

H, W = 2**12, 2**12
y_FOV = jp.pi / 4.0
resize_factor = 2
image_size = (H, W)
render_kwargs = dict(mask=None, shadows=True, lights=lights)
render_kwargs.update(tiles=(1, 1), chunk_size=1024)
render_args = image_size, y_FOV, world_to_camera


def get_point_on_sphere(transform, theta, phi):
    x = jp.sin(theta) * jp.cos(phi)
    y = jp.cos(theta)
    z = jp.sin(theta) * jp.sin(phi)
    local_point = jp.array([x, y, z, 1.0])
    world_point_homogenous = transform @ local_point
    return world_point_homogenous[:3]


def project_to_sphere_surface(point, center, radius):
    direction = center - point
    direction = direction / jp.maximum(jp.linalg.norm(direction), 1e-8)
    return center - (radius * direction)


def highlight_cell(image, row, col, color, box_size=50, line_width=5):
    step = box_size + line_width
    row_index = row - 1
    col_index = col - 1
    y_start = line_width + (row_index * step)
    x_start = line_width + (col_index * step)
    y_end = y_start + box_size
    x_end = x_start + box_size
    image_cell = image.at[y_start:y_end, x_start:x_end]
    return image_cell.set(color)


def make_ray(start, end, material, width=0.02, cone_height=0.15):
    diff = end - start
    length = jp.linalg.norm(diff)
    transform = make_ray_transform(start, diff / length)
    head_length = jp.minimum(cone_height, length * 0.5)
    shaft_length = length - head_length
    shaft = make_ray_shaft(transform, shaft_length, material, width)
    head = make_ray_head(transform, shaft_length, head_length, material, width)
    return paz.graphics.Group([shaft, head])


def make_ray_transform(start, direction):
    x_axis, y_axis, z_axis = make_ray_axes(direction)
    transform = jp.eye(4)
    transform = transform.at[:3, 0].set(x_axis)
    transform = transform.at[:3, 1].set(y_axis)
    transform = transform.at[:3, 2].set(z_axis)
    return transform.at[:3, 3].set(start)


def make_ray_axes(direction):
    y_axis = direction
    x_reference = jp.array([1.0, 0.0, 0.0])
    y_reference = jp.array([0.0, 1.0, 0.0])
    is_aligned = jp.abs(y_axis[1]) > 0.99
    ref_axis = jp.where(is_aligned, x_reference, y_reference)
    x_axis = jp.cross(y_axis, ref_axis)
    x_axis = x_axis / jp.linalg.norm(x_axis)
    z_axis = jp.cross(x_axis, y_axis)
    z_axis = z_axis / jp.linalg.norm(z_axis)
    return x_axis, y_axis, z_axis


def make_ray_shaft(transform, length, material, width):
    shaft_size = jp.array([width, length / 2.0, width])
    shaft_shift = jp.array([0.0, length / 2.0, 0.0])
    shaft_scale = paz.SE3.scaling(shaft_size)
    shaft_pose = paz.SE3.translation(shaft_shift)
    shaft_transform = transform @ shaft_pose @ shaft_scale
    return paz.graphics.Cylinder(shaft_transform, material)


def make_ray_head(transform, shaft_length, cone_height, material, width):
    radius = width * 2.5
    head_height = cone_height / 2.0
    head_center = shaft_length + head_height
    head_size = jp.array([radius, head_height, radius])
    head_shift = jp.array([0.0, head_center, 0.0])
    head_scale = paz.SE3.scaling(head_size)
    head_pose = paz.SE3.translation(head_shift)
    return paz.graphics.Cone(transform @ head_pose @ head_scale, material)


def make_checkered_image(box_size=50, rows=8, cols=8):
    checkered = jp.indices((rows, cols)).sum(axis=0) % 2
    image_channels = []
    for channel in range(3):
        checkered_channel = jp.kron(checkered, jp.ones((box_size, box_size)))
        green_channel = GREEN[channel] * checkered_channel
        white_channel = WHITE[channel] * (1 - checkered_channel)
        checkered_channel = green_channel + white_channel
        image_channels.append(jp.expand_dims(checkered_channel, axis=-1))
    return jp.concatenate(image_channels, axis=-1)


def make_grid(box_size, line_width, rows, cols, line_color, fill_color):
    step_size = line_width + box_size
    image_height = (rows * step_size) + line_width
    image_width = (cols * step_size) + line_width
    y, x = jp.indices((image_height, image_width))
    position_in_step_x = x % step_size
    position_in_step_y = y % step_size
    is_line_pixel_x = position_in_step_x < line_width
    is_line_pixel_y = position_in_step_y < line_width
    is_grid_line = jp.expand_dims(is_line_pixel_x | is_line_pixel_y, axis=-1)
    return (is_grid_line * line_color) + ((1 - is_grid_line) * fill_color)


def make_planar_pattern(image):
    shift = paz.SE3.translation(jp.array([1.0, 0.0, 1.0]))
    scale = paz.SE3.scaling(jp.full(3, 2.0))
    return paz.graphics.PlanarPattern(image, shift @ scale)


# camera --------------------------------------------------
camera = paz.graphics.load("assets/camera")
camera_shift = 1.8 * jp.array([-1.0, 1.0, -2.0])
camera_pose = paz.SE3.view_transform(camera_shift, jp.zeros(3), world_up)
camera_pose = jp.linalg.inv(camera_pose)
camera_size = paz.SE3.scaling(jp.full(3, 0.2))
camera_angle = paz.SE3.rotation_y(jp.pi)
camera = camera._replace(transform=camera_pose @ camera_angle @ camera_size)

# image plane --------------------------------------------
zero_material = paz.graphics.Material(jp.zeros(3), 0.85, 0.1, 0.0, 100)
image_plane_size = paz.SE3.scaling(jp.array([1.0, 0.05, 1.0]))
pattern = make_grid(50, 5, 6, 6, GRAY, paz.graphics.WHITE)
HIGHLIGHT_BLUE = 0.9 * jp.array([0.36, 0.77, 0.96])
pattern = highlight_cell(pattern, 3, 5, HIGHLIGHT_BLUE)
pattern = highlight_cell(pattern, 4, 3, HIGHLIGHT_BLUE)
pattern = make_planar_pattern(pattern)

image_plane = paz.graphics.Cube(image_plane_size, zero_material, pattern)

plane_pose = paz.SE3.view_transform(0.6 * camera_shift, jp.zeros(3), world_up)
plane_pose = jp.linalg.inv(plane_pose)
image_angle = paz.SE3.rotation_x(jp.pi / 2.0)
image_plane_transform = plane_pose @ image_angle @ image_plane_size
image_plane = image_plane._replace(transform=image_plane_transform)

# axes -------------------------------------------------
axes = paz.graphics.load("assets/axes")
axes_shift = paz.SE3.translation(jp.array([0.0, 0.2, 0.0]))
axes_scale = paz.SE3.scaling(jp.full(3, 0.15))
axes = axes._replace(transform=axes_shift @ axes_scale)

# floor ------------------------------------------------
pattern = make_grid(50, 5, 10, 10, GREEN, paz.graphics.WHITE)
pattern = make_planar_pattern(pattern)
floor_size = paz.SE3.scaling(jp.array([1.5, 0.1, 1.5]))
floor_args = (jp.full(3, 0.85), 0.4, 0.1, 0.0, 100)
floor_material = paz.graphics.Material(*floor_args)
floor = paz.graphics.Cube(floor_size, floor_material, pattern)
floor = floor._replace(transform=floor_size)

# light -------------------------------------------------
yellow = jp.array([1.0, 0.65, 0.0])
lamp_position = 0.88 * light_position
light_material = paz.graphics.Material(yellow, 0.9, 0.0, 0.0, 100)
light_radius = 0.2
light_scale = paz.SE3.scaling(jp.full(3, light_radius))
light_shift = paz.SE3.translation(lamp_position)
light = paz.graphics.Sphere(light_shift @ light_scale, light_material)

x, height, z = lamp_position
radius = 0.05
stand_shift = jp.array([x, (height / 2.0), z])
stand_shift = paz.SE3.translation(stand_shift)
stand_scale = paz.SE3.scaling(jp.array([radius, height / 2.0, radius]))
stand_material = paz.graphics.Material(GRAY, 0.9, 0.0, 0.0, 200)
stand = paz.graphics.Cylinder(stand_shift @ stand_scale, stand_material)

# shape 0
checkered_image = make_checkered_image()
sphere_pattern = paz.graphics.SphericalPattern(checkered_image)
sphere_scale = paz.SE3.scaling(sphere_radius := 0.35)
sphere_shift = paz.SE3.translation(jp.array([1.00, sphere_radius, 0.7]))
sphere_transform = sphere_shift @ sphere_scale
sphere = paz.graphics.Sphere(sphere_transform, zero_material, sphere_pattern)

# shape 1
cone_scale = paz.SE3.scaling(cone_size := 0.45)
cone_shift = paz.SE3.translation(jp.array([-0.6, cone_size, -0.4]))
checkered_image = make_checkered_image(50, 4, 4)
planar_pattern = paz.graphics.PlanarPattern(checkered_image)
cone = paz.graphics.Cone(cone_shift @ cone_scale, zero_material, planar_pattern)


light_blue = 0.8 * jp.array([0.36, 0.77, 0.96])
ray_material = paz.graphics.Material(light_blue, 0.9, 0, 0, 100)
camera_position = paz.SE3.get_position_vector(camera_pose)
camera_position = 0.88 * camera_position
ray_0_end = get_point_on_sphere(sphere_transform, jp.pi / 10, 5 * jp.pi / 4)
ray_0 = make_ray(camera_position, ray_0_end, ray_material)

ray_1_end = jp.array([-1.4, 0.1, 0.15])
ray_1 = make_ray(camera_position, ray_1_end, ray_material)

red = 0.8 * paz.graphics.RED
shadow_ray_material = paz.graphics.Material(red, 0.9, 0, 0, 100)
shadow_ray_args = (shadow_ray_material, 0.8 * 0.02, 0.8 * 0.15)
ray_0_light_args = (ray_0_end, lamp_position, light_radius)
shadow_target_0 = project_to_sphere_surface(*ray_0_light_args)
shadow_ray_0 = make_ray(ray_0_end, shadow_target_0, *shadow_ray_args)

ray_1_light_args = (ray_1_end, lamp_position, light_radius)
shadow_target_1 = project_to_sphere_surface(*ray_1_light_args)
shadow_ray_1 = make_ray(ray_1_end, shadow_target_1, *shadow_ray_args)

base_shapes = [camera, image_plane, axes, floor, light, stand, sphere, cone]
ray_shapes = [ray_0, ray_1, shadow_ray_0, shadow_ray_1]
scene = paz.graphics.Scene(base_shapes + ray_shapes)
shadow_mask = jp.array([True] + 5 * [False] + 2 * [True] + 4 * [False])

render = jax.jit(partial(paz.graphics.render, *render_args, **render_kwargs))
image, _ = render(scene=scene, shadow_mask=shadow_mask)
image = paz.image.denormalize(image)
image = paz.image.resize_opencv(image, (H // resize_factor, W // resize_factor))
paz.image.write("ray_tracing.png", image)
