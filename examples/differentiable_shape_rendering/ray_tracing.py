from functools import partial
import jax.numpy as jp
import jax
import paz

camera_origin = 0.78 * jp.array([0, 5.0, 5.0])
camera_target = jp.array([0.0, 0.0, 0.0])

world_up = jp.array([0.0, 1.0, 0.0])
light_position = 0.5 * jp.array([3.0, 5.0, -3.65])
lights = paz.graphics.PointLight(jp.full(3, 1.1), light_position)

world_to_camera = paz.SE3.view_transform(camera_origin, camera_target, world_up)
camera_to_world = jp.linalg.inv(world_to_camera)
camera_shift = paz.SE3.translation(jp.array([0.0, 0.0, -1.3]))
camera_to_world = camera_shift @ camera_to_world
world_to_camera = jp.linalg.inv(camera_to_world)

# H, W, y_FOV, resize_factor = image_size = 2 * 1024, 2 * 1024, jp.pi / 4.0, 1
H, W, y_FOV, resize_factor = image_size = 1024, 1024, jp.pi / 4.0, 1
render_kwargs = {"mask": None, "shadows": True, "lights": lights}
rays = paz.graphics.camera.build_rays(image_size, y_FOV, world_to_camera)
render_args = ((H, W), world_to_camera, rays)
GREEN = (85 / 255, 181 / 255, 103 / 255)  # YlGnL
WHITE = (1.0, 1.0, 1.0)


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


def highlight_cell(img, row, col, color, box_size=50, line_width=5):
    step = box_size + line_width
    r_idx = row - 1
    c_idx = col - 1
    y_start = line_width + (r_idx * step)
    x_start = line_width + (c_idx * step)
    return img.at[
        y_start : (y_start + box_size), x_start : (x_start + box_size)
    ].set(color)


def make_ray(point_A, point_B, material, width=0.02, cone_height=0.15):
    diff = point_B - point_A
    length = jp.linalg.norm(diff)
    direction = diff / length
    y_axis = direction
    ref_axis = jp.where(
        jp.abs(y_axis[1]) > 0.99,
        jp.array([1.0, 0.0, 0.0]),
        jp.array([0.0, 1.0, 0.0]),
    )
    x_axis = jp.cross(y_axis, ref_axis)
    x_axis = x_axis / jp.linalg.norm(x_axis)
    z_axis = jp.cross(x_axis, y_axis)
    z_axis = z_axis / jp.linalg.norm(z_axis)
    transform = jp.eye(4)
    transform = transform.at[:3, 0].set(x_axis)
    transform = transform.at[:3, 1].set(y_axis)
    transform = transform.at[:3, 2].set(z_axis)
    transform = transform.at[:3, 3].set(point_A)
    safe_cone_h = jp.minimum(cone_height, length * 0.5)
    cyl_len = length - safe_cone_h
    cyl_scale = paz.SE3.scaling(jp.array([width, cyl_len / 2.0, width]))
    cyl_shift = paz.SE3.translation(jp.array([0.0, cyl_len / 2.0, 0.0]))
    cyl_transform = transform @ cyl_shift @ cyl_scale
    shaft = paz.graphics.Cylinder(cyl_transform, material)
    cone_scale = paz.SE3.scaling(
        jp.array([width * 2.5, safe_cone_h / 2.0, width * 2.5])
    )
    cone_shift = paz.SE3.translation(
        jp.array([0.0, cyl_len + (safe_cone_h / 2.0), 0.0])
    )
    cone_transform = transform @ cone_shift @ cone_scale
    head = paz.graphics.Cone(cone_transform, material)

    return paz.graphics.Group([shaft, head])


def CheckeredImage(box_size=50, rows=8, cols=8, color_A=GREEN, color_B=WHITE):
    checkered = jp.indices((rows, cols)).sum(axis=0) % 2
    image_channels = []
    for channel_arg in range(3):
        checkered_channel = jp.kron(checkered, jp.ones((box_size, box_size)))
        checkered_color_A = color_A[channel_arg] * checkered_channel
        checkered_color_B = color_B[channel_arg] * (1 - checkered_channel)
        checkered_channel = checkered_color_A + checkered_color_B
        image_channels.append(jp.expand_dims(checkered_channel, axis=-1))
    return jp.concatenate(image_channels, axis=-1)


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


# camera --------------------------------------------------
camera = paz.graphics.load("assets/camera")
camera_shift = 1.8 * jp.array([-1.0, 1.0, -2.0])
camera_pose = paz.SE3.view_transform(camera_shift, jp.zeros(3), world_up)
camera_pose = jp.linalg.inv(camera_pose)
camera_size = paz.SE3.scaling(jp.full(3, 0.2))
camera_angle = paz.SE3.rotation_y(jp.pi)
camera = camera._replace(transform=camera_pose @ camera_angle @ camera_size)

# image plane --------------------------------------------
GRAY = jp.array([0.662, 0.647, 0.576])
zero_material = paz.graphics.Material(jp.zeros(3), 0.85, 0.1, 0.0, 100)
image_plane_size = paz.SE3.scaling(jp.array([1.0, 0.05, 1.0]))
pattern = make_grid(50, 5, 6, 6, GRAY, paz.graphics.WHITE)
HIGHLIGHT_BLUE = 0.9 * jp.array([0.36, 0.77, 0.96])
pattern = highlight_cell(pattern, 3, 5, HIGHLIGHT_BLUE)
pattern = highlight_cell(pattern, 4, 3, HIGHLIGHT_BLUE)
pattern = paz.graphics.PlanarPattern(
    pattern,
    paz.SE3.translation(jp.array([1.0, 0.0, 1.0]))
    @ paz.SE3.scaling(jp.full(3, 2.0)),
)

image_plane = paz.graphics.Cube(image_plane_size, zero_material, pattern)

plane_pose = paz.SE3.view_transform(0.6 * camera_shift, jp.zeros(3), world_up)
plane_pose = jp.linalg.inv(plane_pose)
image_angle = paz.SE3.rotation_x(jp.pi / 2.0)
image_plane = image_plane._replace(
    transform=plane_pose @ image_angle @ image_plane_size
)

# axes -------------------------------------------------
axes = paz.graphics.load("assets/axes")
axes = axes._replace(
    transform=paz.SE3.translation(jp.array([0.0, 0.2, 0.0]))
    @ paz.SE3.scaling(jp.full(3, 0.15))
)

# floor ------------------------------------------------
GREEN = jp.array([85 / 255, 181 / 255, 103 / 255])  # YlGnL
image_plane_size = paz.SE3.scaling(jp.array([1.0, 0.05, 1.0]))
pattern = make_grid(50, 5, 10, 10, GREEN, paz.graphics.WHITE)
pattern = paz.graphics.PlanarPattern(
    pattern,
    paz.SE3.translation(jp.array([1.0, 0.0, 1.0]))
    @ paz.SE3.scaling(jp.full(3, 2.0)),
)
floor_size = paz.SE3.scaling(jp.array([1.5, 0.1, 1.5]))
floor_material = paz.graphics.Material(jp.full(3, 0.85), 0.4, 0.1, 0.0, 100)
floor = paz.graphics.Cube(floor_size, floor_material, pattern)
floor = floor._replace(transform=floor_size)

# light -------------------------------------------------
yellow = jp.array([1.0, 0.65, 0.0])
# _light_position = 1.1 * light_position
_light_position = 0.88 * light_position
light_material = paz.graphics.Material(yellow, 0.9, 0.0, 0.0, 100)
light_radius = 0.2
light_scale = paz.SE3.scaling(jp.full(3, light_radius))
light_shift = paz.SE3.translation(_light_position)
light = paz.graphics.Sphere(light_shift @ light_scale, light_material)

x, height, z = _light_position
radius = 0.05
stand_shift = jp.array([x, (height / 2.0), z])
stand_shift = paz.SE3.translation(stand_shift)
stand_scale = paz.SE3.scaling(jp.array([radius, height / 2.0, radius]))
stand_material = paz.graphics.Material(GRAY, 0.9, 0.0, 0.0, 200)
stand = paz.graphics.Cylinder(stand_shift @ stand_scale, stand_material)

# shape 0
checkered_image = CheckeredImage()
sphere_pattern = paz.graphics.SphericalPattern(checkered_image)
sphere_scale = paz.SE3.scaling(sphere_radius := 0.35)
sphere_shift = paz.SE3.translation(jp.array([1.00, sphere_radius, 0.7]))
sphere_transform = sphere_shift @ sphere_scale
sphere = paz.graphics.Sphere(sphere_transform, zero_material, sphere_pattern)

# shape 1
cone_scale = paz.SE3.scaling(cone_size := 0.45)
cone_shift = paz.SE3.translation(jp.array([-0.6, cone_size, -0.4]))
checkered_image = CheckeredImage(50, 4, 4)
planar_pattern = paz.graphics.PlanarPattern(checkered_image)
planar_pattern._replace(transform=paz.SE3.scaling(1))
cone = paz.graphics.Cone(cone_shift @ cone_scale, zero_material, planar_pattern)


light_blue = 0.8 * jp.array([0.36, 0.77, 0.96])
ray_material = paz.graphics.Material(light_blue, 0.9, 0, 0, 100)
# ray_material = paz.graphics.Material(light_blue)
camera_position = paz.SE3.get_position_vector(camera_pose)
# camera_position = 0.955 * camera_position
camera_position = 0.88 * camera_position
# ray_0_point_B = get_point_on_sphere(sphere_transform, jp.pi / 5, 1.4 * jp.pi)
ray_0_point_B = get_point_on_sphere(sphere_transform, jp.pi / 10, 5 * jp.pi / 4)
ray_0 = make_ray(camera_position, ray_0_point_B, ray_material)

# ray_1_point_B = jp.array([-1.1, 0.1, -0.05])
ray_1_point_B = jp.array([-1.4, 0.1, 0.15])
ray_1 = make_ray(camera_position, ray_1_point_B, ray_material)

red = 0.8 * paz.graphics.RED
shadow_ray_material = paz.graphics.Material(red, 0.9, 0, 0, 100)
shadow_ray_args = (shadow_ray_material, 0.8 * 0.02, 0.8 * 0.15)
shadow_target_0 = project_to_sphere_surface(
    ray_0_point_B, _light_position, light_radius
)
shadow_ray_0 = make_ray(ray_0_point_B, shadow_target_0, *shadow_ray_args)

shadow_target_1 = project_to_sphere_surface(
    ray_1_point_B, _light_position, light_radius
)
shadow_ray_1 = make_ray(ray_1_point_B, shadow_target_1, *shadow_ray_args)

real_light = paz.graphics.Sphere(
    paz.SE3.translation(light_position) @ paz.SE3.scaling(light_radius),
    ray_material,
)


shapes = [
    camera,
    image_plane,
    axes,
    floor,
    light,
    stand,
    sphere,
    cone,
    ray_0,
    ray_1,
    shadow_ray_0,
    shadow_ray_1,
]

scene = paz.graphics.Scene(shapes)
shadow_mask = jp.array(
    [
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
)

render = jax.jit(partial(paz.graphics.render, *render_args, **render_kwargs))
image, depth = render(scene=scene, shadow_mask=shadow_mask)
image = paz.image.denormalize(image)
image = paz.image.resize_opencv(image, (H // resize_factor, W // resize_factor))
paz.image.write("ray_tracing.png", image)
