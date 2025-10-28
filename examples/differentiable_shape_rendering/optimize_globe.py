import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
from jax.experimental.compilation_cache.compilation_cache import set_cache_dir
import jax.numpy as jp
import jax
import paz
import optax
from paz import SE3
from paz.graphics import (
    PointLight,
    Sphere,
    Material,
    Cylinder,
    Pattern,
    SphericalPattern,
    PlanarPattern,
    Cone,
    Cube,
    WHITE,
    BLACK,
)

set_cache_dir(paz.logger.make_directory("cache"))


def CheckeredImage(box_size=50, rows=8, cols=8, color_A=BLACK, color_B=WHITE):
    checkered = jp.indices((rows, cols)).sum(axis=0) % 2
    image_channels = []
    for channel_arg in range(3):
        checkered_channel = jp.kron(checkered, jp.ones((box_size, box_size)))
        checkered_color_A = color_A[channel_arg] * checkered_channel
        checkered_color_B = color_B[channel_arg] * (1 - checkered_channel)
        checkered_channel = checkered_color_A + checkered_color_B
        image_channels.append(jp.expand_dims(checkered_channel, axis=-1))
    return jp.concatenate(image_channels, axis=-1)


def Button(material, radius=1.1, angle=60, height=0.075, size=0.05):
    x = jp.cos(jp.deg2rad(angle))
    y = jp.sin(jp.deg2rad(angle))
    shift = SE3.translation(radius * jp.array([x, y, 0]))
    scale = paz.SE3.scaling(jp.array([size, height, size]))
    angle = paz.SE3.rotation_z(-jp.deg2rad(30))
    bottom = paz.graphics.Cylinder(shift @ angle @ scale, material)
    top_shift = SE3.translation((radius + height) * jp.array([x, y, 0]))
    top = Sphere(top_shift @ angle @ scale, material)
    return paz.graphics.Group([bottom, top], jp.eye(4))


def Base(transform, base_material, pattern, pattern_neck):
    bottom_shift = SE3.translation([0.0, 0.0, 0.0])
    bottom_scale = SE3.scaling([1.0, 0.05, 1.0])
    bottom_disk = Cylinder(bottom_shift @ bottom_scale, base_material, pattern)

    concave_shift = SE3.translation([0.0, 0.25, 0.0])
    concave_scale = SE3.scaling([0.975, 0.2, 0.975])
    concave_curve = Cone(concave_shift @ concave_scale, base_material, pattern)

    middle_shift = SE3.translation([0.0, 0.15, 0.0])
    middle_scale = SE3.scaling([0.7, 0.05, 0.7])
    middle_disk = Cylinder(middle_shift @ middle_scale, base_material, pattern)

    upper_shift = SE3.translation([0.0, 0.40, 0.0])
    upper_scale = SE3.scaling([0.675, 0.2, 0.675])
    upper_curve = Cone(upper_shift @ upper_scale, base_material, pattern)

    neck_shift = SE3.translation([0.0, 0.425, 0.0])
    neck_scale = SE3.scaling([0.2, 0.20, 0.2])

    neck_curve = Cylinder(neck_shift @ neck_scale, base_material, pattern_neck)
    shapes = [bottom_disk, concave_curve, middle_disk, upper_curve, neck_curve]
    return paz.graphics.Group(shapes, transform)


def Segment(segment_arg, scaling, num_segments, arc_radius, material, pattern):
    angle = (segment_arg / num_segments) * jp.pi - (jp.pi / 2.0)
    x = arc_radius * jp.cos(angle)
    z = arc_radius * jp.sin(angle)
    position = SE3.translation(jp.array([x, 0.0, z]))
    rotation_y = SE3.rotation_y(-angle + (jp.pi / 2.0))
    rotation_z = SE3.rotation_z(jp.pi / 2.0)
    rotation = rotation_y @ rotation_z
    return Cylinder(position @ rotation @ scaling, material, pattern)


def Arc(
    radius, num_segments, segment_length, segment_radius, material, pattern
):
    pattern = pattern._replace(type=paz.graphics.PLANAR_PATTERN)
    segment_size = SE3.scaling([segment_radius, segment_length, segment_radius])
    _segment = paz.lock(
        Segment, segment_size, num_segments, radius, material, pattern
    )
    segments = [_segment(arg) for arg in range(num_segments + 1)]
    standup = SE3.rotation_x(jp.pi / 2.0)
    rotate = SE3.rotation_z(jp.deg2rad(-30))
    return paz.graphics.Group(segments, rotate @ standup)


GREEN = (85 / 255, 181 / 255, 103 / 255)  # YlGnL
# GREEN = (239 / 255, 249 / 255, 179 / 255)  # YlGnD
# GREEN = (0.415, 0.749, 0.639)
GRAY = (0.662, 0.647, 0.576)
# GRAY = (0.75, 0.75, 0.75)
zero_material = Material(jp.zeros(3), 0.85, 0.1, 0.0, 100)

# GLOBE
squares = CheckeredImage(25, 20, 20, GREEN, WHITE)

pattern_scale = SE3.scaling(jp.full(3, 1.05))
pattern_transform = SE3.rotation_z(-jp.deg2rad(30)) @ pattern_scale
globe_pattern = SphericalPattern(squares, pattern_transform)
globe_material = zero_material
globe = Sphere(SE3.scaling(jp.full(3, 1.05)), globe_material, globe_pattern)


# ARC
zero_pattern = Pattern()
pattern_scale = SE3.scaling(jp.full(3, 40.0))
pattern_transform = SE3.rotation_z(-jp.deg2rad(30)) @ pattern_scale
checkboard_pattern_planar = SphericalPattern(squares, pattern_transform)
# gold_material = Material(jp.array([0.9, 0.7, 0.2]), 0.3, 0.8, 0.8, 200.0)
gold_material = Material(jp.array([0.9, 0.7, 0.2]))
arc = Arc(1.175, 50, 0.1, 0.06, zero_material, checkboard_pattern_planar)

# BASE
squares = CheckeredImage(100, 10, 10, GREEN, WHITE)
pattern_scale = SE3.scaling(jp.array([1.0, 1.0, 1.0]))

neck_pattern = SphericalPattern(squares, pattern_scale)
base_pattern = PlanarPattern(squares, SE3.scaling(jp.full(3, 4.1)))
base_material = zero_material

base = Base(
    SE3.translation([0.0, -1.7, 0.0]),
    base_material,
    base_pattern,
    neck_pattern,
)


# BUTTONS
button_material = Material(jp.ones(3), 0.85, 0.1, 0.0, 100)

top_button = Button(button_material, angle=60)
bottom_button = Button(button_material, angle=240)

# FLOOR
floor_image = CheckeredImage(50, 8, 8, WHITE, GRAY)
floor_pattern = PlanarPattern(floor_image, jp.eye(4))
floor_material = zero_material
floor_shift = SE3.translation(jp.array([0.0, -1.725, 0.0]))
floor_scale = SE3.scaling(jp.array([10.0, 0.05, 10.0]))
floor = Cube(floor_shift @ floor_scale, floor_material, floor_pattern)

wall_image = CheckeredImage(50, 8, 8, WHITE, GRAY)
wall_pattern = PlanarPattern(wall_image)
wall_material = zero_material

wall_scale = SE3.scaling(jp.array([10.0, 0.05, 10.0]))
wall_angle = SE3.rotation_x(jp.pi / 2.0)
wall_shift = SE3.translation(jp.array([1.25, 1.9, -3.5]))
wall_transform = wall_shift @ wall_angle @ wall_scale
wall = paz.graphics.Cube(wall_transform, wall_material, wall_pattern)

shapes = [base, globe, arc, top_button, bottom_button, floor, wall]
scene = paz.graphics.Scene(shapes)

camera_pose = SE3.view_transform(
    jp.array([1.1, 1.1, 4.0]),
    jp.array([0.0, -0.5, 0.0]),
    jp.array([0.0, 1.0, 0.0]),
)

lights = PointLight(jp.array([1.0, 1.0, 1.0]), jp.array([5.0, 4.0, 8.0]))

ratio = 8
H, W = 1024 // ratio, 1024 // ratio
y_FOV = jp.pi / 4.0
rays = paz.graphics.camera.build_rays((H, W), y_FOV, camera_pose)
render = jax.jit(
    paz.partial(
        paz.graphics.render,
        image_shape=(H, W),
        lights=lights,
        world_to_camera=camera_pose,
        rays=rays,
        mask=None,
        shadows=False,
    )
)
image, depth = render(scene=scene)
# image = paz.image.resize_opencv(paz.image.denormalize(image), (H // 4, W // 2))
# paz.image.write("optimized_image.png", image)


def get_patterns(scene):
    patterns = []
    for node in scene.nodes:
        if isinstance(node, paz.graphics.Shape):
            patterns.append(node.pattern.image)
        if isinstance(node, paz.graphics.Group):
            group_patterns = []
            for shape in node.shapes:
                group_patterns.append(shape.pattern.image)
            patterns.append(group_patterns)
    return patterns


def set_patterns(scene, pattern_images):
    for node_arg in range(len(scene.nodes)):
        pattern_image = pattern_images[node_arg]
        node = scene.nodes[node_arg]
        if isinstance(node, paz.graphics.Shape):
            new_pattern = scene.nodes[node_arg].pattern._replace(
                image=pattern_image
            )
            scene.nodes[node_arg] = scene.nodes[node_arg]._replace(
                pattern=new_pattern
            )
        if isinstance(node, paz.graphics.Group):
            for shape_arg in range(len(node.shapes)):
                new_pattern = (
                    scene.nodes[node_arg]
                    .shapes[shape_arg]
                    .pattern._replace(image=pattern_image[shape_arg])
                )
                scene.nodes[node_arg].shapes[shape_arg] = (
                    scene.nodes[node_arg]
                    .shapes[shape_arg]
                    ._replace(pattern=new_pattern)
                )
    return scene


def render_patterns(pattern_images, scene, render):
    scene = set_patterns(scene, pattern_images)
    image, _ = render(scene=scene)
    return image


def Loss(pattern_images, scene, render, true_image):
    pred_image = render_patterns(pattern_images, scene, render)
    return jp.mean(jp.abs(true_image - pred_image))


true_image = paz.image.load("globe.png")
true_image = paz.to_jax(paz.image.resize_opencv(true_image, (H, W)))
true_image = paz.image.normalize(true_image)
compute_loss = paz.lock(Loss, scene, render, true_image)
compute_gradients = jax.jit(jax.value_and_grad(compute_loss))
filepath = paz.logger.make_directory("experiments")
optimizer = optax.adam(1e-3)
max_num_steps = 20_000
parameters = get_patterns(scene)
optimizer_state = optimizer.init(parameters)
for step_arg in range(max_num_steps):
    loss, gradients = compute_gradients(parameters)
    delta, state = optimizer.update(gradients, optimizer_state, parameters)
    parameters = optax.apply_updates(parameters, delta)
    print(f"LOSS {loss}")
    if step_arg % 10 == 0 or step_arg == 99:
        image = render_patterns(parameters, scene, render)
        image = paz.image.denormalize(image)
        paz.image.write(
            os.path.join(filepath, f"globe_{step_arg:03d}.png"), image
        )
    if loss < 0.005:
        print("Converged")
        break
