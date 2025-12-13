import os
import jax.numpy as jp
import jax
import paz

# --- Setup ---
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
from jax.experimental.compilation_cache.compilation_cache import set_cache_dir

set_cache_dir(paz.logger.make_directory("cache"))

# --- Colors ---
COLOR_SKIN = jp.array([0.9, 0.85, 0.75])
COLOR_EYES = jp.array([0.1, 0.1, 0.1])

# Rainbow/Vibrant colors for the shell rings (to test patterns clearly)
COLORS_SHELL = [
    [0.05, 0.20, 0.30],  # 1. Dark Petrol / Navy
    [0.25, 0.55, 0.60],  # 2. Teal / Medium Blue-Green
    [0.60, 0.65, 0.75],  # 3. Soft Grey-Blue
    [0.80, 0.90, 0.85],  # 4. Pale Mint / Light Aqua
    [0.95, 0.90, 0.75],  # 5. Cream / Pale Yellow
    [0.05, 0.20, 0.30],  # 1. Dark Petrol / Navy
    [0.25, 0.55, 0.60],  # 2. Teal / Medium Blue-Green
    [0.60, 0.65, 0.75],  # 3. Soft Grey-Blue
]

COLORS_SHELL = [
    [0.05, 0.20, 0.30],  # 1. Dark Petrol / Navy
    [0.25, 0.55, 0.60],  # 2. Teal / Medium Blue-Green
    [0.25, 0.55, 0.60],  # 2. Teal / Medium Blue-Green
    [0.60, 0.65, 0.75],  # 3. Soft Grey-Blue
    [0.60, 0.65, 0.75],  # 3. Soft Grey-Blue
    [0.68, 0.78, 0.73],  # 4. Standard mint
    [0.68, 0.78, 0.73],  # 4. Standard mint  # 4. Standard mint
    [0.76, 0.72, 0.60],  # 5. Cream yellow
    # [0.76, 0.72, 0.60],  # 5. Cream yellow
    # [0.85, 0.81, 0.68],
    # [0.05, 0.20, 0.30],  # 1. Dark Petrol / Navy
    # [0.76, 0.72, 0.60]   # 5. Cream yellow
]


# --- Pattern Generator ---
def build_shell_pattern(colors, radius_thresholds):
    """
    Generates a spherical pattern with 8 distinct concentric rings.
    """
    if len(colors) != 8:
        raise ValueError("The input colors must contain exactly 8 elements.")
    if len(radius_thresholds) != 8:
        raise ValueError(
            "The input radius_thresholds must contain exactly 8 elements."
        )

    resolution = 512
    vertical_indices, horizontal_indices = jp.indices((resolution, resolution))

    # Center and normalize coordinates to -1..1
    center_coordinate = resolution / 2.0
    normalized_y = (vertical_indices - center_coordinate) / center_coordinate
    normalized_x = (horizontal_indices - center_coordinate) / center_coordinate

    # Calculate distance from center
    radius_from_center = jp.sqrt(normalized_x**2 + normalized_y**2)

    final_image = jp.zeros((resolution, resolution, 3))
    previous_radius_threshold = 0.0

    for ring_index in range(8):
        current_radius_threshold = radius_thresholds[ring_index]

        # Create mask for the current ring
        ring_mask = (radius_from_center >= previous_radius_threshold) & (
            radius_from_center < current_radius_threshold
        )

        color_vector = jp.array(colors[ring_index])
        ring_contribution = jp.expand_dims(ring_mask, axis=-1) * color_vector
        final_image = final_image + ring_contribution

        previous_radius_threshold = current_radius_threshold

    # Rotate so the 'pole' of the pattern (the center of rings) points UP (Y-axis)
    # The default Sphere UVs usually map the pole to Y.
    # If the rings appear on the side, we adjust this rotation.
    pattern_transform = paz.SE3.rotation_x(-jp.pi / 2.0)
    scale = paz.SE3.scaling(0.5 * jp.array([1.0, 2.0, 1.0]))

    return paz.graphics.SphericalPattern(final_image, pattern_transform @ scale)


# --- Materials ---
yarn_material = paz.graphics.Material(
    color=COLOR_SKIN, ambient=0.4, diffuse=0.8, specular=0.0, shininess=0.0
)

eye_material = paz.graphics.Material(
    color=COLOR_EYES, ambient=0.1, diffuse=0.1, specular=0.9, shininess=100.0
)

shell_base_mat = paz.graphics.Material(jp.zeros(3), 0.6, 0.8, 0.0, 0.0)

# Define custom ring sizes (Thresholds from 0.0 to ~1.0)
# Smaller numbers = tighter rings in the center.
# ring_radii = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
ring_radii = [0.15, 0.28, 0.32, 0.44, 0.50, 0.60, 0.88, 1.6]
ring_radii = [0.15, 0.28, 0.40, 0.52, 0.64, 0.76, 0.88, 1.7]
ring_radii = [x / 2.0 for x in ring_radii]
shell_pattern = build_shell_pattern(COLORS_SHELL, ring_radii)


# --- Geometry Construction ---
def build_turtle():
    shapes = []

    # 1. Shell (The Body)
    # FIX: Use UNIFORM scaling for X and Z (1.75) to keep rings circular.
    # We still flatten it in Y (0.85).
    shell_transform = paz.SE3.translation(
        jp.array([0.0, 0.5, 0.0])
    ) @ paz.SE3.scaling(jp.array([1.75, 0.85, 1.75]))
    shell = paz.graphics.Sphere(shell_transform, shell_base_mat, shell_pattern)
    shapes.append(shell)

    # 2. Head
    head_transform = paz.SE3.translation(
        jp.array([0.0, 0.6, 1.95])
    ) @ paz.SE3.scaling(jp.full(3, 0.75))
    head = paz.graphics.Sphere(head_transform, yarn_material)
    shapes.append(head)

    # 3. Eyes
    eye_offset_x = 0.35
    eye_offset_z = 2.50
    eye_height = 0.9

    shapes.append(
        paz.graphics.Sphere(
            paz.SE3.translation(
                jp.array([-eye_offset_x, eye_height, eye_offset_z])
            )
            @ paz.SE3.scaling(jp.full(3, 0.08)),
            eye_material,
        )
    )
    shapes.append(
        paz.graphics.Sphere(
            paz.SE3.translation(
                jp.array([eye_offset_x, eye_height, eye_offset_z])
            )
            @ paz.SE3.scaling(jp.full(3, 0.08)),
            eye_material,
        )
    )

    # 4. Flippers (Legs)
    def make_flipper(x, z, angle_y):
        base_tf = paz.SE3.translation(jp.array([x, 0.1, z]))
        # Rotation Y controls direction, Rotation X tilts it down
        rot_tf = paz.SE3.rotation_y(angle_y) @ paz.SE3.rotation_x(
            jp.deg2rad(-20)
        )
        scale_tf = paz.SE3.scaling(jp.array([0.65, 0.2, 1.0]))

        return paz.graphics.Sphere(base_tf @ rot_tf @ scale_tf, yarn_material)

    # Adjusted positions for the slightly wider/rounder shell
    shapes.append(make_flipper(-1.7, 1.1, jp.deg2rad(35)))  # Front Left
    shapes.append(make_flipper(1.7, 1.1, jp.deg2rad(-35)))  # Front Right
    shapes.append(make_flipper(-1.7, -1.2, jp.deg2rad(145)))  # Back Left
    shapes.append(make_flipper(1.7, -1.2, jp.deg2rad(-145)))  # Back Right

    # 5. Tail
    tail_transform = (
        paz.SE3.translation(jp.array([0.0, 0.2, -1.9]))
        @ paz.SE3.rotation_x(jp.deg2rad(-90))
        @ paz.SE3.scaling(jp.array([0.2, 0.4, 0.2]))
    )
    tail = paz.graphics.Cone(tail_transform, yarn_material)
    shapes.append(tail)

    return shapes


# --- Floor ---
floor_mat = paz.graphics.Material(
    jp.array([0.6, 0.4, 0.25]), 0.5, 0.5, 0.0, 10.0
)
floor = paz.graphics.Plane(
    paz.SE3.translation(jp.array([0.0, 0.0, 0.0])), floor_mat
)

# --- Assemble Scene ---
turtle_parts = build_turtle()
scene = paz.graphics.Scene(turtle_parts)

# --- Camera & Render ---
camera_pose = paz.SE3.view_transform(
    0.7 * jp.array([3.0, 6.0, 4.0]),
    jp.array([0.0, 0.0, 0.0]),
    jp.array([0.0, 1.0, 0.0]),
)

lights = [
    paz.graphics.PointLight(jp.ones(3) * 0.8, jp.array([5.0, 8.0, 5.0])),
    # paz.graphics.PointLight(jp.ones(3) * 0.3, jp.array([-5.0, 4.0, -5.0])),
]

H, W = 1024 // 2, 1024 // 2
y_FOV = jp.pi / 4.0
rays = paz.graphics.camera.build_rays((H, W), y_FOV, camera_pose)

render = jax.jit(
    paz.partial(
        paz.graphics.render,
        image_shape=(H, W),
        world_to_camera=camera_pose,
        rays=rays,
        lights=lights,
        shadows=True,
    )
)

print("Rendering Turtle with uniform shell...")
image, depth = render(scene=scene, mask=None)
image = paz.image.denormalize(image)
paz.image.show(image)
paz.image.write("turtle_uniform.png", image)
