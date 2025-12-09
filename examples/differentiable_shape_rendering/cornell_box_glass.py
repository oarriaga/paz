# shadows (262144, 3)
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
from jax.experimental.compilation_cache.compilation_cache import set_cache_dir
import jax.numpy as jp
import jax
import paz

set_cache_dir(paz.logger.make_directory("cache"))

# --- Configuration ---
# Set to True for a floor with a pattern of squares.
# Set to False for a floor with the classical white material.
USE_CHECKERED_FLOOR = False

# --- Colors & Patterns ---
GREEN_CHECKER = (85 / 255, 181 / 255, 103 / 255)
WHITE = (1.0, 1.0, 1.0)


def CheckeredImage(
    box_size=50, rows=8, cols=8, color_A=GREEN_CHECKER, color_B=WHITE
):
    checkered = jp.indices((rows, cols)).sum(axis=0) % 2
    image_channels = []
    for channel_arg in range(3):
        checkered_channel = jp.kron(checkered, jp.ones((box_size, box_size)))
        checkered_color_A = color_A[channel_arg] * checkered_channel
        checkered_color_B = color_B[channel_arg] * (1 - checkered_channel)
        checkered_channel = checkered_color_A + checkered_color_B
        image_channels.append(jp.expand_dims(checkered_channel, axis=-1))
    return jp.concatenate(image_channels, axis=-1)


# --- Materials ---

# Glass Material (for all primitives)
# Updated to use 'transparency' instead of 'refractive'
glass_material = paz.graphics.Material(
    color=jp.array([0.9, 0.9, 1.0]),
    ambient=0.1,
    diffuse=0.1,
    specular=0.9,
    shininess=200.0,
    transparency=0.9,
    refractive_index=1.5,
)

mirror_material = paz.graphics.Material(
    color=jp.array([0.0, 0.0, 0.0]),
    ambient=0.0,
    diffuse=0.8,
    specular=1.0,
    shininess=300.0,
    reflective=1.0,
    transparency=0.0,
    refractive_index=1.0,
)

clear_glass_material = paz.graphics.Material(
    color=jp.array([0.0, 0.0, 0.0]),  # Black: Don't add surface pigment
    ambient=0.0,  # 0.0: No background glow
    diffuse=0.0,  # 0.0: No "chalky" scattering
    specular=1.0,  # 1.0: Sharp, bright highlights from lights
    shininess=300.0,  # 300+: Very tight, small highlights
    reflective=1.0,  # 1.0: Let the Fresnel effect control reflection
    transparency=1.0,  # 1.0: Fully transparent
    refractive_index=1.0,  # 1.5: Glass (causes the bending/lensing)
)

gold_material = paz.graphics.Material(
    color=jp.array([1.0, 0.77, 0.34]),
    ambient=0.3,
    diffuse=0.3,
    specular=1.0,
    shininess=200.0,
    reflective=0.7,
    transparency=0.0,
    refractive_index=1.0,
)


# shape_material = mirror_material
# shape_material = glass_material
shape_material = gold_material

# Cornell Box Materials
white_wall_material = paz.graphics.Material(
    color=0.8 * jp.array([0.73, 0.73, 0.73]),
    ambient=0.3,
    diffuse=0.9,
    specular=0.0,
    shininess=0.0,
)

red_wall_material = paz.graphics.Material(
    color=jp.array([0.65, 0.05, 0.05]),
    ambient=0.1,
    diffuse=0.9,
    specular=0.0,
    shininess=10.0,
)

green_wall_material = paz.graphics.Material(
    color=jp.array([0.12, 0.45, 0.15]),
    ambient=0.1,
    diffuse=0.9,
    specular=0.0,
    shininess=10.0,
)

# Default material for checkered pattern (neutral)
default_pattern_material = paz.graphics.Material(
    jp.zeros(3), 0.3, 0.1, 0.0, 100
)


# --- Scene Construction ---

# Box Dimensions
BOX_SIZE = 10.0
HALF_SIZE = BOX_SIZE / 2.0
# Floor at Y=0, Ceiling at Y=10. Center X=0, Center Z approx 0 or -5 depending on depth.
# Let's align floor to Y=0.
# Left X = -5, Right X = 5.
# Back Z = -5.

# 1. Floor
if USE_CHECKERED_FLOOR:
    checkered_image = CheckeredImage()
    planar_pattern = paz.graphics.PlanarPattern(checkered_image)
    floor = paz.graphics.Plane(
        paz.SE3.identity(), default_pattern_material, planar_pattern
    )
else:
    floor = paz.graphics.Plane(paz.SE3.identity(), white_wall_material)

# 2. Ceiling (Y=10, Normal pointing down)
ceiling = paz.graphics.Plane(
    paz.SE3.translation(jp.array([0.0, BOX_SIZE, 0.0]))
    @ paz.SE3.rotation_x(jp.pi),
    white_wall_material,
)

# 3. Back Wall (Z=-5, Normal pointing forward/Z+)
back_wall = paz.graphics.Plane(
    paz.SE3.translation(jp.array([0.0, HALF_SIZE, -HALF_SIZE]))
    @ paz.SE3.rotation_x(jp.pi / 2),
    white_wall_material,
)

# 4. Left Wall (X=-5, Red, Normal pointing Right/X+)
left_wall = paz.graphics.Plane(
    paz.SE3.translation(jp.array([-HALF_SIZE, HALF_SIZE, 0.0]))
    @ paz.SE3.rotation_z(-jp.pi / 2),
    red_wall_material,
)

# 5. Right Wall (X=5, Green, Normal pointing Left/X-)
right_wall = paz.graphics.Plane(
    paz.SE3.translation(jp.array([HALF_SIZE, HALF_SIZE, 0.0]))
    @ paz.SE3.rotation_z(jp.pi / 2),
    green_wall_material,
)

# Primitives (Scaled to fit in the box)
# Sphere
shape_sphere = paz.graphics.Sphere(
    paz.SE3.translation(jp.array([-2.0, 1.5, -2.0]))
    @ paz.SE3.scaling(jp.full(3, 1.5)),
    shape_material,
)

# Cube
shape_cube = paz.graphics.Cube(
    paz.SE3.translation(jp.array([2.0, 1.5, -2.0]))
    @ paz.SE3.rotation_y(jp.pi / 6)
    @ paz.SE3.scaling(jp.full(3, 1.5)),
    shape_material,
)

# Cylinder
shape_cylinder = paz.graphics.Cylinder(
    paz.SE3.translation(jp.array([-2.0, 1.0, 2.0]))
    @ paz.SE3.scaling(jp.full(3, 1.0)),
    shape_material,
)

# Cone
shape_cone = paz.graphics.Cone(
    # paz.SE3.translation(jp.array([2.0, 1.50, 2.0]))
    paz.SE3.translation(jp.array([2.0, 1.51, 2.0]))
    @ paz.SE3.scaling(jp.full(3, 1.5)),
    shape_material,
)

scene_objects = [
    floor,
    ceiling,
    back_wall,
    left_wall,
    right_wall,
    shape_sphere,
    shape_cube,
    shape_cylinder,
    shape_cone,
]

scene = paz.graphics.Scene(scene_objects)

# --- Camera & Lighting ---

# Camera positioned outside the open front, looking in.
camera_pose = paz.SE3.view_transform(
    jp.array([0.0, HALF_SIZE, 13.0]),  # Position
    jp.array([0.0, HALF_SIZE, 0.0]),  # Target
    jp.array([0.0, 1.0, 0.0]),  # Up
)

# Lights
# One main light near the ceiling
lights = [
    paz.graphics.PointLight(
        jp.ones(3) * 0.8, jp.array([0.0, BOX_SIZE - 1.0, 0.0])
    ),
    # Optional fill light
    # paz.graphics.PointLight(jp.ones(3) * 0.2, jp.array([0.0, HALF_SIZE, 10.0])),
]

H, W = 1024 // 2, 1024 // 2
y_FOV = jp.pi / 3.5
rays = paz.graphics.camera.build_rays((H, W), y_FOV, camera_pose)

# --- Rendering ---

render = jax.jit(
    paz.partial(
        paz.graphics.render,
        image_shape=(H, W),
        world_to_camera=camera_pose,
        rays=rays,
        lights=lights,
        shadows=True,
        # shadows=False,
        num_bounces=5,
    )
)

print(
    f"Rendering Cornell Box scene with {'Checkered' if USE_CHECKERED_FLOOR else 'White'} floor..."
)
image, depth = render(scene=scene, mask=None)
image = paz.image.denormalize(image)
paz.image.show(image)
paz.image.write("cornell_box_mirror.png", image)

# Interactive Viewer
paz.graphics.viewer(scene, camera_pose, True, lights)
