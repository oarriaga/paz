from functools import partial
import jax.numpy as jp
from paz import SE3
import jax
import paz


def Ball(material, stretch_factor=1.3):
    scale = SE3.scaling(jp.array([1.0, stretch_factor, 1.0]))
    shift = SE3.translation(jp.array([0.0, stretch_factor, 0.0]))
    return paz.graphics.Group([paz.graphics.Sphere(shift @ scale, material)])


def Knot(material, height=0.15, width=0.1, body_bottom_y=0.0):
    angle = SE3.rotation_x(jp.pi)
    scale = SE3.scaling(jp.array([width, height / 2.0, width]))
    pos_y = body_bottom_y - (height / 2.0)
    shift = SE3.translation(jp.array([0.0, pos_y, 0.0]))
    transform = shift @ angle @ scale
    return paz.graphics.Group([paz.graphics.Cone(transform, material)])


def compute_orthonormal_basis(up_axis):
    right_axis = jp.array([1.0, 0.0, 0.0])
    forward_axis = paz.algebra.normalize(jp.cross(right_axis, up_axis))
    return jp.stack([right_axis, up_axis, forward_axis], axis=1)


def build_segment_pose(start_point, final_point, thickness):
    up_axis, length = paz.algebra.normalize_and_norm(final_point - start_point)
    rotation_matrix = compute_orthonormal_basis(up_axis)
    pose = SE3.to_affine_matrix(rotation_matrix, start_point)
    scale = SE3.scaling(jp.array([thickness, length / 2.0, thickness]))
    shift = SE3.translation(jp.array([0.0, length / 2.0, 0.0]))
    return pose @ shift @ scale


def String(key, material, start_position, thickness, num_parts, size, noise):
    segments, world_down = [], jp.array([0.0, -1.0, 0.0])
    for k in jax.random.split(key, num_parts):
        _noise = jax.random.uniform(k, (3,), float, -noise, noise).at[1].set(0)
        y_axis = paz.algebra.normalize(world_down + _noise)
        final_position = start_position + (y_axis * size)
        pose = build_segment_pose(start_position, final_position, thickness)
        segments.append(paz.graphics.Cylinder(pose, material))
        start_position = final_position
    return paz.graphics.Group(segments)


def Balloon(key, pose, sphere_material, string_material):
    ball = Ball(sphere_material)
    knot = Knot(sphere_material, body_bottom_y=0.05)
    string = String(key, string_material, jp.zeros(3), 0.015, 14, 0.28, 0.05)
    return paz.graphics.Group([string, ball, knot], pose)


keys = jax.random.split(jax.random.PRNGKey(777), 6)
camera_origin = jp.array([0.0, 2.5, 5.0])
camera_target = jp.array([-0.12, 2.25, 0.0])
world_up = jp.array([0.0, 1.0, 0.0])
world_to_camera = SE3.view_transform(camera_origin, camera_target, world_up)
H, W, y_FOV = 1024 // 2, 1024 // 2, jp.pi / 3.0
rays = paz.graphics.camera.build_rays((H, W), y_FOV, world_to_camera)
lights = paz.graphics.PointLight(jp.full(3, 0.8), jp.array([5.0, 8.0, 5.0]))
render_kwargs = {"lights": lights, "mask": None, "shadows": False}
render_args = ((H, W), world_to_camera, rays)
render = jax.jit(partial(paz.graphics.render, *render_args, **render_kwargs))

color_0 = jp.array([0.90, 0.15, 0.25])  # RED
color_1 = jp.array([0.20, 0.60, 0.85])  # BLUE
color_2 = jp.array([0.50, 0.25, 0.60])  # PURPLE
color_3 = jp.array([1.00, 0.55, 0.15])  # ORGANGE
color_4 = jp.array([0.95, 0.75, 0.10])  # YELLOW
color_5 = jp.array([0.45, 0.75, 0.35])  # GREEN
string_color = jp.array([0.85, 0.75, 0.50])
colors = [color_0, color_1, color_2, color_3, color_4, color_5]

pose_0 = SE3.translation(jp.array([0.90, 2.6, -0.5])) @ SE3.rotation_z(-0.2)
pose_1 = SE3.translation(jp.array([-0.5, 2.4, -0.5])) @ SE3.rotation_z(0.20)
pose_2 = SE3.translation(jp.array([-1.3, 1.8, 0.20])) @ SE3.rotation_z(0.30)
pose_3 = SE3.translation(jp.array([-0.7, 1.2, 0.80])) @ SE3.rotation_z(0.20)
pose_4 = SE3.translation(jp.array([0.80, 1.2, 0.80])) @ SE3.rotation_z(-0.25)
pose_5 = SE3.translation(jp.array([0.00, 0.2, -0.2]))
poses = [pose_0, pose_1, pose_2, pose_3, pose_4, pose_5]

ambient, diffuse, specular = 0.4, 0.8, (0.7, 2.0)
phong_components = {
    "ambient": (ambient, 0.0, 0.0, 0.0),
    "diffuse": (0.0, diffuse, 0.0, 0.0),
    "specular": (0.0, 0.0, *specular),
    "total": (ambient, diffuse, *specular),
}

for image_name, phong_args in phong_components.items():
    materials = [paz.graphics.Material(color, *phong_args) for color in colors]
    string_material = paz.graphics.Material(string_color, *phong_args)
    iterator = zip(keys, poses, materials)
    shapes = [Balloon(*args, string_material) for args in iterator]
    image, depth = render(scene=paz.graphics.Scene(shapes))
    image = paz.image.denormalize(image)
    image = paz.image.resize_opencv(image, (H // 2, W // 2))
    paz.image.write(f"{image_name}.png", image)
