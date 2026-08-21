import jax
import jax.numpy as jp
import paz

BLUE = jp.array([0.324, 0.692, 0.863])  # MyBlue
GREEN = jp.array([154 / 255, 213 / 255, 135 / 255])  # YlGnI
YELLOW = jp.array([1.0, 0.65, 0.0])  # MyYellow
RED = jp.array([171 / 255, 62 / 255, 66 / 255])  # cherry

H, W = 280, 960
Y_FOV = 0.37
SCALE = 4
BOUNCES = 4
FLOOR_REFLECTIVE = 0.3
SHAPE_REFLECTIVE = 0.12


def main():
    path = paz.logger.make_directory("realistic_primitives")
    scene = build_scene()
    light = build_area_light()
    render = build_render(light, shadows=True, num_bounces=BOUNCES, scale=SCALE)
    image, _ = render(scene=scene)
    save_image(path, "realistic_primitives.png", image, SCALE)


def build_render(lights, shadows, num_bounces, scale):
    shape = (H * scale, W * scale)
    args = shape, Y_FOV, CAMERA_POSE
    kwargs = dict(lights=lights, tiles=(1, 1), chunk_size=2**10)
    kwargs.update(shadows=shadows, num_bounces=num_bounces, mask=None)
    return jax.jit(paz.partial(paz.graphics.render, *args, **kwargs))


def save_image(path, name, image, scale):
    image = paz.image.denormalize(image)
    if scale > 1:
        image = paz.image.resize(image, (H, W), "bilinear")
    paz.image.write(f"{path}/{name}", image)


def build_scene():
    floor = build_floor()
    sphere = build_sphere(-4.5)
    cylinder = build_cylinder(-1.5)
    cone = build_cone(1.5)
    cube = build_cube(4.5)
    return paz.graphics.Scene([floor, sphere, cylinder, cone, cube])


def build_sphere(x):
    return paz.graphics.Sphere(rest_pose(x, 1.15), shape_material(GREEN))


def build_cylinder(x):
    return paz.graphics.Cylinder(rest_pose(x, 1.0), shape_material(BLUE))


def build_cone(x):
    scale = paz.SE3.scaling(jp.array([1.25, 1.12, 1.25]))
    pose = paz.SE3.translation(jp.array([x, 1.12, 0.0])) @ scale
    return paz.graphics.Cone(pose, shape_material(YELLOW))


def build_cube(x):
    return paz.graphics.Cube(rest_pose(x, 1.0), shape_material(RED))


def build_floor():
    return paz.graphics.Plane(material=floor_material())


def rest_pose(x, size):
    shift = paz.SE3.translation(jp.array([x, size, 0.0]))
    return shift @ paz.SE3.scaling(jp.full(3, size))


def shape_material(color):
    args = color, 0.14, 0.75, 0.6, 200.0, SHAPE_REFLECTIVE
    return paz.graphics.Material(*args)


def floor_material():
    args = jp.ones(3), 0.025, 0.67, 0.0, 100.0, FLOOR_REFLECTIVE
    return paz.graphics.Material(*args)


def build_area_light():
    intensity = jp.array([1.5, 1.5, 1.5])
    corner = jp.array([4.75, 9.0, 4.75])
    edge1 = jp.array([2.5, 0.0, 0.0])
    edge2 = jp.array([0.0, 0.0, 2.5])
    args = intensity, corner, edge1, edge2, 10, 10, jax.random.key(0)
    return paz.graphics.AreaLight(*args)


def build_camera_pose():
    position = jp.array([-1.2, 5.6, 8.6])
    target = jp.array([-0.54, 0.85, -0.04])
    up = jp.array([0.0, 1.0, 0.0])
    return paz.SE3.view_transform(position, target, up)


CAMERA_POSE = build_camera_pose()


if __name__ == "__main__":
    main()
