import jax
import jax.numpy as jp
import paz
import paz.graphics.renderer as paz_renderer

soft_occlusion = paz_renderer.compute_soft_occlusion
paz_renderer.compute_soft_occlusion = paz.partial(soft_occlusion, slope=1.0)

H, W = 480, 640


def main():
    path = paz.logger.make_directory("phong_realism_levels")
    ambient_scene = build_ambient_scene()
    phong_scene = build_phong_scene(0.0)
    reflective_scene = build_phong_scene(0.3)
    point_light = build_point_light()
    area_light = build_area_light()
    stage_specs = [
        ("01_ambient.png", ambient_scene, point_light, False, 1, 1),
        ("02_phong.png", phong_scene, point_light, False, 1, 1),
        ("03_shadows.png", phong_scene, point_light, True, 1, 1),
        ("04_soft_shadows.png", phong_scene, area_light, True, 1, 1),
        ("05_ssaa_reflection.png", reflective_scene, area_light, True, 4, 4),
    ]
    for spec in stage_specs:
        render_level(path, *spec)


def render_level(path, name, scene, lights, shadows, num_bounces, scale):
    render = build_render(lights, shadows, num_bounces, scale)
    image, _ = render(scene=scene)
    save_image(path, name, image, scale)


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


def build_ambient_scene():
    return build_scene(*build_ambient_materials())


def build_phong_scene(reflective):
    return build_scene(*build_materials(reflective))


def build_scene(floor_material, red_material, blue_material):
    floor = paz.graphics.Plane(material=floor_material)
    red_sphere = place_sphere(red_material, -0.5, 0.5, 0.5)
    blue_sphere = place_sphere(blue_material, 0.25, 0.33, 0.33)
    return paz.graphics.Scene([floor, red_sphere, blue_sphere])


def build_ambient_materials():
    floor, red, blue = build_materials(0.0)
    floor = to_ambient_only(floor)._replace(ambient=0.6)
    red = to_ambient_only(red)._replace(ambient=0.14)
    blue = to_ambient_only(blue)._replace(ambient=0.14)
    return floor, red, blue


def build_materials(reflective):
    floor_args = jp.ones(3), 0.025, 0.67, 0.0, 100.0, reflective
    red_args = jp.array([1.0, 0.0, 0.0]), 0.1, 0.6, 0.8, 100.0, reflective
    blue_args = jp.array([0.5, 0.5, 1.0]), 0.1, 0.6, 0.8, 100.0, reflective
    floor = paz.graphics.Material(*floor_args)
    red = paz.graphics.Material(*red_args)
    blue = paz.graphics.Material(*blue_args)
    return floor, red, blue


def to_ambient_only(material):
    return material._replace(diffuse=0.0, specular=0.0, reflective=0.0)


def build_point_light():
    intensity = jp.array([1.5, 1.5, 1.5])
    position = jp.array([0.0, 3.0, 4.0])
    return paz.graphics.PointLight(intensity, position)


def build_area_light():
    intensity = jp.array([1.5, 1.5, 1.5])
    corner = jp.array([-1.0, 2.0, 4.0])
    edge1 = jp.array([2.0, 0.0, 0.0])
    edge2 = jp.array([0.0, 2.0, 0.0])
    args = intensity, corner, edge1, edge2, 10, 10, jax.random.key(0)
    return paz.graphics.AreaLight(*args)


def build_camera_pose():
    position = jp.array([2.3, 1.5, 2.5])
    target = jp.array([0.0, 0.5, 0.0])
    up = jp.array([0.0, 1.0, 0.0])
    return paz.SE3.view_transform(position, target, up)


def place_sphere(material, x, y, radius):
    shift = paz.SE3.translation(jp.array([x, y, 0.0]))
    scale = paz.SE3.scaling(jp.full(3, radius))
    return paz.graphics.Sphere(shift @ scale, material)


def compute_y_FOV(H, W):
    horizontal_FOV = jp.pi / 6.0
    return 2 * jp.arctan(jp.tan(horizontal_FOV / 2) * H / W)


CAMERA_POSE = build_camera_pose()
Y_FOV = compute_y_FOV(H, W)


if __name__ == "__main__":
    main()
