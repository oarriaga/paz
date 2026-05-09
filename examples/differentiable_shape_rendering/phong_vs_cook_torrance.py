import jax
import jax.numpy as jp
import numpy as np
import paz


def build_cook_torrance_dielectric(color, roughness):
    return paz.graphics.CookTorranceMaterial(
        color=jp.array(color),
        ambient=0.1,
        base_reflectance=0.04,
        roughness=roughness,
        metallic=0.0,
    )


def build_cook_torrance_metal(color, roughness):
    return paz.graphics.CookTorranceMaterial(
        color=jp.array(color),
        ambient=0.05,
        base_reflectance=0.04,
        roughness=roughness,
        metallic=1.0,
    )


def build_phong(color, ambient, diffuse, specular, shininess):
    return paz.graphics.Material(
        color=jp.array(color),
        ambient=ambient,
        diffuse=diffuse,
        specular=specular,
        shininess=shininess,
    )


def place_sphere(material, x_position, radius=0.6):
    shift = paz.SE3.translation(jp.array([x_position, radius, 0.0]))
    scale = paz.SE3.scaling(jp.full(3, radius))
    return paz.graphics.Sphere(shift @ scale, material)


def build_scene(floor_material, sphere_materials):
    floor = paz.graphics.Plane(material=floor_material)
    spheres = [
        place_sphere(sphere_materials[0], x_position=-1.5),
        place_sphere(sphere_materials[1], x_position=0.0),
        place_sphere(sphere_materials[2], x_position=1.5),
    ]
    return paz.graphics.Scene([floor] + spheres)


red = (0.85, 0.18, 0.18)
gold = (0.94, 0.78, 0.34)
blue = (0.2, 0.4, 0.8)
gray = (0.6, 0.6, 0.6)

cook_torrance_floor = build_cook_torrance_dielectric(gray, roughness=0.8)
cook_torrance_spheres = [
    build_cook_torrance_dielectric(red, roughness=0.35),
    build_cook_torrance_metal(gold, roughness=0.2),
    build_cook_torrance_metal(blue, roughness=0.45),
]

phong_floor = build_phong(gray, 0.05, 0.7, 0.05, 10.0)
phong_spheres = [
    build_phong(red, 0.08, 0.85, 0.25, 50.0),
    build_phong(gold, 0.02, 0.15, 1.0, 50.0),
    build_phong(blue, 0.02, 0.18, 0.9, 12.0),
]

phong_scene = build_scene(phong_floor, phong_spheres)
cook_torrance_scene = build_scene(cook_torrance_floor, cook_torrance_spheres)

camera_pose = paz.SE3.view_transform(
    jp.array([0.0, 2.0, 6.0]),
    jp.array([0.0, 0.5, 0.0]),
    jp.array([0.0, 1.0, 0.0]),
)

cook_torrance_lights = [
    paz.graphics.PointLight(jp.full(3, 1.2), jp.array([3.0, 5.0, 4.0])),
    paz.graphics.PointLight(jp.full(3, 0.8), jp.array([-3.0, 4.0, 2.0])),
]

phong_lights = [
    paz.graphics.PointLight(jp.full(3, 0.65), jp.array([3.0, 5.0, 4.0])),
    paz.graphics.PointLight(jp.full(3, 0.42), jp.array([-3.0, 4.0, 2.0])),
]

H, W = 2**10, 2**10
y_FOV = jp.pi / 4.0
render_args = (H, W), y_FOV, camera_pose


def build_render(lights):
    render_kwargs = dict(lights=lights, tiles=(1, 1), chunk_size=512)
    function = paz.partial(
        paz.graphics.render,
        *render_args,
        **render_kwargs,
        shadows=True,
    )
    return jax.jit(function)


phong_render = build_render(phong_lights)
cook_torrance_render = build_render(cook_torrance_lights)


def render_image(render_fn, scene):
    image, _ = render_fn(scene=scene, mask=None)
    return paz.image.resize(image, (H // 2, W // 2), "bilinear")


phong_image = render_image(phong_render, phong_scene)
cook_torrance_image = render_image(cook_torrance_render, cook_torrance_scene)

paz.image.write("phong.png", paz.image.denormalize(phong_image))
paz.image.write("cook_torrance.png", paz.image.denormalize(cook_torrance_image))

images = np.array([paz.to_numpy(phong_image), paz.to_numpy(cook_torrance_image)])
mosaic = paz.draw.mosaic(paz.to_jax(images), (1, 2), 8, 1)
paz.image.write("phong_vs_cook_torrance.png", paz.image.denormalize(mosaic))
