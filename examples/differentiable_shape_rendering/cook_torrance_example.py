import jax
import jax.numpy as jp
import paz


def build_dielectric(color, roughness):
    return paz.graphics.CookTorranceMaterial(
        color=jp.array(color),
        ambient=0.1,
        base_reflectance=0.04,
        roughness=roughness,
        metallic=0.0,
    )


def build_metal(color, roughness):
    return paz.graphics.CookTorranceMaterial(
        color=jp.array(color),
        ambient=0.05,
        base_reflectance=0.04,
        roughness=roughness,
        metallic=1.0,
    )


def place_sphere(material, x_position, radius=0.6):
    shift = paz.SE3.translation(jp.array([x_position, radius, 0.0]))
    scale = paz.SE3.scaling(jp.full(3, radius))
    return paz.graphics.Sphere(shift @ scale, material)


floor_material = build_dielectric((0.6, 0.6, 0.6), roughness=0.8)
floor = paz.graphics.Plane(material=floor_material)

red_dielectric = build_dielectric((0.85, 0.18, 0.18), roughness=0.35)
gold_metal = build_metal((0.94, 0.78, 0.34), roughness=0.2)
blue_metal = build_metal((0.2, 0.4, 0.8), roughness=0.45)

spheres = [
    place_sphere(red_dielectric, x_position=-1.5),
    place_sphere(gold_metal, x_position=0.0),
    place_sphere(blue_metal, x_position=1.5),
]

scene = paz.graphics.Scene([floor] + spheres)

camera_pose = paz.SE3.view_transform(
    jp.array([0.0, 2.0, 6.0]),
    jp.array([0.0, 0.5, 0.0]),
    jp.array([0.0, 1.0, 0.0]),
)

lights = [
    paz.graphics.PointLight(jp.full(3, 0.8), jp.array([3.0, 5.0, 4.0])),
    paz.graphics.PointLight(jp.full(3, 0.5), jp.array([-3.0, 4.0, 2.0])),
]

H, W = 2**10, 2**10
y_FOV = jp.pi / 4.0
render_args = (H, W), y_FOV, camera_pose
render_kwargs = dict(lights=lights, tiles=(1, 1), chunk_size=512)
render = jax.jit(
    paz.partial(
        paz.graphics.render,
        *render_args,
        **render_kwargs,
        shadows=True,
    )
)

image, _ = render(scene=scene, mask=None)
image = paz.image.denormalize(image)
image = paz.image.resize_opencv(image, (H // 2, W // 2))
paz.image.write("cook_torrance.png", image)
