from functools import partial
import jax.numpy as jp
import jax
import paz

camera_origin = jp.array([0, -2.0, 4.0])
camera_target = jp.array([0.0, 0.0, 0.0])
world_up = jp.array([0.0, 1.0, 0.0])
lights = paz.graphics.PointLight(jp.full(3, 1.2), jp.array([3.0, 3.0, 5.0]))
camera_pose = paz.SE3.view_transform(camera_origin, camera_target, world_up)
H, W, y_FOV = image_size = 1024, 1024, jp.pi / 4.0
render_kwargs = {"lights": lights, "mask": None, "shadows": False}
rays = paz.graphics.camera.build_rays(image_size, y_FOV, camera_pose)
render_args = ((H, W), camera_pose, rays)

material = paz.graphics.Material(paz.graphics.RED, 0.3, 0.5, 0.8, 16)
shape = paz.graphics.Sphere(jp.eye(4), material)
scene = paz.graphics.Scene([shape])

render = jax.jit(partial(paz.graphics.render, *render_args, **render_kwargs))
image, depth = render(scene=scene)
paz.image.show(paz.image.denormalize(image))
