from functools import partial

import jax
import jax.numpy as jp

import paz


def orbit_pose(camera_origin, angle):
    x, height, z = camera_origin
    radius = jp.linalg.norm(jp.array([x, z]))
    pos = jp.array([radius * jp.cos(angle), height, radius * jp.sin(angle)])
    return paz.SE3.view_transform(pos, jp.zeros(3), jp.array([0.0, 1.0, 0.0]))


def render_orbit(
    image_shape, y_FOV, scene, lights, shadows, camera_origin, num_views
):
    shapes = paz.graphics.scene.flatten_scene(scene)
    num_bounces = paz.graphics.scene.compute_bounces(shapes)

    @jax.jit
    def render_frame(pose):
        args = image_shape, y_FOV, pose, scene, None, lights, (1, 1), 1024
        image, _ = paz.graphics.render(*args, shadows, None, num_bounces)
        return jp.clip(image, 0.0, 1.0)

    angles = jp.linspace(1.5 * jp.pi, 3.5 * jp.pi, num_views)
    build = partial(orbit_pose, camera_origin)
    return [render_frame(build(angle)) for angle in angles]
