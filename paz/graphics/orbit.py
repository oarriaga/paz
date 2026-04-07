from functools import partial

import jax
import jax.numpy as jp

import paz


def orbit_pose(camera_origin, angle):
    x, height, z = camera_origin
    radius = jp.linalg.norm(jp.array([x, z]))
    pos = jp.array([radius * jp.cos(angle), height, radius * jp.sin(angle)])
    return paz.SE3.view_transform(pos, jp.zeros(3), jp.array([0.0, 1.0, 0.0]))


def render_orbit(image_shape, y_FOV, scene, lights, shadows, camera_origin, num_views):
    H, W = image_shape
    shapes, mask, _, lights = paz.graphics.scene.compile(scene, lights, mask=None)
    bounces = paz.graphics.scene.compute_bounces(shapes)
    identity_rays = paz.graphics.camera.build_rays(image_shape, y_FOV, jp.eye(4))
    render_bounced = paz.graphics.renderer.render_bounced

    @jax.jit
    def render_frame(pose):
        cam_to_world = jp.linalg.inv(pose)
        rays = paz.graphics.geometry.transform_rays(cam_to_world, *identity_rays)
        args = (H, W, pose, rays, shapes, lights, mask, shadows, None, bounces)
        image, _ = render_bounced(*args)
        return jp.clip(image, 0.0, 1.0)

    angles = jp.linspace(1.5 * jp.pi, 3.5 * jp.pi, num_views)
    build = partial(orbit_pose, camera_origin)
    return [render_frame(build(angle)) for angle in angles]
