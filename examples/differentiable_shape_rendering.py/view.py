import jax.numpy as jp
import paz

camera_pose = paz.SE3.view_transform(
    camera_origin=jp.array([0.0, 10.0, 10.0]),
    target_origin=jp.array([0.0, 0.0, 0.0]),
    world_up=jp.array([0.0, 1.0, 0.0]),
)

scene = paz.graphics.load("axes.json")
paz.graphics.viewer(scene, camera_pose)
