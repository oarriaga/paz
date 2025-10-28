# remove mess with inv
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
import jax
import paz
import jax.numpy as jp
import optax

import matplotlib.pyplot as plt

import numpy as np


def _convert_leaf(leaf, float_dtype):
    if isinstance(leaf, jp.ndarray):
        if leaf.dtype == jp.float32:
            return jp.array(leaf, dtype=jp.float32)
        elif leaf.dtype == jp.int32:
            return jp.array(leaf, dtype=jp.int32)
        else:
            raise ValueError
    else:
        return leaf


def tree_to_dtype(pytree, dtype):
    return jax.tree.map(lambda leaf: _convert_leaf(leaf, dtype), pytree)


key = jax.random.PRNGKey(777)
step_size = 0.1
dtype = jp.bfloat16
num_steps = 100
DOF = 2.0
scale = 0.2
shot_arg = 0
num_points = 100
H = 480
W = 640
radius = 0.005
camera_pose = jp.eye(4)
class_arg = 3
shot_arg = 0
max_depth = 1.0
dataset = "fewsol"
y_FOV = paz.datasets.get_y_FOV(dataset)
camera_intrinsics = paz.datasets.get_intrinsics(dataset)
openCV_to_openGL = jp.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


if dataset == "fewsol":
    shot = paz.datasets.fewsol.load(
        "/home/octavio/few-shot_scene_reconstruction/data/FEWSOL/", class_arg
    )[shot_arg]
    true_image, true_depth = shot.image.copy(), shot.depth.copy()
elif dataset == "fsclvr":
    dataset = paz.datasets.fsclvr.load("plain", "train")
    true_image, true_depth, label = dataset[class_arg][shot_arg]
else:
    raise ValueError


pointcloud = paz.pointcloud.from_depth(true_depth, camera_intrinsics)
pointcloud = paz.algebra.transform_points(openCV_to_openGL, pointcloud)
pointcloud = paz.pointcloud.bound(pointcloud, max_depth)
pointcloud_sampled = paz.pointcloud.sample(key, pointcloud, num_points)

optimizer = optax.adam(step_size)
loss = paz.lock(paz.plane.student_t_loss, scale, DOF, pointcloud_sampled)
fit = paz.lock(paz.plane.fit, optimizer, loss, num_steps)
(pred_normal, pred_centroid), losses = paz.time(fit)(key, pointcloud_sampled)
pred_offset = -jp.dot(pred_normal, pred_centroid)

rays = paz.graphics.camera.build_rays((H, W), y_FOV, camera_pose)
lights = [paz.graphics.PointLight(jp.ones(3), jp.ones(3))]

world_up = jp.array([0.0, 1.0, 0.0])
y_axis = pred_normal / jp.linalg.norm(pred_normal)

x_axis = jp.cross(world_up, y_axis)
x_axis = x_axis / jp.linalg.norm(x_axis)
z_axis = jp.cross(y_axis, x_axis)
rotation = jp.stack([x_axis, y_axis, z_axis], axis=1)


plane_pose = jp.eye(4)
plane_pose = plane_pose.at[:3, :3].set(rotation)
plane_pose = plane_pose.at[:3, 3].set(pred_centroid)

rotate = paz.SE3.rotation_y(jp.pi / 2)
scale = paz.SE3.scaling(jp.full(3, radius))

pointcloud_plane = paz.algebra.transform_points(
    jp.linalg.inv(plane_pose), pointcloud
)
mask = pointcloud_plane[:, 1] > 0.01

nodes = [paz.graphics.Plane(plane_pose, paz.graphics.Material(jp.full(3, 0.7)))]
# num_positives = jp.sum(mask)
# num_shape_points = num_positives if num_positives < 100 else 100
positions = paz.pointcloud.sample(key, pointcloud[mask], 50)
# positions = paz.pointcloud.sample(key, pointcloud[mask], num_shape_points)
for position in positions:
    pose = paz.SE3.translation(position) @ scale
    nodes.append(paz.graphics.Sphere(pose))

scene = paz.graphics.Scene(nodes)
# scene = tree_to_dtype(paz.graphics.Scene(nodes), dtype)
render = jax.jit(
    paz.partial(
        paz.graphics.render,
        (H, W),
        # camera_pose.astype(dtype),
        camera_pose,
        rays,
        # tree_to_dtype(rays, dtype),
        # lights=tree_to_dtype(lights, dtype),
        lights=lights,
        mask=None,
        shadows=False,
    )
)
# pred_image, pred_depth = paz.time(render)(scene=scene)
pred_image, pred_depth = render(scene=scene)
print("pred_image.dtype", pred_image.dtype)
pred_image = paz.image.denormalize(pred_image)
true_image = (
    true_image if dataset == "fewsol" else paz.image.denormalize(true_image)
)
mosaic = paz.draw.mosaic(jp.array([true_image, pred_image]), (1, 2))
paz.image.show(mosaic.astype(jp.uint8))
