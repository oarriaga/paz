import os

# os.environ["PYOPENGL_PLATFORM"] = "egl"  # must be set before importing pyrender

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
# os.environ["PYOPENGL_PLATFORM"] = "egl"  # keep this too

import time
import trimesh
import pyrender

import jax
import jax.numpy as jp
import numpy as np

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_platform_name", "cpu")

import paz
from paz.datasets import fewsol
import ellipsoid

concept_arg = 0
shot_arg = 0
seed = 777
max_depth = 1.4
scale = 1e3
num_stdvs = 2.0
x_scale = 1.0
cv_to_gl = jp.diag(jp.array([1.0, -1.0, -1.0, 1.0]))
min_height = 0.01

shots = fewsol.load(concept_arg)
shot = shots[shot_arg]
camera_intrinsics = fewsol.get_intrinsics()

key = jax.random.PRNGKey(seed)
start = time.perf_counter()

pointcloud = paz.pointcloud.from_depth(shot.depth, camera_intrinsics)
pointcloud = paz.pointcloud.bound(pointcloud, max_depth)
pointcloud = paz.algebra.transform_points(cv_to_gl, pointcloud)
pointcloud, floor_to_world = paz.pointcloud.to_ground_frame(pointcloud)
pointcloud = paz.pointcloud.filter_above_height(pointcloud, min_height)
pointcloud = paz.pointcloud.remove_outliers(pointcloud, 2.0)

# TODO we are going to take instead the pointcloud already transformed of the object
shifts, scales = ellipsoid.fit_scene(
    key,
    shot.depth,
    shot.masks,
    scale,
    num_stdvs,
    camera_intrinsics,
    max_depth,
    x_scale,
)
elapsed = time.perf_counter() - start
print(f"elapsed: {elapsed:.1f}s")

points = paz.pointcloud.bound(shot.pointcloud, max_depth)
camera_to_plane = ellipsoid.fit_plane(key, points)
points = paz.pointcloud.transform(points, camera_to_plane)
colors = np.array(paz.pointcloud.color(points, [255, 0, 0]))


def Axis(color=(0, 255, 0), origin_size=0.01):
    mesh = trimesh.creation.axis(origin_color=color, origin_size=origin_size)
    return pyrender.Mesh.from_trimesh(mesh, smooth=False)


scene = pyrender.Scene()
scene.add(pyrender.Mesh.from_points(np.array(points), colors=colors))
scene.add(Axis(), pose=np.eye(4))
scene.add(Axis(), pose=np.array(camera_to_plane))
for shift, axes in zip(shifts, scales):
    mesh = ellipsoid.Mesh(*axes, 10, 10)
    scene.add(
        pyrender.Mesh.from_trimesh(mesh),
        pose=np.array(paz.SE3.translation(shift)),
    )
pyrender.Viewer(scene, use_raymond_lighting=True)
