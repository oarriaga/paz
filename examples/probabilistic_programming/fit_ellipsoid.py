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
import matplotlib.pyplot as plt
import numpy as np

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_platform_name", "cpu")

import paz
from paz.datasets import fewsol
import ellipsoid
from optimizers import LBFGS, LineSearch

concept_arg = 0
shot_arg = 0
seed = 777
max_depth = 1.4
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
scene_pointcloud = paz.pointcloud.bound(pointcloud, max_depth)
object_pointcloud = paz.pointcloud.mask(pointcloud, shot.masks[0], max_depth)
scene_pointcloud = paz.algebra.transform_points(cv_to_gl, scene_pointcloud)
_pointcloud_ground, plane_to_camera = paz.pointcloud.to_ground_frame(
    scene_pointcloud
)
# plane_to_camera maps the fitted ground frame into the GL camera frame.
camera_to_plane = paz.SE3.invert(plane_to_camera)
object_pointcloud = paz.pointcloud.transform(
    paz.algebra.transform_points(cv_to_gl, object_pointcloud),
    camera_to_plane,
)
object_pointcloud = paz.pointcloud.filter_above_height(
    object_pointcloud, min_height
)
object_pointcloud = paz.pointcloud.remove_outliers(object_pointcloud, 2.0)

optimizer = LBFGS(10.0, 10, LineSearch(50))
statistics = ellipsoid.pointcloud_compute_statistics(object_pointcloud)
model, data = ellipsoid.RobustEllipsoid(
    object_pointcloud, statistics, x_scale
)
loss_fn = ellipsoid.build_map_objective(model, data)
shift, axes, result = ellipsoid.fit(
    model, object_pointcloud, num_stdvs, loss_fn, optimizer
)
elapsed = time.perf_counter() - start
print(f"elapsed: {elapsed:.1f}s")

points = object_pointcloud
colors = np.array(paz.pointcloud.color(points, [255, 0, 0]))


def build_forward_fit(model, shift, axes):
    return model.latent_space.Sample(
        x_mean=shift[0],
        y_mean=shift[1],
        z_mean=shift[2],
        x_axis=axes[0],
        y_axis=axes[1],
        z_axis=axes[2],
    )


def plot_prior_predictive_samples(model, key, forward_fit, num_samples=1000):
    samples = model.prior.sample(key, num_samples)
    names = model.latent_space.names
    figure, axes = plt.subplots(2, 3, figsize=(10, 6))
    for axis, name in zip(axes.flatten(), names):
        values = np.array(getattr(samples, name))
        axis.hist(values, bins=40, color="steelblue", alpha=0.7)
        if forward_fit is not None:
            axis.axvline(
                float(getattr(forward_fit, name)),
                color="crimson",
                linewidth=2,
            )
        axis.set_title(name)
    figure.suptitle("Prior predictive samples")
    figure.tight_layout()
    return figure


def plot_loss_curve(losses, title):
    figure, axis = plt.subplots(1, 1, figsize=(6, 4))
    axis.plot(np.array(losses), color="black")
    axis.set_title(title)
    axis.set_xlabel("step")
    axis.set_ylabel("loss")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure


def plot_surface_residuals(pointcloud, shift, axes, title):
    x, y, z = paz.pointcloud.split(pointcloud)
    residuals = ellipsoid.compute_surface_equation(
        x, y, z, shift[0], shift[1], shift[2], axes[0], axes[1], axes[2]
    )
    figure, axis = plt.subplots(1, 1, figsize=(6, 4))
    axis.hist(np.array(residuals), bins=50, color="gray", alpha=0.8)
    axis.set_title(title)
    axis.set_xlabel("residual")
    axis.set_ylabel("count")
    figure.tight_layout()
    return figure


def Axis(color=(0, 255, 0), origin_size=0.01):
    mesh = trimesh.creation.axis(origin_color=color, origin_size=origin_size)
    return pyrender.Mesh.from_trimesh(mesh, smooth=False)


scene = pyrender.Scene()
scene.add(pyrender.Mesh.from_points(np.array(points), colors=colors))
scene.add(Axis(color=(0, 255, 0)), pose=np.eye(4))
scene.add(Axis(color=(0, 0, 255)), pose=np.array(camera_to_plane))
mesh = ellipsoid.Mesh(*axes, 10, 10)
scene.add(
    pyrender.Mesh.from_trimesh(mesh),
    pose=np.array(paz.SE3.translation(shift)),
)

key_prior = jax.random.split(key, 2)[1]
forward_fit = build_forward_fit(model, shift, axes)
plot_prior_predictive_samples(
    model,
    key_prior,
    forward_fit,
    num_samples=2000,
)
plot_loss_curve(result.losses, "Loss curve")
plot_surface_residuals(
    object_pointcloud,
    shift,
    axes,
    "Surface residuals",
)
plt.show()
pyrender.Viewer(scene, use_raymond_lighting=True)
