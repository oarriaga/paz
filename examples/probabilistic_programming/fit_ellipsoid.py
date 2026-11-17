# Fit an ellipsoid to the pointcloud of a FewSOL object by maximizing the
# posterior of its center and axes.

import os

# pyrender needs these before it loads to render on the NVIDIA GPU.
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

import argparse

import jax
import jax.numpy as jp
import numpy as np
import pyrender
import trimesh

import paz
import paz.utils.plot as plot
from paz import distributions as pzd
from paz.datasets import fewsol
from paz.inference.latent_space import to_forward_samples, to_inverse_samples


def load_object_pointcloud(shot, max_depth, min_height):
    cv_to_gl = jp.diag(jp.array([1.0, -1.0, -1.0, 1.0]))
    pointcloud = paz.pointcloud.from_depth(shot.depth, fewsol.get_intrinsics())
    scene = paz.pointcloud.bound(pointcloud, max_depth)
    scene = paz.algebra.transform_points(cv_to_gl, scene)
    _, plane_to_camera = paz.pointcloud.to_ground_frame(scene)
    camera_to_plane = paz.SE3.invert(plane_to_camera)
    points = paz.pointcloud.mask(pointcloud, shot.masks[0], max_depth)
    points = paz.algebra.transform_points(cv_to_gl, points)
    points = paz.pointcloud.transform(points, camera_to_plane)
    points = paz.pointcloud.filter_above_height(points, min_height)
    return paz.pointcloud.remove_outliers(points, 2.0), camera_to_plane


def build_model(pointcloud, mean, stdv):
    # The center should not leave the cloud without strong evidence.
    center = paz.Prior(pzd.Laplace(mean, 0.1 * stdv), name="center")
    axes = paz.Prior(build_axes_distribution(stdv), name="axes")
    surface = paz.Observable(Likelihood(pointcloud), name="surface")
    return paz.PGM([center, axes], [surface(center, axes)], "ellipsoid")


def build_axes_distribution(stdv):
    # Points spread evenly over a sphere have a stdv of radius / sqrt(3).
    mean = jp.sqrt(3.0) * stdv
    return pzd.TruncatedNormal(mean, 0.05 * mean, 1e-6 * mean, 2.0 * mean)


def Likelihood(pointcloud):
    def apply(center, axes):
        residuals = compute_surface_residuals(pointcloud, center, axes)
        return pzd.Normal(residuals, 0.01)

    return apply


def compute_surface_residuals(pointcloud, center, axes):
    return (((pointcloud - center) / axes) ** 2).sum(axis=1) - 1.0


def plot_prior(model, key, fit, num_samples):
    samples = model.prior.sample(key, num_samples)
    figure, axes = plot.subplots(nrows=2, ncols=3, figsize=(10, 6))
    for row, name in enumerate(("center", "axes")):
        for column, axis in enumerate(axes[row]):
            values = getattr(samples, name)[:, column]
            plot.histogram(values, axis, bins=40, color="steelblue")
            fitted_value = getattr(fit, name)[column]
            plot.vline(fitted_value, axis, "crimson", "-", 2)
            axis.set_title(f"{name} {'xyz'[column]}")
    figure.suptitle("Prior samples and fitted values")
    figure.tight_layout()


def build_scene(points, camera_to_plane, center, axes):
    scene = pyrender.Scene()
    colors = paz.to_numpy(paz.pointcloud.color(points, [255, 0, 0]))
    points = pyrender.Mesh.from_points(paz.to_numpy(points), colors=colors)
    scene.add(points)
    scene.add(build_frame((0, 255, 0)), pose=np.eye(4))
    scene.add(build_frame((0, 0, 255)), pose=paz.to_numpy(camera_to_plane))
    ellipsoid = trimesh.creation.uv_sphere(count=[10, 10])
    ellipsoid.apply_scale(paz.to_numpy(axes))
    pose = paz.to_numpy(paz.SE3.translation(center))
    scene.add(pyrender.Mesh.from_trimesh(ellipsoid), pose=pose)
    return scene


def build_frame(color):
    mesh = trimesh.creation.axis(origin_color=color, origin_size=0.01)
    return pyrender.Mesh.from_trimesh(mesh, smooth=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit an ellipsoid")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--concept", type=int, default=19)
    parser.add_argument("--shot", type=int, default=0)
    parser.add_argument("--max_depth", type=float, default=1.5)
    parser.add_argument("--min_height", type=float, default=0.01)
    parser.add_argument("--num_stdvs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=10.0)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--tolerance", type=float, default=1e-2)
    parser.add_argument("--num_prior_samples", type=int, default=200_000)
    args = parser.parse_args()
    jax.config.update("jax_debug_nans", True)

    shot = fewsol.load(args.concept)[args.shot]
    cloud_args = (shot, args.max_depth, args.min_height)
    points, camera_to_plane = load_object_pointcloud(*cloud_args)
    mean, stdv = paz.pointcloud.mean(points), paz.pointcloud.stdv(points)
    model = build_model(points, mean, stdv)
    data = jp.zeros(len(points))

    def compute_loss(sample):
        return -model.log_prob_inverse(sample, data)

    Sample = model.latent_space.Sample
    initial = Sample(center=mean, axes=args.num_stdvs * stdv)
    initial = to_inverse_samples(model.latent_space, initial)
    linesearch = paz.optimization.armijo_linesearch(50)
    stop_args = (args.learning_rate, args.max_steps, args.tolerance)
    optimizer_args = (initial, compute_loss, *stop_args, 10, linesearch)
    parameters, history = paz.optimizers.LBFGS(*optimizer_args)
    fit = to_forward_samples(model.latent_space, parameters)
    print("center:", fit.center)
    print("axes:", fit.axes)

    key = jax.random.PRNGKey(args.seed)
    plot_prior(model, key, fit, args.num_prior_samples)
    figure, axes = plot.subplots(ncols=2, figsize=(12, 4))
    steps = jp.arange(len(history.losses))
    plot.line(steps, history.losses, axes[0], "black")
    plot.set_labels(axes[0], x="step", y="loss")
    axes[0].set_title("Loss curve")
    residuals = compute_surface_residuals(points, fit.center, fit.axes)
    plot.histogram(residuals, axes[1], density_norm=False, color="gray")
    plot.set_labels(axes[1], x="residual", y="count")
    axes[1].set_title("Surface residuals")
    figure.tight_layout()
    plot.show()

    scene = build_scene(points, camera_to_plane, fit.center, fit.axes)
    pyrender.Viewer(scene, use_raymond_lighting=True)
