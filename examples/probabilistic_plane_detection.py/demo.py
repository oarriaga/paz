import jax
import paz
import jax.numpy as jp
import optax
import matplotlib.pyplot as plt


key = jax.random.PRNGKey(777)
step_size = 0.1
num_steps = 100
DOF = 2.0
scale = 0.2
conept_arg = 0
shot_arg = 0

dataset = paz.datasets.fsclvr.load("plain", "train")
image, depth, label = dataset[0][0]
camera_intrinsics = paz.datasets.fsclvr.get_intrinsics()
data = paz.pointcloud.from_depth(depth, camera_intrinsics)
data = paz.pointcloud.sample(key, data, 10_000)
plt.imshow(depth)
plt.show()


optimizer = optax.adam(step_size)
loss = paz.lock(paz.plane.student_t_loss, scale, DOF, data)
_fit = paz.lock(paz.plane.fit, optimizer, loss, num_steps)
(pred_normal, pred_centroid), losses = paz.time(_fit)(key, data)
pred_offset = -jp.dot(pred_normal, pred_centroid)


def plot_results(inliers, normal, offset):
    figure = plt.figure(figsize=(12, 10))
    axis = figure.add_subplot(111, projection="3d")

    axis.scatter(
        inliers[:, 0],
        inliers[:, 1],
        inliers[:, 2],
        c="blue",
        label="Inliers",
        s=10,
        alpha=0.6,
    )
    xlim = axis.get_xlim()
    ylim = axis.get_ylim()
    x_plane, y_plane = jp.meshgrid(
        jp.linspace(xlim[0], xlim[1], 10), jp.linspace(ylim[0], ylim[1], 10)
    )

    n = normal
    d = offset
    z_plane = (-n[0] * x_plane - n[1] * y_plane - d) / n[2]

    axis.plot_surface(
        x_plane,
        y_plane,
        z_plane,
        alpha=0.3,
        color="green",
        rstride=100,
        cstride=100,
        label="Fitted Plane",
    )

    axis.set_xlabel("X axis")
    axis.set_ylabel("Y axis")
    axis.set_zlabel("Z axis")
    axis.set_title("Robust 3D Plane Fitting")
    axis.legend(
        handles=[
            plt.Line2D([0], [0], linestyle="none", c="b", marker="o"),
            plt.Line2D([0], [0], linestyle="-", c="g", alpha=0.3, lw=4),
        ],
        labels=["Pointcloud", "Fitted Plane"],
    )
    plt.show()


plot_results(data, pred_normal, pred_offset)
