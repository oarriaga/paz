import jax.numpy as jp
import matplotlib.pyplot as plt


def plot_results(inliers, outliers, normal, offset):
    """Visualizes the point cloud and the fitted plane."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the inlier and outlier points
    ax.scatter(
        inliers[:, 0],
        inliers[:, 1],
        inliers[:, 2],
        c="blue",
        label="Inliers",
        s=10,
        alpha=0.6,
    )
    ax.scatter(
        outliers[:, 0],
        outliers[:, 1],
        outliers[:, 2],
        c="red",
        label="Outliers",
        s=30,
        alpha=0.8,
    )

    # Create a meshgrid to plot the fitted plane surface
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_plane, y_plane = jp.meshgrid(
        jp.linspace(xlim[0], xlim[1], 10), jp.linspace(ylim[0], ylim[1], 10)
    )

    # Calculate z values for the plane using the fitted parameters
    n = normal
    d = offset
    # z = (-n_x*x - n_y*y - d) / n_z
    z_plane = (-n[0] * x_plane - n[1] * y_plane - d) / n[2]

    # Plot the plane surface
    ax.plot_surface(
        x_plane,
        y_plane,
        z_plane,
        alpha=0.3,
        color="green",
        rstride=100,
        cstride=100,
        label="Fitted Plane",
    )

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title("Robust 3D Plane Fitting with JAX and TFP")
    # Workaround for legend with plot_surface
    ax.legend(
        handles=[
            plt.Line2D([0], [0], linestyle="none", c="b", marker="o"),
            plt.Line2D([0], [0], linestyle="none", c="r", marker="o"),
            plt.Line2D([0], [0], linestyle="-", c="g", alpha=0.3, lw=4),
        ],
        labels=["Inliers", "Outliers", "Fitted Plane"],
    )
    plt.show()
