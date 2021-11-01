import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_colored_sphere(num_points=200):
    range_phi = np.linspace(0, np.pi, num_points)
    range_psi = np.linspace(0, 2*np.pi, num_points)

    mesh_phi, mesh_psi = np.meshgrid(range_phi, range_psi)
    coordinate_grid = np.array([mesh_phi, mesh_psi])
    coordinate_grid = np.reshape(np.transpose(coordinate_grid, (1, 2, 0)), (-1, 2))

    xs = np.sin(coordinate_grid[:, 0])*np.cos(coordinate_grid[:, 1])
    ys = np.sin(coordinate_grid[:, 0])*np.sin(coordinate_grid[:, 1])
    zs = np.cos(coordinate_grid[:, 0])

    normalized_coordinate_grid = np.ones((len(coordinate_grid), 3))
    normalized_coordinate_grid[:, 0] = coordinate_grid[:, 0] / np.pi
    normalized_coordinate_grid[:, 1] = coordinate_grid[:, 1]/(2*np.pi)

    xs = (xs + 1.) / 2.0
    ys = (ys + 1.) / 2.0
    zs = (zs + 1.) / 2.0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c=np.column_stack((xs, ys, zs)))
    ax.set_xlabel("Normalized x-coordinate")
    ax.set_ylabel("Normalized y-coordinate")
    ax.set_zlabel("Normalized z-coordinate")
    plt.show()


if __name__ == "__main__":
    plot_colored_sphere()

