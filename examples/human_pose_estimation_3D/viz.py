import numpy as np


def show3Dpose(keypoints, ax, lcolor="#3498db", rcolor="#e74c3c"):
    """Draws 3D human-pose skeletons on a matplotlib 3D axis. Left-side bones
    use `lcolor`, right-side bones use `rcolor`."""
    keypoints = np.reshape(keypoints, (keypoints.shape[0], -1, 3))
    if keypoints.shape[1] == 16:
        start = [0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14]
        end = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    else:
        start = [0, 1, 2, 0, 6, 7, 0, 12, 13, 13, 17, 18, 13, 25, 26]
        end = [1, 2, 3, 6, 7, 8, 12, 13, 15, 17, 18, 19, 25, 26, 27]
    left_side = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    for person in keypoints:
        for bone in range(len(start)):
            xs, ys, zs = [np.array([person[start[bone], axis],
                                    person[end[bone], axis]])
                          for axis in range(3)]
            ax.plot(xs, ys, zs, lw=2, c=lcolor if left_side[bone] else rcolor)
    set_axis_limits(ax, keypoints[0, 0])


def set_axis_limits(ax, root, radius=750):
    ax.set_xlim3d([-radius + root[0], radius + root[0]])
    ax.set_ylim3d([-radius + root[1], radius + root[1]])
    ax.set_zlim3d([-radius + root[2], radius + root[2]])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
