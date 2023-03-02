"""Functions to visualize human poses"""
import numpy as np


def show3Dpose(keypoints, ax, lcolor="#3498db", rcolor="#e74c3c",
               add_labels=False):  # blue, orange
    """Visualize a 3d skeleton

    # Arguments
        channels: 48x1 vector. The pose to plot.
        ax: matplotlib 3d axis to draw on
        lcolor: color for left part of the body
        rcolor: color for right part of the body
        add_labels: whether to add coordinate labels
    """
    if keypoints.shape[1] == 48 or keypoints.shape[1] == 16:
        keypoints = np.reshape(keypoints, (keypoints.shape[0], 16, -1))
        start_joints = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13,
                                 14])
        end_joints = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                               15])
    else:
        keypoints = np.reshape(keypoints, (keypoints.shape[0], 32, -1))
        start_joints = np.array([0, 1, 2, 0, 6, 7, 0, 12, 13, 13, 17, 18, 13,
                                 25, 26])
        end_joints = np.array(
            [1, 2, 3, 6, 7, 8, 12, 13, 15, 17, 18, 19, 25, 26, 27])
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    def plot_person(person, lcolor="#3498db", rcolor="#e74c3c"):
        for i in np.arange(len(start_joints)):
            x, y, z = [np.array([person[start_joints[i], j],
                                 person[end_joints[i], j]]) for j in range(3)]
            ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)

    i = 0
    for person in keypoints:
        if i == 1:
            plot_person(person, lcolor="#9b59b6", rcolor="#2ecc71")
        else:
            plot_person(person)
        i += 1
    RADIUS = 750
    x, y, z = keypoints[0, 0, 0], keypoints[0, 0, 1], keypoints[0, 0, 2]
    ax.set_xlim3d([-RADIUS + x, RADIUS + x])
    ax.set_zlim3d([-RADIUS + z, RADIUS + z])
    ax.set_ylim3d([-RADIUS + y, RADIUS + y])
    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_zaxis.set_pane_color(white)


def show2Dpose(keypoints, ax, lcolor="#3498db", rcolor="#e74c3c",
               add_labels=False):
    """Visualize a 2d skeleton

    # Arguments
        keypoints: nx64 vector. n is num persons detected to plot.
        ax: matplotlib axis to draw on
        lcolor: color for left part of the body
        rcolor: color for right part of the body
        add_labels: whether to add coordinate labels
    # Returns
        Nothing. Draws on ax.
    """
    if keypoints.shape[1] == 32:
        keypoints = np.reshape(keypoints, (keypoints.shape[0], 16, -1))
        start_joints = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13,
                                 14])
        end_joints = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                               15])
    else:
        keypoints = np.reshape(keypoints, (keypoints.shape[0], 32, -1))
        start_joints = np.array([0, 1, 2, 0, 6, 7, 0, 12, 13, 13, 17, 18, 13,
                                 25, 26])
        end_joints = np.array([1, 2, 3, 6, 7, 8, 12, 13, 15, 17, 18, 19, 25,
                               26, 27])
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    def plot_person(person):
        for i in np.arange(len(start_joints)):
            x, y = [np.array([person[start_joints[i], j],
                              person[end_joints[i], j]]) for j in range(2)]
            ax.plot(x, y, lw=2, c=lcolor if LR[i] else rcolor)

    for person in keypoints:
        plot_person(person)
    RADIUS = 350  # space around the subject
    x, y = keypoints[0, 0, 0], keypoints[0, 0, 1]
    ax.set_xlim([-RADIUS + x, RADIUS + x])
    ax.set_ylim([-RADIUS + y, RADIUS + y])
    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("z")
    ax.set_aspect('equal')
