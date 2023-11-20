"""Functions to visualize human poses"""

import numpy as np
import data_utils


def show3Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):  # blue, orange
    """Visualize a 3d skeleton

    Args
        channels: 48x1 vector. The pose to plot.
        ax: matplotlib 3d axis to draw on
        lcolor: color for left part of the body
        rcolor: color for right part of the body
        add_labels: whether to add coordinate labels
    Returns
        Nothing. Draws on ax.
    """

    # assert channels.shape[1] == 16 * 3, "channels should have 48 entries, it has %d instead" % channels.size

    if channels.shape[1] == 48:
        vals = np.reshape(channels, (channels.shape[0], 16, -1))
        # print(f"\nvals 3d {vals} {vals.shape}", flush=True)
        I = np.array([1, 2, 3, 1, 5, 6, 1, 8, 9, 9, 11, 12, 9, 14, 15]) - 1  # start points
        J = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]) - 1  # end points

    else:
        # print(f"ELSE loop p3d shape is : {channels.shape}", flush=True)
        vals = np.reshape(channels, (channels.shape[0], 32, -1))
        I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 14, 18, 19, 14, 26, 27]) - 1  # start points
        J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 16, 18, 19, 20, 26, 27, 28]) - 1  # end points
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    def plot_person(person, lcolor="#3498db", rcolor="#e74c3c"):
        for i in np.arange(len(I)):
            x, y, z = [np.array([person[I[i], j], person[J[i], j]]) for j in range(3)]
            ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)

    i = 0
    # Now we plot all persons one by one in same plot
    for person in vals:
        if i == 1:
            plot_person(person, lcolor="#9b59b6", rcolor="#2ecc71")
        else:
            plot_person(person)
        i += 1

    RADIUS = 750  # space around the subject
    xroot, yroot, zroot = vals[0, 0, 0], vals[0, 0, 1], vals[0, 0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_zaxis.set_pane_color(white)
    # Keep y (xz) pane


def show2Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
    """Visualize a 2d skeleton

    Args
        channels: nx64 vector. n is num persons detected to plot.
        ax: matplotlib axis to draw on
        lcolor: color for left part of the body
        rcolor: color for right part of the body
        add_labels: whether to add coordinate labels
    Returns
        Nothing. Draws on ax.
    """

    if channels.shape[1] == 32:
        vals = np.reshape(channels, (channels.shape[0], 16, -1))
        # print(f"\nvals 2d f loop {vals} {vals.shape}")
        I = np.array([1, 2, 3, 1, 5, 6, 1, 8, 9, 9, 11, 12, 9, 14, 15]) - 1  # start points
        J = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]) - 1  # end points

    else:
        # print(f"ELSE loop p2d shape is : {channels.shape}")
        vals = np.reshape(channels, (channels.shape[0], 32, -1))
        # print(f"\nvals 2d else loop {vals} {vals.shape}")
        I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 14, 18, 19, 14, 26, 27]) - 1  # start points
        J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 16, 18, 19, 20, 26, 27, 28]) - 1  # end points
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    def plot_person(person):
        for i in np.arange(len(I)):
            x, y = [np.array([person[I[i], j], person[J[i], j]]) for j in range(2)]
            ax.plot(x, y, lw=2, c=lcolor if LR[i] else rcolor)

    # Now we plot all persons one by one in same plot
    for person in vals:
        plot_person(person)

    # Get rid of the ticks
    # ax.set_xticks([])
    # ax.set_yticks([])
    #
    # # Get rid of tick labels
    # ax.get_xaxis().set_ticklabels([])
    # ax.get_yaxis().set_ticklabels([])

    RADIUS = 350  # space around the subject
    xroot, yroot = vals[0, 0, 0], vals[0, 0, 1]
    ax.set_xlim([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim([-RADIUS + yroot, RADIUS + yroot])

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("z")

    ax.set_aspect('equal')
