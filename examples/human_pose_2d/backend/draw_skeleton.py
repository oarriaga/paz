import cv2
import numpy as np
from dataset import VISUALISATION_CONFIG


def add_joints(image, joints, color, dataset):
    part_arg = VISUALISATION_CONFIG[dataset]['part_arg']
    part_orders = VISUALISATION_CONFIG[dataset]['part_orders']

    def link(a, b, color):
        if part_arg[a] < joints.shape[0] and part_arg[b] < joints.shape[0]:
            joint_a = joints[part_arg[a]]
            joint_b = joints[part_arg[b]]
            if joint_a[2] > 0 and joint_b[2] > 0:
                cv2.line(image, (int(joint_a[0]), int(joint_a[1])),
                                (int(joint_b[0]), int(joint_b[1])),
                         color, 2)

    # add joints
    for joint in joints:
        if joint[2] > 0:
            cv2.circle(image, (int(joint[0]), int(joint[1])), 1, color, 2)

    # add link
    for pair in part_orders:
        link(pair[0], pair[1], color)

    return image


def draw_skeleton(image, joints, dataset):
    for person in joints:
        color = np.random.randint(0, 255, size=3)
        color = [int(i) for i in color]
        add_joints(image, person, color, dataset=dataset)
    return image
