import numpy as np
from paz.backend.image import draw
from dataset import VISUALISATION_CONFIG


def transform_point(point, transform):
    point = np.array([point[0], point[1], 1.]).T
    point_transformed = np.dot(transform, point)
    return point_transformed


def transform_joints(grouped_joints, transform):
    transformed_joints = []
    for joints in grouped_joints:
        for joint in joints:
            joint[0:2] = transform_point(joint[0:2], transform)[:2]
        transformed_joints.append(joints[:, :3])
    return transformed_joints


def extract_joints(joints):
    for joints_arg in range(len(joints)):
        joints[joints_arg] = joints[joints_arg][:, :2]
    return joints


def link_joints(a, b, image, joints, color, dataset):
    part_arg = VISUALISATION_CONFIG[dataset]['part_arg']
    if part_arg[a] < joints.shape[0] and part_arg[b] < joints.shape[0]:
        joint_a = joints[part_arg[a]]
        joint_b = joints[part_arg[b]]
        if joint_a[2] > 0 and joint_b[2] > 0:
            draw.draw_line(image, (int(joint_a[0]), int(joint_a[1])),
                                  (int(joint_b[0]), int(joint_b[1])), color, 2)
    return image


def annotate_joints(image, joints, color):
    for joint in joints:
        if joint[2] > 0:
            draw.draw_circle(image, (int(joint[0]), int(joint[1])), color, 2)
    return image


def add_joints(image, joints, color, dataset):
    part_orders = VISUALISATION_CONFIG[dataset]['part_orders']
    image = annotate_joints(image, joints, color)
    for pair in part_orders:
        image = link_joints(pair[0], pair[1], image, joints, color, dataset)
    return image


def draw_skeleton(image, joints, dataset):
    for person in joints:
        color = np.random.randint(0, 255, size=3)
        color = [int(i) for i in color]
        add_joints(image, person, color, dataset=dataset)
    return image
