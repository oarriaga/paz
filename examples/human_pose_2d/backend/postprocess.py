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


def annotate_joints(image, joints):
    color = [0, 0, 0]
    for joint in joints:
        if joint[2] > 0:
            draw.draw_circle(image, (int(joint[0]), int(joint[1])), color, 2)
    return image


def add_joints(image, joints, dataset):
    part_orders = VISUALISATION_CONFIG[dataset]['part_orders']
    part_color = VISUALISATION_CONFIG[dataset]['part_color']
    for pair_arg, pair in enumerate(part_orders):
        color = part_color[pair_arg]
        image = link_joints(pair[0], pair[1], image, joints, color, dataset)
    image = annotate_joints(image, joints)
    return image


def draw_skeleton(image, joints, dataset):
    for person in joints:
        add_joints(image, person, dataset=dataset)
    return image
