import cv2
import numpy as np


coco_part_labels = [
    'nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r',
    'sho_l', 'sho_r', 'elb_l', 'elb_r', 'wri_l', 'wri_r',
    'hip_l', 'hip_r', 'kne_l', 'kne_r', 'ank_l', 'ank_r'
]
coco_part_arg = {
    b: a for a, b in enumerate(coco_part_labels)
}
coco_part_orders = [
    ('nose', 'eye_l'), ('eye_l', 'eye_r'), ('eye_r', 'nose'),
    ('eye_l', 'ear_l'), ('eye_r', 'ear_r'), ('ear_l', 'sho_l'),
    ('ear_r', 'sho_r'), ('sho_l', 'sho_r'), ('sho_l', 'hip_l'),
    ('sho_r', 'hip_r'), ('hip_l', 'hip_r'), ('sho_l', 'elb_l'),
    ('elb_l', 'wri_l'), ('sho_r', 'elb_r'), ('elb_r', 'wri_r'),
    ('hip_l', 'kne_l'), ('kne_l', 'ank_l'), ('hip_r', 'kne_r'),
    ('kne_r', 'ank_r')
]


VIS_CONFIG = {
    'COCO': {
        'part_labels': coco_part_labels,
        'part_arg': coco_part_arg,
        'part_orders': coco_part_orders
    }}


def add_joints(image, joints, color, dataset):
    part_arg = VIS_CONFIG[dataset]['part_arg']
    part_orders = VIS_CONFIG[dataset]['part_orders']

    def link(a, b, color):
        if part_arg[a] < joints.shape[0] and part_arg[b] < joints.shape[0]:
            jointa = joints[part_arg[a]]
            jointb = joints[part_arg[b]]
            if jointa[2] > 0 and jointb[2] > 0:
                cv2.line(image, (int(jointa[0]), int(jointa[1])),
                                (int(jointb[0]), int(jointb[1])),
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
