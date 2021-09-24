num_joints = 17
joint_order = [i-1 for i in [1, 2, 3, 4, 5, 6, 7, 12,
                             13, 8, 9, 10, 11, 14, 15, 16, 17]]

JOINT_CONFIG = {
    'COCO': [
        0, 1, 2, 3, 4, 5, 6, 11, 12, 7, 8, 9, 10, 13, 14, 15, 16
    ],
    'COCO_WITH_CENTER': [
        1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17
        ]
}


FLIP_CONFIG = {
    'COCO': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
    ],
    'COCO_WITH_CENTER': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17
    ]
}


def get_joint_info(data_info):
    dataset = data_info['data']
    data_with_center = data_info['data_with_center']
    if data_with_center:
        joint_order = JOINT_CONFIG[dataset + '_WITH_CENTER']
        fliped_joint_order = FLIP_CONFIG[dataset + '_WITH_CENTER']
    else:
        joint_order = JOINT_CONFIG[dataset]
        fliped_joint_order = FLIP_CONFIG[dataset]

    num_joints = len(joint_order)
    return joint_order, num_joints, fliped_joint_order
