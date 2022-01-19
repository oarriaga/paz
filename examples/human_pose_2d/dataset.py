JOINT_CONFIG = {
    'COCO': [
        0, 1, 2, 3, 4, 5, 6, 11, 12, 7, 8, 9, 10, 13, 14, 15, 16],
    'COCO_WITH_CENTER': [
        0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17]}


FLIP_CONFIG = {
    'COCO': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
    'COCO_WITH_CENTER': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17]}


coco_part_labels = ['nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r', 'sholder_l',
                    'sholder_r', 'elbow_l', 'elbow_r', 'wrist_l', 'wrist_r',
                    'hip_l', 'hip_r', 'knee_l', 'knee_r', 'ankle_l', 'ankle_r']

coco_part_arg = {b: a for a, b in enumerate(coco_part_labels)}

coco_part_orders = [('nose', 'eye_l'), ('eye_l', 'eye_r'),
                    ('eye_r', 'nose'), ('eye_l', 'ear_l'),
                    ('eye_r', 'ear_r'), ('ear_l', 'sholder_l'),
                    ('ear_r', 'sholder_r'), ('sholder_l', 'sholder_r'),
                    ('sholder_l', 'hip_l'), ('sholder_r', 'hip_r'),
                    ('hip_l', 'hip_r'), ('sholder_l', 'elbow_l'),
                    ('elbow_l', 'wrist_l'), ('sholder_r', 'elbow_r'),
                    ('elbow_r', 'wrist_r'), ('hip_l', 'knee_l'),
                    ('knee_l', 'ankle_l'), ('hip_r', 'knee_r'),
                    ('knee_r', 'ankle_r')]

VISUALISATION_CONFIG = {
    'COCO': {'part_labels': coco_part_labels,
             'part_arg': coco_part_arg,
             'part_orders': coco_part_orders}}


class GetJointInfo():
    def __init__(self, dataset_name, data_with_center):
        super(GetJointInfo, self).__init__()
        self.dataset_name = dataset_name
        self.data_with_center = data_with_center

    def get_join_order(self):
        if self.dataset_name == 'COCO':
            joint_order = [0, 1, 2, 3, 4, 5, 6, 11, 12,
                           7, 8, 9, 10, 13, 14, 15, 16]
            flipped_joint_order = [0, 2, 1, 4, 3, 6, 5, 8, 7,
                                   10, 9, 12, 11, 14, 13, 16, 15]
        return joint_order, flipped_joint_order

    def get_join_order_with_center(self):
        if self.dataset_name == 'COCO':
            joint_order = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13,
                           8, 9, 10, 11, 14, 15, 16, 17]
            flipped_joint_order = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10,
                                   9, 12, 11, 14, 13, 16, 15, 17]
        return joint_order, flipped_joint_order

    def get_joint_parts_info(self):
        if self.dataset_name == 'COCO':
            part_labels = ['nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r',
                           'sholder_l', 'sholder_r', 'elbow_l', 'elbow_r',
                           'wrist_l', 'wrist_r', 'hip_l', 'hip_r', 'knee_l',
                           'knee_r', 'ankle_l', 'ankle_r']

            part_arg = {b: a for a, b in enumerate(coco_part_labels)}

            part_orders = [('nose', 'eye_l'), ('eye_l', 'eye_r'),
                           ('eye_r', 'nose'), ('eye_l', 'ear_l'),
                           ('eye_r', 'ear_r'), ('ear_l', 'sholder_l'),
                           ('ear_r', 'sholder_r'), ('sholder_l', 'sholder_r'),
                           ('sholder_l', 'hip_l'), ('sholder_r', 'hip_r'),
                           ('hip_l', 'hip_r'), ('sholder_l', 'elbow_l'),
                           ('elbow_l', 'wrist_l'), ('sholder_r', 'elbow_r'),
                           ('elbow_r', 'wrist_r'), ('hip_l', 'knee_l'),
                           ('knee_l', 'ankle_l'), ('hip_r', 'knee_r'),
                           ('knee_r', 'ankle_r')]

            joint_parts_info = {'part_labels': part_labels,
                                'part_arg': part_arg,
                                'part_orders': part_orders}
        return joint_parts_info
