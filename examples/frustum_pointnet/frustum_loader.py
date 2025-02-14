import tensorflow as tf
import tensorflow.python.keras.backend as K
import numpy as np

from dataset_utils import NUM_HEADING_BIN
from paz.optimization.losses.frustumpointnet_loss import ExtractBox3DCorners


class frustum_data_loader(object):
    def __init__(self, batch_size=32):
        # g_type_object_of_interest = ['animal', 'bicycle', 'bus', 'car',
        #                              'emergency_vehicle', 'motorcycle',
        #                              'other_vehicle', 'pedestrian', 'truck']
        self.g_type_object_of_interest = ['car', 'cyclist', 'pedestrian']
        self.NUM_CLASS = len(self.g_type_object_of_interest)
        self.NUM_POINT = 1024
        self.NUM_CHANNELS_OF_PC = 3
        self.batch_size = batch_size

    def parse_data(self, raw_record):

        example = self.parse_frustum_point_record(raw_record)
        return example['frustum_point_cloud'], \
               tf.cast(example['one_hot_vec'], tf.float32), \
               tf.cast(example['seg_label'], tf.int32), \
               example['box3d_center'], \
               tf.cast(example['angle_class'], tf.int32), \
               example['angle_residual'], \
               tf.cast(example['size_class'], tf.int32), \
               example['size_residual']

    def parse_frustum_point_record(self, tfexample_message: str):

        keys_to_features = {
            "size_class": tf.io.FixedLenFeature((), tf.int64,
                                                tf.zeros((), tf.int64)),
            "size_residual": tf.io.FixedLenFeature((3,), tf.float32,
                                                   tf.zeros((3,), tf.float32)),
            "seg_label": tf.io.FixedLenFeature((self.NUM_POINT,), tf.int64,
                                               tf.zeros((self.NUM_POINT,),
                                                        tf.int64)),
            "frustum_point_cloud": tf.io.FixedLenFeature(
                (self.NUM_POINT, self.NUM_CHANNELS_OF_PC), tf.float32),
            "rot_angle": tf.io.FixedLenFeature((), tf.float32,
                                               tf.zeros((), tf.float32)),
            "angle_class": tf.io.FixedLenFeature((), tf.int64,
                                                 tf.zeros((), tf.int64)),
            "angle_residual": tf.io.FixedLenFeature((), tf.float32,
                                                    tf.zeros((), tf.float32)),
            "one_hot_vec": tf.io.FixedLenFeature((self.NUM_CLASS,), tf.int64),
            "box3d_center": tf.io.FixedLenFeature((3,), tf.float32,
                                                  tf.zeros((3,), tf.float32)),
        }
        parsed_example = tf.io.parse_single_example(tfexample_message,
                                                    keys_to_features)
        return parsed_example

    def parse_test_data(self, raw_record):
        example = self.parse_frustum_point_test_record(raw_record)
        print(example)
        return example['frustum_point_cloud'], tf.cast(example['one_hot_vec'],
                                                       tf.float32), \
               tf.cast(example['rot_angle'], tf.float32), tf.cast(
            example['prob'], tf.float32), \
               example['type_name'], example['sample_token'], example['box_2d']

    def parse_frustum_point_test_record(self, tfexample_message: str):

        keys_to_features = {
            "frustum_point_cloud": tf.io.FixedLenFeature(
                (self.NUM_POINT, self.NUM_CHANNELS_OF_PC), tf.float32),
            "rot_angle": tf.io.FixedLenFeature((), tf.float32,
                                               tf.zeros((), tf.float32)),
            "one_hot_vec": tf.io.FixedLenFeature((self.NUM_CLASS,), tf.int64),
            "prob": tf.io.FixedLenFeature((), tf.float32,
                                          tf.zeros((), tf.float32)),
            "type_name": tf.io.FixedLenFeature((), tf.int64),
            "sample_token": tf.io.FixedLenFeature((), tf.int64),
            "box_2d": tf.io.FixedLenFeature((4,), tf.float32)
        }
        parsed_example = tf.io.parse_single_example(tfexample_message,
                                                    keys_to_features)
        return parsed_example

    def load_data(self, tfrec_path, operation):
        if operation == 'train':
            train_dataset = tf.data.TFRecordDataset(tfrec_path)
            parsed_train_dataset = train_dataset.map(self.parse_data)
            parsed_train_dataset = parsed_train_dataset.batch(
                self.batch_size, drop_remainder=True)
            return parsed_train_dataset
        elif operation == 'validation':
            val_dataset = tf.data.TFRecordDataset(tfrec_path)
            parsed_val_dataset = val_dataset.map(self.parse_data)
            parsed_val_dataset = parsed_val_dataset.batch(
                self.batch_size, drop_remainder=True)
            return parsed_val_dataset
        else:
            test_dataset = tf.data.TFRecordDataset(tfrec_path)
            parsed_test_dataset = test_dataset.map(self.parse_test_data)
            parsed_test_dataset = parsed_test_dataset.batch(
                self.batch_size, drop_remainder=True)
            return parsed_test_dataset


def make_train_iterator(dataset):
    iterator = dataset.make_one_shot_iterator()
    next_val = iterator.get_next()

    with K.get_session().as_default() as sess:
        while True:
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl = sess.run(next_val)

            hcls_onehot = tf.one_hot(tf.cast(heading_class_label_pl, tf.int64),
                                     depth=NUM_HEADING_BIN, on_value=1,
                                     off_value=0,
                                     axis=-1)  # BxNUM_HEADING_BIN
            heading_residual_normalized_label = heading_residual_label_pl / (
                        np.pi / NUM_HEADING_BIN)

            corners_3d = ExtractBox3DCorners(centers_pl,
                                             heading_residual_label_pl,
                                             size_residual_label_pl)

            x_train = {'frustum_point_cloud': pointclouds_pl,
                       'one_hot_vector': one_hot_vec_pl}

            y_train = {'seg_logits': labels_pl,
                       'center': centers_pl,
                       'seg_pc_centroid': centers_pl,
                       'heading_class': heading_class_label_pl,
                       'heading residuals': heading_residual_label_pl,
                       'size_class': size_class_label_pl,
                       'size_residuals': size_residual_label_pl,
                       'corners_3d_pred': corners_3d}

            yield x_train, y_train


def make_test_iterator(dataset):
    iterator = dataset.make_one_shot_iterator()
    next_val = iterator.get_next()

    with K.get_session().as_default() as sess:
        while True:
            pointclouds_pl, one_hot_vec_pl, rot_angle, prob_pl, class_pl, \
            tokens_pl, box2d_pl = sess.run(next_val)
            data_dict = {"frustum_point_cloud": pointclouds_pl,
                         "one_hot_vector": one_hot_vec_pl,
                         "rot_angle": rot_angle,
                         "rgb_prob": prob_pl, "cls_type": class_pl,
                         "token": tokens_pl,
                         "box_2D": box2d_pl}

            yield data_dict
