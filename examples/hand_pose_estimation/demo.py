import argparse
import tensorflow as tf
import numpy as np

from HandPoseEstimation import ColorHandPoseNet

model = ColorHandPoseNet()
coords_xyz_rel_normed, keypoints_scoremap, hand_mask, image_crop, center, \
scale_crop = model.predict([img, np.array([[1.0, 0.0]])], batch_size=1)
keypoints_resized = tf.compat.v1.image.resize_images(keypoints_scoremap[-1],
                                                     size=(256, 256))
keypoint_coords3d = np.squeeze(coords_xyz_rel_normed)
keypoint_coords_crop = detect_keypoints(keypoints_resized)
keypoint_coords = transform_cropped_coords(keypoint_coords_crop, center,
                                           scale_crop, 256)
