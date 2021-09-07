import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Reshape, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras import Model
from backend import extract_bounding_box, flip_right_hand
from backend import crop_image_from_coordinates, object_scoremap
from backend import get_rotation_matrix


def Hand_Segmentation_Net(input_shape=(320, 320, 3), load_pretrained=True):
    image = Input(shape=input_shape, name='image')

    X_1 = Conv2D(64, kernel_size=3, padding='same',
                 name='conv1_1')(image)
    X_1 = LeakyReLU(alpha=0.01)(X_1)

    X_2 = Conv2D(64, 3, padding='same', name='conv1_2')(X_1)
    X_2 = LeakyReLU(alpha=0.01)(X_2)

    X_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X_2)

    X_4 = Conv2D(128, 3, padding='same', name='conv1_3')(X_3)
    X_4 = LeakyReLU(alpha=0.01)(X_4)

    X_5 = Conv2D(128, 3, padding='same', name='conv1_4')(X_4)
    X_5 = LeakyReLU(alpha=0.01)(X_5)

    X_6 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X_5)

    X_7 = Conv2D(256, 3, padding='same', name='conv1_5')(X_6)
    X_7 = LeakyReLU(alpha=0.01)(X_7)

    X_8 = Conv2D(256, 3, padding='same', name='conv1_6')(X_7)
    X_8 = LeakyReLU(alpha=0.01)(X_8)

    X_9 = Conv2D(256, 3, padding='same', name='conv1_7')(X_8)
    X_9 = LeakyReLU(alpha=0.01)(X_9)

    X_10 = Conv2D(256, 3, padding='same', name='conv1_8')(X_9)
    X_10 = LeakyReLU(alpha=0.01)(X_10)

    X_11 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X_10)

    X_12 = Conv2D(512, 3, padding='same', name='conv1_9')(X_11)
    X_12 = LeakyReLU(alpha=0.01)(X_12)

    X_13 = Conv2D(512, 3, padding='same', name='conv1_10')(X_12)
    X_13 = LeakyReLU(alpha=0.01)(X_13)

    X_14 = Conv2D(512, 3, padding='same', name='conv1_11')(X_13)
    X_14 = LeakyReLU(alpha=0.01)(X_14)

    X_15 = Conv2D(512, 3, padding='same', name='conv1_12')(X_14)
    X_15 = LeakyReLU(alpha=0.01)(X_15)

    X_16 = Conv2D(512, 3, padding='same', name='conv1_13')(X_15)
    X_16 = LeakyReLU(alpha=0.01)(X_16)

    X_17 = Conv2D(128, 3, padding='same', name='conv1_14')(X_16)
    X_17 = LeakyReLU(alpha=0.01)(X_17)

    X_18 = Conv2D(512, 1, padding='same', name='conv1_15')(X_17)
    X_18 = LeakyReLU(alpha=0.01)(X_18)

    raw_segmented_image = Conv2D(2, 1, padding='same', activation=None,
                                 name='conv1_16')(X_18)

    segmentation_net = Model(inputs={'image': image},
                             outputs={'image': image,
                                      'raw_segmentation_map':
                                          raw_segmented_image},
                             name='HandSegNet')

    if load_pretrained:
        weight_path = segmentation_net.name + '-pretrained_weights.h5'
        segmentation_net.load_weights(weight_path)

    return segmentation_net


def PoseNet(input_shape=None, load_pretrained=True):
    if input_shape is None:
        input_shape = [256, 256, 3]
    scoremap_list = list()
    cropped_image = Input(shape=input_shape, name='cropped_image')

    X_1 = Conv2D(64, kernel_size=3, padding='same', name='conv2_1')(
        cropped_image)
    X_1 = LeakyReLU(alpha=0.01)(X_1)

    X_2 = Conv2D(64, kernel_size=3, padding='same', name='conv2_2')(X_1)
    X_2 = LeakyReLU(alpha=0.01)(X_2)

    X_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X_2)

    X_4 = Conv2D(128, kernel_size=3, padding='same', name='conv2_3')(X_3)
    X_4 = LeakyReLU(alpha=0.01)(X_4)

    X_5 = Conv2D(128, kernel_size=3, padding='same', name='conv2_4')(X_4)
    X_5 = LeakyReLU(alpha=0.01)(X_5)

    X_6 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X_5)

    X_7 = Conv2D(256, kernel_size=3, padding='same', name='conv2_5')(X_6)
    X_7 = LeakyReLU(alpha=0.01)(X_7)

    X_8 = Conv2D(256, kernel_size=3, padding='same', name='conv2_6')(X_7)
    X_8 = LeakyReLU(alpha=0.01)(X_8)

    X_9 = Conv2D(256, kernel_size=3, padding='same', name='conv2_7')(X_8)
    X_9 = LeakyReLU(alpha=0.01)(X_9)

    X_10 = Conv2D(256, kernel_size=3, padding='same', name='conv2_8')(X_9)
    X_10 = LeakyReLU(alpha=0.01)(X_10)

    X_11 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X_10)

    X_12 = Conv2D(512, kernel_size=3, padding='same', name='conv2_9')(X_11)
    X_12 = LeakyReLU(alpha=0.01)(X_12)

    X_13 = Conv2D(512, kernel_size=3, padding='same', name='conv2_10')(X_12)
    X_13 = LeakyReLU(alpha=0.01)(X_13)

    X_14 = Conv2D(256, kernel_size=3, padding='same', name='conv2_11')(X_13)
    X_14 = LeakyReLU(alpha=0.01)(X_14)

    X_15 = Conv2D(256, kernel_size=3, padding='same', name='conv2_12')(X_14)
    X_15 = LeakyReLU(alpha=0.01)(X_15)

    X_16 = Conv2D(256, kernel_size=3, padding='same', name='conv2_13')(X_15)
    X_16 = LeakyReLU(alpha=0.01)(X_16)

    X_17 = Conv2D(256, kernel_size=3, padding='same', name='conv2_14')(X_16)
    X_17 = LeakyReLU(alpha=0.01)(X_17)

    X_18 = Conv2D(128, kernel_size=3, padding='same', name='conv2_15')(X_17)
    X_18 = LeakyReLU(alpha=0.01)(X_18)

    X_19 = Conv2D(512, kernel_size=1, name='conv2_16')(X_18)
    X_19 = LeakyReLU(alpha=0.01)(X_19)

    X_20 = Conv2D(21, kernel_size=1, name='conv2_17')(X_19)
    scoremap_list.append(X_20)

    X_21 = Concatenate(axis=3)([scoremap_list[-1], X_18])

    X_22 = Conv2D(128, kernel_size=7, padding='same',
                  name='conv2_18')(X_21)
    X_22 = LeakyReLU(alpha=0.01)(X_22)

    X_23 = Conv2D(128, kernel_size=7, padding='same', name='conv2_19')(X_22)
    X_23 = LeakyReLU(alpha=0.01)(X_23)

    X_24 = Conv2D(128, kernel_size=7, padding='same', name='conv2_20')(X_23)
    X_24 = LeakyReLU(alpha=0.01)(X_24)

    X_25 = Conv2D(128, kernel_size=7, padding='same', name='conv2_21')(X_24)
    X_25 = LeakyReLU(alpha=0.01)(X_25)

    X_26 = Conv2D(128, kernel_size=7, padding='same', name='conv2_22')(X_25)
    X_26 = LeakyReLU(alpha=0.01)(X_26)

    X_27 = Conv2D(128, kernel_size=1, name='conv2_23')(X_26)
    X_27 = LeakyReLU(alpha=0.01)(X_27)

    X_28 = Conv2D(21, kernel_size=1, padding='same', name='conv2_24')(X_27)

    scoremap_list.append(X_28)

    X_29 = Concatenate(axis=3)([scoremap_list[-1], X_18])

    X_30 = Conv2D(128, kernel_size=7, padding='same', name='conv2_25')(X_29)
    X_30 = LeakyReLU(alpha=0.01)(X_30)

    X_31 = Conv2D(128, kernel_size=7, padding='same', name='conv2_26')(X_30)
    X_31 = LeakyReLU(alpha=0.01)(X_31)

    X_32 = Conv2D(128, kernel_size=7, padding='same', name='conv2_27')(X_31)
    X_32 = LeakyReLU(alpha=0.01)(X_32)

    X_33 = Conv2D(128, kernel_size=7, padding='same', name='conv2_28')(X_32)
    X_33 = LeakyReLU(alpha=0.01)(X_33)

    X_34 = Conv2D(128, kernel_size=7, padding='same', name='conv2_29')(X_33)
    X_34 = LeakyReLU(alpha=0.01)(X_34)

    X_35 = Conv2D(128, kernel_size=1, name='conv2_30')(X_34)
    X_35 = LeakyReLU(alpha=0.01)(X_35)

    score_maps = Conv2D(21, kernel_size=1, name='conv2_31')(X_35)
    scoremap_list.append(score_maps)

    PoseNet = Model(inputs={'cropped_image': cropped_image},
                    outputs={'score_maps': scoremap_list[-1]}, name='PoseNet')

    if load_pretrained:
        weight_path = PoseNet.name + '-pretrained_weights.h5'
        PoseNet.load_weights(weight_path)

    return PoseNet


def PosePriorNet(keypoint_heatmaps_shape=None, hand_side_shape=None,
                 num_keypoints=21, load_pretrained=True):
    if hand_side_shape is None:
        hand_side_shape = [2]
    if keypoint_heatmaps_shape is None:
        keypoint_heatmaps_shape = [32, 32, 21]
    score_maps = Input(shape=keypoint_heatmaps_shape)
    hand_side = Input(shape=hand_side_shape)

    X_1 = Conv2D(32, 3, padding='same', name='conv3_1')(score_maps)
    X_1 = LeakyReLU(alpha=0.01)(X_1)

    X_2 = Conv2D(32, 3, padding='same', strides=2, name='conv3_2')(X_1)
    X_2 = LeakyReLU(alpha=0.01)(X_2)

    X_3 = Conv2D(64, 3, padding='same', name='conv3_3')(X_2)
    X_3 = LeakyReLU(alpha=0.01)(X_3)

    X_4 = Conv2D(64, 3, padding='same', strides=2, name='conv3_4')(X_3)
    X_4 = LeakyReLU(alpha=0.01)(X_4)

    X_5 = Conv2D(128, 3, padding='same', name='conv3_5')(X_4)
    X_5 = LeakyReLU(alpha=0.01)(X_5)

    X_6 = Conv2D(128, 3, padding='same', strides=2, name='conv3_6')(X_5)
    X_6 = LeakyReLU(alpha=0.01)(X_6)

    X_7 = Reshape([-1])(X_6)
    X_7 = Concatenate(axis=1)([X_7, hand_side])

    X_8 = Dense(512, name='dense3_1')(X_7)
    X_8 = LeakyReLU(alpha=0.01)(X_8)
    X_8 = Dropout(rate=0.2)(X_8)

    X_9 = Dense(512, name='dense3_2')(X_8)
    X_9 = LeakyReLU(alpha=0.01)(X_9)
    X_9 = Dropout(rate=0.2)(X_9)

    X_10 = Dense(num_keypoints * 3, name='dense3_3')(X_9)

    hand_keypoints = Reshape((21, 3), name='reshape3_1')(X_10)
    PosePriorNet = Model(inputs={'score_maps': score_maps,
                                 'hand_side': hand_side},
                         outputs={'canonical_coordinates': hand_keypoints},
                         name='PosePriorNet')
    if load_pretrained:
        weight_path = PosePriorNet.name + '-pretrained_weights.h5'
        PosePriorNet.load_weights(weight_path)

    return PosePriorNet


def ViewPointNet(keypoint_heat_maps_shape=None, hand_side_shape=None,
                 load_pretrained=True):
    if hand_side_shape is None:
        hand_side_shape = [2]
    if keypoint_heat_maps_shape is None:
        keypoint_heat_maps_shape = [32, 32, 21]
    score_maps = Input(shape=keypoint_heat_maps_shape,
                              name='score_maps')
    hand_side = Input(shape=hand_side_shape, name='hand_side')

    X_1 = Conv2D(64, 3, padding='same')(score_maps)
    X_1 = LeakyReLU(alpha=0.01)(X_1)

    X_2 = Conv2D(64, 3, strides=2, padding='same')(X_1)
    X_2 = LeakyReLU(alpha=0.01)(X_2)

    X_3 = Conv2D(128, 3, padding='same')(X_2)
    X_3 = LeakyReLU(alpha=0.01)(X_3)

    X_4 = Conv2D(128, 3, strides=2, padding='same')(X_3)
    X_4 = LeakyReLU(alpha=0.01)(X_4)

    X_5 = Conv2D(256, 3, padding='same')(X_4)
    X_5 = LeakyReLU(alpha=0.01)(X_5)

    X_6 = Conv2D(256, 3, strides=2, padding='same')(X_5)
    X_6 = LeakyReLU(alpha=0.01)(X_6)
    X_6 = Reshape([-1])(X_6)
    X_6 = Concatenate(axis=1)([X_6, hand_side])

    X_7 = Dense(256)(X_6)
    X_7 = LeakyReLU(alpha=0.01)(X_7)

    X_8 = Dense(128)(X_7)
    X_8 = LeakyReLU(alpha=0.01)(X_8)

    ux = Dense(1)(X_8)
    uy = Dense(1)(X_8)
    uz = Dense(1)(X_8)

    axis_angles = Concatenate(axis=1)([ux, uy, uz])

    ViewPointNet = Model(inputs={'score_maps': score_maps,
                                 'hand_side': hand_side},
                         outputs={'rotation_parameters': axis_angles,
                                  'hand_side': hand_side},
                         name='ViewPointNet')
    if load_pretrained:
        weight_path = ViewPointNet.name + '-pretrained_weights.h5'
        ViewPointNet.load_weights(weight_path)

    return ViewPointNet


def ColorHandPoseNet(image_shape=None, hand_side_shape=None,
                     use_pretrained=False, crop_size=None, num_keypoints=None):
    if image_shape is None:
        image_shape = [320, 320, 3]
    if hand_side_shape is None:
        hand_side_shape = [2]

    image = Input(shape=image_shape, name='image')
    hand_side = Input(shape=hand_side_shape, name='hand_side')

    HandSegNet = Hand_Segmentation_Net(load_pretrained=use_pretrained)
    HandPoseNet = PoseNet(load_pretrained=use_pretrained)
    HandPosePriorNet = PosePriorNet(load_pretrained=use_pretrained)
    HandViewPointNet = ViewPointNet(load_pretrained=use_pretrained)

    if crop_size is None:
        crop_size = 256
    else:
        crop_size = crop_size

    if num_keypoints is None:
        num_keypoints = 21
    else:
        num_keypoints = num_keypoints

    raw_segmented_image = HandSegNet({'image': image})
    hand_scoremap = tf.image.resize(raw_segmented_image,
                                    (image_shape[0], image_shape[1]))
    hand_mask = object_scoremap(hand_scoremap)
    center, _, crop_size_best = extract_bounding_box(hand_mask)
    crop_size_best *= 1.25
    scale_crop = tf.minimum(tf.maximum(crop_size / crop_size_best, 0.25), 5.0)
    image_crop = crop_image_from_coordinates(image, center, crop_size,
                                             scale=scale_crop)

    keypoints_scoremap = HandPoseNet({'cropped_image': image_crop})

    canonical_coordinates = HandPosePriorNet(
        {'score_maps': keypoints_scoremap[-1], 'hand_side': hand_side})

    rotation_parameters = HandViewPointNet(
        {'score_maps': keypoints_scoremap[-1], 'hand_side': hand_side})

    rotation_matrix = get_rotation_matrix(rotation_parameters)

    cond_right = tf.equal(tf.argmax(hand_side, 1), 1)
    cond_right_all = tf.tile(tf.reshape(cond_right, [-1, 1, 1]),
                             [1, num_keypoints, 3])

    coord_xyz_can_flip = flip_right_hand(canonical_coordinates, cond_right_all)

    coord_xyz_rel_normed = tf.matmul(coord_xyz_can_flip, rotation_matrix)

    ColorHandPoseNet = Model([image, hand_side],
                             [coord_xyz_rel_normed, keypoints_scoremap,
                              hand_mask, image_crop, center, scale_crop],
                             name='ColorHandPoseNet')

    return ColorHandPoseNet


if __name__ == '__main__':
    ColorHandPoseNet()
