import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Reshape, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, \
    LeakyReLU
from tensorflow.keras import Model
from backend import extract_bounding_box, crop_image_from_xy, flip_right_hand


def Hand_Segmentation_Net(input_shape=None):
    if input_shape is None:
        input_shape = [320, 320, 3]
    image = Input(shape=input_shape, name='image')
    X_1 = Conv2D(64, kernel_size=3, activation='relu', padding='same',
                 name='conv1_1')(image)
    X_2 = Conv2D(64, 3, activation='relu', padding='same', name='conv1_2')(X_1)
    X_3 = MaxPooling2D(pool_size=(2, 2))(X_2)
    X_4 = Conv2D(128, 3, padding='same', activation='relu', name='conv1_3')(X_3)
    X_5 = Conv2D(128, 3, padding='same', activation='relu', name='conv1_4')(X_4)
    X_6 = MaxPooling2D(pool_size=(2, 2))(X_5)
    X_7 = Conv2D(256, 3, padding='same', activation='relu', name='conv1_5')(X_6)
    X_8 = Conv2D(256, 3, padding='same', activation='relu', name='conv1_6')(X_7)
    X_9 = Conv2D(256, 3, padding='same', activation='relu', name='conv1_7')(X_8)
    X_10 = Conv2D(256, 3, padding='same', activation='relu',
                  name='conv1_8')(X_9)
    X_11 = MaxPooling2D(pool_size=(2, 2))(X_10)

    X_12 = Conv2D(512, 3, padding='same', activation='relu',
                  name='conv1_9')(X_11)
    X_13 = Conv2D(512, 3, padding='same', activation='relu',
                  name='conv1_10')(X_12)
    X_14 = Conv2D(512, 3, padding='same', activation='relu',
                  name='conv1_11')(X_13)
    X_15 = Conv2D(512, 3, padding='same', activation='relu',
                  name='conv1_12')(X_14)
    X_16 = Conv2D(512, 3, padding='same', activation='relu',
                  name='conv1_13')(X_15)
    X_17 = Conv2D(2, 1, padding='same', activation=None, name='conv1_14')(X_16)
    raw_segmented_image = UpSampling2D(size=(8, 8), interpolation='bilinear')(
        X_17)

    segmentation_net = Model(inputs=image, outputs=raw_segmented_image,
                             name='HandSegNet')

    return segmentation_net


def PoseNet(input_shape=None):
    if input_shape is None:
        input_shape = [256, 256]
    segmented_image = Input(shape=input_shape, name='segmented_image')
    X_1 = Conv2D(64, 3, activation='relu', padding='same',
                 name='conv2_1')(segmented_image)
    X_2 = Conv2D(64, 3, activation='relu', padding='same', name='conv2_2')(X_1)
    X_3 = MaxPooling2D(pool_size=(2, 2))(X_2)
    X_4 = Conv2D(128, 3, activation='relu', padding='same', name='conv2_3')(X_3)
    X_5 = Conv2D(128, 3, activation='relu', padding='same', name='conv2_4')(X_4)
    X_6 = MaxPooling2D(pool_size=(2, 2))(X_5)
    X_7 = Conv2D(256, 3, activation='relu', padding='same', name='conv2_5')(X_6)
    X_8 = Conv2D(256, 3, activation='relu', padding='same', name='conv2_6')(X_7)
    X_9 = Conv2D(256, 3, activation='relu', padding='same', name='conv2_7')(X_8)
    X_10 = Conv2D(256, 3, activation='relu', padding='same',
                  name='conv2_8')(X_9)
    X_11 = MaxPooling2D(pool_size=(2, 2))(X_10)
    X_12 = Conv2D(512, 3, activation='relu', padding='same',
                  name='conv2_9')(X_11)
    X_13 = Conv2D(512, 3, activation='relu', padding='same',
                  name='conv2_10')(X_12)
    X_14 = Conv2D(512, 3, activation='relu', padding='same',
                  name='conv2_11')(X_13)
    X_15 = Conv2D(512, 3, activation='relu', padding='same',
                  name='conv2_12')(X_14)
    X_16 = Conv2D(512, 3, activation='relu', padding='same',
                  name='conv2_13')(X_15)
    X_17 = Conv2D(21, 1, padding='same', activation=None, name='conv2_14')(X_16)

    X_18 = Concatenate(axis=-1)([X_16, X_17])

    X_19 = Conv2D(128, 7, activation='relu', padding='same',
                  name='conv2_15')(X_18)
    X_20 = Conv2D(128, 7, activation='relu', padding='same',
                  name='conv2_16')(X_19)
    X_21 = Conv2D(128, 7, activation='relu', padding='same',
                  name='conv2_17')(X_20)
    X_22 = Conv2D(128, 7, activation='relu', padding='same',
                  name='conv2_18')(X_21)
    X_23 = Conv2D(128, 7, activation='relu', padding='same',
                  name='conv2_19')(X_22)
    X_24 = Conv2D(21, 1, padding='same', activation=None, name='conv2_20')(X_23)

    X_25 = Concatenate(axis=-1)([X_16, X_17, X_24])

    X_26 = Conv2D(128, 7, activation='relu', padding='same',
                  name='conv2_21')(X_25)
    X_27 = Conv2D(128, 7, activation='relu', padding='same',
                  name='conv2_22')(X_26)
    X_28 = Conv2D(128, 7, activation='relu', padding='same',
                  name='conv2_23')(X_27)
    X_29 = Conv2D(128, 7, activation='relu', padding='same',
                  name='conv2_24')(X_28)
    X_30 = Conv2D(128, 7, activation='relu', padding='same',
                  name='conv2_25')(X_29)
    score_maps = Conv2D(21, 1, padding='same', activation=None,
                        name='conv2_26')(X_30)

    PoseNet = Model(inputs=segmented_image, outputs=score_maps, name='PoseNet')

    return PoseNet


def PosePriorNet(score_maps_shape=None, hand_side_shape=None,
                 num_keypoints=21):
    if hand_side_shape is None:
        hand_side_shape = [2]
    if score_maps_shape is None:
        score_maps_shape = [21, 256, 256]
    score_maps = Input(shape=score_maps_shape, name='score_maps')
    hand_side = Input(shape=hand_side_shape, name='hand_side')
    batch_size = score_maps.shape[0]
    X_1 = Conv2D(32, 3, activation='relu', padding='same',
                 name='conv3_1')(score_maps)
    X_2 = Conv2D(32, 3, activation='relu', padding='same', strides=2,
                 name='conv3_2')(X_1)
    X_3 = Conv2D(64, 3, activation='relu', padding='same',
                 name='conv3_3')(X_2)
    X_4 = Conv2D(64, 3, activation='relu', padding='same', strides=2,
                 name='conv3_4')(X_3)
    X_5 = Conv2D(128, 3, activation='relu', padding='same',
                 name='conv3_5')(X_4)
    X_6 = Conv2D(128, 3, padding='same', activation=None, strides=2,
                 name='conv3_6')(X_5)

    X_7 = Reshape([batch_size, -1])(X_6)
    X_7 = Concatenate(axis=1)([X_7, hand_side])

    X_8 = Dense(512, activation='relu')(X_7)
    X_8 = Dropout(rate=0.2)(X_8)

    X_9 = Dense(512, activation='relu')(X_8)
    X_9 = Dropout(rate=0.2)(X_9)

    X_10 = Dense(num_keypoints)(X_9)
    hand_keypoints = Reshape([21, 3])(X_10)
    PosePriorNet = Model(inputs=[score_maps, hand_side], outputs=hand_keypoints,
                         name='PosePriorNet')
    return PosePriorNet


def ViewPointNet(keypoint_heat_maps_shape=None, hand_side_shape=None):
    if hand_side_shape is None:
        hand_side_shape = [2]
    if keypoint_heat_maps_shape is None:
        keypoint_heat_maps_shape = [21, 256, 256]
    keypoint_heat_maps = Input(shape=keypoint_heat_maps_shape, name='heat_maps')
    hand_side = Input(shape=hand_side_shape, name='hand_side')
    batch_size = keypoint_heat_maps.shape[0]

    X_1 = Conv2D(64, 3)(keypoint_heat_maps)
    X_1 = LeakyReLU()(X_1)

    X_2 = Conv2D(64, 3, strides=2)(X_1)
    X_2 = LeakyReLU()(X_2)

    X_3 = Conv2D(128, 3)(X_2)
    X_3 = LeakyReLU()(X_3)

    X_4 = Conv2D(128, 3, strides=2)(X_3)
    X_4 = LeakyReLU()(X_4)

    X_5 = Conv2D(256, 3)(X_4)
    X_5 = LeakyReLU()(X_5)

    X_6 = Conv2D(128, 3, strides=2)(X_5)
    X_6 = LeakyReLU()(X_6)

    X_6 = Reshape([batch_size, -1])(X_6)
    X_6 = Concatenate(axis=1)([X_6, hand_side])

    X_7 = Dense(256)(X_6)
    X_7 = LeakyReLU()(X_7)

    X_8 = Dense(128)(X_7)
    X_8 = LeakyReLU()(X_8)

    ux = Dense(1)(X_8)
    uy = Dense(1)(X_8)
    uz = Dense(1)(X_8)

    axis_angles = Concatenate(axis=1)([ux, uy, uz])
    ViewPointNet = Model(inputs=[keypoint_heat_maps, hand_side],
                         outputs=axis_angles, name='ViewPointNet')

    return ViewPointNet


def ColorHandPoseNet(image_shape=None, hand_side_shape=None, weight_path=None,
                     crop_size=None, num_keypoints=None):
    if image_shape is None:
        image_shape = [320, 320, 3]
    if hand_side_shape is None:
        hand_side_shape = [2]
    HandSegNet = Hand_Segmentation_Net()
    HandPoseNet = PoseNet()
    HandPosePriorNet = PosePriorNet()
    HandViewPointNet = ViewPointNet()
    if weight_path is not None:
        HandSegNet.load_weights(weight_path + 'hand_segnet.ckpt')
        HandPoseNet.load_weights(weight_path + 'hand_posenet.ckpt')
        HandPosePriorNet.load_weights(weight_path + 'hand_pose_prior_net.ckpt')
        HandViewPointNet.load_weights(weight_path + 'hand_viewpoint_net.ckpt')
    else:
        raise AttributeError('weight_path cannot be {}'.format(weight_path))

    if crop_size is None:
        crop_size = 256
    else:
        crop_size = crop_size

    if num_keypoints is None:
        num_keypoints = 21
    else:
        num_keypoints = num_keypoints

    image = Input(shape=image_shape, name='image')
    hand_side = Input(shape=hand_side_shape, name='hand_side')

    hand_scoremap = HandSegNet(image)
    hand_scoremap = tf.nn.softmax(hand_scoremap)
    hand_mask = tf.argmax(hand_scoremap)

    center, _, crop_size_best = extract_bounding_box(hand_mask)
    crop_size_best *= 1.25
    scale_crop = tf.minimum(tf.maximum(crop_size / crop_size_best, 0.25), 5.0)
    image_crop = crop_image_from_xy(image, center, crop_size, scale=scale_crop)

    keypoints_scoremap = HandPoseNet(image_crop)
    keypoints_scoremap = keypoints_scoremap[-1]

    canonical_coordinates = HandPosePriorNet(keypoints_scoremap, hand_side)

    rotation_parameters = HandViewPointNet(keypoints_scoremap, hand_side)

    cond_right = tf.equal(tf.argmax(hand_side, 1), 1)
    cond_right_all = tf.tile(tf.reshape(cond_right, [-1, 1, 1]),
                             [1, num_keypoints, 3])
    coord_xyz_can_flip = flip_right_hand(canonical_coordinates, cond_right_all)

    # rotate view back
    coord_xyz_rel_normed = tf.matmul(coord_xyz_can_flip, rotation_parameters)

    return coord_xyz_rel_normed


if __name__ == '__main__':
    ColorHandPoseNet()
