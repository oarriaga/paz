import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Reshape, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras import Model
from tensorflow.keras.utils import get_file


BASE_WEIGHT_PATH = ('https://github.com/oarriaga/altamira-data/'
                    'releases/download/v0.1/')


def HandSegmentationNet(input_shape=(320, 320, 3), weights='RHD'):
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

    if weights is not None:
        model_name = '_'.join([segmentation_net.name, weights])

    if weights is not None:
        weights_url = BASE_WEIGHT_PATH + model_name + '_weights.hdf5'
        weights_path = get_file(os.path.basename(weights_url), weights_url,
                                cache_subdir='paz/models')
        segmentation_net.load_weights(weights_path)

    return segmentation_net


def PoseNet(input_shape=(256, 256, 3), weights='RHD'):
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

    if weights is not None:
        model_name = '_'.join([PoseNet.name, weights])

    if weights is not None:
        weights_url = BASE_WEIGHT_PATH + model_name + '_weights.hdf5'
        weights_path = get_file(os.path.basename(weights_url), weights_url,
                                cache_subdir='paz/models')
        PoseNet.load_weights(weights_path)

    return PoseNet


def PosePriorNet(keypoint_heatmaps_shape=(32, 32, 21), hand_side_shape=(2,),
                 num_keypoints=21, weights='RHD'):
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

    if weights is not None:
        model_name = '_'.join([PosePriorNet.name, weights])

    if weights is not None:
        weights_url = BASE_WEIGHT_PATH + model_name + '_weights.hdf5'
        weights_path = get_file(os.path.basename(weights_url), weights_url,
                                cache_subdir='paz/models')
        PosePriorNet.load_weights(weights_path)

    return PosePriorNet


def ViewPointNet(keypoint_heat_maps_shape=(32, 32, 21), hand_side_shape=(2,),
                 weights='RHD'):
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
    if weights is not None:
        model_name = '_'.join([ViewPointNet.name, weights])

    if weights is not None:
        weights_url = BASE_WEIGHT_PATH + model_name + '_weights.hdf5'
        weights_path = get_file(os.path.basename(weights_url), weights_url,
                                cache_subdir='paz/models')
        ViewPointNet.load_weights(weights_path)

    return ViewPointNet
