from tensorflow.keras.layers import Concatenate, Dense, Dropout, Reshape, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras import Model
from tensorflow.keras.utils import get_file

BASE_WEIGHT_PATH = (
    'https://github.com/oarriaga/altamira-data/releases/download/v0.11/')


def HandSegmentationNet(input_shape=(320, 320, 3), weights='RHDv2'):
    image = Input(shape=input_shape, name='image')

    X = Conv2D(64, kernel_size=3, padding='same', name='conv1_1')(image)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(64, 3, padding='same', name='conv1_2')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    X = Conv2D(128, 3, padding='same', name='conv1_3')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(128, 3, padding='same', name='conv1_4')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    X = Conv2D(256, 3, padding='same', name='conv1_5')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(256, 3, padding='same', name='conv1_6')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(256, 3, padding='same', name='conv1_7')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(256, 3, padding='same', name='conv1_8')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    X = Conv2D(512, 3, padding='same', name='conv1_9')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(512, 3, padding='same', name='conv1_10')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(512, 3, padding='same', name='conv1_11')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(512, 3, padding='same', name='conv1_12')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(512, 3, padding='same', name='conv1_13')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(128, 3, padding='same', name='conv1_14')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(512, 1, padding='same', name='conv1_15')(X)
    X = LeakyReLU(alpha=0.01)(X)

    raw_segmented_image = Conv2D(2, 1, padding='same', activation=None,
                                 name='conv1_16')(X)

    segmentation_net = Model(inputs={'image': image},
                             outputs={'image': image,
                                      'raw_segmentation_map':
                                          raw_segmented_image},
                             name='HandSegNet')

    if weights is not None:
        model_filename = [segmentation_net.name, str(weights)]
        model_filename = '_'.join(['-'.join(model_filename), 'weights.hdf5'])
        weights_path = get_file(model_filename,
                                BASE_WEIGHT_PATH + model_filename,
                                cache_subdir='paz/models')
        print('Loading %s model weights' % weights_path)
        segmentation_net.load_weights(weights_path)

    return segmentation_net


def PoseNet(input_shape=(256, 256, 3), weights='RHDv2'):
    cropped_image = Input(shape=input_shape, name='cropped_image')

    X = Conv2D(64, kernel_size=3, padding='same', name='conv2_1')(
        cropped_image)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(64, kernel_size=3, padding='same', name='conv2_2')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    X = Conv2D(128, kernel_size=3, padding='same', name='conv2_3')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(128, kernel_size=3, padding='same', name='conv2_4')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    X = Conv2D(256, kernel_size=3, padding='same', name='conv2_5')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(256, kernel_size=3, padding='same', name='conv2_6')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(256, kernel_size=3, padding='same', name='conv2_7')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(256, kernel_size=3, padding='same', name='conv2_8')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    X = Conv2D(512, kernel_size=3, padding='same', name='conv2_9')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(512, kernel_size=3, padding='same', name='conv2_10')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(256, kernel_size=3, padding='same', name='conv2_11')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(256, kernel_size=3, padding='same', name='conv2_12')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(256, kernel_size=3, padding='same', name='conv2_13')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(256, kernel_size=3, padding='same', name='conv2_14')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(128, kernel_size=3, padding='same', name='conv2_15')(X)
    X = LeakyReLU(alpha=0.01)(X)
    skip_connection = X

    X = Conv2D(512, kernel_size=1, name='conv2_16')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(21, kernel_size=1, name='conv2_17')(X)

    X = Concatenate(axis=3)([X, skip_connection])

    X = Conv2D(128, kernel_size=7, padding='same', name='conv2_18')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(128, kernel_size=7, padding='same', name='conv2_19')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(128, kernel_size=7, padding='same', name='conv2_20')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(128, kernel_size=7, padding='same', name='conv2_21')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(128, kernel_size=7, padding='same', name='conv2_22')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(128, kernel_size=1, name='conv2_23')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(21, kernel_size=1, padding='same', name='conv2_24')(X)

    X = Concatenate(axis=3)([X, skip_connection])

    X = Conv2D(128, kernel_size=7, padding='same', name='conv2_25')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(128, kernel_size=7, padding='same', name='conv2_26')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(128, kernel_size=7, padding='same', name='conv2_27')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(128, kernel_size=7, padding='same', name='conv2_28')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(128, kernel_size=7, padding='same', name='conv2_29')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(128, kernel_size=1, name='conv2_30')(X)
    X = LeakyReLU(alpha=0.01)(X)

    score_maps = Conv2D(21, kernel_size=1, name='conv2_31')(X)

    PoseNet = Model(inputs={'cropped_image': cropped_image},
                    outputs={'score_maps': score_maps}, name='PoseNet')

    if weights is not None:
        model_filename = [PoseNet.name, str(weights)]
        model_filename = '_'.join(['-'.join(model_filename), 'weights.hdf5'])
        weights_path = get_file(model_filename,
                                BASE_WEIGHT_PATH + model_filename,
                                cache_subdir='paz/models')
        print('Loading %s model weights' % weights_path)
        PoseNet.load_weights(weights_path)

    return PoseNet


def PosePriorNet(keypoint_heatmaps_shape=(32, 32, 21), hand_side_shape=(2,),
                 num_keypoints=21, weights='RHDv2'):
    score_maps = Input(shape=keypoint_heatmaps_shape)
    hand_side = Input(shape=hand_side_shape)

    X = Conv2D(32, 3, padding='same', name='conv3_1')(score_maps)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(32, 3, padding='same', strides=2, name='conv3_2')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(64, 3, padding='same', name='conv3_3')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(64, 3, padding='same', strides=2, name='conv3_4')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(128, 3, padding='same', name='conv3_5')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(128, 3, padding='same', strides=2, name='conv3_6')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Reshape([-1])(X)
    X = Concatenate(axis=1)([X, hand_side])

    X = Dense(512, name='dense3_1')(X)
    X = LeakyReLU(alpha=0.01)(X)
    X = Dropout(rate=0.2)(X)

    X = Dense(512, name='dense3_2')(X)
    X = LeakyReLU(alpha=0.01)(X)
    X = Dropout(rate=0.2)(X)

    X = Dense(num_keypoints * 3, name='dense3_3')(X)

    hand_keypoints = Reshape((21, 3), name='reshape3_1')(X)
    PosePriorNet = Model(inputs={'score_maps': score_maps,
                                 'hand_side': hand_side},
                         outputs={'canonical_coordinates': hand_keypoints},
                         name='PosePriorNet')

    if weights is not None:
        model_filename = [PosePriorNet.name, str(weights)]
        model_filename = '_'.join(['-'.join(model_filename), 'weights.hdf5'])
        weights_path = get_file(model_filename,
                                BASE_WEIGHT_PATH + model_filename,
                                cache_subdir='paz/models')
        print('Loading %s model weights' % weights_path)
        PosePriorNet.load_weights(weights_path)

    return PosePriorNet


def ViewPointNet(keypoint_heat_maps_shape=(32, 32, 21), hand_side_shape=(2,),
                 weights='RHDv2'):
    score_maps = Input(shape=keypoint_heat_maps_shape,
                       name='score_maps')
    hand_side = Input(shape=hand_side_shape, name='hand_side')

    X = Conv2D(64, 3, padding='same')(score_maps)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(64, 3, strides=2, padding='same')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(128, 3, padding='same')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(128, 3, strides=2, padding='same')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(256, 3, padding='same')(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(256, 3, strides=2, padding='same')(X)
    X = LeakyReLU(alpha=0.01)(X)
    X = Reshape([-1])(X)
    X = Concatenate(axis=1)([X, hand_side])

    X = Dense(256)(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Dense(128)(X)
    X = LeakyReLU(alpha=0.01)(X)

    ux = Dense(1)(X)
    uy = Dense(1)(X)
    uz = Dense(1)(X)

    axis_angles = Concatenate(axis=1)([ux, uy, uz])

    ViewPointNet = Model(inputs={'score_maps': score_maps,
                                 'hand_side': hand_side},
                         outputs={'rotation_parameters': axis_angles[0],
                                  'hand_side': hand_side},
                         name='ViewPointNet')

    if weights is not None:
        model_filename = [ViewPointNet.name, str(weights)]
        model_filename = '_'.join(['-'.join(model_filename), 'weights.hdf5'])
        weights_path = get_file(model_filename,
                                BASE_WEIGHT_PATH + model_filename,
                                cache_subdir='paz/models')
        print('Loading %s model weights' % weights_path)
        ViewPointNet.load_weights(weights_path)

    return ViewPointNet
