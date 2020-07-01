import os
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import get_file

from ..layers import Conv2DNormalization
from .utils import create_multibox_head
from .utils import create_prior_boxes


BASE_WEIGHT_PATH = ('https://github.com/oarriaga/altamira-data/'
                    'releases/download/v0.1/')


def SSD512(num_classes=81, weights='COCO', input_shape=(512, 512, 3),
           num_priors=[4, 6, 6, 6, 6, 4, 4], l2_loss=0.0005,
           return_base=False, trainable_base=True):
    """Single-shot-multibox detector for 512x512x3 BGR input images.
    # Arguments
        num_classes: Integer. Specifies the number of class labels.
        weights: String or None. If string should be a valid dataset name.
            Current valid datasets include `COCO` and `YCBVideo`.
        input_shape: List of integers. Input shape to the model including only
            spatial and channel resolution e.g. (512, 512, 3).
        num_priors: List of integers. Number of default box shapes
            used in each detection layer.
        l2_loss: Float. l2 regularization loss for convolutional layers.
        return_base: Boolean. If `True` the model returned is just
            the original base.
        trainable_base: Boolean. If `True` the base model
            weights are also trained.

    # Reference
        - [SSD: Single Shot MultiBox
            Detector](https://arxiv.org/abs/1512.02325)
    """

    datasets = {'COCO', 'YCBVideo', None}
    if not (weights in datasets or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `COCO`, '
                         'YCBVideo or the path to the weights '
                         'file to be loaded.')

    if weights == 'COCO' and num_classes != 81:
        raise ValueError('If using `weights` as `"COCO"` '
                         '`num_classes` should be 81')

    if weights == 'YCBVideo' and num_classes != 22:
        raise ValueError('If using `weights` as `"YCBVideo"` '
                         '`num_classes` should be 22')

    image = Input(shape=input_shape, name='image')

    # Block 1 -----------------------------------------------------------------
    conv1_1 = Conv2D(64, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv1_1')(image)
    conv1_2 = Conv2D(64, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                         padding='same', )(conv1_2)

    # Block 2 -----------------------------------------------------------------
    conv2_1 = Conv2D(128, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                         padding='same')(conv2_2)

    # Block 3 -----------------------------------------------------------------
    conv3_1 = Conv2D(256, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                         padding='same')(conv3_3)

    # Block 4 -----------------------------------------------------------------
    conv4_1 = Conv2D(512, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv4_3')(conv4_2)
    conv4_3_norm = Conv2DNormalization(20, name='branch_1')(conv4_3)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                         padding='same')(conv4_3)

    # Block 5 -----------------------------------------------------------------
    conv5_1 = Conv2D(512, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                         padding='same')(conv5_3)

    # Dense 6/7 --------------------------------------------------------------
    pool5z = ZeroPadding2D(padding=(6, 6))(pool5)
    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6),
                 padding='valid', activation='relu',
                 kernel_regularizer=l2(l2_loss),
                 trainable=trainable_base,
                 name='fc6')(pool5z)
    fc7 = Conv2D(1024, (1, 1), padding='same',
                 activation='relu',
                 kernel_regularizer=l2(l2_loss),
                 trainable=trainable_base,
                 name='branch_2')(fc6)

    # EXTRA layers in SSD -----------------------------------------------------

    # Block 6 -----------------------------------------------------------------
    conv6_1 = Conv2D(256, (1, 1), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     name='conv6_1')(fc7)
    conv6_1z = ZeroPadding2D()(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), padding='valid',
                     activation='relu', name='branch_3',
                     kernel_regularizer=l2(l2_loss))(conv6_1z)

    # Block 7 -----------------------------------------------------------------
    conv7_1 = Conv2D(128, (1, 1), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     name='conv7_1')(conv6_2)
    conv7_1z = ZeroPadding2D()(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), padding='valid', strides=(2, 2),
                     activation='relu', name='branch_4',
                     kernel_regularizer=l2(l2_loss))(conv7_1z)

    # Block 8 -----------------------------------------------------------------
    conv8_1 = Conv2D(128, (1, 1), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     name='conv8_1')(conv7_2)
    conv8_1z = ZeroPadding2D()(conv8_1)
    conv8_2 = Conv2D(256, (3, 3), padding='valid', strides=(2, 2),
                     activation='relu', name='branch_5',
                     kernel_regularizer=l2(l2_loss))(conv8_1z)

    # Block 9 -----------------------------------------------------------------
    conv9_1 = Conv2D(128, (1, 1), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     name='conv9_1')(conv8_2)
    conv9_1z = ZeroPadding2D()(conv9_1)
    conv9_2 = Conv2D(256, (3, 3), padding='valid', strides=(2, 2),
                     activation='relu', name='branch_6',
                     kernel_regularizer=l2(l2_loss))(conv9_1z)

    # Block 10 ----------------------------------------------------------------
    conv10_1 = Conv2D(128, (1, 1), padding='same', activation='relu',
                      kernel_regularizer=l2(l2_loss),
                      name='conv10_1')(conv9_2)
    conv10_1z = ZeroPadding2D()(conv10_1)
    conv10_2 = Conv2D(256, (4, 4), padding='valid', strides=(1, 1),
                      activation='relu', name='branch_7',
                      kernel_regularizer=l2(l2_loss))(conv10_1z)

    branch_tensors = [conv4_3_norm, fc7, conv6_2, conv7_2,
                      conv8_2, conv9_2, conv10_2]
    if return_base:
        output_tensor = branch_tensors

    else:
        output_tensor = create_multibox_head(
            branch_tensors, num_classes, num_priors, l2_loss)

    model = Model(inputs=image, outputs=output_tensor, name='SSD512')

    if weights is not None:
        model_name = '_'.join(['SSD512', weights])

    if weights is not None:
        weights_url = BASE_WEIGHT_PATH + model_name + '_weights.hdf5'
        weights_path = get_file(os.path.basename(weights_url), weights_url,
                                cache_subdir='paz/models')
        model.load_weights(weights_path)

    model.prior_boxes = create_prior_boxes('COCO')
    return model
