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


WEIGHT_PATH = (
    'https://github.com/oarriaga/altamira-data/releases/download/v0.2/')


def SSD300(num_classes=21, base_weights='VOC', head_weights='VOC',
           input_shape=(300, 300, 3), num_priors=[4, 6, 6, 6, 4, 4],
           l2_loss=0.0005, return_base=False, trainable_base=True):

    """Single-shot-multibox detector for 300x300x3 BGR input images.
    # Arguments
        num_classes: Integer. Specifies the number of class labels.
        base_weights: String or None. If string should be a valid dataset name.
            Current valid datasets include `VOC` `FAT` and `VGG`.
        head_weights: String or None. If string should be a valid dataset name.
            Current valid datasets include `VOC` and `FAT`.
        input_shape: List of integers. Input shape to the model including only
            spatial and channel resolution e.g. (300, 300, 3).
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

    if base_weights not in ['VGG', 'VOC', 'FAT', None]:
        raise ValueError('Invalid `base_weights`:', base_weights)

    if head_weights not in ['VOC', 'FAT', None]:
        raise ValueError('Invalid `base_weights`:', base_weights)

    if ((base_weights == 'VGG') and (head_weights is not None)):
        raise NotImplementedError('Invalid `base_weights` with head_weights')

    if ((base_weights is None) and (head_weights is not None)):
        raise NotImplementedError('Invalid `base_weights` with head_weights')

    if ((num_classes != 21) and (head_weights == 'VOC')):
        raise ValueError('Invalid `head_weights` with given `num_classes`')

    if ((num_classes != 22) and (head_weights == 'FAT')):
        raise ValueError('Invalid `head_weights` with given `num_classes`')

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
                     kernel_regularizer=l2(l2_loss))(fc7)
    conv6_1z = ZeroPadding2D()(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), padding='valid',
                     activation='relu', name='branch_3',
                     kernel_regularizer=l2(l2_loss))(conv6_1z)

    # Block 7 -----------------------------------------------------------------
    conv7_1 = Conv2D(128, (1, 1), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss))(conv6_2)
    conv7_1z = ZeroPadding2D()(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), padding='valid', strides=(2, 2),
                     activation='relu', name='branch_4',
                     kernel_regularizer=l2(l2_loss))(conv7_1z)

    # Block 8 -----------------------------------------------------------------
    conv8_1 = Conv2D(128, (1, 1), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss))(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), padding='valid', strides=(1, 1),
                     activation='relu', name='branch_5',
                     kernel_regularizer=l2(l2_loss))(conv8_1)

    # Block 9 -----------------------------------------------------------------
    conv9_1 = Conv2D(128, (1, 1), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss))(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), padding='valid', strides=(1, 1),
                     activation='relu', name='branch_6',
                     kernel_regularizer=l2(l2_loss))(conv9_1)

    branch_tensors = [conv4_3_norm, fc7, conv6_2, conv7_2, conv8_2, conv9_2]
    if return_base:
        outputs = branch_tensors
    else:
        outputs = create_multibox_head(
            branch_tensors, num_classes, num_priors, l2_loss)

    model = Model(inputs=image, outputs=outputs, name='SSD300')

    if ((base_weights is not None) or (head_weights is not None)):
        model_filename = ['SSD300', str(base_weights), str(head_weights)]
        model_filename = '_'.join(['-'.join(model_filename), 'weights.hdf5'])
        weights_path = get_file(model_filename, WEIGHT_PATH + model_filename,
                                cache_subdir='paz/models')
        print('Loading %s model weights' % weights_path)
        finetunning_model_names = ['SSD300-VGG-None_weights.hdf5',
                                   'SSD300-VOC-None_weights.hdf5']
        by_name = True if model_filename in finetunning_model_names else False
        model.load_weights(weights_path, by_name)
    model.prior_boxes = create_prior_boxes('VOC')
    return model
