import os
from tensorflow.keras.utils import get_file
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model


WEIGHT_PATH = ('https://github.com/oarriaga/altamira-data/releases/download'
               '/v0.10/HigherHRNet.hdf5')


def stem(tensor, filters):
    x = ZeroPadding2D(padding=(1, 1), name='pad')(tensor)
    x = Conv2D(filters, 3, strides=2, use_bias=False, name='conv1')(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-05,
                           name='bn1')(x, training=False)
    x = ReLU(name='relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pad_1')(x)
    x = Conv2D(filters, 3, strides=2, use_bias=False, name='conv2')(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-05, name='bn2')(x)
    x = ReLU()(x)
    return x


def bottleneck(tensor, filters, expansion, downsample=None, name=None):
    residual = tensor
    x = Conv2D(filters, 1, use_bias=False, name=name + '.conv1')(tensor)
    x = BatchNormalization(momentum=0.1, epsilon=1e-05, name=name + '.bn1')(x)
    x = ReLU(name=name + '.relu')(x)
    x = Conv2D(filters, 3, padding='same',
               use_bias=False, name=name + '.conv2')(x)
    x = BatchNormalization(momentum=0.1, epsilon=1.0e-5, name=name + '.bn2')(x)
    x = ReLU()(x)
    x = Conv2D(filters * expansion, 1, use_bias=False, name=name + '.conv3')(x)
    x = BatchNormalization(momentum=0.1, epsilon=1.0e-5, name=name + '.bn3')(x)
    if downsample is not None:
        x1 = Conv2D(256, 1, use_bias=False,
                    name='layer1.0.downsample.0')(tensor)
        residual = BatchNormalization(momentum=0.1, epsilon=1e-05,
                                      name='layer1.0.downsample.1')(x1)
    x = Add()([x, residual])
    x = ReLU()(x)
    return x


def basic_block(tensor, filters, name=None):
    residual = tensor
    x = Conv2D(filters, 3, padding='same',
               use_bias=False, name=name + '.conv1')(tensor)
    x = BatchNormalization(momentum=0.1, epsilon=1e-05, name=name + '.bn1')(x)
    x = ReLU(name=name + '.relu')(x)
    x = Conv2D(filters, 3, padding='same',
               use_bias=False, name=name + '.conv2')(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-05, name=name + '.bn2')(x)
    x = Add()([x, residual])
    x = ReLU()(x)
    return x


def transition_block(tensor, alpha, name):
    in_channels = K.int_shape(tensor)[-1]
    if in_channels == 256:
        filters = 32 * alpha
    else:
        filters = in_channels * alpha
    x = ZeroPadding2D(padding=(1, 1))(tensor)
    x = Conv2D(filters, 3, strides=2, use_bias=False, name=name + '0.0')(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-05, name=name + '0.1')(x)
    x = ReLU()(x)
    return x


def blocks_in_branch(tensors, stage, in_channels, name):
    if stage != len(tensors):
        raise ValueError('''outputs {} feed to fuse_layers must to be same as
                         num_branches {}'''.format(tensors, stage))
    for arg in range(stage):
        filters = in_channels * (2 ** arg)
        tensors[arg] = basic_block(tensors[arg], filters,
                                   name=name[:18] + str(arg) + '.' + name[18:])
    return tensors


def final_layers(num_keypoints, with_AE_loss=None, num_deconv=1):
    final_layers = []
    if with_AE_loss[0]:
        output_channels = num_keypoints * 2
    else:
        output_channels = num_keypoints
    x = Conv2D(output_channels, 1, padding='same', name='final_layers.0')
    final_layers.append(x)

    for arg in range(num_deconv):
        if with_AE_loss[arg + 1]:
            output_channels = num_keypoints * 2
        else:
            output_channels = num_keypoints
        x1 = Conv2D(output_channels, 1, padding='same', name='final_layers.1')
        final_layers.append(x1)
    return final_layers


def deconv_layers(tensor, output_channels, num_deconv=1):
    for arg in range(num_deconv):
        x = Conv2DTranspose(output_channels, 4, strides=2,
                            padding='same', use_bias=False,
                            name='deconv_layers.0.0.0')(tensor)
        x = BatchNormalization(momentum=0.1, epsilon=1e-05,
                               name='deconv_layers.0.0.1')(x)
        x = ReLU()(x)
        for block in range(4):
            x = basic_block(
                x, output_channels,
                name='deconv_layers.0.' + str(block + 1) + '.' + '0')
    return x


def get_names(name, branch_arg, stage_arg, counter, iterations=0):
    name1 = '.'.join((name, str(branch_arg), str(stage_arg),
                     str(iterations + counter)))
    name2 = '.'.join((name, str(branch_arg), str(stage_arg),
                     str(iterations + counter + .1)))
    return [name1, name2]


def fuse_layers(tensors, stage, output_branches, filters=32, name=None):
    if stage != len(tensors):
        raise ValueError('''outputs {} feed to fuse_layers must to be same as
                         num_branches {}'''.format(tensors, stage))
    all_tensors = []
    for branch_arg in range(output_branches):
        x_to_y_tensors = []
        for stage_arg in range(stage):
            # step: how much the feature map is upsampled or downsampled
            steps = stage_arg - branch_arg
            if steps == 0:
                y = tensors[branch_arg]

            elif steps > 0:  # upsample
                name0 = '.'.join((name, str(branch_arg), str(stage_arg)))
                y = upsample(tensors[stage_arg], filters * (2 ** branch_arg),
                             size=(2**steps, 2**steps), name=name0)

            elif steps < 0:  # downsample
                y_flag = False
                iterations = 0
                for k in range((-1 * steps) - 1):
                    iterations += 1
                    if y_flag:
                        name1 = get_names(name, branch_arg, stage_arg, 1.0)
                        y = downsample(y, filters * (2 ** stage_arg), name1)

                    else:
                        name2 = get_names(name, branch_arg, stage_arg, 0.0)
                        y = downsample(tensors[stage_arg],
                                       filters * (2 ** stage_arg), name2,
                                       with_padding=False)
                    y = ReLU()(y)
                    y_flag = True

                if not y_flag:
                    tensors[stage_arg] = ZeroPadding2D()(tensors[stage_arg])
                    name3 = get_names(name, branch_arg, stage_arg, 0.0)
                    y = downsample(tensors[stage_arg],
                                   filters * (2 ** branch_arg), name3,
                                   with_padding=False)

                else:
                    name4 = get_names(name, branch_arg, stage_arg,
                                      .0, iterations)
                    y = downsample(y, filters * (2 ** branch_arg), name4)
            x_to_y_tensors.append(y)

        all_tensors.append(x_to_y_tensors)

    x_fused = []
    for x_tensor_arg in range(len(all_tensors)):
        for y_tensor_arg in range(len(x_to_y_tensors)):
            if y_tensor_arg == 0:
                x_fused.append(all_tensors[x_tensor_arg][0])
            else:
                x = Add()([x_fused[x_tensor_arg],
                          all_tensors[x_tensor_arg][y_tensor_arg]])
                x_fused[x_tensor_arg] = x

    for x_fused_arg in range(len(x_fused)):
        x_fused[x_fused_arg] = ReLU()(x_fused[x_fused_arg])
    return x_fused


def upsample(tensor, filters, size, name=None):
    x = Conv2D(filters, 1, use_bias=False, name=name + '.0')(tensor)
    x = BatchNormalization(momentum=0.1, epsilon=1e-05, name=name + '.1')(x)
    x = UpSampling2D(size=size, interpolation='nearest', name=name + '.2')(x)
    return x


def downsample(tensor, filters, name=None, with_padding=True):
    if with_padding:
        tensor = ZeroPadding2D(padding=(1, 1))(tensor)
    x = Conv2D(filters, 3, strides=2, use_bias=False, name=name[0])(tensor)
    x = BatchNormalization(momentum=0.1, epsilon=1e-05, name=name[1])(x)
    return x


def HigherHRNet(weights='COCO', input_shape=(None, None, 3), num_keypoints=17,
                with_AE_loss=[True, False]):
    """Human pose estimation detector for any input size of images.
    # Arguments
        weights: String or None. If string should be a valid dataset name.
            Current valid datasets include `COCO`.
        input_shape: List of integers. Input shape to the model including only
            spatial and channel resolution e.g. (512, 512, 3).
        num_keypoints: Int. Number of joints.
        with_AE_loss: List of boolean.

    # Reference
        - [HigherHRNet: Scale-Aware Representation Learning for Bottom-Up
           Human Pose Estimation](https://arxiv.org/abs/1908.10357)
    """

    image = Input(shape=input_shape, name='image')
    x = stem(image, 64)
    # print(f"check 1 stem TF ==> {x.get_shape()}")

    # First group of bottleneck (resnet) modules
    # Stage 1 -----------------------------------------------------------------
    x = bottleneck(x, filters=64, expansion=4,
                   downsample=True, name='layer1' + '.0')
    for block in range(3):
        x = bottleneck(x, filters=64, expansion=4,
                       downsample=None, name='layer1' + '.' + str(block + 1))

    x_list = []
    # Creation of the first two branches (one full and one half resolution)
    x1 = Conv2D(32, 3, strides=1, padding='same',
                use_bias=False, name='transition1.0.0')(x)
    x1 = BatchNormalization(momentum=0.1, name='transition1.0.1')(x1)
    x1 = ReLU()(x1)

    x_list.append(x1)
    x_list.append(transition_block(x, 2, name='transition1.1.'))

    # Stage 2 -----------------------------------------------------------------
    for block in range(4):
        x_list = blocks_in_branch(x_list, stage=2, in_channels=32,
                                  name='stage2.0.branches.' + str(block))
    x_list = fuse_layers(x_list, stage=2, output_branches=2,
                         name='stage2.0.fuse_layers')
    x_list.append(transition_block(x_list[1], 2, name='transition2.2.'))

    # Stage 3 -----------------------------------------------------------------
    for module in range(4):
        for block in range(4):
            name = 'stage3.' + str(module) + '.branches.' + str(block)
            x_list = blocks_in_branch(x_list, stage=3,
                                      in_channels=32, name=name)
        x_list = fuse_layers(x_list, stage=3, output_branches=3,
                             name='stage3.' + str(module) + '.fuse_layers')
    x_list.append(transition_block(x_list[2], 2, name='transition3.3.'))

    # Stage 4 -----------------------------------------------------------------
    for module in range(3):
        for block in range(4):
            name = 'stage4.' + str(module) + '.branches.' + str(block)
            x_list = blocks_in_branch(x_list, stage=4,
                                      in_channels=32, name=name)
        if module == 2:
            name = 'stage4.' + str(module) + '.fuse_layers'
            x_list = fuse_layers(x_list, stage=4, output_branches=1, name=name)
        else:
            name = 'stage4.' + str(module) + '.fuse_layers'
            x_list = fuse_layers(x_list, stage=4, output_branches=4, name=name)

    final_outputs = []
    x2 = x_list[0]
    output = final_layers(num_keypoints, with_AE_loss=with_AE_loss)[0](x2)
    final_outputs.append(output)

    x2 = concatenate((x2, output), -1)
    x2 = deconv_layers(x2, 32)
    x2 = final_layers(num_keypoints, with_AE_loss=with_AE_loss)[1](x2)
    final_outputs.append(x2)

    model = Model(image, outputs=final_outputs, name='HigherHRNet')

    if(weights == 'COCO'):
        URL = ('https://github.com/oarriaga/altamira-data/releases/download'
               '/v0.10/HigherHRNet_weights.hdf5')
        filename = os.path.basename(URL)
        weights_path = get_file(filename, URL, cache_subdir='paz/models')
        print('==> Loading %s model weights' % weights_path)
        model.load_weights(weights_path)
    return model
