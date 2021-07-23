import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    ReLU,
    Input,
    Conv2DTranspose,
    UpSampling2D,
    Add,
    concatenate,
    ZeroPadding2D
)
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils import conv_utils
import numpy as np

BN_MOMENTUM = 0.1


def conv3x3(out_planes, stride, name):
    """3x3 convolution with padding"""
    return Conv2D(out_planes, kernel_size=3, strides=stride, padding='valid', use_bias=False, name=name)


def stem(x, filters):
    x = ZeroPadding2D(padding=(1, 1), name='pad')(x)
    x = Conv2D(filters, 3, strides=2, padding='valid', use_bias=False, name='conv1')(x)
    x = BatchNormalization(momentum=BN_MOMENTUM, epsilon=1e-05, name='bn1')(x, training=False)
    x = ReLU(name='relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pad_1')(x)
    x = Conv2D(filters, 3, strides=2, padding='valid', use_bias=False, name='conv2')(x)
    x = BatchNormalization(momentum=BN_MOMENTUM, epsilon=1e-05, name='bn2')(x)
    x = ReLU()(x)
    return x


def BasicBlock(x, filters, stride, name):
    residual = x

    x = Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False, name=name+'.conv1')(x)
    x = BatchNormalization(momentum=BN_MOMENTUM, epsilon=1e-05, name=name+'.bn1')(x)
    x = ReLU(name=name+'.relu')(x)
    x = Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False, name=name+'.conv2')(x)
    x = BatchNormalization(momentum=BN_MOMENTUM, epsilon=1e-05, name=name+'.bn2')(x)
    x = Add()([x, residual])
    x = ReLU()(x)
    return x


def Bottleneck(x, filters=64, stride=1, expansion=4, downsample=None, name=None):
    residual = x

    out = Conv2D(filters, 1, use_bias=False, name=name+'.conv1')(x)
    out = BatchNormalization(momentum=BN_MOMENTUM, epsilon=1e-05, name=name+'.bn1')(out)
    out = ReLU(name=name+'.relu')(out)
    out = Conv2D(filters, 3, strides=stride, padding='same', use_bias=False, name=name+'.conv2')(out)
    out = BatchNormalization(momentum=BN_MOMENTUM, epsilon=1.0e-5, name=name+'.bn2')(out)
    out = ReLU()(out)
    out = Conv2D(filters * expansion, 1, use_bias=False, name=name+'.conv3')(out)
    out = BatchNormalization(momentum=BN_MOMENTUM, epsilon=1.0e-5, name=name+'.bn3')(out)

    if downsample is not None:
        y = Conv2D(256, 1, strides=1, use_bias=False, name='layer1' + '.0.downsample.0')(x)
        residual = BatchNormalization(momentum=BN_MOMENTUM, epsilon=1e-05, name='layer1' + '.0.downsample.1')(y)

    out = Add()([out, residual])
    out = ReLU()(out)
    return out


def transition_block(x, alpha, name):
    in_channels = K.int_shape(x)[-1]
    if in_channels == 256:
        filters = 32 * alpha
    else:
        filters = in_channels * alpha
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters, 3, strides=2, padding='valid', use_bias=False, name=name+'0.0')(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-05, name=name+'0.1')(x)
    x = ReLU()(x)
    return x


# HRNet
def blocks_in_branch(x, stage, in_channels, name):
    assert stage == len(x), \
        "outputs {} feed to fuse_layers must to be same as num_branches {}".format(x, stage)
    for i in range(stage):
        c = in_channels * (2 ** i)
        x[i] = BasicBlock(x[i], c, stride=1, name=name[:18]+str(i)+'.'+name[18:])
    return x


def fuse_layers(tensors, stage, output_branches, c=32, name=None):
    assert stage == len(tensors), \
     "outputs {} feed to fuse_layers must to be same as num_branches {}".format(x, stage)
    all_tensors = []
    for i in range(output_branches):
        x_to_y_tensors = []
        for j in range(stage):  # for each branch
            if i == j:
                y = tensors[i]

            elif i < j:  # upsample
                y = Conv2D(c * (2 ** i), kernel_size=1, strides=1, padding='valid', use_bias=False, name=name+str(i)+'.'+str(j)+'.0')(tensors[j])
                y = BatchNormalization(momentum=0.1, epsilon=1e-05, name=name+str(i)+'.'+str(j)+'.1')(y)
                y = UpSampling2D(size=(2 ** (j - i), 2 ** (j - i)), interpolation='nearest', name=name+str(i)+'.'+str(j)+'.2')(y)  # CHECK formula

            elif i > j:  # downsample
                # print(f"\ny shape before pad {y.shape}")
                y_new = False
                for_loop = 0
                for k in range(i - j - 1):  # else
                    for_loop += 1
                    if y_new:
                        y = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(y)
                    y = conv3x3(c * (2 ** j), stride=2, name=name+str(i)+'.'+str(j)+'.1.0')(y) if y_new \
                        else conv3x3(c * (2 ** j), stride=2, name=name+str(i)+'.'+str(j)+'.0.0')(tensors[j])
                    y = BatchNormalization(momentum=0.1, epsilon=1e-05, name=name+str(i)+'.'+str(j)+'.0.1')(y) if not y_new \
                        else BatchNormalization(momentum=0.1, epsilon=1e-05, name=name+str(i)+'.'+str(j)+'.1.1')(y)
                    y = ReLU()(y)
                    y_new = True

                if not y_new:
                    tensors[j] = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(tensors[j])
                else:
                    y = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(y)
                y = conv3x3(c * (2 ** i), stride=2, name=name+str(i)+'.'+str(j)+'.0.0')(tensors[j]) if not y_new \
                    else conv3x3(c * (2 ** i), stride=2, name=name+str(i)+'.'+str(j)+'.'+str(for_loop)+'.0')(y)  # if
                y = BatchNormalization(momentum=0.1, epsilon=1e-05, name=name+str(i)+'.'+str(j)+'.0.1')(y) if not y_new \
                    else BatchNormalization(momentum=0.1, epsilon=1e-05, name=name+str(i)+'.'+str(j)+'.'+str(for_loop)+'.1')(y)
            x_to_y_tensors.append(y)

        all_tensors.append(x_to_y_tensors)

    x_fused = []

    for i in range(len(all_tensors)):
        for j in range(0, len(x_to_y_tensors)):
            if j == 0:
                x_fused.append(all_tensors[i][0])
            else:
                x_fused[i] = Add()([x_fused[i], all_tensors[i][j]])

    for i in range(len(x_fused)):
        x_fused[i] = ReLU()(x_fused[i])

    return x_fused


def final_layers(num_keypoints, with_AE_loss=None, num_deconv=1):
    dim_tag = num_keypoints

    final_layers = []
    output_channels = num_keypoints + dim_tag if with_AE_loss[0] else num_keypoints
    final_layers.append(Conv2D(output_channels, 1, strides=1, padding='same', name='final_layers.0'))

    for i in range(num_deconv):
        output_channels = num_keypoints + dim_tag if with_AE_loss[i + 1] else num_keypoints
        final_layers.append(Conv2D(output_channels, 1, strides=1, padding='same', name='final_layers.1'))

    return final_layers


def deconv_layers(x, output_channels, num_deconv=1, name=None):
    # Hinweis: adding output_padding param in ConvTranspose2D layer messes output shape when output_padding is 0
    # maybe using output_padding=None might work but for now if not needed just dont write it.
    for i in range(num_deconv):
        x = Conv2DTranspose(output_channels, 4, strides=2, padding='same', use_bias=False, name='deconv_layers.0.0.0')(x)
        x = BatchNormalization(momentum=BN_MOMENTUM, epsilon=1e-05, name='deconv_layers.0.0.1')(x)
        x = ReLU()(x)

        for block in range(4):
            x = BasicBlock(x, output_channels, stride=1, name='deconv_layers.0.'+str(block+1)+'.'+'0')
    return x


def HigherHRNet(input_shape=(None, None, 3), num_keypoints=17):

    # Stem net
    inputs = Input(shape=input_shape, name='image')
    x = stem(inputs, 64)
    print(f"check 1 stem TF ==> {x.get_shape()}")

    # Stage 1       - First group of bottleneck (resnet) modules
    x = Bottleneck(x, filters=64, stride=1, expansion=4, downsample=True, name='layer1' + '.0')
    for block in range(3):
        x = Bottleneck(x, filters=64, stride=1, expansion=4, downsample=None, name='layer1'+'.'+str(block+1))

    x_list = []

    # Creation of the first two branches (one full and one half resolution)
    x1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name='transition1.0.0')(x)
    x1 = BatchNormalization(momentum=BN_MOMENTUM, name='transition1.0.1')(x1)
    x1 = ReLU()(x1)
    x_list.append(x1)
    x_list.append(transition_block(x, 2, name='transition1.1.'))

    # Stage 2   stage2.0.branches.0.0.conv1    stage2.0.branches.1.0.conv1
    for block in range(4):
        x_list = blocks_in_branch(x_list, stage=2, in_channels=32, name='stage2.0.branches.'+str(block))
    x_list = fuse_layers(x_list, stage=2, output_branches=2, name='stage2.0.fuse_layers.')
    x_list.append(transition_block(x_list[1], 2, name='transition2.2.'))

    # Stage 3
    for module in range(4):
        for block in range(4):
            x_list = blocks_in_branch(x_list, stage=3, in_channels=32, name='stage3.'+str(module)+'.branches.'+str(block))
        x_list = fuse_layers(x_list, stage=3, output_branches=3, name='stage3.'+str(module)+'.fuse_layers.')
    x_list.append(transition_block(x_list[2], 2, name='transition3.3.'))

    # Stage 4
    for module in range(3):
        for block in range(4):
            x_list = blocks_in_branch(x_list, stage=4, in_channels=32, name='stage4.'+str(module)+'.branches.'+str(block))
        x_list = fuse_layers(x_list, stage=4, output_branches=1, name='stage4.'+str(module)+'.fuse_layers.') \
            if module == 2 else fuse_layers(x_list, stage=4, output_branches=4, name='stage4.'+str(module)+'.fuse_layers.')

    final_outputs = []
    x_ = x_list[0]
    y = final_layers(num_keypoints, with_AE_loss=[True, False])[0](x_)
    final_outputs.append(y)
    cat_output = True

    for i in range(1):
        if cat_output:
            x_ = concatenate((x_, y), -1)
        x_ = deconv_layers(x_, 32)
        y = final_layers(num_keypoints, with_AE_loss=[True, False])[i + 1](x_)
        final_outputs.append(y)
    #print(f"\nFINAL OUTPUTS {len(final_outputs)} {final_outputs[0].shape} {final_outputs[1].shape}\n")

    m = Model(inputs, outputs=final_outputs, name='HigherHRNet')
    return m