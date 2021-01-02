import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import get_file

from paz.models.detection.model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from paz.models.detection.model_util import point_cloud_masking
from paz.models.detection.model_util import placeholder_inputs, parse_output_to_tensors, get_loss
from paz.optimization.losses.frustumpointnet_loss import FPointNet_loss
# from paz.models.detection.model_util import FPointNet_loss


# FPointNet_loss = FrustumPointNetLoss()


def Instance_Seg_Net(point_cloud, one_hot_vec):
    batch_size = point_cloud.get_shape().as_list()[0]
    num_points = point_cloud.get_shape().as_list()[1]

    net = tf.expand_dims(point_cloud, 2)

    net = Conv2D(64, kernel_size=[1, 1], padding='VALID', strides=[1, 1], activation='relu', name='conv1_1')(net)

    net = Conv2D(64, kernel_size=[1, 1], padding='VALID', strides=[1, 1], activation='relu', name='conv1_2')(net)

    point_feat = Conv2D(64, kernel_size=[1, 1], padding='VALID', strides=[1, 1], activation='relu', name='conv1_3')(net)

    net = Conv2D(128, kernel_size=[1, 1], padding='VALID', strides=[1, 1], activation='relu', name='conv1_4')(
        point_feat)

    net = Conv2D(1024, kernel_size=[1, 1], padding='VALID', strides=[1, 1], activation='relu', name='conv1_5')(net)

    global_feat = MaxPooling2D(pool_size=[num_points, 1], padding='VALID')(net)

    global_feat = tf.concat([global_feat, tf.expand_dims(tf.expand_dims(one_hot_vec, 1), 1)], axis=3)
    global_feat_expand = tf.tile(global_feat, [1, num_points, 1, 1])
    concat_feat = tf.concat(axis=3, values=[point_feat, global_feat_expand])

    net = Conv2D(512, kernel_size=[1, 1], padding='VALID', strides=[1, 1], activation='relu', name='conv1_6')(
        concat_feat)

    net = Conv2D(256, kernel_size=[1, 1], padding='VALID', strides=[1, 1], activation='relu', name='conv1_7')(net)

    net = Conv2D(128, kernel_size=[1, 1], padding='VALID', strides=[1, 1], activation='relu', name='conv1_8')(net)

    net = Conv2D(128, kernel_size=[1, 1], padding='VALID', strides=[1, 1], activation='relu', name='conv1_9')(net)

    net = Dropout(rate=0.5)(net)

    logits = Conv2D(2, kernel_size=[1, 1], padding='VALID', strides=[1, 1], activation=None, name='conv1_10')(net)

    logits = tf.squeeze(logits, [2])  # BxNxC

    return logits


def Box_Est_Net(object_point_cloud, one_hot_vec):
    num_point = object_point_cloud.get_shape()[1]
    net = tf.expand_dims(object_point_cloud, 2)

    net = Conv2D(128, kernel_size=[1, 1], padding='VALID', strides=[1, 1], activation='relu', name='conv2_1')(net)

    net = Conv2D(128, kernel_size=[1, 1], padding='VALID', strides=[1, 1], activation='relu', name='conv2_2')(net)

    net = Conv2D(256, kernel_size=[1, 1], padding='VALID', strides=[1, 1], activation='relu', name='conv2_3')(net)

    net = Conv2D(512, kernel_size=[1, 1], padding='VALID', strides=[1, 1], activation='relu', name='conv2_4')(net)

    net = MaxPooling2D(pool_size=[num_point, 1], padding='VALID')(net)

    net = tf.squeeze(net, axis=[1, 2])
    net = tf.concat([net, one_hot_vec], axis=1)

    net = Dense(units=512, activation='relu')(net)
    net = Dense(units=256, activation='relu')(net)

    output = Dense(units=3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4, activation=None)(net)

    return output


def TNet(object_point_cloud, one_hot_vec):
    num_point = object_point_cloud.get_shape()[1]
    net = tf.expand_dims(object_point_cloud, 2)

    net = Conv2D(128, kernel_size=[1, 1], padding='VALID', strides=[1, 1], activation='relu', name='conv3_1')(net)

    net = Conv2D(128, kernel_size=[1, 1], padding='VALID', strides=[1, 1], activation='relu', name='conv3_2')(net)

    net = Conv2D(256, kernel_size=[1, 1], padding='VALID', strides=[1, 1], activation='relu', name='conv3_3')(net)

    net = MaxPooling2D(pool_size=[num_point, 1], padding='VALID')(net)

    net = tf.squeeze(net, axis=[1, 2])
    net = tf.concat([net, one_hot_vec], axis=1)

    net = Dense(units=256, activation='relu')(net)
    net = Dense(units=128, activation='relu')(net)
    predicted_center = Dense(units=3, activation=None)(net)

    return predicted_center


def FrustumPointNetModel(point_cloud_shape=(1024, 3), one_hot_vec_shape=(3,), mask_label_shape=(1024,),
                         center_label_shape=(3,), heading_class_label_shape=(), heading_residual_label_shape=(),
                         size_class_label_shape=(), size_residual_label_shape=(3,), batch_size=32):
    end_points = {}

    point_cloud = Input(point_cloud_shape, name="frustum_point_cloud", batch_size=batch_size)
    one_hot_vec = Input(one_hot_vec_shape, name="one_hot_vec", batch_size=batch_size)
    mask_label = Input(mask_label_shape, name="seg_label", batch_size=batch_size)
    center_label = Input(center_label_shape, name="box3d_center", batch_size=batch_size)
    heading_class_label = Input(heading_class_label_shape, name="angle_class", batch_size=batch_size)
    heading_residual_label = Input(heading_residual_label_shape, name="angle_residual", batch_size=batch_size)
    size_class_label = Input(size_class_label_shape, name="size_class", batch_size=batch_size)
    size_residual_label = Input(size_residual_label_shape, name="size_residual", batch_size=batch_size)

    logits = Instance_Seg_Net(point_cloud, one_hot_vec)  # bs,n,2
    end_points['mask_logits'] = logits

    # Mask Point Centroid
    object_point_cloud_xyz, mask_xyz_mean, end_points = point_cloud_masking(point_cloud, logits, end_points)

    # T-Net
    center_delta = TNet(object_point_cloud_xyz, one_hot_vec)  # (32,3)

    stage1_center = center_delta + mask_xyz_mean  # Bx3
    end_points['stage1_center'] = stage1_center
    # Get object point cloud in object coordinate
    object_point_cloud_xyz_new = object_point_cloud_xyz - tf.expand_dims(center_delta, 1)

    # 3D Box Estimation
    box_pred = Box_Est_Net(object_point_cloud_xyz_new, one_hot_vec)  # (32, 59)

    end_points = parse_output_to_tensors(box_pred, end_points)
    end_points['center'] = end_points['center_boxnet'] + stage1_center  # Bx3

    logits = end_points['mask_logits']
    mask = end_points['mask']
    stage1_center = end_points['stage1_center']
    center_boxnet = end_points['center_boxnet']
    heading_scores = end_points['heading_scores']  # BxNUM_HEADING_BIN
    heading_residuals_normalized = end_points['heading_residuals_normalized']
    heading_residuals = end_points['heading_residuals']
    size_scores = end_points['size_scores']
    size_residuals_normalized = end_points['size_residuals_normalized']
    size_residuals = end_points['size_residuals']
    center = end_points['center']

    logits = Lambda(lambda x: x, name="InsSeg_out")(logits)
    box3d_center = Lambda(lambda x: x, name="center_out")(center)
    heading_scores = Lambda(lambda x: x, name="heading_scores")(heading_scores)
    heading_residual = Lambda(lambda x: x, name="heading_residual")(heading_residuals)
    heading_residuals_normalized = Lambda(lambda x: x, name="heading_residual_norm")(heading_residuals_normalized)
    size_scores = Lambda(lambda x: x, name="size_scores")(size_scores)
    size_residual = Lambda(lambda x: x, name="size_residual")(size_residuals)
    size_residuals_normalized = Lambda(lambda x: x, name="size_residual_norm")(size_residuals_normalized)

    loss = Lambda(FPointNet_loss, output_shape=(1,), name='fp_loss',
                  arguments={'corner_loss_weight': 10.0, 'box_loss_weight': 1.0})([mask_label, center_label,
                                                                                   heading_class_label,
                                                                                   heading_residual_label,
                                                                                   size_class_label,
                                                                                   size_residual_label, end_points])

    training_model = Model([point_cloud, one_hot_vec, mask_label, center_label, heading_class_label,
                            heading_residual_label, size_class_label, size_residual_label], loss,
                           name='f_pointnet_train')
    det_model = Model(inputs=[point_cloud, one_hot_vec],
                      outputs=[logits, box3d_center, heading_scores, heading_residual, size_scores, size_residual],
                      name='f_pointnet_inference')
    training_model.summary()
    return training_model, det_model


if __name__ == '__main__':
    FrustumPointNetModel()
