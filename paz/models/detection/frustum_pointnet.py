import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model

from examples.frustum_pointnet.frustum_loader import g_mean_size_arr
from paz.optimization.losses.frustumpointnet_loss import FrustumPointNetLoss


def ModelOutputToTensor(output, IntermediateOutputs, NUM_HEADING_BIN=12,
                        NUM_SIZE_CLUSTER=8):
    """ Parse batch output to separate tensors (added to IntermediateOutputs)
    Input:
        output: TF tensor in shape (B,3+2*NUM_HEADING_BIN+4*NUM_SIZE_CLUSTER)
        IntermediateOutputs: dict
    Output:
        IntermediateOutputs: dict

    """
    batch_size = output.get_shape()[0]
    center = tf.slice(output, [0, 0], [-1, 3])
    IntermediateOutputs['center_boxnet'] = center

    heading_scores = tf.slice(output, [0, 3], [-1, NUM_HEADING_BIN])
    heading_residuals_normalized = tf.slice(output, [0, 3 + NUM_HEADING_BIN],
                                            [-1, NUM_HEADING_BIN])
    IntermediateOutputs['heading_scores'] = heading_scores  # BxNUM_HEADING_BIN
    IntermediateOutputs['heading_residuals_normalized'] = \
        heading_residuals_normalized  # BxNUM_HEADING_BIN (-1 to 1)
    IntermediateOutputs['heading_residuals'] = \
        heading_residuals_normalized * (np.pi / NUM_HEADING_BIN)

    size_scores = tf.slice(output, [0, 3 + NUM_HEADING_BIN * 2],
                           [-1, NUM_SIZE_CLUSTER])  # BxNUM_SIZE_CLUSTER
    size_residuals_normalized = tf.slice(output,
                                         [0,
                                          3 + NUM_HEADING_BIN * 2 +
                                          NUM_SIZE_CLUSTER],
                                         [-1, NUM_SIZE_CLUSTER * 3])

    size_residuals_normalized = tf.reshape(size_residuals_normalized,
                                           [batch_size, NUM_SIZE_CLUSTER,
                                            3])  # BxNUM_SIZE_CLUSTERx3
    IntermediateOutputs['size_scores'] = size_scores
    IntermediateOutputs['size_residuals_normalized'] = size_residuals_normalized
    IntermediateOutputs['size_residuals'] = \
        size_residuals_normalized * tf.expand_dims(
            tf.constant(g_mean_size_arr,  dtype=tf.float32), 0)

    return IntermediateOutputs


def ObjectPointCloudMasking(point_cloud, mask, npoints=512):
    """ Gather object point clouds according to predicted masks.
    Input:
        point_cloud: TF tensor in shape (B,N,C)
        mask: TF tensor in shape (B,N) of 0 (not pick) or 1 (pick)
        npoints: int scalar, maximum number of points to keep (default: 512)
    Output:
        object_pc: TF tensor in shape (B,npoint,C)
        indices: TF int tensor in shape (B,npoint,2)
    """

    def mask_to_indices(mask):
        indices = np.zeros((mask.shape[0], npoints, 2), dtype=np.int32)
        for i in range(mask.shape[0]):
            pos_indices = np.where(mask[i, :] > 0.5)[0]
            # skip cases when pos_indices is empty
            if len(pos_indices) > 0:
                if len(pos_indices) > npoints:
                    choice = np.random.choice(len(pos_indices),
                                              npoints, replace=False)
                else:
                    choice = np.random.choice(len(pos_indices),
                                              npoints - len(pos_indices),
                                              replace=True)
                    choice = np.concatenate((np.arange(len(pos_indices)),
                                             choice))
                np.random.shuffle(choice)
                indices[i, :, 1] = pos_indices[choice]
            indices[i, :, 0] = i
        return indices

    indices = tf.py_function(mask_to_indices, [mask], tf.int32)
    object_pc = tf.gather_nd(point_cloud, indices)
    return object_pc, indices


def PointCloudTranslation(point_cloud, logits, IntermediateOutputs,
                          NUM_OBJECT_POINT=512, xyz_only=True):
    """ Select point cloud with predicted 3D mask,
    translate coordinates to the masked points centroid.

    Used from Frustum PointNet

    Input:
        point_cloud: TF tensor in shape (B,N,C)
        logits: TF tensor in shape (B,N,2)
        IntermediateOutputs: dict
        xyz_only: boolean, if True only return XYZ channels
    Output:
        object_point_cloud: TF tensor in shape (B,M,3)
            for simplicity we only keep XYZ here
            M = NUM_OBJECT_POINT as a hyper-parameter
        mask_xyz_mean: TF tensor in shape (B,3)
    """
    batch_size = point_cloud.get_shape()[0]
    num_point = point_cloud.get_shape()[1]
    mask = tf.slice(logits, [0, 0, 0], [-1, -1, 1]) < \
           tf.slice(logits, [0, 0, 1], [-1, -1, 1])
    mask = tf.cast(mask, dtype=tf.float32)  # BxNx1
    mask_count = tf.tile(tf.math.reduce_sum(mask, axis=1, keepdims=True),
                         [1, 1, 3])  # Bx1x3
    point_cloud_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])  # BxNx3
    mask_xyz_mean = tf.math.reduce_sum(
        tf.tile(mask, [1, 1, 3]) * point_cloud_xyz, axis=1,
        keepdims=True)  # Bx1x3
    mask = tf.squeeze(mask, axis=[2])  # BxN
    IntermediateOutputs['mask'] = mask
    mask_xyz_mean = mask_xyz_mean / tf.maximum(mask_count, 1)  # Bx1x3

    # Translate to masked points' centroid
    point_cloud_xyz_stage1 = point_cloud_xyz - \
                             tf.tile(mask_xyz_mean, [1, num_point, 1])

    if xyz_only:
        point_cloud_stage1 = point_cloud_xyz_stage1
    else:
        point_cloud_features = tf.slice(point_cloud, [0, 0, 3], [-1, -1, -1])
        point_cloud_stage1 = tf.concat(
            [point_cloud_xyz_stage1, point_cloud_features], axis=-1)
    num_channels = point_cloud_stage1.get_shape()[2]

    object_point_cloud, _ = ObjectPointCloudMasking(point_cloud_stage1,
                                                    mask, NUM_OBJECT_POINT)
    object_point_cloud.set_shape([batch_size, NUM_OBJECT_POINT, num_channels])

    return object_point_cloud, tf.squeeze(mask_xyz_mean, axis=1), \
           IntermediateOutputs


def InstanceSegmentationNet(point_cloud, one_hot_vector):
    """Instance Segmentation network with PointNet backbone.
    # Arguments
        point_cloud: Tensor, Frustum Point Cloud extracted using 2D object
            detection and projected pointcloud.
        one_hot_vector: Tensor. One hot vector of the class to which the
            frustum pointcloud belongs to.

    # Reference
        - [PointNet](https://arxiv.org/abs/1612.00593)
        - [Frustum PointNet](https://arxiv.org/abs/1711.08488)
    """
    num_points = point_cloud.get_shape().as_list()[1]

    input = tf.expand_dims(point_cloud, 2)

    x = Conv2D(64, 1, (1, 1), 'valid', activation='relu', name='conv1_1')(input)

    x = Conv2D(64, 1, (1, 1), 'valid', activation='relu', name='conv1_2')(x)

    point_feat = Conv2D(64, 1, (1, 1), 'valid', activation='relu',
                        name='conv1_3')(x)

    x = Conv2D(128, 1, (1, 1), 'valid', activation='relu',
               name='conv1_4')(point_feat)

    x = Conv2D(1024, 1, (1, 1), 'valid', activation='relu',
               name='conv1_5')(x)

    global_feat = MaxPooling2D(pool_size=[num_points, 1], padding='VALID')(x)

    global_feat = tf.concat([global_feat, tf.expand_dims(tf.expand_dims(
        one_hot_vector, 1), 1)], axis=3)

    global_feat_expand = tf.tile(global_feat, [1, num_points, 1, 1])

    concat_feat = tf.concat(axis=3, values=[point_feat, global_feat_expand])

    x = Conv2D(512, 1, (1, 1), 'valid', activation='relu',
               name='conv1_6')(concat_feat)

    x = Conv2D(256, 1, (1, 1), 'valid', activation='relu', name='conv1_7')(x)

    x = Conv2D(128, 1, (1, 1), 'valid', activation='relu', name='conv1_8')(x)

    x = Conv2D(128, 1, (1, 1), 'valid', activation='relu', name='conv1_9')(x)

    x = Dropout(rate=0.5)(x)

    logits = Conv2D(2, 1, (1, 1), 'valid', activation=None, name='conv1_10')(x)

    logits = tf.squeeze(logits, [2])  # BxNxC

    return logits


def BoxEstimationNetwork(object_point_cloud, one_hot_vec, NUM_HEADING_BIN=12,
                         NUM_SIZE_CLUSTER=8):
    """Amodal Box Regression network which provides Box parameters as output
        # Arguments
            object_point_cloud: Tensor.  Point Cloud after performing instance
                segmentation.
            one_hot_vector: Tensor. One hot vector of the class to which the
                pointcloud belongs to.

        # Reference
            - [Frustum PointNet](https://arxiv.org/abs/1711.08488)
    """
    num_point = object_point_cloud.get_shape()[1]
    input = tf.expand_dims(object_point_cloud, 2)

    x = Conv2D(128, 1, (1, 1), 'valid', activation='relu',
               name='conv2_1')(input)

    x = Conv2D(128, 1, (1, 1), 'valid', activation='relu', name='conv2_2')(x)

    x = Conv2D(256, 1, (1, 1), 'valid', activation='relu', name='conv2_3')(x)

    x = Conv2D(512, 1, (1, 1), 'valid', activation='relu', name='conv2_4')(x)

    x = MaxPooling2D(pool_size=[num_point, 1], padding='VALID')(x)

    x = tf.squeeze(x, axis=[1, 2])
    x = tf.concat([x, one_hot_vec], axis=1)

    x = Dense(units=512, activation='relu')(x)
    x = Dense(units=256, activation='relu')(x)

    output = Dense(units=3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4,
                   activation=None)(x)

    return output


def SpatialTransformerNetwork(object_point_cloud, one_hot_vec):
    """A Transformer Network to project object pointcloud into space
    invariant frame
        # Arguments
            object_point_cloud: Tensor.  Point Cloud after performing instance
                segmentation.
            one_hot_vector: Tensor. One hot vector of the class to which the
                pointcloud belongs to.

        # Reference
            - [Frustum PointNet](https://arxiv.org/abs/1711.08488)
    """
    num_point = object_point_cloud.get_shape()[1]
    input = tf.expand_dims(object_point_cloud, 2)

    x = Conv2D(128, 1, (1, 1), 'valid', activation='relu',
               name='conv3_1')(input)

    x = Conv2D(128, 1, (1, 1), 'valid', activation='relu', name='conv3_2')(x)
    x = Conv2D(256, 1, (1, 1), 'valid', activation='relu', name='conv3_3')(x)
    x = MaxPooling2D(pool_size=[num_point, 1], padding='VALID')(x)

    x = tf.squeeze(x, axis=[1, 2])
    x = tf.concat([x, one_hot_vec], axis=1)

    x = Dense(units=256, activation='relu')(x)
    x = Dense(units=128, activation='relu')(x)
    predicted_center = Dense(units=3, activation=None)(x)

    return predicted_center


def FrustumPointNetModel(point_cloud_shape=(1024, 3), one_hot_vec_shape=(3,),
                         mask_label_shape=(1024,), center_label_shape=(3,),
                         heading_class_label_shape=(),
                         heading_residual_label_shape=(),
                         size_class_label_shape=(),
                         size_residual_label_shape=(3,), batch_size=32):
    """ Loss functions for 3D object detection.
        Input:
            Frustum_Point_Cloud: TF int32 tensor in shape (B,N)
            One_hot_vector: TF int32 tensor in shape (B,N)
            mask_label: TF int32 tensor in shape (B,N)
            center_label: TF tensor in shape (B,3)
            heading_class_label: TF int32 tensor in shape (B,)
            heading_residual_label: TF tensor in shape (B,)
            size_class_label: TF tensor int32 in shape (B,)
            size_residual_label: TF tensor tensor in shape (B,)
        Output:
            Training Model: Frustum PointNet model with embedded loss layer
                            that is used for training
            Detection Model: Frustum PointNet model that can output Bounding
                             box parameters
        References:
            - [Frustum PointNets for 3D Object Detection from RGB-D Data]
            (https://arxiv.org/abs/1711.08488)
    """
    IntermediateOutputs = {}

    point_cloud = Input(point_cloud_shape, name="frustum_point_cloud",
                        batch_size=batch_size)
    one_hot_vector = Input(one_hot_vec_shape, name="one_hot_vec",
                        batch_size=batch_size)
    mask_label = Input(mask_label_shape, name="segmentation_label",
                       batch_size=batch_size)
    center_label = Input(center_label_shape, name="box3d_center",
                         batch_size=batch_size)
    heading_class_label = Input(heading_class_label_shape, name="angle_class",
                                batch_size=batch_size)
    heading_residual_label = Input(heading_residual_label_shape,
                                   name="angle_residual", batch_size=batch_size)
    size_class_label = Input(size_class_label_shape, name="size_class",
                             batch_size=batch_size)
    size_residual_label = Input(size_residual_label_shape, name="size_residual",
                                batch_size=batch_size)

    logits = InstanceSegmentationNet(point_cloud, one_hot_vector)  # bs,n,2
    IntermediateOutputs['mask_logits'] = logits

    # Mask Point Centroid
    object_point_cloud_xyz, mask_xyz_mean, IntermediateOutputs = \
        PointCloudTranslation(point_cloud, logits, IntermediateOutputs)

    # T-Net
    center_delta = SpatialTransformerNetwork(object_point_cloud_xyz,
                                             one_hot_vector)  # (32,3)

    stage1_center = center_delta + mask_xyz_mean  # Bx3
    IntermediateOutputs['stage1_center'] = stage1_center
    # Get object point cloud in object coordinate
    object_point_cloud_xyz_new = object_point_cloud_xyz - tf.expand_dims(
        center_delta, 1)

    # 3D Box Estimation
    box_pred = BoxEstimationNetwork(object_point_cloud_xyz_new, one_hot_vector)

    IntermediateOutputs = ModelOutputToTensor(box_pred, IntermediateOutputs)
    IntermediateOutputs['center'] = IntermediateOutputs['center_boxnet'] + \
                                    stage1_center  # Bx3

    logits = IntermediateOutputs['mask_logits']
    heading_scores = IntermediateOutputs['heading_scores']  # BxNUM_HEADING_BIN
    heading_residual = IntermediateOutputs['heading_residuals']
    size_scores = IntermediateOutputs['size_scores']
    size_residual = IntermediateOutputs['size_residuals']
    box3d_center = IntermediateOutputs['center']

    loss = FrustumPointNetLoss().TotalLoss([mask_label, center_label,
                                            heading_class_label,
                                            heading_residual_label,
                                            size_class_label,
                                            size_residual_label,
                                            IntermediateOutputs])

    training_model = Model([point_cloud, one_hot_vector, mask_label,
                            center_label, heading_class_label,
                            heading_residual_label, size_class_label,
                            size_residual_label], loss,
                           name='f_pointnet_train')

    det_model = Model(inputs=[point_cloud, one_hot_vector],
                      outputs=[logits, box3d_center, heading_scores,
                               heading_residual, size_scores, size_residual],
                      name='f_pointnet_inference')
    training_model.summary()

    return training_model, det_model


if __name__ == '__main__':
    FrustumPointNetModel()
