import tensorflow as tf
import tensorflow.keras.backend as K


def smooth_L1_loss(y_true, y_pred):
    diff = tf.math.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), 'float32')
    loss = (less_than_one * 0.5 * diff ** 2) + \
           (1 - less_than_one) * (diff - 0.5)
    return loss


def reshape_data(target_ids, target_masks, y_pred):
    target_ids = K.reshape(target_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks,
                             (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred,
                       (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    y_pred = tf.transpose(y_pred, [0, 3, 1, 2])
    return target_ids, target_masks, y_pred


def call_ROIs():
    """ Decorator function to call the output ROIs.

    # Returns:
        output ROIs [No. of ROIs before nms (y1, x1, y2, x2)]
    """
    def _call_ROIs(value):
        return value * 1

    return _call_ROIs


def gnd_truth_call(image):
    """Decorator function used to call the norm_boxes_graph function.

    # Arguments
        image: Input image in original form [H, W, C].
        boxes: Bounding box in original form [N, (y1, x1, y2, x2)].
    # Returns
        bounding box: Bounding box in normalised form [N, (y1, x1, y2, x2)].
    """
    shape = tf.shape(image)[1:3]

    def _gnd_truth_call(boxes):
        return norm_boxes_graph(boxes, shape)

    return _gnd_truth_call
