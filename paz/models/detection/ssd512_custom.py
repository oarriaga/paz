import os
from tensorflow.keras import Model
from tensorflow.keras.utils import get_file
from .ssd512 import SSD512
from paz.models.detection.utils import create_multibox_head


def SSD512Custom(num_classes, weight_path, num_priors=[4, 6, 6, 6, 6, 4, 4],
                 l2_loss=5e-4, trainable_base=False):
    """Custom Single-shot-multibox detector for 512x512x3 BGR input images.
    # Arguments
        num_classes: Integer. Specifies the number of class labels.
        weight_path: String. Weight path trained on custom dataset.
        num_priors: List of integers. Number of default box shapes
            used in each detection layer.
        l2_loss: Float. l2 regularization loss for convolutional layers.
        trainable_base: Boolean. If `True` the base model
            weights are also trained.

    # Reference
        - [SSD: Single Shot MultiBox
            Detector](https://arxiv.org/abs/1512.02325)
    """
    base_model = SSD512(weights='COCO', trainable_base=trainable_base)
    branch_names = ['branch_1', 'branch_2', 'branch_3', 'branch_4',
                    'branch_5', 'branch_6', 'branch_7']
    branch_tensors = []
    for branch_name in branch_names:
        branch_layer = base_model.get_layer(branch_name)
        branch_tensors.append(branch_layer.output)

    output_tensor = create_multibox_head(
        branch_tensors, num_classes, num_priors, l2_loss)
    model = Model(base_model.input, output_tensor, name='SSD512Custom')
    model.prior_boxes = base_model.prior_boxes
    filename = os.path.basename(weight_path)
    weights_path = get_file(filename, weight_path, cache_subdir='paz/models')
    print('==> Loading %s model weights' % weights_path)
    model.load_weights(weights_path)
    return model
