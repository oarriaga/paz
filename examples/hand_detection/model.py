from tensorflow.keras import Model
from paz.models import SSD512
from paz.models.detection.utils import create_multibox_head


def SSD512Custom(num_classes, num_priors=[4, 6, 6, 6, 6, 4, 4], l2_loss=5e-4):
    base_model = SSD512(weights='COCO', trainable_base=False)
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
    return model
