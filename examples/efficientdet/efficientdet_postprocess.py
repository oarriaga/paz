import paz.processors as pr
from paz.abstract import SequentialProcessor
from utils import get_class_name_efficientdet


def efficientdet_postprocess(model, outputs, image_scales, raw_images=None):
    postprocessing = SequentialProcessor(
        [pr.Squeeze(axis=None),
         pr.DecodeBoxes(model.prior_boxes, variances=[1, 1, 1, 1]),
         pr.ScaleBox(image_scales), pr.NonMaximumSuppressionPerClass(0.4),
         pr.FilterBoxes(get_class_name_efficientdet('COCO'), 0.4)])
    outputs = postprocessing(outputs)
    draw_boxes2D = pr.DrawBoxes2D(get_class_name_efficientdet('COCO'))
    image = draw_boxes2D(raw_images.astype('uint8'), outputs)
    return image, outputs
