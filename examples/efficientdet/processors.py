from paz.abstract import Processor
from boxes import match


class MatchBoxes(Processor):
    """Match prior boxes with ground truth boxes.

    # Arguments
        prior_boxes: Numpy array of shape (num_boxes, 4).
        iou: Float in [0, 1]. Intersection over union in which prior boxes
            will be considered positive. A positive box is box with a class
            different than `background`.
        variance: List of two floats.
    """
    def __init__(self, prior_boxes, iou=.5):
        self.prior_boxes = prior_boxes
        self.iou = iou
        super(MatchBoxes, self).__init__()

    def call(self, boxes):
        boxes = match(boxes, self.prior_boxes, self.iou)
        return boxes
