import numpy as np

from ..core.sequencer import Sequencer
from ..core.ops.numpy_ops import match
from ..core.ops.numpy_ops import to_one_hot
from ..core.ops.opencv_ops import load_image


class SingleShotSequencer(Sequencer):
    """ Detection sequencer used for single shot architectures.
    # Arguments
        data: List of dictionaries with keys `inputs` and `targets` containing
            each of those a dictionary with specific data type
            e.g. `RGB`, `box_data`.
            These elements are often created using one of the default datasets
            loader: e.g. VOC, YCBVideo, FAT located in paz/datasets/
        batch_size: Int. Batch size to be dispatched for optimization.
        prior_boxes: Prior boxes of single shot model. Needed for matching
            with ground truth boxes.
        pipeline: Function that takes as input
            (image_array, box_coordinates and class_ids) and returns a
            dictionary with the modified/augmented data. This dictionary
            should contain as keys `image`, `boxes2D`.
        num_classes: Integer. Number of classes.
    """
    def __init__(self, data, batch_size, prior_boxes, pipeline, num_classes):
        self.prior_boxes = prior_boxes
        self.num_classes = num_classes
        self.pipeline = pipeline
        super(SingleShotSequencer, self).__init__(data, batch_size)

    def _preprocess_sample(self, sample):
        image_path = sample['inputs']['image']
        ground_truth_data = sample['targets']['box_data']
        image_array = load_image(image_path)
        box_coordinates, class_ids = self._unpack_data(ground_truth_data)
        data = (image_array, box_coordinates, class_ids)
        image_array, box_coordinates, class_ids = self.pipeline(*data)
        box_data = np.concatenate([box_coordinates, class_ids], axis=1)
        box_data = match(box_data, self.prior_boxes, .5, [.1, .2])
        box_coordinates, class_ids = self._unpack_data(box_data)
        class_ids = to_one_hot(class_ids, self.num_classes)
        box_data = np.concatenate(
            [box_coordinates, class_ids], axis=-1)
        return image_array, box_data

    def _unpack_data(self, ground_truth_data):
        num_objects = len(ground_truth_data)
        box_coordinates = np.zeros((num_objects, 4))
        class_labels = np.zeros((num_objects, 1), dtype=np.uint32)
        for box_arg, box_data in enumerate(ground_truth_data):
            box_coordinates[box_arg] = box_data[:4]
            class_labels[box_arg] = box_data[4:]
        return box_coordinates, class_labels
