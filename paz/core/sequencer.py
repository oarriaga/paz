from tensorflow.keras.utils import Sequence
import numpy as np


class Sequencer(Sequence):
    """Abstract base sequencer class for dispatching batches.

    # Arguments
        data: List of dictionaries. The length of the list corresponds to the
            amount of samples in the data. Inside each sample there should
            be a dictionary with `keys` indicating the data types/topics
            e.g. ``image``, ``depth``, ``boxes`` and as `values` of these
            `keys` the corresponding data e.g. numpy arrays.
        processor: Function that takes a sample (dictionary) as input and
            returns a list of outputs.
        tensor_shapes: List of lists. Each element of the list contains a list
            indicating the shape of the data
            e.g. ((32, 128, 128, 1), (32, 8732, 21))
            In the case above the first shape could correspond to an image with
            the shape indicating (batch_shape, height, width, num_channels),
            while the second shape indicates the bounding boxes in that image.
            (batch_size, num_bounding_boxes, box_coordinates + num_classes)
            The shapes must correspond to the output shapes of the processor.
        tensor_names: List of strings. Each string is the name of the tensor
            in which that data topic will be given to.
            The list must have the same order as the processor output types.
        batch_size: Integer. Batch size.
    """
    def __init__(self, data, processor, batch_size=32, tensor_names=None):

        self.data = data
        self.batch_size = batch_size
        self.processor = processor
        self.tensor_names = tensor_names
        if self.tensor_names is None:
            self.tensor_names = data[0].keys()
        self.shapes = self.processor.output_shapes

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, batch_index):
        batch_arg_A = self.batch_size * (batch_index)
        batch_arg_B = self.batch_size * (batch_index + 1)
        batch_samples = self.data[batch_arg_A:batch_arg_B]
        batches = [np.zeros((self.batch_size, *size)) for size in self.shapes]
        for sample_arg, unprocessed_sample in enumerate(batch_samples):
            sample = self.processor(**unprocessed_sample)
            for data_arg, data in enumerate(sample):
                batches[data_arg][sample_arg] = data
        # return dict(zip(self.tensor_names, batches))
        # dictionaries are not supported as output
        return batches[0], batches[1]
