from tensorflow.keras.utils import Sequence
import time
import numpy as np
from .processor import SequentialProcessor

import matplotlib.pyplot as plt


class SequenceExtra(Sequence):
    def __init__(self, pipeline, batch_size, as_list=False):
        if not isinstance(pipeline, SequentialProcessor):
            raise ValueError('``processor`` must be a ``SequentialProcessor``')
        self.output_wrapper = pipeline.processors[-1]
        self.pipeline = pipeline
        self.inputs_name_to_shape = self.output_wrapper.inputs_name_to_shape
        self.labels_name_to_shape = self.output_wrapper.labels_name_to_shape
        self.ordered_input_names = self.output_wrapper.ordered_input_names
        self.ordered_label_names = self.output_wrapper.ordered_label_names
        self.batch_size = batch_size
        self.as_list = as_list

    def make_empty_batches(self, name_to_shape):
        batch = {}
        for name, shape in name_to_shape.items():
            batch[name] = np.zeros((self.batch_size, *shape))
        return batch

    def _to_list(self, batch, names):
        return [batch[name] for name in names]

    def _place_sample(self, sample, sample_arg, batch):
        for name, data in sample.items():
            batch[name][sample_arg] = data

    def _get_unprocessed_batch(self, data, batch_index):
        batch_arg_A = self.batch_size * (batch_index)
        batch_arg_B = self.batch_size * (batch_index + 1)
        unprocessed_batch = data[batch_arg_A:batch_arg_B]
        return unprocessed_batch

    def __getitem__(self, batch_index):
        inputs = self.make_empty_batches(self.inputs_name_to_shape)
        labels = self.make_empty_batches(self.labels_name_to_shape)
        inputs, labels = self.process_batch(inputs, labels, batch_index)
        if self.as_list:
            inputs = self._to_list(inputs, self.ordered_input_names)
            labels = self._to_list(labels, self.ordered_label_names)
        return inputs, labels

    def process_batch(self, inputs, labels, batch_index=None):
        raise NotImplementedError


class ProcessingSequence(SequenceExtra):
    """Sequence generator used for processing samples given in ``data``.

    # Arguments
        processor: Function, used for processing elements of ``data``.
        batch_size: Int.
        data: List. Each element of the list is processed by ``processor``.
        as_list: Bool, if True ``inputs`` and ``labels`` are dispatched as
            lists. If false ``inputs`` and ``labels`` are dispatched as
            dictionaries.
    """
    def __init__(self, processor, batch_size, data, as_list=False):
        self.data = data
        super(ProcessingSequence, self).__init__(
            processor, batch_size, as_list)

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def process_batch(self, inputs, labels, batch_index):
        unprocessed_batch = self._get_unprocessed_batch(self.data, batch_index)

        for sample_arg, unprocessed_sample in enumerate(unprocessed_batch):
            sample = self.pipeline(unprocessed_sample.copy())
            self._place_sample(sample['inputs'], sample_arg, inputs)
            self._place_sample(sample['labels'], sample_arg, labels)
        return inputs, labels


class GeneratingSequence(SequenceExtra):
    """Sequence generator used for generating samples.

    # Arguments
        processor: Function used for generating and processing ``samples``.
        batch_size: Int.
        num_steps: Int. Number of steps for each epoch.
        as_list: Bool, if True ``inputs`` and ``labels`` are dispatched as
            lists. If false ``inputs`` and ``labels`` are dispatched as
            dictionaries.
    """
    def __init__(self, processor, batch_size, num_steps, as_list=False):
        self.num_steps = num_steps
        super(GeneratingSequence, self).__init__(
            processor, batch_size, as_list)

    def __len__(self):
        return self.num_steps

    def process_batch(self, inputs, labels, batch_index):
        for sample_arg in range(self.batch_size):
            sample = self.pipeline()
            self._place_sample(sample['inputs'], sample_arg, inputs)
            self._place_sample(sample['labels'], sample_arg, labels)
        return inputs, labels


class GeneratingSequencePix2Pose(SequenceExtra):
    """Sequence generator used for generating samples.

    # Arguments
        processor: Function used for generating and processing ``samples``.
        batch_size: Int.
        num_steps: Int. Number of steps for each epoch.
        as_list: Bool, if True ``inputs`` and ``labels`` are dispatched as
            lists. If false ``inputs`` and ``labels`` are dispatched as
            dictionaries.
    """
    def __init__(self, processor, model, batch_size, num_steps, color_loss_fn, as_list=False, rotation_matrices=None):
        self.num_steps = num_steps
        self.model = model
        self.color_loss_fn = color_loss_fn
        self.rotation_matrices = rotation_matrices
        super(GeneratingSequencePix2Pose, self).__init__(
            processor, batch_size, as_list)

    def __len__(self):
        return self.num_steps

    def rotate_image(self, image, rotation_matrix):
        # Bring the image in the range between 0 and 1
        image = (image + 1) * 0.5

        mask_image = (np.sum(image, axis=-1) != 0).astype(float)
        mask_image = np.repeat(mask_image[..., np.newaxis], 3, axis=-1)
        image_colors_rotated = image + np.ones_like(image) * 0.0001
        image_colors_rotated = np.einsum('ij,klj->kli', rotation_matrix, image_colors_rotated)
        image_colors_rotated = np.where(np.less(image_colors_rotated, 0),
                                        np.ones_like(image_colors_rotated) + image_colors_rotated, image_colors_rotated)
        image_colors_rotated = np.clip(image_colors_rotated, a_min=0.0, a_max=1.0)
        image_colors_rotated = image_colors_rotated * mask_image

        # Bring the image again in the range between -1 and 1
        image_colors_rotated = (image_colors_rotated * 2) - 1
        return image_colors_rotated

    def process_batch(self, inputs, labels, batch_index):
        start = time.time()
        input_images, samples = list(), list()
        for sample_arg in range(self.batch_size):
            sample = self.pipeline()
            samples.append(sample)
            input_image = sample['inputs'][self.ordered_input_names[0]]
            input_images.append(input_image)

        end = time.time()
        print("Time batch generation: {}".format(end - start))

        input_images = np.asarray(input_images)
        predictions = self.model.predict(input_images)

        # Calculate the errors between the target output and the predicted output
        for sample_arg in range(self.batch_size):
            sample = samples[sample_arg]

            # List of tuples of the form (error, error_image)
            stored_errors = []

            # Iterate over all rotation matrices to find the object position
            # with the smallest error
            for rotation_matrix in self.rotation_matrices:
                color_image_rotated = self.rotate_image(sample['labels']['color_output'], rotation_matrix)
                error_image = np.sum(predictions['color_output_0'][sample_arg] - color_image_rotated, axis=-1, keepdims=True)
                error_value = self.color_loss_fn(np.expand_dims(color_image_rotated, axis=0).astype(np.single), np.expand_dims(predictions['color_output_0'][sample_arg], axis=0).astype(np.single))
                stored_errors.append((error_value, error_image))

            minimal_error_pair = min(stored_errors, key=lambda t: t[0])
            sample['labels'][self.ordered_label_names[0]] = minimal_error_pair[1]
            self._place_sample(sample['inputs'], sample_arg, inputs)
            self._place_sample(sample['labels'], sample_arg, labels)

        return inputs, labels


class GeneratingSequencePix2PoseMultipleHypotheses(SequenceExtra):
    """Sequence generator used for generating samples.

    # Arguments
        processor: Function used for generating and processing ``samples``.
        batch_size: Int.
        num_steps: Int. Number of steps for each epoch.
        as_list: Bool, if True ``inputs`` and ``labels`` are dispatched as
            lists. If false ``inputs`` and ``labels`` are dispatched as
            dictionaries.
    """
    def __init__(self, processor, model, batch_size, num_steps, multipleHypotheses, color_output_layer_name, error_output_layer_name, color_loss_fn, as_list=False):
        self.num_steps = num_steps
        self.model = model
        self.multipleHypotheses = multipleHypotheses
        self.color_output_layer_names = multipleHypotheses.names_hypotheses_layers[color_output_layer_name]
        self.error_output_layer_names = multipleHypotheses.names_hypotheses_layers[error_output_layer_name]
        self.color_loss_fn = color_loss_fn

        super(GeneratingSequencePix2PoseMultipleHypotheses, self).__init__(processor, batch_size, as_list)

        error_output_shape = [multipleHypotheses.M] + self.output_wrapper.labels_name_to_shape[error_output_layer_name]
        self.labels_name_to_shape[error_output_layer_name] = error_output_shape

    def __len__(self):
        return self.num_steps

    def process_batch(self, inputs, labels, batch_index):
        input_images, samples = list(), list()
        for sample_arg in range(self.batch_size):
            sample = self.pipeline()
            samples.append(sample)
            input_image = sample['inputs'][self.ordered_input_names[0]]
            input_images.append(input_image)

        input_images = np.asarray(input_images)
        # We cannot use model.predict here because in this case
        # BatchNormalisation behaves weird
        predictions = self.model(input_images, training=True)
        # Turn all the EagerTensors into Numpy Arrays
        predictions = {k: v.numpy() for k, v in predictions.items()}

        # Calculate the errors between the target output and the predicted output
        for sample_arg in range(self.batch_size):
            sample = samples[sample_arg]

            # List of tuples of the form (error, error_image)
            stored_errors = []

            # Iterate over all color outputs to find the one with the smallest
            # error compared to the real image
            for i, color_output_layer_name in enumerate(self.color_output_layer_names):
                prediction_color_output = predictions[color_output_layer_name][sample_arg]
                real_color_output = sample['labels'][self.ordered_label_names[1]]
                error_image = np.sum(prediction_color_output - real_color_output, axis=-1, keepdims=True)
                error_value = self.color_loss_fn(np.expand_dims(real_color_output, axis=0).astype(np.single), np.expand_dims(prediction_color_output, axis=0).astype(np.single))
                stored_errors.append((i, error_value, error_image))

            # Cool trick: Only for the error output where the corresponding color output is the
            # smallest we place the difference between real color image and predicted color image
            # as the image to train on. Otherwise just place the predicted error image --> nothing to
            # train on
            error_outputs = list()
            # Put all the predictions in a list
            for error_output_layer_name in self.error_output_layer_names:
                error_outputs.append(predictions[error_output_layer_name][sample_arg])

            # Just replace one (minimum error) with the actual error image to train on
            minimal_error = min(stored_errors, key=lambda t: t[1])
            error_outputs[minimal_error[0]] = minimal_error[2]

            sample['labels'][self.ordered_label_names[0]] = np.stack(error_outputs, axis=0)

            print("Minimal error: {}".format(minimal_error[0]))

            self._place_sample(sample['inputs'], sample_arg, inputs)
            self._place_sample(sample['labels'], sample_arg, labels)

        return inputs, labels