from copy import deepcopy
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
#tf.config.run_functions_eagerly(True)


#class MultiplyHypothesesAmbiguities():
#    def __init__(self, predictions=None):
#        self.predictions = predictions


class MultipleHypotheses():

    def __init__(self, M=5, epsilon=0.05):
        # Set the current predictions before every batch
        self.predictions = None

        # Dictionary where the key is the original name of the layer
        # and the values are the names of the newly added hypotheses layers
        self.names_hypotheses_layers = dict()

        # Names of the hypotheses layers in the right order
        self.ordered_hypotheses_layer_names = list()

        # Names of the layers to apply the multiple hypotheses loss to
        self.layer_names_multiple_hypotheses_loss = dict()

        # Number of hypotheses
        self.M = M

        # Epsilon (see paper)
        self.epsilon = epsilon

        self.ordered_hypotheses_layer_names = list()
        self.names_hypotheses_layers = dict()

    def multiple_hypotheses_output(self, previous_layer, output_layers_dict):
        output_layers_list = list()
        output_layers_names = list()

        for m in range(self.M):
            for name, layer_orig in output_layers_dict.items():
                hypotheses_layer_name = name + "_" + str(m)
                layer = deepcopy(layer_orig)
                layer._name = hypotheses_layer_name
                output_layers_list.append(layer(previous_layer))
                output_layers_names.append(hypotheses_layer_name)

                if name in self.names_hypotheses_layers:
                    self.names_hypotheses_layers[name].append(hypotheses_layer_name)
                else:
                    self.names_hypotheses_layers[name] = [hypotheses_layer_name]

                self.ordered_hypotheses_layer_names.append(hypotheses_layer_name)

        return output_layers_list, output_layers_names

    def loss_multiple_hypotheses_wrapped(self, loss_fn, original_layer_name):
        def loss_unwrapped(real_image, predicted_image):

            layer_names_multiple_hypotheses_loss = self.names_hypotheses_layers[original_layer_name]
            print("Layer multiple hypotheses names: {}".format(layer_names_multiple_hypotheses_loss))
            predictions = [self.predictions[layer_name] for layer_name in layer_names_multiple_hypotheses_loss]

            # Apply the loss function to all all outputs
            losses = list()
            for predicted_image_multiple_hypotheses in predictions:
                losses.append(loss_fn(real_image, predicted_image_multiple_hypotheses))

            # Stack all losses to a single tensor
            losses = tf.stack(losses)
            print("Losses: {}".format(losses))

            # Find the loss of this specific output
            current_loss = loss_fn(real_image, predicted_image)
            print("Current loss: {}".format(current_loss))

            # Find the minimal loss for each batch
            min_loss = tf.math.reduce_min(losses, axis=0)

            mask_loss = tf.math.less_equal(current_loss, min_loss)
            total_loss = tf.where(mask_loss, current_loss, tf.zeros_like(current_loss))

            # Add average loss of other hypotheses to some degree
            total_loss = (1 - self.epsilon*(self.M/(self.M - 1)))*total_loss + self.epsilon/(self.M-1)*tf.math.reduce_sum(losses, axis=0)
            total_loss = tf.where(mask_loss, total_loss, tf.zeros_like(total_loss))
            print("Total loss: {}".format(total_loss))

            # Just return a single scalar over the whole batch
            # If all the values in total_loss are zero, just return a zero.
            # Otherwise return the average of the non-zero values
            if tf.math.reduce_sum(total_loss) == 0:
                return tf.constant(0., dtype=tf.float32)
            else:
                non_zero_mask = tf.math.not_equal(total_loss, 0)
                non_zero_loss = tf.boolean_mask(total_loss, non_zero_mask)
                return tf.math.reduce_mean(non_zero_loss)

        return loss_unwrapped

    def map_ordered_names_to_output_tensors(self, output_tensors):
        ordered_names_to_output_tensors_dict = dict()
        for output_tensor, ordered_hypotheses_layer_name in zip(output_tensors, self.ordered_hypotheses_layer_names):
            ordered_names_to_output_tensors_dict[ordered_hypotheses_layer_name] = output_tensor
        return ordered_names_to_output_tensors_dict

    def map_layer_names_to_attributes(self, layer_names_to_attributes_dict):
        ''' Receives a dict with {original_layer_name: attribute} and a returns a dict
        that maps all of the new hypotheses layer names to their corresponding attribute'''
        hypotheses_layer_names_to_attributes_dict = dict()
        for orig_layer_name in layer_names_to_attributes_dict:
            for hypotheses_layer_name in self.names_hypotheses_layers[orig_layer_name]:
                hypotheses_layer_names_to_attributes_dict[hypotheses_layer_name] = layer_names_to_attributes_dict[orig_layer_name]

        return hypotheses_layer_names_to_attributes_dict


class MultipleHypothesesCallback(Callback):

    def __init__(self, multipleHypotheses):
        self.multipleHypotheses = multipleHypotheses

    def on_train_batch_begin(self, predictions, logs=None):
        self.multipleHypotheses.predictions = predictions
