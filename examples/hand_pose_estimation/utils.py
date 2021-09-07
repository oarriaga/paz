import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage import transform


def load_pretrained_weights(weights_path, model, num_layers):
    with tf.compat.v1.Session() as sess:

        # import graph
        saver = tf.compat.v1.train.import_meta_graph(weights_path)
        sess.run(tf.compat.v1.global_variables_initializer())
        # load weights for graph
        saver.restore(sess, weights_path[:-5])

        # get all global variables (including model variables)
        global_variables = tf.compat.v1.global_variables()

        # get their name and value and put them into dictionary
        sess.as_default()

        model_variables = {}
        for variable in global_variables:
            try:
                model_variables[variable.name] = variable.eval()
            except:
                print("For var={}, an exception occurred".format(variable.name))

        layer_count = 1  # skip Input layer
        for key_count, weights in enumerate(model_variables.items()):
            if layer_count > num_layers:
                break

            while not model.layers[layer_count].trainable_weights:
                layer_count = layer_count + 1

            if key_count % 2 == 0:
                kernel = weights[1]
                print(kernel.shape)
            else:
                bias = weights[1]
                print(bias.shape)
                model.layers[layer_count].set_weights([kernel, bias])
                layer_count = layer_count + 1

    return model


def visualize_heatmaps(heatmaps):
    """Visualize all 21 heatmaps in a 7x3 grid"""

    fig, axes = plt.subplots(7, 3, figsize=(16, 16))
    print(heatmaps.shape)
    # heatmaps = np.expand_dims(heatmaps, axis=0)

    for i in range(heatmaps.shape[3]):
        img_row = int(i / 3)
        img_col = i % 3

        heatmap = heatmaps[:, :, :, i]

        heatmap = (heatmap - tf.reduce_min(heatmap)) / (
                    tf.reduce_max(heatmap) - tf.reduce_min(heatmap))

        axes[img_row, img_col].imshow(np.squeeze(heatmap), cmap='jet')
    plt.show()


def show_mask(image, name='image', wait=True):
    """Shows RGB image in an external window.

    # Arguments
        image: Numpy array
        name: String indicating the window name.
        wait: Boolean. If ''True'' window stays open until user presses a key.
            If ''False'' windows closes immediately.
    """
    if image.dtype != np.uint8:
        raise ValueError('``image`` must be of type ``uint8``')
    cv2.imshow(name, image)
    if wait:
        while True:
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    return np_image
