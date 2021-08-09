import tensorflow as tf


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
