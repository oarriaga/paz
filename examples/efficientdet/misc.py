import h5py
import tensorflow as tf

# Mock input image.
mock_input_image = tf.random.uniform((64, 512, 512, 3),
                                     dtype=tf.dtypes.float32,
                                     seed=1)


def read_hdf5(path):
    """A function to read weights from h5 file."""
    weights = {}
    keys = []
    with h5py.File(path, 'r') as f:
        f.visit(keys.append)
        for key in keys:
            if ':' in key:
                weights[f[key].name] = f[key].value
    return weights


def load_pretrained_weights(model, weight_file_path):
    """
    A self-made manual method to copy weights from
    the official EfficientDet to this implementation.
    """
    pretrained_weights = read_hdf5(weight_file_path)
    assert len(model.weights) == len(pretrained_weights)
    str_appender = ['efficientnet-b0/',
                    'resample_p6/',
                    'fpn_cells/',
                    'class_net/',
                    'box_net/']
    for n, i in enumerate(model.weights):
        name = i.name
        for appenders in str_appender:
            if appenders in name:
                name = '/' + appenders + name
        if 'batch_normalization' in name:
            name = name.replace('batch_normalization',
                                'tpu_batch_normalization')
        if name in list(pretrained_weights.keys()):
            if model.weights[n].shape == pretrained_weights[name].shape:
                model.weights[n] = (pretrained_weights[name])
            else:
                ValueError('Shape mismatch for weights of same name.')
        else:
            ValueError("Weight with %s not found." % name)
    return model
