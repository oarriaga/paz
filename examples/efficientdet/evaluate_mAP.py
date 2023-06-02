import os
import tensorflow as tf
from paz.datasets import VOC
from paz.optimization.callbacks import EvaluateMAP
from paz.pipelines import DetectSingleShot
from efficientdet import EFFICIENTDETD0

if __name__ == "__main__":
    weights_path = "./trained_models/efficientdet-d0/"
    evaluation_frequency = 1

    if '.hdf5' in weights_path:
        evaluation_frequency = 1
        weights = [weights_path.split('/')[-1]]
        weights_path = "/".join(weights_path.split('/')[0:-1]) + "/"
    else:
        list_of_files = sorted(filter(lambda x: os.path.isfile(
            os.path.join(weights_path, x)),
            os.listdir(weights_path)))
        weights = [weight_file
                   for weight_file in list_of_files
                   if '.hdf5' in weight_file]

    gpus = tf.config.experimental.list_physical_devices('GPU')
    data_names = [['VOC2007', 'VOC2012'], 'VOC2007']
    data_splits = [['trainval', 'trainval'], 'test']
    data_managers, datasets, evaluation_data_managers = [], [], []
    for data_name, data_split in zip(data_names, data_splits):
        data_manager = VOC('VOCdevkit/', data_split, name=data_name)
        data_managers.append(data_manager)
        datasets.append(data_manager.load_data())
        if data_split == 'test':
            eval_data_manager = VOC(
                'VOCdevkit/', data_split, name=data_name, evaluate=True)
            evaluation_data_managers.append(eval_data_manager)

    model = EFFICIENTDETD0(num_classes=21, base_weights='COCO',
                           head_weights=None)
    for weight in weights[::evaluation_frequency]:
        model.load_weights(weights_path + weight)
        try:
            evaluate = EvaluateMAP(
                evaluation_data_managers[0], DetectSingleShot(
                    model, data_managers[0].class_names, 0.01, 0.45),
                evaluation_frequency, './trained_models/', 0.5)
            epoch = int(weight.split('.')[1].split('-')[0]) - 2
            evaluate.on_epoch_end(epoch, None)
        except:
            pass
