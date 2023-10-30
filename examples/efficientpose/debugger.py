
import argparse
import numpy as np
import cv2
from paz.backend.image import show_image
from pose import get_class_names
from paz.abstract import ProcessingSequence
from paz.processors import TRAIN, VAL
from pose import (AugmentPose, EFFICIENTPOSEA, LINEMOD_CAMERA_MATRIX,
                  LINEMOD_OBJECT_SIZES, EfficientPosePreprocess,
                  EfficientPosePostprocess)
from linemod import LINEMOD
import numpy as np
from paz.abstract import Processor
import paz.processors as pr
from paz.backend.image import lincolor
from efficientpose import EFFICIENTPOSEA
from processors import DrawPose6D


description = 'Training script for single-shot object detection models'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-dp', '--data_path', default='Linemod_preprocessed/',
                    type=str, help='Path for writing model weights and logs')
parser.add_argument('-id', '--object_id', default='08',
                    type=str, help='ID of the object to train')
parser.add_argument('-bs', '--batch_size', default=1, type=int,
                    help='Batch size for training')
args = parser.parse_args()

data_splits = ['train', 'test']
data_names = ['LINEMOD', 'LINEMOD']

# loading datasets
data_managers, datasets, evaluation_data_managers = [], [], []
for data_name, data_split in zip(data_names, data_splits):
    data_manager = LINEMOD(args.data_path, args.object_id,
                           data_split, name=data_name)
    data_managers.append(data_manager)
    datasets.append(data_manager.load_data())
    if data_split == 'test':
        eval_data_manager = LINEMOD(
            args.data_path, args.object_id, data_split,
            name=data_name, evaluate=True)
        evaluation_data_managers.append(eval_data_manager)

num_classes = data_managers[0].num_classes

# instantiating model
model = EFFICIENTPOSEA(num_classes, base_weights='COCO', head_weights=None)

# setting data augmentation pipeline
augmentators = []
for split in [TRAIN, VAL]:
    augmentator = AugmentPose(model, split, size=512, num_classes=num_classes)
    augmentators.append(augmentator)

sequencers = []
for data, augmentator in zip(datasets, augmentators):
    sequencer = ProcessingSequence(augmentator, args.batch_size, data)
    sequencers.append(sequencer)


class DetectAndEstimateEfficientPose(Processor):
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 LINEMOD_CAMERA_MATRIX, LINEMOD_OBJECT_SIZES, preprocess=None,
                 postprocess=None, variances=[0.1, 0.1, 0.2, 0.2],
                 show_boxes2D=False, show_poses6D=True):
        self.model = model
        self.class_names = class_names
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.variances = variances
        self.class_to_sizes = LINEMOD_OBJECT_SIZES
        self.camera_matrix = LINEMOD_CAMERA_MATRIX
        self.colors = lincolor(len(self.class_to_sizes.keys()))
        self.show_boxes2D = show_boxes2D
        self.show_poses6D = show_poses6D
        if preprocess is None:
            self.preprocess = EfficientPosePreprocess(model)
        if postprocess is None:
            self.postprocess = EfficientPosePostprocess(
                model, class_names, score_thresh, nms_thresh, class_arg=0)

        super(DetectAndEstimateEfficientPose, self).__init__()
        self.draw_boxes2D = pr.DrawBoxes2D(self.class_names)
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'poses6D'])

    def _build_draw_pose6D(self, name_to_size, camera_parameter):
        name_to_draw = {}
        iterator = zip(name_to_size.items(), self.colors)
        for (name, object_size), box_color in iterator:
            draw = DrawPose6D(object_size, camera_parameter, box_color)
            name_to_draw[name] = draw
        return name_to_draw

    def call(self, image, detections, transformations):
        preprocessed_data = self.preprocess(normalized_image)
        _, _, camera_parameter = preprocessed_data
        outputs = detections, transformations
        boxes2D, poses6D = self.postprocess(
            image, outputs, 1.0, camera_parameter)
        if self.show_boxes2D:
            image = self.draw_boxes2D(image, boxes2D)

        if self.show_poses6D:
            self.draw_pose6D = self._build_draw_pose6D(
                self.class_to_sizes, self.camera_matrix)
            for box2D, pose6D in zip(boxes2D, poses6D):
                image = self.draw_pose6D[box2D.class_name](image, pose6D)
        return self.wrap(image, boxes2D, poses6D)


class EFFICIENTPOSEALINEMODDRILLER(DetectAndEstimateEfficientPose):
    def __init__(self, score_thresh=0.60, nms_thresh=0.45,
                 show_boxes2D=False, show_poses6D=True):
        names = get_class_names('LINEMOD_EFFICIENTPOSE_DRILLER')
        model = EFFICIENTPOSEA(num_classes=len(names), base_weights='COCO',
                               head_weights=None,  momentum=0.99,
                               epsilon=0.001, activation='softmax')
        super(EFFICIENTPOSEALINEMODDRILLER, self).__init__(
            model, names, score_thresh, nms_thresh,
            LINEMOD_CAMERA_MATRIX, LINEMOD_OBJECT_SIZES,
            show_boxes2D=show_boxes2D, show_poses6D=show_poses6D)

detect = EFFICIENTPOSEALINEMODDRILLER(score_thresh=0.5, nms_thresh=0.45,
                                      show_boxes2D=True, show_poses6D=False)

# Display input image
for i in range(len(sequencers[0])):
    seq = sequencers[0][i]
    image = seq[0]['image']
    normalized_image = 255 * (image - image.min()) / (image.max() - image.min())
    normalized_image = normalized_image.astype(np.uint8)[0]
    # cv2.imshow('Input Image', normalized_image)

    # Display 2D bounding box image
    boxes = seq[1]['boxes']

    # Display matched
    rotations = seq[1]['transformation'][0, :, :3]
    translations = seq[1]['transformation'][0, :, 6:9]
    transformation = np.concatenate((rotations, translations), axis=1)
    transformation = np.expand_dims(transformation, axis=0)
    inferences = detect(normalized_image, boxes, transformation)


    show_image(inferences['image'])
    # cv2.waitKey(10)
# sequencers[0][0][1]['boxes']
print("k")
