import cv2
import numpy as np
import paz.processors as pr
from paz.abstract import Processor
from paz.backend.image import show_image
from paz.abstract import ProcessingSequence
from linemod import (Linemod, LINEMOD_CAMERA_MATRIX, LINEMOD_OBJECT_SIZES,
                     RGB_LINEMOD_MEAN)
from paz.models.pose_estimation import EfficientPosePhi0
from paz.pipelines.pose import AugmentEfficientPose, EfficientPosePreprocess
from anchors import build_translation_anchors

raw_image_shape = (640, 480)
input_shape = 512


class EstimateEfficientPose(Processor):
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 object_sizes=LINEMOD_OBJECT_SIZES, mean=RGB_LINEMOD_MEAN,
                 camera_matrix=LINEMOD_CAMERA_MATRIX,
                 variances=[0.1, 0.1, 0.2, 0.2], show_boxes2D=False,
                 show_poses6D=True):
        self.model = model
        self.class_names = class_names
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.variances = variances
        self.class_to_sizes = object_sizes
        self.camera_matrix = camera_matrix
        self.show_boxes2D = show_boxes2D
        self.show_poses6D = show_poses6D
        self.preprocess = EfficientPosePreprocess(model, mean)
        self.postprocess = EfficientPosePostprocess(
                model, class_names, score_thresh, nms_thresh, class_arg=0)

        super(EstimateEfficientPose, self).__init__()
        self.draw_boxes2D = pr.DrawBoxes2D(self.class_names)
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'poses6D'])

    def _build_draw_pose6D(self, name_to_size, camera_parameter):
        name_to_draw = {}
        for name, object_size in name_to_size.items():
            draw = pr.DrawPose6D(object_size, camera_parameter)
            name_to_draw[name] = draw
        return name_to_draw

    def call(self, image, detections, transformations):
        outputs = detections, transformations
        boxes2D, poses6D = self.postprocess(outputs)
        if self.show_boxes2D:
            image = self.draw_boxes2D(image, boxes2D)

        if self.show_poses6D:
            self.draw_pose6D = self._build_draw_pose6D(
                self.class_to_sizes, self.camera_matrix)
            for box2D, pose6D in zip(boxes2D, poses6D):
                image = self.draw_pose6D[box2D.class_name](image, pose6D)
        return self.wrap(image, boxes2D, poses6D)


class EfficientPosePostprocess(Processor):
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 variances=[0.1, 0.1, 0.2, 0.2], class_arg=None,
                 num_pose_dims=3):
        super(EfficientPosePostprocess, self).__init__()
        self.num_pose_dims = num_pose_dims
        self.postprocess_1 = pr.SequentialProcessor([
            pr.Squeeze(axis=None),
            pr.DecodeBoxes(model.prior_boxes, variances),
            pr.RemoveClass(class_names, class_arg)])
        self.postprocess_2 = pr.SequentialProcessor([
            pr.NonMaximumSuppressionPerClass(nms_thresh),
            pr.MergeNMSBoxWithClass(),
            pr.FilterBoxes(class_names, score_thresh)])
        self.to_boxes2D = pr.ToBoxes2D(class_names)
        self.round_boxes = pr.RoundBoxes2D()
        self.denormalize = DenormalizeBoxes2D()
        self.compute_common_rows = pr.ComputeCommonRowIndices()
        self.squeeze = pr.Squeeze(axis=0)
        self.transform_rotations = pr.Scale(np.pi)
        self.to_pose_6D = pr.ToPose6D(class_names)

    def call(self, model_output):
        detections, transformations = model_output
        box_data = self.postprocess_1(detections)
        box_data_all = box_data
        box_data = self.postprocess_2(box_data)
        boxes2D = self.to_boxes2D(box_data)
        boxes2D = self.denormalize(boxes2D)
        boxes2D = self.round_boxes(boxes2D)

        rotations = transformations[:, :, :self.num_pose_dims]
        translations = transformations[:, :, self.num_pose_dims:]
        poses6D = []
        if len(boxes2D) > 0:
            selected_indices = self.compute_common_rows(box_data_all, box_data)
            rotations = self.squeeze(rotations)
            rotations = rotations[selected_indices]
            rotations = self.transform_rotations(rotations)
            translations = self.squeeze(translations)
            translations = translations[selected_indices]

        poses6D = self.to_pose_6D(box_data, rotations, translations)
        return boxes2D, poses6D


class DenormalizeBoxes2D(Processor):
    def __init__(self):
        super(DenormalizeBoxes2D, self).__init__()

    def call(self, boxes2D):
        for box2D in boxes2D:
            box2D.coordinates = denormalize_box(box2D.coordinates)
        return boxes2D


def denormalize_box(box):
    x_min, y_min, x_max, y_max = box[:4]
    x_min = int(x_min * raw_image_shape[0])
    y_min = int(y_min * raw_image_shape[0])
    x_max = int(x_max * raw_image_shape[0])
    y_max = int(y_max * raw_image_shape[0])
    return (x_min, y_min, x_max, y_max)


class EfficientPosePhi0LinemodDebug(EstimateEfficientPose):
    def __init__(self, score_thresh=0.60, nms_thresh=0.45,
                 show_boxes2D=False, show_poses6D=True):
        names = ['background', 'driller']
        model = EfficientPosePhi0(build_translation_anchors,
                                  num_classes=len(names), base_weights='COCO',
                                  head_weights=None)
        super(EfficientPosePhi0LinemodDebug, self).__init__(
            model, names, score_thresh, nms_thresh,
            LINEMOD_OBJECT_SIZES, LINEMOD_CAMERA_MATRIX,
            show_boxes2D=show_boxes2D, show_poses6D=show_poses6D)


def deprocess_image(image):
    image = image[:384, :, :]
    image = cv2.resize(image, (640, 480))
    image = 255 * (image - image.min()) / (image.max() - image.min())
    image = image.astype(np.uint8)
    return image


if __name__ == '__main__':
    data_path = 'Linemod_preprocessed/'
    object_id = '08'
    split = pr.TRAIN
    data_split = 'train'
    data_name = 'Linemod'

    # loading datasets
    data_manager = Linemod(data_path, object_id, data_split, name=data_name)
    dataset = data_manager.load_data()
    num_classes = data_manager.num_classes

    # instantiating model
    model = EfficientPosePhi0(build_translation_anchors, num_classes,
                              base_weights='COCO', head_weights=None)

    # setting data augmentation pipeline
    augmentator = AugmentEfficientPose(model, RGB_LINEMOD_MEAN,
                                       LINEMOD_CAMERA_MATRIX, split,
                                       size=input_shape,
                                       num_classes=num_classes)
    sequencer = ProcessingSequence(augmentator, 1, dataset)

    detect = EfficientPosePhi0LinemodDebug(show_boxes2D=True,
                                           show_poses6D=True)
    for sequence in sequencer:
        image = sequence[0]['image'][0]
        image = deprocess_image(image)
        boxes = sequence[1]['boxes'][0]
        rotations = sequence[1]['transformation'][0][:, :3]
        translations = sequence[1]['transformation'][0][:, 6:9]
        transformation = np.concatenate((rotations, translations), axis=1)
        transformation = np.expand_dims(transformation, axis=0)
        inferences = detect(image, boxes, transformation)
        show_image(inferences['image'])
