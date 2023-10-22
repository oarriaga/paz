import paz.processors as pr
from paz.abstract import SequentialProcessor
from paz.pipelines.detection import PreprocessBoxes
from pose import (EfficientPosePreprocess, RGB_LINEMOD_MEAN,
                  LINEMOD_CAMERA_MATRIX)
from processors import (MatchPoses, TransformRotation,
                        ConcatenatePoses, ConcatenateScale)


class AugmentPose(SequentialProcessor):
    def __init__(self, model, num_classes=8, size=512, mean=RGB_LINEMOD_MEAN,
                 camera_matrix=LINEMOD_CAMERA_MATRIX, IOU=.5,
                 variances=[0.1, 0.1, 0.2, 0.2], num_pose_dims=3):
        super(AugmentPose, self).__init__()
        self.preprocess_image = EfficientPosePreprocess(model, mean,
                                                        camera_matrix)

        # box processors
        self.scale_boxes = pr.ScaleBox()
        args = (num_classes, model.prior_boxes, IOU, variances)
        self.preprocess_boxes = PreprocessBoxes(*args)

        # pose processors
        self.match_poses = MatchPoses(model.prior_boxes)
        self.concatenate_poses = ConcatenatePoses()
        self.concatenate_scale = ConcatenateScale()

        # pipeline
        self.add(pr.UnpackDictionary(['image', 'boxes', 'rotation',
                                      'translation_raw', 'class']))
        self.add(pr.ControlMap(pr.LoadImage(), [0], [0]))
        self.add(pr.ControlMap(self.preprocess_image, [0], [0, 1, 2]))
        self.add(pr.ControlMap(self.scale_boxes, [3, 1], [3], keep={1: 1}))
        self.add(pr.ControlMap(self.preprocess_boxes, [4], [5], keep={4: 4}))
        self.add(pr.ControlMap(TransformRotation(num_pose_dims), [3], [3]))
        self.add(pr.ControlMap(self.match_poses, [4, 3], [3], keep={4: 4}))
        self.add(pr.ControlMap(self.match_poses, [4, 5], [8]))
        self.add(pr.ControlMap(self.concatenate_poses, [3, 6], [8]))
        self.add(pr.ControlMap(self.concatenate_scale, [5, 1], [8]))
        self.add(pr.SequenceWrapper(
            {0: {'image': [size, size, 3]}},
            {2: {'boxes': [len(model.prior_boxes), 4 + num_classes]},
             4: {'transformation': [len(model.prior_boxes),
                                    3 * num_pose_dims + 2]}}))
