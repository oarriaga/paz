import numpy as np

from layer import SegmentationDilation
from paz import processors as pr
from paz.abstract import Processor, SequentialProcessor
from pipelines import PostProcessKeypoints
from processors import ExtractBoundingbox, AdjustCropSize, CropImage
from processors import MergeDictionaries, ExtractKeypoints
from processors import Resize_image, TransformKeypoints


class DetectHandKeypoints(Processor):
    def __init__(self, handsegnet, posenet, posepriornet, viewpointnet,
                 image_size=320, crop_shape=(256, 256), num_keypoints=21):
        super(DetectHandKeypoints, self).__init__()
        preprocess_image = SequentialProcessor(
            [pr.NormalizeImage(),
             pr.ResizeImage((image_size, image_size)),
             pr.ExpandDims(0)]
        )
        postprocess_segmentation = SequentialProcessor(
            [pr.UnpackDictionary(['image', 'raw_segmentation_map']),
             pr.ControlMap(Resize_image(size=(image_size, image_size)),
                           [1], [1]),
             pr.ControlMap(SegmentationDilation(), [1], [1]),
             pr.ControlMap(ExtractBoundingbox(), [1], [2, 3, 4],
                           keep={1: 1}),
             pr.ControlMap(AdjustCropSize(), [4], [4]),
             pr.ControlMap(CropImage(crop_size=crop_shape[0]), [0, 2, 4],
                           [0], keep={2: 2, 4: 4})]
        )
        self.localize_hand = pr.Predict(handsegnet, preprocess=preprocess_image,
                                        postprocess=postprocess_segmentation)

        self.merge_dictionaries = MergeDictionaries()

        self.predict_keypoints2D = pr.Predict(posenet)
        self.predict_keypoints3D = pr.Predict(posepriornet)
        self.predict_keypoints_angles = pr.Predict(viewpointnet)
        self.postprocess_keypoints = PostProcessKeypoints()
        self.resize = Resize_image(crop_shape)
        self.extract_2D_keypoints = ExtractKeypoints()
        self.transform_keypoints = TransformKeypoints()
        self.draw_keypoint = pr.DrawKeypoints2D(num_keypoints)
        self.denormalize = pr.DenormalizeImage()
        self.wrap = pr.WrapOutput(['image', 'keypoints2D'])

    def call(self, image, hand_side=np.array([[1.0, 0.0]])):
        hand_crop, segmentation_map, center, _, crop_size_best = \
            self.localize_hand(image)

        score_maps = self.predict_keypoints2D(hand_crop)

        hand_side = {'hand_side': hand_side}
        score_maps = self.merge_dictionaries([score_maps, hand_side])

        score_maps_resized = self.resize(score_maps['score_maps'])
        keypoints_2D = self.extract_2D_keypoints(score_maps_resized)

        rotation_parameters = self.predict_keypoints3D(score_maps)
        viewpoints = self.predict_keypoints_angles(score_maps)

        canonical_keypoints = self.merge_dictionaries([rotation_parameters,
                                                       viewpoints])
        relative_keypoints = self.postprocess_keypoints(canonical_keypoints)
        tranformed_keypoints_2D = \
            self.transform_keypoints(keypoints_2D, center, crop_size_best, 256)
        image = self.draw_keypoint(np.squeeze(hand_crop), keypoints_2D)
        image = self.denormalize(image)
        output = self.wrap(image.astype('uint8'), keypoints_2D)
        return output
