import numpy as np

from paz import processors as pr
from paz.abstract import SequentialProcessor, Processor

from HandPoseEstimation import Hand_Segmentation_Net, PosePriorNet, PoseNet
from HandPoseEstimation import ViewPointNet

from hand_keypoints_loader import RenderedHandLoader

from processors import AdjustCropSize, CropImage, CanonicaltoRelativeFrame
from processors import HandSegmentationMap, ExtractBoundingbox, Resize_image
from processors import Merge_Dictionaries, GetRotationMatrix, ExtractKeypoints

from pipelines import preprocess_image, PostprocessSegmentation
from pipelines import Process2DKeypoints, PostProcessKeypoints


class DetectHandKeypoints(Processor):
    def __init__(self, handsegnet, posenet, posepriornet, viewpointnet,
                 shape=(256,256), num_keypoints=21):
        super(DetectHandKeypoints, self).__init__()
        self.preprocess_image = preprocess_image()
        self.postprocess_segmentation = PostprocessSegmentation(handsegnet)
        self.process_keypoints = Process2DKeypoints(posenet)
        self.predict_keypoints3D = pr.Predict(posepriornet)
        self.predict_keypoints_angles = pr.Predict(viewpointnet)
        self.postprocess_keypoints = PostProcessKeypoints()
        self.resize = pr.ResizeImage(shape)
        self.extract_2D_keypoints = ExtractKeypoints()
        self.draw_keypoint = pr.DrawKeypoints2D(num_keypoints)
        self.wrap = pr.WrapOutput(['image', 'keypoints2D'])

    def call(self, image, hand_side = np.array([[1.0, 0.0]])):
        image = self.preprocess_image(image)
        hand_crop, _, _ = self.postprocess_segmentation(image)
        score_maps = self.process_keypoints(hand_crop)
        hand_side = {'hand_side': hand_side}
        score_maps = Merge_Dictionaries()([score_maps, hand_side])
        score_maps_resized = self.resize(score_maps[-1])
        keypoints_2D = self.extract_2D_keypoints(score_maps_resized)
        canonical_coordinates = self.predict_keypoints3D(score_maps)
        viewpoints = self.predict_keypoints_angles(score_maps)
        canonical_keypoints = Merge_Dictionaries()([canonical_coordinates,
                                                    viewpoints])
        relative_keypoints = self.postprocess_keypoints(canonical_keypoints)
        self.draw_keypoint(hand_crop, keypoints_2D)
        return self.wrap(hand_crop, keypoints_2D)