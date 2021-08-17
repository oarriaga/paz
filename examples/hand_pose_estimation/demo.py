import argparse
import numpy as np

from HandPoseEstimation import Hand_Segmentation_Net, PosePriorNet, PoseNet
from HandPoseEstimation import ViewPointNet
from hand_keypoints_loader import RenderedHandLoader
from paz import processors as pr
from paz.abstract import SequentialProcessor
from processors import AdjustCropSize, CropImage, CanonicaltoRelativeFrame
from processors import HandSegmentationMap, ExtractBoundingbox, Resize_image
from processors import Merge_Dictionaries, GetRotationMatrix

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='Path to dataset')
args = parser.parse_args()

data_manager = RenderedHandLoader(args.data_path, 'val')
data = data_manager.load_data()


class preprocess_image(SequentialProcessor):
    def __init__(self, image_size=320):
        super(preprocess_image, self).__init__()
        self.add(pr.UnpackDictionary(['image_path']))
        self.add(pr.LoadImage())
        self.add(pr.ResizeImage((image_size, image_size)))
        self.add(pr.ConvertColorSpace(pr.RGB2BGR))
        self.add(pr.SubtractMeanImage(pr.BGR_IMAGENET_MEAN))
        self.add(pr.ExpandDims(0))


class PostprocessSegmentation(SequentialProcessor):
    def __init__(self, HandSegNet, image_size=320, crop_size=256):
        super(PostprocessSegmentation, self).__init__()
        self.add(pr.Predict(HandSegNet))
        self.add(pr.UnpackDictionary(['image', 'raw_segmentation_map']))
        self.add(pr.ControlMap(Resize_image(size=(image_size, image_size)),
                               [1], [1]))
        self.add(pr.ControlMap(HandSegmentationMap(), [1], [1]))
        self.add(pr.ControlMap(ExtractBoundingbox(), [1], [2, 3, 4],
                               keep={1: 1}))
        self.add(pr.ControlMap(AdjustCropSize(), [4], [3]))
        self.add(pr.ControlMap(CropImage(crop_size=crop_size), [0, 2, 3],
                               [0]))
        self.add(pr.ControlMap(pr.CastImage('uint8'), [0], [0]))


class Process2DKeypoints(SequentialProcessor):
    def __init__(self, PoseNet):
        super(Process2DKeypoints, self).__init__()
        self.add(pr.Predict(PoseNet))


class PostProcessKeypoints(SequentialProcessor):
    def __init__(self, number_of_keypoints=21):
        super(PostProcessKeypoints, self).__init__()
        self.add(pr.UnpackDictionary(['canonical_coordinates',
                                      'rotation_parameters', 'hand_side']))
        self.add(pr.ControlMap(GetRotationMatrix(), [1], [1]))
        self.add(pr.ControlMap(CanonicaltoRelativeFrame(number_of_keypoints),
                               [0, 1, 2], [0]))


use_pretrained = True
HandSegNet = Hand_Segmentation_Net(load_pretrained=use_pretrained)
HandPoseNet = PoseNet(load_pretrained=use_pretrained)
HandPosePriorNet = PosePriorNet(load_pretrained=use_pretrained)
HandViewPointNet = ViewPointNet(load_pretrained=use_pretrained)

hand_localization = PostprocessSegmentation(HandSegNet)
postprocess_keypoints = PostProcessKeypoints()

# for sample in data:
image = preprocess_image()(data[0])
hand_crop, _, _ = hand_localization(image)
score_maps = Process2DKeypoints(HandPoseNet)(hand_crop)
hand_side = {'hand_side': np.array([[1.0, 0.0]])}
score_maps = Merge_Dictionaries()([score_maps, hand_side])
canonical_coordinates = pr.Predict(HandPosePriorNet)(score_maps)
viewpoints = pr.Predict(HandViewPointNet)(score_maps)
canonical_keypoints = Merge_Dictionaries()([canonical_coordinates,
                                            viewpoints])
relative_keypoints = postprocess_keypoints(canonical_keypoints)

