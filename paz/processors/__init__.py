# imports are done directly to keep user's auto-complete clean

from .detection import SquareBoxes2D
from .detection import DenormalizeBoxes2D
from .detection import RoundBoxes2D
from .detection import ClipBoxes2D
from .detection import FilterClassBoxes2D
from .detection import CropBoxes2D
from .detection import ToBoxes2D
from .detection import MatchBoxes
from .detection import EncodeBoxes
from .detection import DecodeBoxes
from .detection import NonMaximumSuppressionPerClass
from .detection import FilterBoxes
from .detection import ApplyOffsets
from .detection import CropImage

from .draw import DrawBoxes2D
from .draw import DrawKeypoints2D
from .draw import DrawBoxes3D
from .draw import DrawRandomPolygon

from .image import CastImage
from .image import SubtractMeanImage
from .image import AddMeanImage
from .image import NormalizeImage
from .image import DenormalizeImage
from .image import LoadImage
from .image import RandomSaturation
from .image import RandomBrightness
from .image import RandomContrast
from .image import RandomHue
from .image import ResizeImage
from .image import ResizeImages
from .image import RandomImageBlur
from .image import RandomFlipImageLeftRight
from .image import ConvertColorSpace
from .image import ShowImage
from .image import ImageDataProcessor
from .image import BGR_IMAGENET_MEAN
from .image import RGB_IMAGENET_MEAN

from .geometric import RandomFlipBoxesLeftRight
from .geometric import ToAbsoluteBoxCoordinates
from .geometric import ToNormalizedBoxCoordinates
from .geometric import RandomSampleCrop
from .geometric import Expand
from .geometric import ApplyTranslation
from .geometric import ApplyRandomTranslation

from .keypoints import Render
from .keypoints import DenormalizeKeypoints

from .standard import ControlMap
from .standard import ExpandDomain
from .standard import CopyDomain
from .standard import ExtendInputs
from .standard import SequenceWrapper
from .standard import Predict
from .standard import ToClassName
from .standard import ExpandDims
from .standard import BoxClassToOneHotVector
from .standard import Squeeze
from .standard import Copy
from .standard import Lambda
from .standard import UnpackDictionary
from .standard import WrapOutput
from .standard import Concatenate
from .standard import SelectElement

from .pose import SolvePNP
from .pose import UPNP

from ..backend.image.opencv_image import RGB2BGR
from ..backend.image.opencv_image import BGR2RGB
from ..backend.image.opencv_image import RGB2GRAY
from ..backend.image.opencv_image import RGB2HSV
from ..backend.image.opencv_image import HSV2RGB

TRAIN = 0
VAL = 1
TEST = 2
