from HandPoseEstimation import HandSegmentationNet, PosePriorNet, PoseNet
from HandPoseEstimation import ViewPointNet
from paz.backend.image.opencv_image import load_image, show_image
from pipelines import DetectHandKeypoints

use_pretrained = True
HandSegNet = HandSegmentationNet(weights=None)
HandPoseNet = PoseNet(weights=None)
HandPosePriorNet = PosePriorNet(weights=None)
HandViewPointNet = ViewPointNet(weights=None)
HandSegNet.load_weights('./pretrained_weights/HandSegNet-RHDv2_weights.hdf5')
HandPoseNet.load_weights('./pretrained_weights/PoseNet-RHDv2_weights.hdf5')
HandPosePriorNet.load_weights(
    './pretrained_weights/PosePriorNet-RHDv2_weights.hdf5')
HandViewPointNet.load_weights(
    './pretrained_weights/ViewPointNet-RHDv2_weights.hdf5')
pipeline = DetectHandKeypoints(HandSegNet, HandPoseNet, HandPosePriorNet,
                               HandViewPointNet)

img = load_image('./sample.jpg')
detection = pipeline(img)

show_image(detection['image'].astype('uint8'))

img = load_image('./images/00149.png')
detection = pipeline(img)

show_image(detection['image'].astype('uint8'))

img = load_image('./images/img5.png')
detection = pipeline(img)

show_image(detection['image'].astype('uint8'))