from HandPoseEstimation import HandSegmentationNet, PosePriorNet, PoseNet
from HandPoseEstimation import ViewPointNet
from paz.backend.image.opencv_image import load_image, show_image, write_image
from pipelines import DetectHandKeypoints

use_pretrained = True
HandSegNet = HandSegmentationNet()
HandPoseNet = PoseNet()
HandPosePriorNet = PosePriorNet()
HandViewPointNet = ViewPointNet()

pipeline = DetectHandKeypoints(HandSegNet, HandPoseNet, HandPosePriorNet,
                               HandViewPointNet)

image = load_image('./sample.jpg')
detection = pipeline(image)

show_image(detection['image'].astype('uint8'))
write_image('./detection.jpg', detection['image'].astype('uint8'))
