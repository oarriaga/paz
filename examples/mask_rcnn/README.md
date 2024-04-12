# Mask R-CNN 5 stage Architecture:
1. **Backbone:**
A backbone is the main feature extractor of Mask R-CNN. Common choices of this 
part are residual networks (ResNets) with or without FPN. For simplicity, 
ResNet without FPN as a backbone is taken. When a raw image is fed into a 
ResNet backbone, data goes through multiple residual bottleneck blocks, 
and turns into a feature map. Feature map from the final convolutional layer of
the backbone contains abstract informations of an image, e.g., different object
instances, their classes and spatial properties. It is then fed to the RPN.

2. **Region Proposal Network:**
RPN stands for Region Proposal Network. Its function is scanning the feature 
map and proposing regions that may have objects in them (Region of Interest 
or RoI). Concretely, a convolutional layer processes the feature map, 
outputs a c-channel tensor whose each spacial vector (also have c channels) 
is associated with an anchor center. A set of anchor boxes with different 
scales and aspect ratios are generated given one anchor center. These anchor 
boxes are different areas that evenly distributed over the whole image and 
cover it completely. Then two sibling 1 by 1 convolutional layers process the 
c-channel tensor. One is a binary classifier. It predicts whether each anchor 
box has an object. It maps each c-channel vector to a k-channel vector 
(represents k anchor boxes with different scales and aspect ratios sharing 
one anchor center). The other is a object bounding-box regressor. It predicts 
the offsets between the true object bounding-box and the anchor box. It maps 
each c-channel vector to a 4k-channel vector. For those overlapped bounding-
boxes that may suggest the same object, we select ones with the highest 
objectness score, and drop the others. It's the Non-max suppression process. 
Thus a bunch of proposed RoIs is obtained.

3. **ROIALign:**
RoIAlign or Region of Interest alignment extracts feature vectors from a 
feature map based on RoI proposed by RPN, and turn them into a fix-sized tensor 
for further processes. This operation can be illustrated by the above figure. 
RoI with their corresponding areas in the feature map by scaling is aligned. 
These regions come in different locations, scales and aspect radios. To get 
feature tensors of uniform shape, we sample over relevant aligned areas of 
the feature map. The white-bordered grid represents the feature map. The 
black-bordered grids represent RoIs. We divide each RoI into a fixed number of 
bins. In each bin, there are 4 dots representing sample locations. 
Feature vectors are sampled on the feature map grid around each dot and compute 
their bilinear interpolation as the dot vector. Then we pool dot vectors within 
one bin to get a smaller fix-sized feature map for each RoI. Each RoI's feature 
map is put into a set of residual bottleneck blocks to extract features 
further. The results represent every RoI's finer feature map and will be 
processed by two following parallel branches: object detection branch and mask 
generation branch.

4. **Object Detection branch:**
After we get individual RoI feature map, we can predict its object category 
and a finer instance bounding-box. This branch is a fully-connected layer 
that maps feature vectors to the final n classes and 4n instance bounding-box 
coordinates.
5. **Mask generation branch:**
On the mask generation branch, we feed RoI feature map to a transposed 
convolutional layer and a convolutional layer successively. This branch is a 
fully convolutional network. One binary segmentation mask is generated for 
one class. Then we pick the output mask according to the class prediction in 
object detection branch. In this way, per-pixel's mask prediction can avoid 
competition between different classes.

## **Losses Used:**
1. **rpn_class_loss** : How well the Region Proposal Network separates
background with objects.
2. **rpn_bbox_loss** : How well the Region Proposal Network localise objects.
3. **mrcnn_bbox_loss** : How well the Mask RCNN localise objects.
4. **mrcnn_class_loss** : How well the Mask RCNN recognise each class of 
object.
5. **mrcnn_mask_loss** : How well the Mask RCNN segment objects. 
## **How to use?**

Run the  shapes_train.py file for training  on the custom Shapes Dataset and 
use shapes_demo.py to test the results.

**NOTE: Specifiy the path of the saved shapes weights after training while 
testing.**

Use the coco_demo.py file to test on television test image using coco 
pretrained weights.

## **TODO List:**

1. Modify the train file to train for COCO dataset.
2. Refractor the utils.py file and use paz library for display function.
3. Recheck the x and y coordinates of bounding box functions in 
mask_rcnn.backened to match the paz format. And retrain again. 