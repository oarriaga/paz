import os
from pipelines import DetectHumanPose2D
from tensorflow.keras.utils import get_file
from paz.backend.image import write_image, load_image
from dataset import JOINT_CONFIG, FLIP_CONFIG


URL = ('https://github.com/oarriaga/altamira-data/releases/download'
       '/v0.10/single_person_test_pose.png')
filename = os.path.basename(URL)
fullpath = get_file(filename, URL, cache_subdir='paz/tests')
image = load_image(fullpath)

dataset = 'COCO'
data_with_center = False
if data_with_center:
    joint_order = JOINT_CONFIG[dataset + '_WITH_CENTER']
    flipped_joint_order = FLIP_CONFIG[dataset + '_WITH_CENTER']
else:
    joint_order = JOINT_CONFIG[dataset]
    flipped_joint_order = FLIP_CONFIG[dataset]


detect = DetectHumanPose2D(joint_order, flipped_joint_order,
                           dataset, data_with_center)
output = detect(image)
write_image('output/result.jpg', output['image'])
