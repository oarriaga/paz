#!/bin/bash
keypoints_path="/home/dfki.uni-bremen.de/loarriagacamargo/.keras/paz/models/keypointnet-shared_10_035_power_drill/keypoints_mean.txt"


python3 train.py --keypoints_path $keypoints_path --images_directory /home/dfki.uni-bremen.de/loarriagacamargo/Documents/poseur/data/background_images/ --obj_path /home/dfki.uni-bremen.de/loarriagacamargo/.keras/paz/datasets/ycb/models/035_power_drill/textured.obj --model KeypointNet2D --class_name 035_power_drill
