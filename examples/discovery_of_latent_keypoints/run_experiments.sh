#!/bin/bash

class_name="obj_000001"
obj_path="$HOME/.keras/paz/datasets/tless/models_cad/$class_name.ply"
python3 discover_latent_keypoints.py --obj_path $obj_path --class_name $class_name -d 80 -kp 8
python3 discover_latent_keypoints.py --obj_path $obj_path --class_name $class_name -d 80 -kp 10
python3 discover_latent_keypoints.py --obj_path $obj_path --class_name $class_name -d 80 -kp 15
python3 discover_latent_keypoints.py --obj_path $obj_path --class_name $class_name -d 80 -kp 20
python3 discover_latent_keypoints.py --obj_path $obj_path --class_name $class_name -d 80 -kp 30
