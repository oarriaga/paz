#!/bin/bash

# class_name="obj_000001"
# obj_path="$HOME/.keras/paz/datasets/tless/models_cad/$class_name.ply"
class_name="035_power_drill_test"
obj_path="/home/robot/datasets/3d_models/klt.obj"
python3 discover_latent_keypoints.py --filepath $obj_path --class_name $class_name --depth 0.30 --num_keypoints 8 --batch_size 10 --smooth False --rotation_noise 0.0 --loss_weights 1.0 1.0 10.0 0.2 0.5
# python3 discover_latent_keypoints.py --obj_path $obj_path --class_name $class_name -d 80 -kp 10
# python3 discover_latent_keypoints.py --obj_path $obj_path --class_name $class_name -d 80 -kp 15
# python3 discover_latent_keypoints.py --obj_path $obj_path --class_name $class_name -d 80 -kp 20
# python3 discover_latent_keypoints.py --obj_path $obj_path --class_name $class_name -d 80 -kp 30
