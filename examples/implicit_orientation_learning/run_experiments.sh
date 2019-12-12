#!/bin/bash
class_name="035_power_drill"
images_directory="$HOME/.keras/paz/datasets/voc-backgrounds/"
object_path="$HOME/.keras/paz/datasets/ycb/models/$class_name/textured.obj"

python3 train.py --images_directory $images_directory --obj_path $object_path --class_name $class_name -st 4
