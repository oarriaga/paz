import numpy as np
import re

filepath = "/media/fabian/Data/Masterarbeit/data/VOCdevkit/VOC2012/ImageSets/Main/bird_val.txt"
filepath_save = "/media/fabian/Data/Masterarbeit/data/VOCdevkit/VOC2012/ImageSets/Main/bird_val_cleaned.txt"

with open(filepath) as f:
    lines = f.readlines()

file_save = open(filepath_save, "w")

for line in lines:
    line_splitted = re.split("\ +", line)
    if line_splitted[1].strip() == "1":
        file_save.write(line.split(" ")[0] + '\n')

file_save.close()