dataset_synthesis.py gives an example of training image synthesis with the OBJ file of 3D model using pyrender. The images information including image size, bounding box coordinates and class name is recorded in annotation.txt, which is in the same folder as the synthesized images.

data_manager.py provides the methods reading the annotations in the format that dataset_synthesis.py creates.
