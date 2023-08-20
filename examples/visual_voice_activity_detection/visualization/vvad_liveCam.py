'''
This is a live Demo using the webcam
'''

# System imports
import os

# 3rd party imports
from vvadlrs3 import pretrained_models, sample, dlibmodels
from vvadlrs3.utils.imageUtils import cropImage
import cv2
import numpy as np
import dlib
from dvg_ringbuffer import RingBuffer


# local imports

# end file header
__author__ = 'Adrian Lubitz'

# Sliding window approach

# load pretrained model
model = pretrained_models.getFaceImageModel()

print('Model Overview')
print(model.summary())
print('Model input shape')
print(model.layers[0].input_shape)


# create a webcambuffer
cap = cv2.VideoCapture(0)


# create a sample
k = 36  # Number of frames used for inference
shape = (96, 96)  # Resolution of the input imgae for the prediction
featureType = 'faceImage'  # Type of the features that will be created from the Image
# TODO: this should actually only be needed if not using faceImage type
shapeModelPath = str(dlibmodels.SHAPE_PREDICTOR_68_FACE_LANDMARKS())
ffg = sample.FaceFeatureGenerator(
    featureType, shapeModelPath=shapeModelPath, shape=shape)

# TODO: Fist approach only with a detector - later we can try FaceTracker for multiple faces?
detector = dlib.get_frontal_face_detector()

# Ringbuffer for features
rb = RingBuffer(36, dtype=(np.uint8, (96, 96, 3)))

# tracker = FaceTracker(init_pos, relative=relative)
# for x, image in enumerate(frames):
#     face, boundingBox = tracker.getNextFace(image)
#     if boundingBox:  # check if tracker was successfull
#         self.data.append(ffg.getFeatures(face))
#     else:
#         print("did not get a face for frame number {}".format(x))
#         break

# create_sample_from_buffer
# infer if sample is valid
# annotate and visualize output image(first of sample data) - do I need to save the boundingbox as well in the sample?


# TODO: This shows the camera output right now - should show predictiions at some point
samples = []
while(True):
    # TODO: this is problematic because it runs on the default(30FPS) framerate but we need 25fps for ideal results
    ret, frame = cap.read()     # Capture frame-by-frame
    dets = detector(frame, 1)   # Detect faces
    if dets:
        features = ffg.getFeatures(cropImage(frame, dets[0]))
        # fill ringbuffer
        rb.append(features)

        if rb.is_full:
            y = model.predict(np.array([rb]))
            s = sample.FeatureizedSample()
            s.data = np.copy(rb)
            s.label = y > 0.5
            s.featureType = featureType
            samples.append(s)
            print(f"added sample {len(samples)}, label: {s.label}, score: {y}")
        cv2.imshow('frame', frame)
    else:
        # empty ringbuffer - to prevent glitches
        rb.clear()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


output_path = "default_fps_samples"
if not os.path.exists(output_path):
    os.makedirs(output_path)
for i, sample in enumerate(samples):
    sample.save(os.path.join(output_path, str(i)) + '.pickle')