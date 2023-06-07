import argparse
import time
import cv2
# from paz.pipelines import SSD512HandDetection
from paz.backend.camera import VideoPlayer, Camera


parser = argparse.ArgumentParser(description='Minimal hand detection')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

device_id = 0
# camera = cv2.VideoCapture(device_id)
# if camera is None or not camera.isOpened():
#     raise ValueError('Unable to open device', device_id)
# camera.set(3, 640)#width
# camera.set(4, 480)#height
# camera.set(10, 100)#brightness
# time.sleep(1)
# image = camera.read()[1]
# # print(image)
# camera.release()

# pipeline = SSD512HandDetection()
camera = Camera(0)
image = camera.take_photo()

# print(image)


cv2.imshow("image", image)
cv2.waitKey(1)

# player = VideoPlayer((640, 480), pipeline, camera)
# player.run()
