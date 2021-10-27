from paz.backend.image import show_image
import numpy as np
import cv2


# def calibrate_camera(square_size, pattern_shape=(5, 5)):

pattern_size = (5, 7)
square_size_mm = 35
window_size, zero_zone = (11, 11), (-1, -1)

# constructing default 3D points
point3D = np.zeros((np.prod(pattern_size), 3), np.float32)
xy_coordinates = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T
point3D[:, :2] = xy_coordinates.reshape(-1, 2) * square_size_mm

camera = cv2.VideoCapture(0)
cv2.namedWindow('camera_window')
# 2D points in image plane, 3D points in real world space, images, counter
image_points, points3D, images, image_counter = [], [], [], 0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
print('Press `Escape` to quit')
while True:

    frame = camera.read()[1]
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    show_image(image_gray, wait=False)
    chessboard_found, corners = cv2.findChessboardCorners(
        image_gray, pattern_size, None)
    print(chessboard_found)
    if chessboard_found:
        points3D.append(point3D)
        refined_corners = cv2.cornerSubPix(
            image_gray, corners, window_size, zero_zone, criteria)
        image_points.append(refined_corners)
        frame = cv2.drawChessboardCorners(
            frame, pattern_size, refined_corners, chessboard_found)
        show_image(frame)
        image_counter = image_counter + 1

    cv2.imshow('camera_window', frame)
    keystroke = cv2.waitKey(1)

    if keystroke % 256 == 27:
        print('`Escape` key hit, closing...')
        break

camera.release()
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    points3D, image_points, image_gray.shape[::-1], None, None)
print(ret, mtx, dist, rvecs, tvecs)
print(mtx)
# fx = 659.10
# fy = 668.76
# cx = 276.76
# cy = 252.35
# ret = 0.6814
# dist = [9.86e-3, 1.41, 1.08e-2, 2.431e-3, -7.05]
