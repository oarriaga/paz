import cv2
import numpy as np

from ..backend.image import resize_image, convert_color_space, show_image
from ..backend.image import BGR2RGB


class Camera(object):
    """Camera abstract class.
    By default this camera uses the openCV functionality.
    It can be inherited to overwrite methods in case another camera API exists.
    """
    def __init__(self, device_id=0, name='Camera', intrinsics=None,
                 distortion=None):
        # TODO load parameters from camera name. Use ``load`` method.
        self.device_id = device_id
        self.name = name
        self.intrinsics = intrinsics
        self.distortion = None
        self._camera = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def intrinsics(self):
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, value):
        if value is None:
            value = np.zeros((4))
        self._intrinsics = value

    @property
    def distortion(self):
        return self._distortion

    @distortion.setter
    def distortion(self, distortion):
        self._distortion = distortion

    def start(self):
        """ Starts capturing device

        # Returns
            Camera object.
        """
        self._camera = cv2.VideoCapture(self.device_id)
        if self._camera is None or not self._camera.isOpened():
            raise ValueError('Unable to open device', self.device_id)
        return self._camera

    def stop(self):
        """ Stops capturing device.
        """
        return self._camera.release()

    def read(self):
        """Reads camera input and returns a frame.

        # Returns
            Image array.
        """
        frame = self._camera.read()[1]
        return frame

    def is_open(self):
        """Checks if camera is open.

        # Returns
            Boolean
        """
        return self._camera.isOpened()

    def calibrate(self):
        raise NotImplementedError

    def save(self, filepath):
        raise NotImplementedError

    def load(self, filepath):
        raise NotImplementedError

    def intrinsics_from_HFOV(self, HFOV=70, image_shape=None):
        """Computes camera intrinsics using horizontal field of view (HFOV).

        # Arguments
            HFOV: Angle in degrees of horizontal field of view.
            image_shape: List of two floats [height, width].

        # Returns
            camera intrinsics array (3, 3).

        # Notes:

                       \           /      ^
                        \         /       |
                         \ lens  /        | w/2
        horizontal field  \     / alpha/2 |
        of view (alpha)____\( )/_________ |      image
                           /( )\          |      plane
                          /     <-- f --> |
                         /       \        |
                        /         \       |
                       /           \      v

                    Pinhole camera model

        From the image above we know that: tan(alpha/2) = w/2f
        -> f = w/2 * (1/tan(alpha/2))

        alpha in webcams and phones is often between 50 and 70 degrees.
        -> 0.7 w <= f <= w
        """
        if image_shape is None:
            self.start()
            height, width = self.read().shape[0:2]
            self.stop()
        else:
            height, width = image_shape[:2]

        focal_length = (width / 2) * (1 / np.tan(np.deg2rad(HFOV / 2.0)))
        intrinsics = np.array([[focal_length, 0, width / 2.0],
                               [0, focal_length, height / 2.0],
                               [0, 0, 1.0]])
        self.intrinsics = intrinsics

    def take_photo(self):
        """Starts camera, reads buffer and returns an image.
        """
        self.start()
        image = self.read()
        # all pipelines start with RGB
        image = convert_color_space(image, BGR2RGB)
        self.stop()
        return image


class VideoPlayer(object):
    """Performs visualization inferences in a real-time video.

    # Properties
        image_size: List of two integers. Output size of the displayed image.
        pipeline: Function. Should take RGB image as input and it should
            output a dictionary with key 'image' containing a visualization
            of the inferences. Built-in pipelines can be found in
            ``paz/processing/pipelines``.

    # Methods
        run()
        record()
    """

    def __init__(self, image_size, pipeline, camera, topic='image'):
        self.image_size = image_size
        self.pipeline = pipeline
        self.camera = camera
        self.topic = topic

    def step(self):
        """ Runs the pipeline process once

        # Returns
            Inferences from ``pipeline``.
        """
        if self.camera.is_open() is False:
            raise ValueError('Camera has not started. Call ``start`` method.')

        frame = self.camera.read()
        if frame is None:
            print('Frame: None')
            return None
        # all pipelines start with an RGB image
        frame = convert_color_space(frame, BGR2RGB)
        return self.pipeline(frame)

    def run(self):
        """Opens camera and starts continuous inference using ``pipeline``,
        until the user presses ``q`` inside the opened window.
        """
        self.camera.start()
        while True:
            output = self.step()
            if output is None:
                continue
            image = resize_image(output[self.topic], tuple(self.image_size))
            show_image(image, 'inference', wait=False)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.camera.stop()
        cv2.destroyAllWindows()

    def record(self, name='video.avi', fps=20, fourCC='XVID'):
        """Opens camera and records continuous inference using ``pipeline``.

        # Arguments
            name: String. Video name. Must include the postfix .avi.
            fps: Int. Frames per second.
            fourCC: String. Indicates the four character code of the video.
            e.g. XVID, MJPG, X264.
        """
        self.camera.start()
        fourCC = cv2.VideoWriter_fourcc(*fourCC)
        writer = cv2.VideoWriter(name, fourCC, fps, self.image_size)
        while True:
            output = self.step()
            if output is None:
                continue
            image = resize_image(output['image'], tuple(self.image_size))
            show_image(image, 'inference', wait=False)
            writer.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.stop()
        writer.release()
        cv2.destroyAllWindows()

    def record_from_file(self, video_file_path, name='video.avi',
                         fps=20, fourCC='XVID'):
        """Load video and records continuous inference using ``pipeline``.

        # Arguments
            video_file_path: String. Path to the video file.
            name: String. Output video name. Must include the postfix .avi.
            fps: Int. Frames per second.
            fourCC: String. Indicates the four character code of the video.
            e.g. XVID, MJPG, X264.
        """

        fourCC = cv2.VideoWriter_fourcc(*fourCC)
        writer = cv2.VideoWriter(name, fourCC, fps, self.image_size)

        video = cv2.VideoCapture(video_file_path)
        if (video.isOpened() is False):
            print("Error opening video  file")

        while video.isOpened():
            is_frame_received, frame = video.read()
            if not is_frame_received:
                print("Frame not received. Exiting ...")
                break
            if is_frame_received is True:
                output = self.pipeline(frame)
                if output is None:
                    continue
                image = resize_image(output['image'], tuple(self.image_size))
                show_image(image, 'inference', wait=False)
                writer.write(image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        writer.release()
        cv2.destroyAllWindows()
