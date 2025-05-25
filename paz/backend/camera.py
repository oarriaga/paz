import jax.numpy as jp
import numpy as np
import cv2
import paz


class Camera(object):
    """Camera abstract class."""

    def __init__(
        self, identifier=0, name="Camera", intrinsics=None, distortion=None
    ):
        self.identifier = identifier
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
            value = jp.zeros((4))
        self._intrinsics = value

    @property
    def distortion(self):
        return self._distortion

    @distortion.setter
    def distortion(self, distortion):
        self._distortion = distortion

    def is_open(self):
        return self._camera.isOpened()

    def is_closed(self):
        return not self.is_open()

    def start(self):
        self._camera = cv2.VideoCapture(self.identifier)
        if (self._camera is None) or self.is_closed():
            raise ValueError("Unable to open device", self.identifier)

    def stop(self):
        return self._camera.release()

    def read(self):
        image = paz.image.BGR_to_RGB(self._camera.read()[1])
        return np.ascontiguousarray(image)
        # return cv2.cvtColor(self._camera.read()[1], cv2.COLOR_BGR2RGB)

    def save(self, filepath):
        raise NotImplementedError

    def load(self, filepath):
        raise NotImplementedError

    def intrinsics_from_HFOV(self, HFOV=70, image_shape=None):
        if image_shape is None:
            self.start()
            H, W = paz.image.get_size(self.read())
            self.stop()
        else:
            H, W = image_shape[:2]
        return paz.pinhole.intrinsics_from_HFOV(H, W, HFOV)

    def take_photo(self):
        """Starts camera, reads buffer and returns an image.

        # Arguments:
            camera: paz.Camera namedtuple.

        # Returns:
            Image array.
        """
        self.start()
        image = self.read()
        self.stop()
        return image

    def calibrate(self, chessboard_size, images=None):
        """Executes camera calibration for a given chessboard size.

        # Arguments
            images: Array of shape (num_images, H, W, 3).
            chessboard_size: Array of shape (H, W).

        # Returns
            camera_matrix: Array of shape (3, 3) representing the camera matrix
            distortion_coefficient: Array of shape (1, 5).
        """
        if images is None:
            raise NotImplementedError

        return paz.pinhole.calibrate(images, chessboard_size)


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

    def __init__(self, image_size, pipeline, camera, topic=0):
        self.image_size = tuple(image_size)
        self.pipeline = pipeline
        self.camera = camera
        self.topic = topic

        if isinstance(topic, str):
            self.get_topic = lambda x: getattr(x, self.topic)
            self.topic_name = self.topic
        elif isinstance(topic, int):
            self.get_topic = lambda x: x[self.topic]
            self.topic_name = str(topic)
        else:
            raise ValueError("topic should be either a string or an integer")

    def step(self):
        """Runs the pipeline process once

        # Returns
            Inferences from ``pipeline``.
        """
        frame = self.camera.read()
        if frame is None:
            outputs = None
        else:
            outputs = self.pipeline(frame)
        return outputs

    def run(self):
        """Opens camera and starts continuous inference using ``pipeline``,
        until the user presses ``q`` inside the opened window.
        """
        self.camera.start()
        while True:
            output = self.step()
            if output is None:
                continue
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            topic = self.get_topic(output)
            image = paz.image.resize(topic, self.image_size).astype("uint8")
            paz.image.show(image, self.topic_name, wait=False)
        self.camera.stop()
        cv2.destroyAllWindows()

    def record(self, name="video.avi", fps=20, fourCC="XVID"):
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
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            image = paz.image.resize(output["image"], self.image_size)
            paz.image.show(image, self.topic_name, wait=False)
            writer.write(paz.image.RGB_to_BGR(image))

        self.camera.stop()
        writer.release()
        cv2.destroyAllWindows()

    def record_from_file(
        self, video_file_path, name="video.avi", fps=20, fourCC="XVID"
    ):
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
        if video.isOpened() is False:
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
                image = resize_image(output["image"], tuple(self.image_size))
                show_image(image, "inference", wait=False)
                image = convert_color_space(image, BGR2RGB)
                writer.write(image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        writer.release()
        cv2.destroyAllWindows()

    def record_frames(self, name="video.avi", fps=20, fourCC="XVID"):
        """Opens camera and records continuous inference frames.

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
            frame = self.camera.read()
            if frame is None:
                print("Frame: None")
                return None
            frame = convert_color_space(frame, BGR2RGB)
            image = resize_image(frame, tuple(self.image_size))
            show_image(image, "frame", wait=False)
            image = convert_color_space(image, BGR2RGB)
            writer.write(image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.camera.stop()
        writer.release()
        cv2.destroyAllWindows()

    def extract_frames_from_video(
        self, video_file_path, frame_selection_arg=20
    ):
        """Load video and split into frames.

        # Arguments
            video_file_path: String. Path to the video file.
            frame_selection_arg: Int. Number of frames to be skipped.
        """

        video = cv2.VideoCapture(video_file_path)
        if video.isOpened() is False:
            print("Error opening video  file")

        frame_arg = 0
        while video.isOpened():
            is_frame_received, frame = video.read()
            if not is_frame_received:
                print("Frame not received. Exiting ...")
                break
            if is_frame_received is True:
                image = resize_image(frame, tuple(self.image_size))
                image_path = os.path.join("./images", str(frame_arg) + ".jpg")
                image = convert_color_space(image, BGR2RGB)
                if frame_arg % frame_selection_arg == 0:
                    write_image(image_path, image)
                frame_arg += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cv2.destroyAllWindows()
