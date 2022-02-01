from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import mean_squared_error
import tensorflow as tf


def compute_weighted_symmetric_loss(RGBA_true, RGB_pred, rotations, beta=3.0):
    """Computes the mininum of all rotated L1 reconstruction losses weighting
        the positive alpha mask values in the predicted RGB image by beta.

    # Arguments
        RGBA_true: Tensor [batch, H, W, 4]. Color with alpha mask label values.
        RGB_pred: Tensor [batch, H, W, 3]. Predicted RGB values.
        rotations: Array (num_symmetries, 3, 3). Rotation matrices
            that when applied lead to the same object view.

    # Returns
        Tensor [batch, H, W] with weighted reconstruction loss values.
    """
    RGB_true, alpha = split_alpha_mask(RGBA_true)
    RGB_true = normalized_image_to_normalized_device_coordinates(RGB_true)
    symmetric_losses = []
    for rotation in rotations:
        RGB_true_rotated = tf.einsum('ij,bklj->bkli', rotation, RGB_true)
        RGB_true_rotated = normalized_device_coordinates_to_normalized_image(
            RGB_true_rotated)
        RGB_true_rotated = tf.clip_by_value(RGB_true_rotated, 0.0, 1.0)
        RGB_true_rotated = RGB_true_rotated * alpha
        RGBA_true_rotated = tf.concat([RGB_true_rotated, alpha], axis=3)
        loss = compute_weighted_reconstruction_loss(
            RGBA_true_rotated, RGB_pred, beta)
        loss = tf.expand_dims(loss, -1)
        symmetric_losses.append(loss)
    symmetric_losses = tf.concat(symmetric_losses, axis=-1)
    minimum_symmetric_loss = tf.reduce_min(symmetric_losses, axis=-1)
    return minimum_symmetric_loss


class WeightedSymmetricReconstruction(Loss):
    """Computes the mininum of all rotated L1 reconstruction losses weighting
        the positive alpha mask values in the predicted RGB image by beta.
    """
    def __init__(self, rotations, beta=3.0):
        super(WeightedSymmetricReconstruction, self).__init__()
        self.rotations = rotations
        self.beta = beta

    def call(self, RGBA_true, RGB_pred):
        loss = compute_weighted_symmetric_loss(
            RGBA_true, RGB_pred, self.rotations, self.beta)
        return loss


def compute_error_prediction_loss(RGBA_true, RGBE_pred):
    """Computes L2 reconstruction loss of predicted error mask.

    # Arguments
        RGBA_true: Tensor [batch, H, W, 4]. Color with alpha mask label values.
        RGBE_pred: Tensor [batch, H, W, 3]. Predicted RGB and error mask.

    # Returns
        Tensor [batch, H, W] with weighted reconstruction loss values.

    """
    RGB_pred, error_pred = split_error_mask(RGBE_pred)
    error_true = compute_weighted_reconstruction_loss(RGBA_true, RGB_pred, 1.0)
    # TODO check we need to set minimum to 1.0?
    error_true = tf.minimum(error_true, 1.0)
    error_loss = mean_squared_error(error_true, error_pred)
    error_loss = tf.expand_dims(error_loss, axis=-1)
    return error_loss


class ErrorPrediction(Loss):
    """Computes L2 reconstruction loss of predicted error mask.

    # Arguments
        RGBA_true: Tensor [batch, H, W, 4]. Color with alpha mask label values.
        RGBE_pred: Tensor [batch, H, W, 3]. Predicted RGB and error mask.

    # Returns
        Tensor [batch, H, W] with weighted reconstruction loss values.

    """
    def __init__(self):
        super(ErrorPrediction, self).__init__()

    def call(self, RGBA_true, RGBE_pred):
        error_loss = compute_error_prediction_loss(RGBA_true, RGBE_pred)
        return error_loss


from paz.backend.image import draw_dot


def draw_points2D_(image, keypoints, colors, radius=1):
    for (u, v), (R, G, B) in zip(keypoints, colors):
        color = (int(R), int(G), int(B))
        draw_dot(image, (u, v), color, radius)
    return image


def rotate_image(image, rotation_matrix):
    """Rotates an image with a symmetry.

    # Arguments
        image: Array (H, W, 3) with domain [0, 255].
        rotation_matrix: Array (3, 3).

    # Returns
        Array (H, W, 3) with domain [0, 255]
    """
    mask_image = np.sum(image, axis=-1, keepdims=True) != 0
    image = image_to_normalized_device_coordinates(image)
    rotated_image = np.einsum('ij,klj->kli', rotation_matrix, image)
    rotated_image = normalized_device_coordinates_to_image(rotated_image)
    rotated_image = np.clip(rotated_image, a_min=0.0, a_max=255.0)
    rotated_image = rotated_image * mask_image
    return rotated_image


class EstimatePoseMasks(Processor):
    def __init__(self, detect, estimate_pose, offsets, draw=True,
                 valid_class_names=['035_power_drill']):
        """Pose estimation pipeline using keypoints.
        """
        super(EstimatePoseMasks, self).__init__()
        self.detect = detect
        self.estimate_pose = estimate_pose
        self.postprocess_boxes = SequentialProcessor(
            [pr.UnpackDictionary(['boxes2D']),
             pr.FilterClassBoxes2D(valid_class_names),
             pr.SquareBoxes2D(),
             pr.OffsetBoxes2D(offsets)])
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'poses6D'])
        self.unwrap = UnwrapDictionary(['pose6D', 'points2D', 'points3D'])
        self.draw_boxes2D = pr.DrawBoxes2D(detect.class_names)
        self.object_sizes = self.estimate_pose.object_sizes
        self.cube_points3D = build_cube_points3D(*self.object_sizes)
        self.draw = draw

    def call(self, image):
        boxes2D = self.postprocess_boxes(self.detect(image))
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)
        poses6D, points = [], []
        for crop, box2D in zip(cropped_images, boxes2D):
            results = self.estimate_pose(crop, box2D)
            pose6D, points2D, points3D = self.unwrap(results)
            poses6D.append(pose6D), points.append([points2D, points3D])
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
            image = draw_masks(image, points, self.object_sizes)
            image = draw_poses6D(image, poses6D, self.cube_points3D,
                                 self.estimate_pose.camera.intrinsics)
        return self.wrap(image, boxes2D, poses6D)


class MultiPix2Pose(Processor):
    def __init__(self, detect, segment, camera, name_to_weights, name_to_sizes,
                 valid_class_names, offsets=[0.2, 0.2], epsilon=0.15, draw=True):
        self.detect = detect
        self.name_to_weights = name_to_weights
        self.name_to_sizes = name_to_sizes
        self.valid_class_names = valid_class_names
        self.pix2points = Pix2Points(segment, np.zeros((3)), epsilon)
        self.predict_pose = SolveChangingObjectPnP(camera.intrinsics)
        self.change_coordinates = pr.ChangeKeypointsCoordinateSystem()
        self.camera = camera
        self.postprocess_boxes = SequentialProcessor(
            [pr.UnpackDictionary(['boxes2D']),
             pr.FilterClassBoxes2D(valid_class_names),
             pr.SquareBoxes2D(),
             pr.OffsetBoxes2D(offsets)])
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.draw_boxes2D = pr.DrawBoxes2D(detect.class_names)
        self.draw = draw
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'poses6D'])
        self.name_to_cube_points3D = {}
        self.mask_to_points2D = RGBMaskToImagePoints2D(
            segment.output_shape[1:3])
        for name in self.name_to_sizes:
            W, H, D = self.name_to_sizes[name]
            cube_points3D = build_cube_points3D(W, H, D)
            self.name_to_cube_points3D[name] = cube_points3D

        self.predict_RGBMask = PredictRGBMask(segment, epsilon)

    def call(self, image):
        boxes2D = self.postprocess_boxes(self.detect(image))
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)
        poses6D, points2D, points3D = [], [], []
        for crop, box2D in zip(cropped_images, boxes2D):
            class_name = box2D.class_name
            name_to_weights = self.name_to_weights[class_name]
            self.pix2points.model.load_weights(name_to_weights)
            object_sizes = self.name_to_sizes[class_name]
            # self.pix2points.object_sizes = object_sizes
            # points = self.pix2points(crop)

            RGB_mask = self.predict_RGBMask(crop)
            H, W, num_channels = crop.shape
            RGB_mask = resize_image(RGB_mask, (W, H))

            self.mask_to_points3D = RGBMaskToObjectPoints3D(object_sizes)
            class_points3D = self.mask_to_points3D(RGB_mask)
            class_points2D = self.mask_to_points2D(RGB_mask)
            class_points2D = normalize_points2D(class_points2D, H, W)

            # from paz.backend.image import show_image
            # show_image((points['RGB_mask'] * 255).astype('uint8'))
            # class_points2D = points['points2D']
            # class_points3D = points['points3D']
            H, W, num_channels = crop.shape
            class_points2D = denormalize_points2D(class_points2D, H, W)
            class_points2D = self.change_coordinates(class_points2D, box2D)
            print(len(class_points3D) > self.predict_pose.MIN_REQUIRED_POINTS)
            print(len(class_points3D), len(class_points2D))
            if len(class_points3D) > self.predict_pose.MIN_REQUIRED_POINTS:
                pose_results = self.predict_pose(class_points3D, class_points2D)
                success, rotation, translation = pose_results
                print('solver success', success)
                # success = True
            else:
                success = False
            if success:
                quaternion = rotation_vector_to_quaternion(rotation)
                pose6D = Pose6D(quaternion, translation, class_name)
            else:
                pose6D = None
            print(success)
            points2D.append(class_points2D)
            points3D.append(class_points3D)
            poses6D.append(pose6D)
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
            for class_points2D, class_points3D, pose6D in zip(points2D, points3D, poses6D):
                class_name = pose6D.class_name
                object_sizes = self.name_to_sizes[class_name]
                colors = points3D_to_RGB(class_points3D, object_sizes)
                image = draw_points2D(image, class_points2D, colors)

            for pose6D in poses6D:
                class_name = pose6D.class_name
                cube_points3D = self.name_to_cube_points3D[class_name]
                image = draw_pose6D(image, pose6D, cube_points3D,
                                    self.camera.intrinsics)
        return {'image': image, 'boxes2D': boxes2D, 'poses6D': poses6D}


class PixelMaskRenderer():
    """Render-ready scene composed of a single object and a single moving camera.

    # Arguments
        path_OBJ: String containing the path to an OBJ file.
        viewport_size: List, specifying [H, W] of rendered image.
        y_fov: Float indicating the vertical field of view in radians.
        distance: List of floats indicating [max_distance, min_distance]
        light: List of floats indicating [max_light, min_light]
        top_only: Boolean. If True images are only take from the top.
        roll: Float, to sample [-roll, roll] rolls of the Z OpenGL camera axis.
        shift: Float, to sample [-shift, shift] to move in X, Y OpenGL axes.
    """
    def __init__(self, path_OBJ, viewport_size=(128, 128), y_fov=3.14159 / 4.0,
                 distance=[0.3, 0.5], light=[0.5, 30], top_only=False,
                 roll=None, shift=None):
        self.distance, self.roll, self.shift = distance, roll, shift
        self.light_intensity, self.top_only = light, top_only
        self._build_scene(path_OBJ, viewport_size, light, y_fov)
        self.renderer = OffscreenRenderer(viewport_size[0], viewport_size[1])
        self.flags_RGBA = RenderFlags.RGBA
        self.flags_FLAT = RenderFlags.RGBA | RenderFlags.FLAT
        self.epsilon = 0.01

    def _build_scene(self, path, size, light, y_fov):
        self.scene = Scene(bg_color=[0, 0, 0, 0])
        self.light = self.scene.add(
            DirectionalLight([1.0, 1.0, 1.0], np.mean(light)))
        self.camera = self.scene.add(
            PerspectiveCamera(y_fov, aspectRatio=np.divide(*size)))
        self.pixel_mesh = self.scene.add(color_object(path))
        self.mesh = self.scene.add(
            Mesh.from_trimesh(trimesh.load(path), smooth=True))
        self.world_origin = self.mesh.mesh.centroid

    def _sample_parameters(self):
        distance = sample_uniformly(self.distance)
        camera_origin = sample_point_in_sphere(distance, self.top_only)
        camera_origin = random_perturbation(camera_origin, self.epsilon)
        light_intensity = sample_uniformly(self.light_intensity)
        return camera_origin, light_intensity

    def render(self):
        camera_origin, intensity = self._sample_parameters()
        camera_to_world, world_to_camera = compute_modelview_matrices(
            camera_origin, self.world_origin, self.roll, self.shift)
        self.light.light.intensity = intensity
        self.scene.set_pose(self.camera, camera_to_world)
        self.scene.set_pose(self.light, camera_to_world)
        self.pixel_mesh.mesh.is_visible = False
        image, depth = self.renderer.render(self.scene, self.flags_RGBA)
        self.pixel_mesh.mesh.is_visible = True
        image, alpha = split_alpha_channel(image)
        self.mesh.mesh.is_visible = False
        RGB_mask, _ = self.renderer.render(self.scene, self.flags_FLAT)
        self.mesh.mesh.is_visible = True
        return image, alpha, RGB_mask
