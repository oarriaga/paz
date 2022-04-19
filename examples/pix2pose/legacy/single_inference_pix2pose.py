class SingleInferenceMultiClassPIX2POSE6D(Processor):
    def __init__(self, name_to_model, name_to_size, camera,
                 epsilon=0.15, resize=False, draw=True):
        super(SingleInferencePIX2POSE6D, self).__init__()
        if set(name_to_model.keys()) != set(name_to_size.keys()):
            raise ValueError('models and sizes must have same class names')
        self.name_to_pix2points = self._build_pix2points(
            name_to_model, name_to_size, epsilon, resize)
        self.name_to_draw = self._build_name_to_draw(name_to_size, camera)
        self.solvePnP = pr.SolveChangingObjectPnPRANSAC(camera.intrinsics)
        self.wrap = pr.WrapOutput(['image', 'points2D', 'points3D', 'pose6D'])
        self.camera = camera
        self.draw = draw

    def _build_pix2points(self, name_to_model, name_to_size, epsilon, resize):
        name_to_pix2points = {}
        for name, model in name_to_model.items():
            pix2points = Pix2Points(model, name_to_size[name], epsilon, resize)
            name_to_pix2points[name] = pix2points
        return name_to_pix2points

    def _build_name_to_draw(self, name_to_size, camera):
        name_to_draw = {}
        for name, object_sizes in name_to_size.items():
            draw = DrawPose6D(object_sizes, camera.intrinsics)
            name_to_draw[name] = draw
        return name_to_draw

    def _single_inference(self, crop, box2D):
        inferences = self.name_to_pix2points[box2D.class_name](crop)
        points2D = inferences['points2D']
        points3D = inferences['points3D']
        points2D = denormalize_keypoints2D(points2D, *crop.shape[:2])
        points2D = translate_points2D_origin(points2D, box2D.coordinates)
        pose6D = None
        if len(points3D) > self.solvePnP.MIN_REQUIRED_POINTS:
            success, R, T = self.solvePnP(points3D, points2D)
            if success:
                pose6D = Pose6D.from_rotation_vector(R, T, self.class_name)
        return points2D, points3D, pose6D

    def call(self, image, boxes2D):
        boxes2D = self.postprocess_boxes(boxes2D)
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)
        poses6D, points2D, points3D = [], [], []
        for crop, box2D in zip(cropped_images, boxes2D):
            set_points2D, set_points3D, pose6D = self._single_inference(
                crop, box2D)
            poses6D.append(pose6D)
            points2D.append(set_points2D)
            points3D.append(set_points3D)
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
            image = self.draw_RGBmask(image, points2D, points3D)
            image = self.draw_poses6D(image, poses6D)
        return self.wrap(image, boxes2D, poses6D)

        if (self.draw and (pose6D is not None)):
            colors = points3D_to_RGB(points3D, self.object_sizes)
            image = draw_points2D(image, points2D, colors)
            image = self.name_to_draw[box2D.class_name](image, pose6D)
        inferences = self.wrap(image, points2D, points3D, pose6D)
        return inferences


