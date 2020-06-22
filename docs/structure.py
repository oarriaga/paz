from paz.backend import boxes
from paz.backend import camera
from paz.backend import keypoints
from paz import models
from paz import processors

EXCLUDE = {}

PAGES = [
    {
        'page': 'backend/boxes.md',
        'functions': [
            boxes.apply_non_max_suppression,
            boxes.apply_offsets,
            boxes.compute_iou,
            boxes.compute_ious,
            boxes.decode,
            boxes.denormalize_box,
            boxes.encode,
            boxes.flip_left_right,
            boxes.make_box_square,
            boxes.match,
            boxes.nms_per_class,
            boxes.to_absolute_coordinates,
            boxes.to_center_form,
            boxes.to_one_hot,
            boxes.to_percent_coordinates,
            boxes.to_point_form
        ],
    },

    {
        'page': 'backend/keypoints.md',
        'functions': [
            keypoints.cascade_classifier,
            keypoints.denormalize_keypoints,
            keypoints.normalize_keypoints,
            keypoints.project_points3D,
            keypoints.solve_PNP
        ],
    },


    {
        'page': 'backend/camera.md',
        'classes': [
            (camera.Camera, [camera.Camera.is_open,
                             camera.Camera.start,
                             camera.Camera.stop]),
            (camera.VideoPlayer, [camera.VideoPlayer.step,
                                  camera.VideoPlayer.run,
                                  camera.VideoPlayer.record])
        ],
    },


    {
        'page': 'models/detection.md',
        'functions': [
            models.detection.SSD300,
            models.detection.SSD512]
    },

    {
        'page': 'processors/image.md',
        'classes': [
            processors.CastImage,
            processors.SubtractMeanImage,
            processors.AddMeanImage,
            processors.NormalizeImage,
            processors.DenormalizeImage,
            processors.LoadImage,
            processors.RandomSaturation,
            processors.RandomBrightness,
            processors.RandomContrast,
            processors.RandomHue,
            processors.ResizeImages,
            processors.ResizeImages,
            processors.RandomImageBlur,
            processors.RandomFlipImageLeftRight,
            processors.ConvertColorSpace,
            processors.ShowImage,
            processors.ImageDataProcessor,
            processors.AlphaBlending,
            processors.RandomImageCrop,
            processors.MakeRandomPlainImage,
            processors.ConcatenateAlphaMask,
            processors.BlendRandomCroppedBackground,
            processors.AddOcclusion
        ]
    }
]
