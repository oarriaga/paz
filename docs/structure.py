from paz.backend import boxes
from paz.backend import camera
from paz.backend import keypoints
from paz.backend import quaternion
from paz.backend.image import draw
from paz.backend.image import opencv_image
from paz.abstract import messages
from paz.abstract import processor
from paz.abstract import loader
from paz.abstract import sequence
from paz import models
from paz import processors
from paz.optimization import losses
from paz.optimization import callbacks

EXCLUDE = {}

# TODO
# backend.processors.standard.py
# backend.datasets *
# backend.pipelines *

# KeypointNet loss parts.

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
        'page': 'backend/quaternion.md',
        'functions': [
            quaternion.rotation_vector_to_quaternion
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
        'page': 'backend/image/draw.md',
        'functions': [
            draw.draw_circle,
            draw.draw_cube,
            draw.draw_dot,
            draw.draw_filled_polygon,
            draw.draw_line,
            draw.draw_random_polygon,
            draw.draw_rectangle,
            draw.lincolor,
            draw.put_text
        ],
    },


    {
        'page': 'backend/image/opencv_image.md',
        'functions': [
            opencv_image.cast_image,
            opencv_image.resize_image,
            opencv_image.convert_color_space,
            opencv_image.load_image,
            opencv_image.random_saturation,
            opencv_image.random_brightness,
            opencv_image.random_contrast,
            opencv_image.random_hue,
            opencv_image.random_flip_left_right,
            opencv_image.show_image,
            opencv_image.warp_affine,
            opencv_image.save_image,
            opencv_image.random_image_crop,
            opencv_image.make_random_plain_image,
            opencv_image.blend_alpha_channel,
            opencv_image.concatenate_alpha_mask,
            opencv_image.split_and_normalize_alpha_channel,
            opencv_image.gaussian_image_blur,
            opencv_image.median_image_blur,
            opencv_image.random_image_blur
        ],
    },


    {
        'page': 'models/classification.md',
        'functions': [
            models.classification.MiniXception
        ],
    },


    {
        'page': 'models/detection.md',
        'functions': [
            models.detection.SSD300,
            models.detection.SSD512,
            models.detection.HaarCascadeDetector
        ],
    },


    {
        'page': 'models/keypoint.md',
        'functions': [
            models.KeypointNet,
            models.KeypointNet2D,
            models.Projector
        ],
    },


    {
        'page': 'models/layers.md',
        'classes': [
            models.layers.Conv2DNormalization,
            models.layers.SubtractScalar,
            models.layers.ExpectedValue2D,
            models.layers.ExpectedDepth
        ],
    },


    {
        'page': 'optimization/callbacks.md',
        'classes': [
            callbacks.DrawInferences,
            callbacks.LearningRateScheduler,
            callbacks.EvaluateMAP
        ],
    },


    {
        'page': 'optimization/losses.md',
        'classes': [
            losses.MultiBoxLoss,
            losses.KeypointNetLoss
        ],
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
    },


    {
        'page': 'processors/draw.md',
        'classes': [
            processors.DrawBoxes2D,
            processors.DrawKeypoints2D,
            processors.DrawBoxes3D,
            processors.DrawRandomPolygon
        ]
    },


    {
        'page': 'processors/geometric.md',
        'classes': [
            processors.RandomFlipBoxesLeftRight,
            processors.ToAbsoluteBoxCoordinates,
            processors.ToNormalizedBoxCoordinates,
            processors.RandomSampleCrop,
            processors.ApplyRandomTranslation,
            processors.ApplyRandomTranslation
        ]
    },



    {
        'page': 'processors/detection.md',
        'classes': [
            processors.SquareBoxes2D,
            processors.DenormalizeBoxes2D,
            processors.RoundBoxes2D,
            processors.ClipBoxes2D,
            processors.FilterClassBoxes2D,
            processors.CropBoxes2D,
            processors.ToBoxes2D,
            processors.MatchBoxes,
            processors.EncodeBoxes,
            processors.DecodeBoxes,
            processors.NonMaximumSuppressionPerClass,
            processors.FilterBoxes,
            processors.ApplyOffsets,
            processors.CropImage
        ]
    },


    {
        'page': 'processors/keypoints.md',
        'classes': [
            processors.ChangeKeypointsCoordinateSystem,
            processors.DenormalizeKeypoints,
            processors.NormalizeKeypoints,
            processors.PartitionKeypoints,
            processors.ProjectKeypoints,
            processors.RemoveKeypointsDepth
        ]
    },

    {
        'page': 'processors/pose.md',
        'classes': [
            processors.SolvePNP
        ]
    },


    {
        'page': 'processors/renderer.md',
        'classes': [
            processors.Render
        ]
    },


    {
        'page': 'abstract/messages.md',
        'classes': [
            (messages.Box2D, [messages.Box2D.contains]),
            messages.Pose6D
        ]
    },



    {
        'page': 'abstract/sequence.md',
        'classes': [
            sequence.ProcessingSequence,
            sequence.GeneratingSequence
        ]
    },


    {
        'page': 'abstract/processor.md',
        'classes': [
            (processor.Processor, [processor.Processor.call]),
            (processor.SequentialProcessor, [
                processor.SequentialProcessor.add,
                processor.SequentialProcessor.remove,
                processor.SequentialProcessor.pop,
                processor.SequentialProcessor.insert,
                processor.SequentialProcessor.get_processor])
        ]
    },

    {
        'page': 'abstract/loader.md',
        'classes': [
            (loader.Loader, [loader.Loader.load_data])
        ]
    },





]
