from paz.backend import boxes
from paz.backend import camera
from paz.backend import keypoints
from paz.backend import quaternion
from paz.backend.image import draw
from paz.abstract import messages
from paz.abstract import processor
from paz.abstract import loader
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
        'page': 'abstract/messages.md',
        'classes': [
            (messages.Box2D, [messages.Box2D.contains]),
            messages.Pose6D
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
