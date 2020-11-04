from paz.backend import boxes
from paz.backend import camera
from paz.backend import render
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
from paz import datasets
from paz import pipelines

EXCLUDE = {}

# TODO
# backend.pipelines *

PAGES = [
    {
        'page': 'backend/boxes.md',
        'functions': [
            boxes.apply_non_max_suppression,
            boxes.offset,
            boxes.clip,
            boxes.compute_iou,
            boxes.compute_ious,
            boxes.decode,
            boxes.denormalize_box,
            boxes.encode,
            boxes.flip_left_right,
            boxes.make_box_square,
            boxes.match,
            boxes.nms_per_class,
            boxes.to_image_coordinates,
            boxes.to_center_form,
            boxes.to_one_hot,
            boxes.to_normalized_coordinates,
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
            keypoints.solve_PNP,
            keypoints.translate_keypoints
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
        'page': 'backend/draw.md',
        'functions': [
            draw.draw_circle,
            draw.draw_cube,
            draw.draw_dot,
            draw.draw_filled_polygon,
            draw.draw_line,
            draw.draw_random_polygon,
            draw.draw_rectangle,
            draw.lincolor,
            draw.put_text,
            draw.make_mosaic
        ],
    },


    {
        'page': 'backend/image.md',
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
            opencv_image.write_image,
            opencv_image.random_shape_crop,
            opencv_image.make_random_plain_image,
            opencv_image.blend_alpha_channel,
            opencv_image.concatenate_alpha_mask,
            opencv_image.split_and_normalize_alpha_channel,
            opencv_image.gaussian_image_blur,
            opencv_image.median_image_blur,
            opencv_image.random_image_blur,
            opencv_image.translate_image,
            opencv_image.sample_scaled_translation,
            opencv_image.get_rotation_matrix
        ],
    },


    {
        'page': 'backend/render.md',
        'functions': [
            render.calculate_norm,
            render.compute_modelview_matrices,
            render.get_look_at_transform,
            render.random_perturbation,
            render.random_translation,
            render.roll_camera,
            render.sample_point_in_full_sphere,
            render.sample_point_in_sphere,
            render.sample_point_in_top_sphere,
            render.sample_uniformly,
            render.scale_translation,
            render.split_alpha_channel,
            render.translate_camera,
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
        'page': 'datasets.md',
        'classes': [
            datasets.VOC,
            datasets.FAT,
            datasets.FER,
            datasets.FERPlus,
            datasets.OpenImages
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
            processors.RandomGaussianBlur,
            processors.RandomFlipImageLeftRight,
            processors.ConvertColorSpace,
            processors.ShowImage,
            processors.ImageDataProcessor,
            processors.AlphaBlending,
            processors.RandomImageCrop,
            processors.RandomShapeCrop,
            processors.MakeRandomPlainImage,
            processors.ConcatenateAlphaMask,
            processors.BlendRandomCroppedBackground,
            processors.AddOcclusion,
            processors.TranslateImage
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
            processors.ToImageBoxCoordinates,
            processors.ToNormalizedBoxCoordinates,
            processors.RandomSampleCrop,
            processors.RandomTranslation,
            processors.RandomRotation,
            processors.RandomKeypointTranslation,
            processors.RandomKeypointRotation
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
            processors.OffsetBoxes2D,
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
            processors.RemoveKeypointsDepth,
            processors.TranslateKeypoints
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
        'page': 'processors/standard.md',
        'classes': [
            processors.ControlMap,
            processors.ExpandDomain,
            processors.CopyDomain,
            processors.ExtendInputs,
            processors.SequenceWrapper,
            processors.Predict,
            processors.ToClassName,
            processors.ExpandDims,
            processors.BoxClassToOneHotVector,
            processors.Squeeze,
            processors.Copy,
            processors.Lambda,
            processors.UnpackDictionary,
            processors.WrapOutput,
            processors.Concatenate,
            processors.SelectElement
        ]
    },


    {
        'page': 'pipelines/image.md',
        'classes': [
            pipelines.AugmentImage,
            pipelines.PreprocessImage,
            pipelines.DecoderPredictor,
            pipelines.EncoderPredictor,
        ]
    },


    {
        'page': 'pipelines/detection.md',
        'classes': [
            pipelines.AugmentBoxes,
            pipelines.AugmentDetection,
            pipelines.PreprocessBoxes,
            pipelines.DetectSingleShot,
            pipelines.DetectHaarCascade,
        ]
    },


    {
        'page': 'pipelines/keypoints.md',
        'classes': [
            pipelines.KeypointNetInference,
            pipelines.KeypointNetSharedAugmentation,
            pipelines.EstimateKeypoints2D,
            pipelines.DetectKeypoints2D,
        ]
    },


    {
        'page': 'pipelines/pose.md',
        'classes': [
            pipelines.EstimatePoseKeypoints,
            pipelines.HeadPoseKeypointNet2D32
        ]
    },


    {
        'page': 'pipelines/renderer.md',
        'classes': [
            pipelines.RandomizeRenderedImage,
            pipelines.RenderTwoViews,
        ]
    },


    {
        'page': 'pipelines/applications.md',
        'classes': [
            pipelines.SSD512COCO,
            pipelines.SSD300VOC,
            pipelines.SSD512YCBVideo,
            pipelines.SSD300FAT,
            pipelines.DetectMiniXceptionFER,
            pipelines.MiniXceptionFER,
            pipelines.FaceKeypointNet2D32,
            pipelines.HeadPoseKeypointNet2D32,
            pipelines.HaarCascadeFrontalFace,
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
