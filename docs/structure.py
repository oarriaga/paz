from paz.backend import angles
from paz.backend import boxes
from paz.backend import camera
from paz.backend import render
from paz.backend import keypoints
from paz.backend import groups
from paz.backend import image
from paz.backend import heatmaps
from paz.backend import standard
from paz.backend.image import draw
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
        'page': 'backend/angles.md',
        'functions': [
            angles.calculate_relative_angle,
            angles.reorder_relative_angles,
            angles.change_link_order,
            angles.is_hand_open
        ],
    },


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
            boxes.to_corner_form,
            boxes.extract_bounding_box_corners
        ],
    },


    {
        'page': 'backend/keypoints.md',
        'functions': [
            keypoints.build_cube_points3D,
            keypoints.normalize_keypoints2D,
            keypoints.denormalize_keypoints2D,
            keypoints.project_to_image,
            keypoints.solve_PnP_RANSAC,
            keypoints.arguments_to_image_points2D,
            keypoints.cascade_classifier,
            keypoints.project_points3D,
            keypoints.solve_PNP,
            keypoints.translate_keypoints,
            keypoints.rotate_point2D,
            keypoints.transform_keypoint,
            keypoints.add_offset_to_point,
            keypoints.translate_points2D_origin,
            keypoints.flip_keypoints_left_right,
            keypoints.compute_orientation_vector,
            keypoints.rotate_keypoints3D,
            keypoints.flip_along_x_axis,
            keypoints.uv_to_vu
        ],
    },


    {
        'page': 'backend/groups.md',
        'functions': [
            groups.rotation_vector_to_quaternion,
            groups.homogenous_quaternion_to_rotation_matrix,
            groups.quaternion_to_rotation_matrix,
            groups.rotation_matrix_to_quaternion,
            groups.get_quaternion_conjugate,
            groups.quaternions_to_rotation_matrices,
            groups.to_affine_matrix,
            groups.to_affine_matrices,
            groups.rotation_vector_to_rotation_matrix,
            groups.build_rotation_matrix_x,
            groups.build_rotation_matrix_y,
            groups.build_rotation_matrix_z,
            groups.compute_norm_SO3,
            groups.calculate_canonical_rotation,
            groups.rotation_matrix_to_axis_angle,
            groups.rotation_matrix_to_compact_axis_angle,
        ],
    },


    {
        'page': 'backend/camera.md',
        'classes': [
            (camera.Camera, [camera.Camera.is_open,
                             camera.Camera.start,
                             camera.Camera.stop,
                             camera.Camera.intrinsics_from_HFOV,
                             camera.Camera.take_photo]),
            (camera.VideoPlayer, [camera.VideoPlayer.step,
                                  camera.VideoPlayer.run,
                                  camera.VideoPlayer.record,
                                  camera.VideoPlayer.record_from_file])
        ],
    },


    {
        'page': 'backend/draw.md',
        'functions': [
            draw.draw_circle,
            draw.draw_square,
            draw.draw_triangle,
            draw.draw_keypoint,
            draw.draw_cube,
            draw.draw_dot,
            draw.draw_filled_polygon,
            draw.draw_line,
            draw.draw_random_polygon,
            draw.draw_rectangle,
            draw.lincolor,
            draw.put_text,
            draw.make_mosaic,
            draw.draw_points2D,
            draw.draw_keypoints_link,
            draw.draw_keypoints,
            draw.points3D_to_RGB,
            draw.draw_RGB_mask,
            draw.draw_RGB_masks
        ],
    },


    {
        'page': 'backend/image.md',
        'functions': [
            image.resize_image,
            image.convert_color_space,
            image.load_image,
            image.show_image,
            image.warp_affine,
            image.write_image,
            image.gaussian_image_blur,
            image.median_image_blur,
            image.get_rotation_matrix,
            image.cast_image,
            image.random_saturation,
            image.random_brightness,
            image.random_contrast,
            image.random_hue,
            image.flip_left_right,
            image.random_flip_left_right,
            image.crop_image,
            image.image_to_normalized_device_coordinates,
            image.normalized_device_coordinates_to_image,
            image.random_shape_crop,
            image.make_random_plain_image,
            image.blend_alpha_channel,
            image.concatenate_alpha_mask,
            image.split_and_normalize_alpha_channel,
            image.random_image_blur,
            image.translate_image,
            image.sample_scaled_translation,
            image.replace_lower_than_threshold,
            image.normalize_min_max,
            image.sample_scaled_translation,
            image.get_rotation_matrix,
            image.calculate_image_center,
            image.get_affine_transform,
            image.get_scaling_factor
        ],
    },


    {
        'page': 'backend/render.md',
        'functions': [
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
        'page': 'backend/heatmaps.md',
        'functions': [
            heatmaps.get_keypoints_heatmap,
            heatmaps.get_tags_heatmap,
            heatmaps.get_keypoints_locations,
            heatmaps.get_top_k_keypoints_numpy,
            heatmaps.get_valid_detections
        ],
    },


    {
        'page': 'backend/standard.md',
        'functions': [
            standard.append_lists,
            standard.append_values,
            standard.get_upper_multiple,
            standard.resize_with_same_aspect_ratio,
            standard.get_transformation_scale,
            standard.compare_vertical_neighbours,
            standard.compare_horizontal_neighbours,
            standard.get_all_indices_of_array,
            standard.gather_nd,
            standard.calculate_norm,
            standard.tensor_to_numpy,
            standard.pad_matrix,
            standard.max_pooling_2d,
            standard.predict
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
            models.Projector,
            models.DetNet,
            models.IKNet,

        ],
    },

    {
        'page': 'models/segmentation.md',
        'functions': [
            models.UNET_VGG16,
            models.UNET_VGG19,
            models.UNET_RESNET50,
            models.UNET
        ],
    },


    {
        'page': 'models/pose_estimation.md',
        'functions': [
            models.HigherHRNet
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
            datasets.OpenImages,
            datasets.CityScapes,
            datasets.Shapes
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
            losses.KeypointNetLoss,
            losses.DiceLoss,
            losses.FocalLoss,
            losses.JaccardLoss,
            losses.WeightedReconstruction,
            losses.WeightedReconstructionWithError
        ],
    },


    {
        'page': 'processors/angles.md',
        'classes': [
            processors.ChangeLinkOrder,
            processors.CalculateRelativeAngles,
            processors.IsHandOpen
        ]
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
            processors.TranslateImage,
            processors.ImageToNormalizedDeviceCoordinates,
            processors.NormalizedDeviceCoordinatesToImage,
            processors.ReplaceLowerThanThreshold,
            processors.GetNonZeroValues,
            processors.GetNonZeroArguments,
            processors.FlipLeftRightImage
        ]
    },


    {
        'page': 'processors/draw.md',
        'classes': [
            processors.DrawBoxes2D,
            processors.DrawKeypoints2D,
            processors.DrawBoxes3D,
            processors.DrawRandomPolygon,
            processors.DrawPose6D,
            processors.DrawPoses6D,
            processors.DrawHumanSkeleton,
            processors.DrawHandSkeleton,
            processors.DrawRGBMask,
            processors.DrawRGBMasks,
            processors.DrawText
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
            processors.RandomKeypointRotation,
            processors.GetTransformationSize,
            processors.GetTransformationScale,
            processors.GetSourceDestinationPoints,
            processors.GetImageCenter,
            processors.WarpAffine,
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
            processors.TranslateKeypoints,
            processors.DenormalizeKeypoints2D,
            processors.NormalizeKeypoints2D,
            processors.ArgumentsToImageKeypoints2D,
            processors.ScaleKeypoints,
            processors.ComputeOrientationVector,
        ]
    },


    {
        'page': 'processors/heatmaps.md',
        'classes': [
            processors.TransposeOutput,
            processors.ScaleOutput,
            processors.GetHeatmaps,
            processors.GetTags,
            processors.RemoveLastElement,
            processors.AggregateResults,
            processors.TopKDetections,
            processors.GroupKeypointsByTag,
            processors.AdjustKeypointsLocations,
            processors.GetScores,
            processors.RefineKeypointsLocations,
            processors.TransformKeypoints,
            processors.ExtractKeypointsLocations,
        ]
    },


    {
        'page': 'processors/munkres.md',
        'classes': [
            processors.Munkres
        ]
    },

    {
        'page': 'processors/pose.md',
        'classes': [
            processors.SolvePNP,
            processors.SolveChangingObjectPnPRANSAC,
            processors.Translation3DFromBoxWidth
        ]
    },


    {
        'page': 'processors/renderer.md',
        'classes': [
            processors.Render
        ]
    },

    {
        'page': 'processors/groups.md',
        'classes': [
            processors.ToAffineMatrix,
            processors.RotationVectorToQuaternion,
            processors.RotationVectorToRotationMatrix,
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
            processors.SelectElement,
            processors.StochasticProcessor,
            processors.Stochastic,
            processors.UnwrapDictionary,
            processors.Scale,
            processors.AppendValues,
            processors.BooleanToTextMessage,
            processors.PrintTopics
        ]
    },


    {
        'page': 'pipelines/angles.md',
        'classes': [
            pipelines.IKNetHandJointAngles
        ]
    },


    {
        'page': 'pipelines/classification.md',
        'classes': [
            pipelines.MiniXceptionFER,
            pipelines.ClassifyHandClosure
        ]
    },


    {
        'page': 'pipelines/detection.md',
        'classes': [
            pipelines.AugmentBoxes,
            pipelines.AugmentDetection,
            pipelines.PreprocessBoxes,
            pipelines.PostprocessBoxes2D,
            pipelines.DetectSingleShot,
            pipelines.DetectHaarCascade,
            pipelines.SSD512HandDetection,
            pipelines.SSD512MinimalHandPose
        ]
    },


    {
        'page': 'pipelines/heatmaps.md',
        'classes': [
            pipelines.GetHeatmapsAndTags
        ]
    },


    {
        'page': 'pipelines/image.md',
        'classes': [
            pipelines.AugmentImage,
            pipelines.PreprocessImage,
            pipelines.DecoderPredictor,
            pipelines.EncoderPredictor,
            pipelines.PreprocessImageHigherHRNet
        ]
    },


    {
        'page': 'pipelines/keypoints.md',
        'classes': [
            pipelines.KeypointNetInference,
            pipelines.KeypointNetSharedAugmentation,
            pipelines.EstimateKeypoints2D,
            pipelines.DetectKeypoints2D,
            pipelines.GetKeypoints,
            pipelines.TransformKeypoints,
            pipelines.HigherHRNetHumanPose2D,
            pipelines.DetNetHandKeypoints,
            pipelines.MinimalHandPoseEstimation,
            pipelines.DetectMinimalHand
        ]
    },


    {
        'page': 'pipelines/pose.md',
        'classes': [
            pipelines.EstimatePoseKeypoints,
            pipelines.HeadPoseKeypointNet2D32,
            pipelines.SingleInstancePIX2POSE6D,
            pipelines.MultiInstancePIX2POSE6D,
            pipelines.MultiInstanceMultiClassPIX2POSE6D
        ]
    },

    {
        'page': 'pipelines/masks.md',
        'classes': [
            pipelines.RGBMaskToImagePoints2D,
            pipelines.RGBMaskToObjectPoints3D,
            pipelines.PredictRGBMask,
            pipelines.Pix2Points
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
            pipelines.HigherHRNetHumanPose2D,
            pipelines.DetectMiniXceptionFER,
            pipelines.MiniXceptionFER,
            pipelines.FaceKeypointNet2D32,
            pipelines.HeadPoseKeypointNet2D32,
            pipelines.HaarCascadeFrontalFace,
            pipelines.SinglePowerDrillPIX2POSE6D,
            pipelines.MultiPowerDrillPIX2POSE6D,
            pipelines.PIX2POSEPowerDrill,
            pipelines.PIX2YCBTools6D,
            pipelines.DetNetHandKeypoints,
            pipelines.MinimalHandPoseEstimation,
            pipelines.DetectMinimalHand,
            pipelines.ClassifyHandClosure,
            pipelines.SSD512MinimalHandPose
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
