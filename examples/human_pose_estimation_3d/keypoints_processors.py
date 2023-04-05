import paz.processors as pr
from backend import standardize, solve_translation3D
from human36m import filter_keypoints2D, unnormalize_data


class FilterKeypoints2D(pr.Processor):
    def __init__(self, args_to_mean, h36m_to_coco_joints2D):
        """ Processor class for the filter_keypoints2D in human36m
        # Arguments
            args_to_mean: keypoints indices
            h36m_to_coco_joints2D: h36m joints indices

        # Returns
            Filtered keypoints2D
        """
        super(FilterKeypoints2D, self).__init__()
        self.h36m_to_coco_joints2D = h36m_to_coco_joints2D
        self.args_to_mean = args_to_mean

    def call(self, keypoints2D):
        return filter_keypoints2D(keypoints2D, self.args_to_mean,
                                  self.h36m_to_coco_joints2D)


class SolveTranslation3D(pr.Processor):
    def __init__(self, camera_intrinsics, args_to_joints3D):
        """ Processor class of the solve translation function in backend
        # Arguments
            camera_intrinsics: camera intrinsics parameters
            args_to_joints3D: joints indices


        """
        super(SolveTranslation3D, self).__init__()
        self.focal_length = camera_intrinsics[0]
        self.image_center = camera_intrinsics[1]
        self.args_to_joints3D = args_to_joints3D

    def call(self, keypoints2D, keypoints3D):
        return solve_translation3D(keypoints2D, keypoints3D, self.focal_length,
                                   self.image_center, self.args_to_joints3D)


class StandardizeKeypoints2D(pr.Processor):
    def __init__(self, data_mean2D, data_stdev2D):
        """ Processor class for standerize
        # Arguments
            data_mean2D: mean 2D
            data_stdev2D: standard deviation 2D
        # Return
            standerized keypoints2D
        """
        self.mean = data_mean2D
        self.stdev = data_stdev2D
        super(StandardizeKeypoints2D, self).__init__()

    def call(self, keypoints2D):
        return standardize(keypoints2D, self.mean, self.stdev)


class UnnormalizeData(pr.Processor):
    def __init__(self, data_mean3D, data_stdev3D, dim_to_use):
        """ Processor class for unnormalize function in human36m.py
        # Arguments
            data_mean3D: mean 3D
            data_stdev3D: standard deviation 3D
            dim_to_use: dimensions to use
        # Return
            Unormalized data
        """
        self.mean = data_mean3D
        self.stdev = data_stdev3D
        self.dim_to_use = dim_to_use
        super(UnnormalizeData, self).__init__()

    def call(self, keypoints2D):
        return unnormalize_data(keypoints2D, self.mean, self.stdev,
                                self.dim_to_use)


class SimpleBaselines3D(pr.Processor):
    def __init__(self, model, data_mean2D, data_stdev2D, data_mean3D,
                 data_stdev3D, dim_to_use3D, args_to_mean,
                 h36m_to_coco_joints2D):
        """ Processor class to predict the 3D keypoints
        # Arguments
            model: Simplebaseline 3D model
            data_mean2D: mean 2D
            data_stdev2D: standard deviation 2D
            data_mean3D: data mean 3D
            data_stdev3D: standar deviation 3D
            dim_to_use: dimensions to use
            args_to_mean: joints indices
            h36m_to_coco_joints2D: h36m data joints indices
        # Return
            keypoints2D: 2D keypoints
            keypoints3D: 3D keypoints
        """
        super(SimpleBaselines3D, self).__init__()
        self.filter = FilterKeypoints2D(args_to_mean, h36m_to_coco_joints2D)
        self.preprocess = StandardizeKeypoints2D(data_mean2D, data_stdev2D)
        self.postprocess = UnnormalizeData(data_mean3D, data_stdev3D,
                                           dim_to_use3D)
        self.predict = pr.Predict(model, self.preprocess, self.postprocess)

    def call(self, keypoints2D):
        keypoints2D = self.filter(keypoints2D)
        keypoints3D = self.predict(keypoints2D)
        return keypoints2D, keypoints3D
