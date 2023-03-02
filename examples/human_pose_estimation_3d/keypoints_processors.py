import paz.processors as pr
from backend import standardize
from human36m import filter_keypoints2D, unnormalize_data


class FilterKeypoints2D(pr.Processor):
    def __init__(self):
        super(FilterKeypoints2D, self).__init__()

    def call(self, keypoints2D):
        return filter_keypoints2D(keypoints2D)


class StandardizeKeypoints2D(pr.Processor):
    def __init__(self, data_mean2D, data_stdev2D):
        self.mean = data_mean2D
        self.stdev = data_stdev2D
        super(StandardizeKeypoints2D, self).__init__()

    def call(self, keypoints2D):
        return standardize(keypoints2D, self.mean, self.stdev)


class UnnormalizeData(pr.Processor):
    def __init__(self, data_mean3D, data_stdev3D, dim_to_use):
        self.mean = data_mean3D
        self.stdv = data_stdev3D
        self.dim_to_use = dim_to_use
        super(UnnormalizeData, self).__init__()

    def call(self, keypoints2D):
        return unnormalize_data(keypoints2D, self.mean, self.stdv,
                                self.dim_to_use)


class SimpleBaselines3D(pr.Processor):
    def __init__(self, model, data_mean2D, data_stdev2D, data_mean3D,
                 data_stdev3D, dim_to_use3D):
        super(SimpleBaselines3D, self).__init__()
        self.filter = FilterKeypoints2D()
        self.preprocess = StandardizeKeypoints2D(data_mean2D, data_stdev2D)
        self.predict = pr.Predict(model)
        self.postprocess = UnnormalizeData(data_mean3D, data_stdev3D,
                                           dim_to_use3D)

    def call(self, keypoints2D):
        keypoints2D = self.filter(keypoints2D)
        normalized_data = self.preprocess(keypoints2D)
        keypoints3D = self.predict(normalized_data)
        keypoints3D = self.postprocess(keypoints3D)
        return keypoints2D, keypoints3D
