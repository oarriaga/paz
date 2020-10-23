from paz import processors as pr
import processors as pe


class CalculateEigenFaces(pr.SequentialProcessor):
    def __init__(self, total_variance=0.95):
        super(CalculateEigenFaces, self).__init__()
        self.total_variance = total_variance
        if not (0 < self.total_variance <= 1.0):
            raise ValueError('Variance must be in (0, 1]')
        self.add(pr.ControlMap(pe.CalculateMeanFace(), [0], [1], {0: 0}))
        self.add(pr.ControlMap(pe.SubtractMeanFace(), [0, 1], [0], {1: 1}))
        self.add(pr.ControlMap(pe.ReshapeFacesToVectors()))
        self.add(pr.ControlMap(pe.ComputeCovarianceMatrix()))
        self.add(pr.ControlMap(pe.ComputeEigenvectors(), [0], [0, 1]))
        self.add(pr.ControlMap(pe.ToDescendingOrder(), [0, 1], [0, 1]))
        filter_variance = pe.FilterVariance(self.total_variance)
        self.add(pr.ControlMap(filter_variance, [0, 1], [0, 1]))


class PostrocessEigenFace(pr.SequentialProcessor):
    def __init__(self, shape=(48, 48)):
        self.shape = shape
        super(PostrocessEigenFace, self).__init__()
        self.add(pe.MinMaxNormalization(255.0))
        self.add(pe.Reshape(shape))
        self.add(pr.CastImage('uint8'))


class ProjectVectorToBase(pr.SequentialProcessor):
    def __init__(self, base, mean_face):
        super(ProjectVectorToBase, self).__init__()
        self.base, self.mean_face = base, mean_face
        self.add(pr.ControlMap(pr.ExpandDims(-1)))
        self.add(pe.SubtractMeanFace())
        self.add(pe.ComputeWeights(self.base))


class CalculateFaceWeights(pr.Processor):
    def __init__(self, base, mean_face, shape=(48, 48), with_crop=True):
        super(CalculateFaceWeights, self).__init__()
        self.base, self.mean_face = base, mean_face
        self.preprocess = pr.SequentialProcessor()
        self.convert_to_gray = pr.ConvertColorSpace(pr.RGB2GRAY)
        if with_crop:
            self.preprocess.add(pe.CropFrontalFace())
        self.preprocess.add(pr.ResizeImage(shape))
        self.preprocess.add(pr.ExpandDims(-1))
        self.subtract = pe.SubtractMeanFace()
        self.project = pe.ProjectToBase(self.base)

    def call(self, face):
        if len(face.shape) != 3:
            raise ValueError('input should have shape [H, W, num_channels]')
        if face.shape[-1] == 3:
            face = self.convert_to_gray(face)
        face = self.preprocess(face)
        face = self.subtract(face, self.mean_face)
        weights = self.project(face)
        return weights
