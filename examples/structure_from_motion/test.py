import numpy as np
import pytest
from backend import center_and_normalize_points
from backend import compute_fundamental_matrix_np
from backend import compute_sampson_distance
from backend import triangulate_points_np
from backend import triangulate_points_cv


@pytest.fixture()
def points2D_a():
    return np.array([[254.15909, 98.73111],
                     [256.721, 152.4705],
                     [257.11206, 151.94768],
                     [260.06412, 93.84321],
                     [260.59418, 81.620544],
                     [260.6075, 97.16983],
                     [264.0722, 188.04288],
                     [265.84265, 205.66505],
                     [267.15305, 104.23621],
                     [269.5142, 81.93885],
                     [269.556, 218.20561],
                     [270.3138, 112.35374],
                     [270.3138, 112.35374],
                     [271.14764, 152.98955],
                     [271.3865, 122.965096],
                     [271.3865, 122.965096],
                     [272.8214, 215.70013],
                     [273.08545, 216.01633],
                     [274.14792, 230.95091],
                     [274.8084, 101.72618]])


@pytest.fixture()
def points2D_b():
    return np.array([[255.07156, 96.87035],
                     [257.68927, 150.35327],
                     [258.06985, 149.83197],
                     [260.89386, 92.202354],
                     [263.41153, 79.403946],
                     [260.8904, 95.524796],
                     [264.363, 185.93474],
                     [266.23245, 203.72972],
                     [267.07404, 102.7987],
                     [272.90857, 80.90839],
                     [269.43213, 216.02907],
                     [270.06955, 110.855736],
                     [270.06955, 110.855736],
                     [271.70123, 151.29602],
                     [271.3043, 121.69999],
                     [271.3043, 121.69999],
                     [272.57043, 213.67328],
                     [272.86646, 214.09387],
                     [273.72516, 229.03004],
                     [274.39767, 100.53689]])


@pytest.fixture()
def projection_matrix_a():
    return np.array([[-4.20287334e+02, 1.41314313e+01,
                      3.13622161e+02, -1.56568189e+03],
                     [-1.73641572e+01, -4.15146145e+02,
                      2.40316799e+02, -1.56288895e+03],
                     [-1.52172269e-02, 1.33204805e-03,
                      9.99883324e-01, -6.05387478e+00]])


@pytest.fixture()
def projection_matrix_b():
    return np.array([[4.13949583e+02, -1.54984249e+01,
                      3.21878147e+02, -4.94474030e+02],
                     [1.17554868e+01, 4.13547166e+02,
                      2.43393817e+02, -6.92859534e+00],
                     [-4.80418501e-03, -8.04420366e-03,
                      9.99956104e-01, -4.25253707e-01]])


@pytest.fixture()
def normalization_matrix():
    return np.array([[0.02804047, 0., -7.47952541],
                     [0., 0.02804047, -4.01244017],
                     [0., 0., 1.]])


@pytest.fixture()
def normalized_points2D_a():
    return np.array([[-0.35278509, -1.24397345],
                     [-0.28094793, 0.2629043],
                     [-0.26998242, 0.24824418],
                     [-0.18720527, -1.38103247],
                     [-0.17234214, -1.72376176],
                     [-0.17196864, -1.28775248],
                     [-0.07481682, 1.26037055],
                     [-0.02517257, 1.75450448],
                     [0.01157166, -1.08960786],
                     [0.07777941, -1.71483631],
                     [0.0789515, 2.10614767],
                     [0.10020057, -0.86198851],
                     [0.10020057, -0.86198851],
                     [0.12358184, 0.2774587],
                     [0.13027958, -0.5644411],
                     [0.13027958, -0.5644411],
                     [0.17051485, 2.03589284],
                     [0.17791894, 2.04475923],
                     [0.2077111, 2.46353187],
                     [0.22623127, -1.15999028]])


@pytest.fixture()
def sampson_distance():
    return np.array([0.96611682, 0.04510975, 0.04155349, 0.69784303,
                     0.62368629, 0.71403622, 0.08400111, 0.05026199,
                     0.45053696, 0.74617261, 0.08947246, 0.26966316,
                     0.26966316, 0.21738575, 0.32017698, 0.32017698,
                     0.08169278, 0.07368215, 0.15867713, 0.04266424])


@pytest.fixture()
def fundamental_matrix():
    return np.array([[3.00833632e-05, 2.74238112e-04, -3.28646028e-02],
                     [-2.69686680e-04, 8.95293926e-07, 6.12415044e-02],
                     [1.70345263e-02, -6.27958463e-02, 2.08534501e+00]])


@pytest.fixture()
def projected_points3D():
    return np.array([[0.36264373, -1.23625341, 3.23886064],
                     [0.39209754, -0.87272199, 3.24149991],
                     [0.39459755, -0.87635155, 3.24132408],
                     [0.40132782, -1.27008061, 3.2400428],
                     [0.40888761, -1.35549116, 3.2404476],
                     [0.40391208, -1.24753678, 3.2383663],
                     [0.4475146, -0.63285506, 3.24035431],
                     [0.46367861, -0.51320109, 3.24930376],
                     [0.44865327, -1.20027067, 3.23935538],
                     [0.47149026, -1.35147047, 3.25064579],
                     [0.48992536, -0.42969631, 3.24058443],
                     [0.47132829, -1.14601123, 3.23854366],
                     [0.47132829, -1.14601123, 3.23854366],
                     [0.48862281, -0.87095875, 3.24227053],
                     [0.48149337, -1.07339978, 3.24183318],
                     [0.48149337, -1.07339978, 3.24183318],
                     [0.51110751, -0.44686137, 3.23777487],
                     [0.51306254, -0.44443822, 3.23999546],
                     [0.52286043, -0.34340138, 3.2406254],
                     [0.49890469, -1.21821233, 3.24087723]])


def test_center_and_normalize_points(points2D_a, normalization_matrix,
                                     normalized_points2D_a):
    T, normalized_points2D = center_and_normalize_points(points2D_a)
    assert np.allclose(T, normalization_matrix)
    assert np.allclose(normalized_points2D, normalized_points2D_a)


def test_compute_fundamental_matrix_np(points2D_a, points2D_b,
                                       fundamental_matrix):
    F = compute_fundamental_matrix_np(points2D_a, points2D_b)
    assert np.allclose(F, fundamental_matrix)


def test_compute_sampson_distance(points2D_a, points2D_b, fundamental_matrix,
                                  sampson_distance):
    F = compute_fundamental_matrix_np(points2D_a, points2D_b)
    distances1 = compute_sampson_distance(F, points2D_a, points2D_b)
    distances2 = compute_sampson_distance(fundamental_matrix,
                                          points2D_a, points2D_b)
    assert np.allclose(distances1, sampson_distance)
    assert np.allclose(distances2, sampson_distance, rtol=1e-03)


def test_triangulate_points_np(projection_matrix_a, projection_matrix_b,
                               points2D_a, points2D_b, projected_points3D):
    points3D_np = triangulate_points_np(
        projection_matrix_a, projection_matrix_b, points2D_a, points2D_b)
    points3D_cv = triangulate_points_cv(
        projection_matrix_a, projection_matrix_b, points2D_a.T, points2D_b.T)
    assert np.allclose(points3D_np, projected_points3D)
    assert np.allclose(points3D_cv, projected_points3D)
    assert np.allclose(points3D_np, points3D_cv)
