"""Utilities to deal with the cameras of human3.6m"""

from xml.dom import minidom
import math
import numpy as np

CAMERA_ID_TO_NAME = {
  1: "54138969",
  2: "55011271",
  3: "58860488",
  4: "60457274",
}


def eulerAnglesToRotationMatrix(theta):
  R_x = np.array([[1, 0, 0],

                  [0, math.cos(theta[0]), -math.sin(theta[0])],

                  [0, math.sin(theta[0]), math.cos(theta[0])]

                  ])
  R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],

                  [0, 1, 0],

                  [-math.sin(theta[1]), 0, math.cos(theta[1])]

                  ])

  R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],

                  [math.sin(theta[2]), math.cos(theta[2]), 0],

                  [0, 0, 1]

                  ])

  R = np.dot(R_x, np.dot(R_y, R_z))
  return R


def project_point_radial( P, R, T, f, c, k, p ):
  """
  Project points from 3d to 2d using camera parameters
  including radial and tangential distortion

  Args
    P: Nx3 points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
  Returns
    Proj: Nx2 points in pixel space
    D: 1xN depth of each point in camera space
    radial: 1xN radial distortion per point
    tan: 1xN tangential distortion per point
    r2: 1xN squared radius of the projected points before distortion
  """

  # P is a matrix of 3-dimensional points
  assert len(P.shape) == 2
  assert P.shape[1] == 3

  # print(f"\n****************In camera loop {P[:32].shape} {P[:32]}*****************\n")

  N = P.shape[0]
  X = R.dot( P.T - T ) # rotate and translate
  XX = X[:2,:] / X[2,:]
  r2 = XX[0,:]**2 + XX[1,:]**2

  radial = 1 + np.einsum( 'ij,ij->j', np.tile(k,(1, N)), np.array([r2, r2**2, r2**3]) )
  tan = p[0]*XX[1,:] + p[1]*XX[0,:]

  XXX = XX * np.tile(radial+tan,(2,1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2 )
  Proj = (f * XXX) + c
  Proj = Proj.T

  D = X[2,]

  return Proj, D, radial, tan, r2


def world_to_camera_frame(P, R, T):
  """
  Convert points from world to camera coordinates

  Args
    P: Nx3 3d points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 3d points in camera coordinates
  """

  assert len(P.shape) == 2
  assert P.shape[1] == 3

  X_cam = R.dot( P.T - T ) # rotate and translate


  return X_cam.T


def camera_to_world_frame(P, R, T):
  """Inverse of world_to_camera_frame

  Args
    P: Nx3 points in camera coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 points in world coordinates
  """

  assert len(P.shape) == 2
  assert P.shape[1] == 3
  # print(f"\n P shape: {P.shape}")
  # print(f"\n R in camera_to_world_frame shape: {R.shape} {type(R)}")
  # print(f"\n T shape: {T.shape}")
  # print(f"\n P.T shape: {(P.T).shape}")
  # print(f"\n R.T shape: {(R.T).shape}")
  # print(f"\n R.T.dot( P.T ) shape: {(R.T.dot( P.T )).shape}")
  X_cam = R.T.dot( P.T ) + T # rotate and translate

  return X_cam.T


def load_camera_params(w0, subject, camera):
  """Load h36m camera parameters

  Args
    w0: 300-long array read from XML metadata
    subect: int subject id
    camera: int camera id
  Returns
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
    name: String with camera id
  """

  # Get the 15 numbers for this subject and camera
  w1 = np.zeros(15)
  start = 6 * ((camera-1)*11 + (subject-1))
  w1[:6] = w0[start:start+6]
  w1[6:] = w0[(265+(camera-1)*9 - 1): (264+camera*9)]

  def rotationMatrix(r):    
    R1, R2, R3 = [np.zeros((3, 3)) for _ in range(3)]

    # [1 0 0; 0 cos(obj.Params(1)) -sin(obj.Params(1)); 0 sin(obj.Params(1)) cos(obj.Params(1))]
    R1[0:] = [1, 0, 0]
    R1[1:] = [0, np.cos(r[0]), -np.sin(r[0])]
    R1[2:] = [0, np.sin(r[0]),  np.cos(r[0])]

    # [cos(obj.Params(2)) 0 sin(obj.Params(2)); 0 1 0; -sin(obj.Params(2)) 0 cos(obj.Params(2))]
    R2[0:] = [ np.cos(r[1]), 0, np.sin(r[1])]
    R2[1:] = [0, 1, 0]
    R2[2:] = [-np.sin(r[1]), 0, np.cos(r[1])]

    # [cos(obj.Params(3)) -sin(obj.Params(3)) 0; sin(obj.Params(3)) cos(obj.Params(3)) 0; 0 0 1];%
    R3[0:] = [np.cos(r[2]), -np.sin(r[2]), 0]
    R3[1:] = [np.sin(r[2]),  np.cos(r[2]), 0]
    R3[2:] = [0, 0, 1]

    return (R1.dot(R2).dot(R3))
    
  R = rotationMatrix(w1)
  # print(f"\n R in load_camera_params: {R} {type(R)}")
  T = w1[3:6][:, np.newaxis]
  f = w1[6:8][:, np.newaxis]
  c = w1[8:10][:, np.newaxis]
  k = w1[10:13][:, np.newaxis]
  p = w1[13:15][:, np.newaxis]
  name = CAMERA_ID_TO_NAME[camera]

  return R, T, f, c, k, p, name


def load_cameras(bpath, subjects=[1,5,6,7,8,9,11]):
  """Loads the cameras of h36m

  Args
    bpath: path to xml file with h36m camera data
    subjects: List of ints representing the subject IDs for which cameras are requested
  Returns
    rcams: dictionary of 4 tuples per subject ID containing its camera parameters for the 4 h36m cams
  """
  rcams = {}

  xmldoc = minidom.parse(bpath)
  string_of_numbers = xmldoc.getElementsByTagName('w0')[0].firstChild.data[1:-1]

  # Parse into floats
  w0 = np.array(list(map(float, string_of_numbers.split(" "))))

  assert len(w0) == 300

  for s in subjects:
    for c in range(4): # There are 4 cameras in human3.6m
      rcams[(s, c+1)] = load_camera_params(w0, s, c+1)

  return rcams


"""
[np.array([[5.46132812e+02, 3.83554688e+02, 8.33929420e-01, 1.59745831e-02,
                        1.63969528e-02],
                       [5.51757812e+02, 3.78632812e+02, 8.70293260e-01, 1.39988316e-02,
                        1.36526786e-02],
                       [5.39804688e+02, 3.79335938e+02, 8.60917926e-01, 1.36770746e-02,
                        1.43025592e-02],
                       [5.60195312e+02, 3.82148438e+02, 7.96310723e-01, 1.12576988e-02,
                        1.21879671e-02],
                       [5.32070312e+02, 3.84257812e+02, 7.91522980e-01, 1.14094652e-02,
                        1.21821724e-02],
                       [5.77070312e+02, 4.18007812e+02, 7.47268856e-01, 1.25813745e-02,
                        1.46950083e-02],
                       [5.18710938e+02, 4.22929688e+02, 7.95782089e-01, 1.47286989e-02,
                        1.25551792e-02],
                       [5.86210938e+02, 4.70039062e+02, 7.53574371e-01, 1.47543736e-02,
                        1.41280089e-02],
                       [5.08867188e+02, 4.73554688e+02, 7.70237863e-01, 1.41594093e-02,
                        1.46752633e-02],
                       [5.96054688e+02, 5.16445312e+02, 7.63570666e-01, 1.27416942e-02,
                        1.39391217e-02],
                       [4.97617188e+02, 5.22070312e+02, 7.91590929e-01, 1.36932237e-02,
                        1.38478344e-02],
                       [5.70039062e+02, 5.14335938e+02, 6.47276998e-01, 1.30958101e-02,
                        1.22159868e-02],
                       [5.30664062e+02, 5.16445312e+02, 6.41340613e-01, 1.21107465e-02,
                        1.29677197e-02],
                       [5.81992188e+02, 5.80429688e+02, 7.29404807e-01, 1.33564528e-02,
                        1.33804791e-02],
                       [5.38398438e+02, 5.83945312e+02, 7.58461654e-01, 1.33801429e-02,
                        1.33158350e-02],
                       [6.00273438e+02, 6.47226562e+02, 7.30164111e-01, 1.35553703e-02,
                        1.25885531e-02],
                       [5.46132812e+02, 6.50039062e+02, 7.33813405e-01, 1.26404017e-02,
                        1.38478614e-02]], dtype=np.float32)] # kashmira
"""