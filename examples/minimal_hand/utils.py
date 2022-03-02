import json

import numpy as np

def load_json(path):
  """
  Load pickle data.
  Parameter
  ---------
  path: Path to pickle file.
  Return
  ------
  Data in pickle file.
  """
  with open(path) as f:
    data = json.load(f)
    x = data
  return data


@staticmethod
def MANO_Test_Pose():
    """
    Return MANO demo POSE
    """
    POSE = np.zeros(48)
    # rotate default mano_handstate pose at its middelfinger rotate up
    POSE[12] = 0.0
    POSE[13] = 0.0
    POSE[14] = 1.0
    POSE[15] = 0.0
    POSE[16] = 0.0
    POSE[17] = 1.0
    POSE[18] = 0.0
    POSE[19] = 0.0
    POSE[20] = 1.0
    return POSE

@staticmethod
def MANO_Test_Quaternions(type=1):
    """
    returns demo output form minimal hand pose detection iknet
    """
    if type == 1:  # wrist
        return [[0.545065, 0.09157968, -0.08761995, -0.8287581],
                [0.51521826, 0.10341978, -0.0889774, -0.84613085],
                [0.14796199, 0.19324194, -0.0020313, -0.9699282],
                [0.4264529, -0.1655271, 0.02198492, 0.8889631],
                [0.5270036, 0.08506694, -0.07981452, -0.84181964],
                [0.15601957, 0.08360817, 0.07223649, -0.98155457],
                [0.23149574, -0.08624467, -0.1788109, 0.9523646],
                [0.53717464, 0.07146199, -0.0884908, -0.83576673],
                [0.1839112, -0.14812586, 0.18513754, -0.95391804],
                [0.01367265, 0.21975, -0.16912761, 0.9606867],
                [0.53122705, 0.07444655, -0.07416294, -0.8406874],
                [0.11037478, 0.02365384, 0.00689615, -0.9935846],
                [0.18242396, 0.0335233, -0.2087681, 0.96021557],
                [0.53914934, 0.11589456, -0.08483, -0.8298738],
                [0.51210964, 0.3087524, -0.21752876, -0.77142525],
                [0.54978365, 0.22628, -0.13179809, -0.79319894],
                [0.48344702, -0.09132235, -0.12148427, 0.86207926],
                [0.40524757, -0.01419774, -0.22037801, 0.88713366],
                [0.10671954, 0.24358426, -0.35025236, 0.8981096],
                [0.33911192, 0.07428803, -0.28663564, 0.89293027],
                [0.48058134, 0.30897322, -0.1592253, -0.80512387]]

    if type == 2:  # peace sign
        return [
            [0.49039537, 0.38119018, -0.41371813, -0.6656154],
            [0.46588847, 0.37707287, -0.42263147, -0.6798139],
            [0.3899733, 0.40524578, -0.42275283, -0.7106172],
            [0.31398332, 0.42147028, -0.43543488, -0.7308719],
            [0.4774761, 0.38166893, -0.42065752, -0.67036766],
            [0.45405883, 0.37312254, -0.43406695, -0.6827854],
            [0.45532864, 0.3795652, -0.4237885, -0.6848426],
            [0.48516688, 0.38969672, -0.41945183, -0.66091573],
            [0.17531246, 0.24589498, 0.12281283, -0.94536674],
            [0.1155391, -0.15757157, -0.29526165, 0.93522316],
            [0.49109262, 0.37801597, -0.41700003, -0.6648632],
            [0.05447021, -0.36720574, 0.02378312, 0.9282388],
            [0.41456276, -0.24038044, -0.41098392, 0.7755302],
            [0.47871348, 0.37387446, -0.4239637, -0.67179316],
            [0.29140592, 0.7114447, -0.50396717, -0.393632],
            [0.5270378, 0.5673109, -0.48269784, -0.40913594],
            [0.30765548, 0.40645477, -0.3898532, -0.766914],
            [0.46456555, 0.38078883, -0.43629164, -0.6699465],
            [0.20912649, -0.12994683, -0.5195667, 0.81818724],
            [0.61401176, -0.11816695, -0.57736945, 0.5250434],
            [0.34455463, 0.71455824, -0.50819343, -0.33530292],
        ]