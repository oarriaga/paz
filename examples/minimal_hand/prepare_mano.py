import pickle

import numpy as np
import json

from config import OFFICIAL_MANO_RIGHT_PATH, \
    HAND_MESH_MODEL_RIGHT_PATH_JSON, OFFICIAL_MANO_LEFT_PATH, HAND_MESH_MODEL_LEFT_PATH_JSON
from kinematics import MANOHandJoints


def prepare_mano_json(left=True):
    """
  Use this function to convert a mano_handstate model (from MANO-Hand Project) to the hand
  model we want to use in the project.
  """

    if left:
        with open(OFFICIAL_MANO_LEFT_PATH, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    else:
        with open(OFFICIAL_MANO_RIGHT_PATH, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

    output = {}
    output['verts'] = data['v_template']
    output['faces'] = data.pop("f")
    output['mesh_basis'] = np.transpose(data["shapedirs"], (2, 0, 1))

    j_regressor = np.zeros([21, 778])
    j_regressor[:16] = data["J_regressor"].toarray()
    for k, v in MANOHandJoints.mesh_mapping.items():
        j_regressor[k, v] = 1
    output['j_regressor'] = j_regressor
    output['joints'] = np.matmul(output['j_regressor'], output['verts'])

    raw_weights = data["weights"]
    weights = [None] * 21
    weights[0] = raw_weights[:, 0]
    for j in 'IMLRT':
        weights[MANOHandJoints.labels.index(j + '0')] = np.zeros(778)
        for k in [1, 2, 3]:
            src_idx = MANOHandJoints.labels.index(j + str(k - 1))
            tar_idx = MANOHandJoints.labels.index(j + str(k))
            weights[tar_idx] = raw_weights[:, src_idx]
    output['weights'] = np.expand_dims(np.stack(weights, -1), -1)

    # save in json_file
    if left:
        with open(HAND_MESH_MODEL_LEFT_PATH_JSON, 'w') as f:
            mano_data_string = convert_to_plain(output)
            json.dump(mano_data_string, f)
    else:
        with open(HAND_MESH_MODEL_RIGHT_PATH_JSON, 'w') as f:
            mano_data_string = convert_to_plain(output)
            json.dump(mano_data_string, f)


def convert_to_plain(hand):
    plain = {}
    for k in ["verts", "faces", "weights", "joints", "mesh_basis", "j_regressor"]:
        plain[k] = np.array(hand[k]).tolist()

    return plain


if __name__ == '__main__':
    prepare_mano_json(left=False)
    prepare_mano_json(left=True)
