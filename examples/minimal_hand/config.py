MANO_CONF_ROOT_DIR = './'

# Attetion: pretrained detnet and iknet are only trained for left model! use left hand
DETECTION_MODEL_PATH = MANO_CONF_ROOT_DIR+'model/detnet/detnet.ckpt'
IK_MODEL_PATH = MANO_CONF_ROOT_DIR+'model/iknet/iknet.ckpt'

# Convert 'HAND_MESH_MODEL_PATH' to 'HAND_MESH_MODEL_PATH_JSON' with 'prepare_mano.py'
HAND_MESH_MODEL_LEFT_PATH_JSON = MANO_CONF_ROOT_DIR+'model/hand_mesh/mano_hand_mesh_left.json'
HAND_MESH_MODEL_RIGHT_PATH_JSON = MANO_CONF_ROOT_DIR+'model/hand_mesh/mano_hand_mesh_right.json'

OFFICIAL_MANO_LEFT_PATH = MANO_CONF_ROOT_DIR+'model/mano/MANO_LEFT.pkl'
OFFICIAL_MANO_RIGHT_PATH = MANO_CONF_ROOT_DIR+'model/mano/MANO_RIGHT.pkl'

IK_UNIT_LENGTH = 0.09473151311686484 # in meter

OFFICIAL_MANO_PATH_LEFT_JSON = MANO_CONF_ROOT_DIR+'model/mano_handstate/mano_left.json'
OFFICIAL_MANO_PATH_RIGHT_JSON = MANO_CONF_ROOT_DIR+'model/mano_handstate/mano_right.json'