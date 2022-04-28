import numpy as np
hand_part_labels = ['wrist',
                    'thumb_cmc',
                    'thumb_mcp',
                    'thumb_ip',
                    'thumb_tip',
                    'index_finger_mcp',
                    'index_finger_pip',
                    'index_finger_dip',
                    'index_finger_tip',
                    'middle_finger_mcp',
                    'middle_finger_pip',
                    'middle_finger_dip',
                    'middle_finger_tip',
                    'ring_finger_mcp',
                    'ring_finger_pip',
                    'ring_finger_dip',
                    'ring_finger_tip',
                    'pinky_mcp',
                    'pinky_pip',
                    'pinky_dip',
                    'pinky_tip']

hand_part_arg = {b: a for a, b in enumerate(hand_part_labels)}

hand_part_orders = [('wrist', 'thumb_cmc'),
                    ('thumb_cmc', 'thumb_mcp'),
                    ('thumb_mcp', 'thumb_ip'),
                    ('thumb_ip', 'thumb_tip'),
                    ('wrist', 'index_finger_mcp'),
                    ('index_finger_mcp', 'index_finger_pip'),
                    ('index_finger_pip', 'index_finger_dip'),
                    ('index_finger_dip', 'index_finger_tip'),
                    ('wrist', 'middle_finger_mcp'),
                    ('middle_finger_mcp', 'middle_finger_pip'),
                    ('middle_finger_pip', 'middle_finger_dip'),
                    ('middle_finger_dip', 'middle_finger_tip'),
                    ('wrist', 'ring_finger_mcp'),
                    ('ring_finger_mcp', 'ring_finger_pip'),
                    ('ring_finger_pip', 'ring_finger_dip'),
                    ('ring_finger_dip', 'ring_finger_tip'),
                    ('wrist', 'pinky_mcp'),
                    ('pinky_mcp', 'pinky_pip'),
                    ('pinky_pip', 'pinky_dip'),
                    ('pinky_dip', 'pinky_tip')]


hand_part_color = [[179, 0, 36],
                   [227, 25, 28],
                   [252, 78, 41],
                   [253, 141, 60],
                   [135, 221, 63],
                   [188, 223, 63],
                   [219, 219, 0],
                   [255, 255, 0],
                   [100, 221, 23],
                   [108, 223, 35],
                   [123, 226, 58],
                   [154, 233, 104],
                   [4, 68, 252],
                   [17, 103, 177],
                   [24, 123, 205],
                   [42, 157, 244],
                   [143, 0, 255],
                   [160, 38, 255],
                   [177, 77, 255],
                   [193, 115, 255]]


hand_joint_color = [[140, 26, 255],
                    [179, 0, 36],
                    [227, 25, 28],
                    [252, 78, 41],
                    [253, 141, 60],
                    [135, 221, 63],
                    [188, 223, 63],
                    [219, 219, 0],
                    [255, 255, 0],
                    [100, 221, 23],
                    [108, 223, 35],
                    [123, 226, 58],
                    [154, 233, 104],
                    [4, 68, 252],
                    [17, 103, 177],
                    [24, 123, 205],
                    [42, 157, 244],
                    [143, 0, 255],
                    [160, 38, 255],
                    [177, 77, 255],
                    [193, 115, 255]]


<<<<<<< HEAD:examples/minimal_hand/joint_config.py
VISUALISATION_CONFIG = {'part_labels': hand_part_labels,
                        'part_arg': hand_part_arg,
                        'part_orders': hand_part_orders,
                        'part_color': hand_part_color,
                        'joint_color': hand_joint_color}


MANO_REF_JOINTS = np.array(
    [[-0.09566993092407175, 0.006383428857461439, 0.006186305280135194],
     [-0.007572684283876889, 0.0011830717890578813, 0.026872294317232474],
     [0.025106219230007748, 0.005192427198442781, 0.029089362428270107],
     [0.04726213151699109, 0.00389400462527089, 0.028975245669040688],
     [-0.001009489532234269, 0.004904465506518265, 0.0028287644658181762],
     [0.03017318285240305, 0.006765794024899131, -0.0027657440521595294],
     [0.053077823086293004, 0.005513689792181309, -0.006710258054895484],
     [-0.026882958864187647, -0.003556899962987172, -0.03702303672314978],
     [-0.009868550726482567, -0.0034950752461879167, -0.0495218116903115],
     [0.0059983504802553515, -0.004186231140635538, -0.05985371909262174],
     [-0.013934376495261512, 0.002426007704596194, -0.020486887752953],
     [0.014379898506751226, 0.004493014962915457, -0.02558542625500547],
     [0.03790041138358198, 0.0028049031381001317, -0.03321924042737473],
     [-0.07158022412142973, -0.009138905684414268, 0.031999152568217934],
     [-0.0519469835801523, -0.008247619132871264, 0.05569870581415224],
     [-0.029729244228165815, -0.01368059029432867, 0.07022282411348789],
     [0.07238572379473107, 0.002952405275404611, 0.027662233800221883],
     [0.0789928213101902, 0.006146648960141516, -0.012040861038314803],
     [0.023687395956832776, -0.005529320599435923, -0.0697884145827113],
     [0.062491898017990564, 0.002426856258013015, -0.04066927095293306],
     [-0.003715698261416634, -0.01635903331447523, 0.09410496964595245]])


IK_UNIT_LENGTH = 0.09473151311686484


class MANOHandJoints:
    num_joints = 21

    labels = [
        'W', #0
        'I0', 'I1', 'I2', #3
        'M0', 'M1', 'M2', #6
        'L0', 'L1', 'L2', #9
        'R0', 'R1', 'R2', #12
        'T0', 'T1', 'T2', #15
        'I3', 'M3', 'L3', 'R3', 'T3' #20, tips are manually added (not in MANO)
    ]

    # finger tips are not joints in MANO, we label them on the mesh manually
    mesh_mapping = {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}

    parents = [
        None,
        0, 1, 2,
        0, 4, 5,
        0, 7, 8,
        0, 10, 11,
        0, 13, 14,
        3, 6, 9, 12, 15
    ]


class MPIIHandJoints:
    num_joints = 21

    labels = [
        'W', #0
        'T0', 'T1', 'T2', 'T3', #4
        'I0', 'I1', 'I2', 'I3', #8
        'M0', 'M1', 'M2', 'M3', #12
        'R0', 'R1', 'R2', 'R3', #16
        'L0', 'L1', 'L2', 'L3', #20
    ]

    parents = [
        None,
        0, 1, 2, 3,
        0, 5, 6, 7,
        0, 9, 10, 11,
        0, 13, 14, 15,
        0, 17, 18, 19
    ]
=======
MINIMAL_HAND_CONFIG = {'part_labels': hand_part_labels,
                       'part_arg': hand_part_arg,
                       'part_orders': hand_part_orders,
                       'part_color': hand_part_color,
                       'joint_color': hand_joint_color}
>>>>>>> master:paz/datasets/CMU_poanoptic.py
