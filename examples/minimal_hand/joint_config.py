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


hand_part_color = [[198, 26, 255],
                   [255, 26, 255],
                   [255, 26, 198],
                   [140, 26, 255],
                   [255, 26, 140],
                   [83, 26, 255],
                   [255, 83, 26],
                   [255, 255, 26],
                   [77, 77, 255],
                   [26, 255, 140],
                   [26, 198, 255],
                   [198, 255, 26],
                   [140, 255, 26],
                   [255, 198, 26],
                   [255, 140, 26],
                   [26, 140, 255],
                   [26, 83, 255],
                   [26, 255, 198],
                   [26, 255, 198],
                   [26, 255, 255]]


hand_joint_color = [[198, 26, 255],
                    [255, 26, 255],
                    [255, 26, 198],
                    [140, 26, 255],
                    [255, 26, 140],
                    [83, 26, 255],
                    [255, 83, 26],
                    [198, 255, 26],
                    [255, 198, 26],
                    [140, 255, 26],
                    [255, 140, 26],
                    [77, 77, 255],
                    [26, 255, 140],
                    [26, 140, 255],
                    [26, 255, 198],
                    [26, 255, 198],
                    [26, 255, 198],
                    [26, 255, 198],
                    [26, 255, 198],
                    [26, 83, 255],
                    [26, 255, 255]]


VISUALISATION_CONFIG = {
    'hand': {'part_labels': hand_part_labels,
             'part_arg': hand_part_arg,
             'part_orders': hand_part_orders,
             'part_color': hand_part_color,
             'joint_color': hand_joint_color}}



"""The 21 hand landmarks."""
# WRIST = 0
# THUMB_CMC = 1
# THUMB_MCP = 2
# THUMB_IP = 3
# THUMB_TIP = 4
# INDEX_FINGER_MCP = 5
# INDEX_FINGER_PIP = 6
# INDEX_FINGER_DIP = 7
# INDEX_FINGER_TIP = 8
# MIDDLE_FINGER_MCP = 9
# MIDDLE_FINGER_PIP = 10
# MIDDLE_FINGER_DIP = 11
# MIDDLE_FINGER_TIP = 12
# RING_FINGER_MCP = 13
# RING_FINGER_PIP = 14
# RING_FINGER_DIP = 15
# RING_FINGER_TIP = 16
# PINKY_MCP = 17
# PINKY_PIP = 18
# PINKY_DIP = 19
# PINKY_TIP = 20

# wrist = 0
# thumb_cmc = 1
# thumb_mcp = 2
# thumb_ip = 3
# thumb_tip = 4
# index_finger_mcp = 5
# index_finger_pip = 6
# index_finger_dip = 7
# index_finger_tip = 8
# middle_finger_mcp = 9
# middle_finger_pip = 10
# middle_finger_dip = 11
# middle_finger_tip = 12
# ring_finger_mcp = 13
# ring_finger_pip = 14
# ring_finger_dip = 15
# ring_finger_tip = 16
# pinky_mcp = 17
# pinky_pip = 18
# pinky_dip = 19
# pinky_tip = 20
