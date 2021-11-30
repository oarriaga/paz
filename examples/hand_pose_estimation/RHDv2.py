KINEMATIC_CHAIN_DICT = {0: 'root',
                        4: 'root', 3: 4, 2: 3, 1: 2,
                        8: 'root', 7: 8, 6: 7, 5: 6,
                        12: 'root', 11: 12, 10: 11, 9: 10,
                        16: 'root', 15: 16, 14: 15, 13: 14,
                        20: 'root', 19: 20, 18: 19, 17: 18}
KINEMATIC_CHAIN_LIST = list(KINEMATIC_CHAIN_DICT.keys())

# Check if usage constants is okay or use them as a parameter to function
LEFT_ROOT_KEYPOINT_ID = 0
LEFT_ALIGNED_KEYPOINT_ID = 12
LEFT_LAST_KEYPOINT_ID = 20

RIGHT_ROOT_KEYPOINT_ID = 21
RIGHT_ALIGNED_KEYPOINT_ID = 33
RIGHT_LAST_KEYPOINT_ID = 41

LEFT_HAND = 0
RIGHT_HAND = 1