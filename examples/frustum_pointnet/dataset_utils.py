import numpy as np

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8
NUM_OBJECT_POINT = 512

type2class = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3,
              'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7}
class2type = {type2class[t]: t for t in type2class}
type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
type_mean_size = {'Car': np.array([3.88311640418, 1.62856739989, 1.52563191]),
                  'Van': np.array([5.06763659, 1.9007158, 2.20532825]),
                  'Truck': np.array([10.13586957, 2.58549199, 3.2520595]),
                  'Pedestrian': np.array([0.84422524, 0.66068622, 1.7625519]),
                  'Person_sitting': np.array([0.80057803, 0.5983815, 1.2745]),
                  'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127]),
                  'Tram': np.array([16.17150617, 2.53246914, 3.53079012]),
                  'Misc': np.array([3.64300781, 1.54298177, 1.92320313])}
g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clustrs
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i, :] = type_mean_size[class2type[i]]