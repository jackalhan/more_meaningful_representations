import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances

vector1 = [3.0, 104.0]
vector2 = [18.0, 90.0]
vectorOther = [1.0, 2.0, 3.0]

v1 = np.array(vector1)
v2 = np.array(vector2)
vOther = np.array(vectorOther)

# v1 = normalize(ques[0][:, np.newaxis], axis=0).ravel()
# v2 = normalize(pars[0][:, np.newaxis], axis=0).ravel()
# x = np.sqrt(np.sum(np.square(np.subtract(v1, v2))))
#
#
# v1 = normalize(ques[0].reshape(1,-1))
# v2 = normalize(pars[0].reshape(1,-1))
# y = euclidean_distances(v1, v2)

def diff_Length_Error():
    raise RuntimeWarning("The length of the two vectors are not the same!")


def euclidean0_0(vector1, vector2):
    ''' calculate the euclidean distance
        input: numpy.arrays or lists
        return: 1. quard distance, 2. euclidean distance
    '''


    quar_distance = 0
    if (len(vector1) != len(vector2)):
        diff_Length_Error()
    zipVector = zip(vector1, vector2)

    for member in zipVector:
        quar_distance += (member[1] - member[0]) ** 2

    return quar_distance, math.sqrt(quar_distance)


def euclidean0_1(vector1, vector2):
    '''calculate the euclidean distance, no numpy
    input: numpy.arrays or lists
    return: euclidean distance
    '''
    dist = [(a - b) ** 2 for a, b in zip(vector1, vector2)]
    dist = math.sqrt(sum(dist))
    return dist


def euclidean2(vector1, vector2):
    '''calculate the euclidean distance, use numpy.dot() function
    input: numpy.arrays or lists
    return: euclidean distance
    '''
    if type(vector1) == list:
        vector1 = np.array(vector1)
    if type(vector2) == list:
        vector2 = np.array(vector2)
    diff = vector2 - vector1
    squareDistance = np.dot(diff.T, diff)
    return squareDistance, math.sqrt(squareDistance)

def euclidean3(vector1, vector2):
    ''' use numpy.linalg.norm to calculate the euclidean distance. '''
    vector1, vector2 = list_to_npArray(vector1, vector2)
    distance = np.linalg.norm(vector1 - vector2, 2,
                              0)  # the third argument "0" means the column, and "1" means the line.
    return distance


def euclidean4(vector1, vector2):
    ''' use scipy to calculate the euclidean distance. '''
    dist = distance.euclidean(vector1, vector2)
    return dist


def euclidean5(vector1, vector2):
    ''' use matplotlib.mlab to calculate the euclidean distance. '''
    vector1, vector2 = list_to_npArray(vector1, vector2)
    dist = plt.mlab.dist(vector1, vector2)
    return dist


def list_to_npArray(vector1, vector2):
    '''convert the list to numpy array'''
    if type(vector1) == list:
        vector1 = np.array(vector1)
    if type(vector2) == list:
        vector2 = np.array(vector2)
    return vector1, vector2