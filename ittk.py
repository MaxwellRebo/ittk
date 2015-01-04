'''
Ittk: Information Theory Toolkit.
2013 Maxwell Rebo.
MIT license.
'''

from __future__ import division

import unittest
import math
import numpy as np
import ittk_helpers as hlp
from ittk_helpers import probs
from numpy import array, shape, where, in1d

# All functions default to log base 2.


def entropy(X):
    X = probs(X)
    total = 0
    for x in X:
        if x == 0:
            continue
        total -= x * np.log2(x)
    return total


# Accepts lists or numpy arrays; will make X and Y into numpy arrays if they're not already
def mutual_information(X, Y, normalized=False, base=2):
    X = hlp.check_numpy_array(X)
    Y = hlp.check_numpy_array(Y)
    numobs = len(X)
    if numobs != len(Y):
        raise Exception("Not matching length")
        return None
    mutual_info = 0.0
    uniq_x = set(X)
    uniq_y = set(Y)
    for x in uniq_x:
        for y in uniq_y:
            px = shape(where(X == x))[1] / numobs
            py = shape(where(Y == y))[1] / numobs
            pxy = len(where(in1d(where(X == x)[0],
                                 where(Y == y)[0]) == True)[0]) / numobs
            if pxy > 0.0:
                mutual_info += pxy * math.log((pxy / (px * py)), base)
    if normalized: mutual_info = mutual_info / np.log2(numobs)
    return mutual_info


# Variation of information
def information_variation(X, Y):
    return entropy(X) + entropy(Y) - (2 * mutual_information(X, Y))


def kldiv(X, Y, isprobs=False):
    if isprobs == False:
        p = probs(X)
        q = probs(Y)
    else:
        p = X
        q = Y
    p, q = hlp.match_arrays(p, q)
    logpq = np.array([])
    for i in range(len(p)):
        if q[i] == 0 or p[i] == 0:
            logpq = np.append(logpq, 0)
        else:
            logpq = np.append(logpq, np.log2(p[i] / q[i]))
    kldivergence = np.dot(p, logpq)
    return kldivergence


# Note: this will reduce the length of the sequence by the number of lag points
# X: numpy array
#Y: numpy array
#lag_points: integer. defaults to 1
def lag(x, y, lag_points=1):
    for i in range(lag_points):
        x.pop(0)
        y.pop()
    return x, y


def lagged_mutual_information(X, Y, lag_points=1):
    X, Y = lag(X, Y, lag_points)
    return mutual_information(X, Y)


####################
# Test cases
####################

class TestMutualInformation(unittest.TestCase):
    def test_mutual_information(self):
        x = np.array([7, 7, 7, 3])
        y = np.array([0, 1, 2, 3])
        mut_inf = mutual_information(x, y)
        self.assertEquals(0.8112781244591329, mut_inf)
        x2 = [1, 0, 1, 1, 0]
        y2 = [1, 1, 1, 0, 0]
        self.assertEquals(mutual_information(x2, y2), 0.01997309402197492)



class TestInformationVariation(unittest.TestCase):
    def test_information_variation(self):
        x = np.array([7, 7, 7, 3])
        y = np.array([0, 1, 2, 3])
        inf_var = information_variation(x, y)
        self.assertEquals(1.1887218755408671, inf_var)


# class TestProbs(unittest.TestCase):
#     def test_probs(self):
#         correct_array = array([0, 0.33333333, 0.33333333, 0.33333333])
#         test_array = hlp.probs(np.array([1, 2, 3]))
#         print test_array
#         print correct_array
#         print type(test_array)
#         print type(correct_array)
#         self.assertTrue(np.array_equal(test_array, correct_array))


if __name__ == '__main__':
    unittest.main()


