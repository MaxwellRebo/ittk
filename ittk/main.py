from __future__ import division

import math
import numpy as np
import ittk_helpers as hlp
from ittk_exceptions import ITTKException
from ittk_helpers import probs
from ittk_helpers import cond_probs
from numpy import array, shape, where, in1d

# Note: All functions default to log base 2.


def check_prob_sum(arr):
    return int(sum(arr)) == 1


def entropy(x):
    x = probs(x)
    total = 0
    for x_i in x:
        if x_i == 0:
            continue
        total -= x_i * np.log2(x_i)
    return total


def mutual_information(x, y, normalized=False, base=2):
    """
    Compute the mutual information between two sets of observations.
    First converts observations to discrete conditional probability distribution, then computes their MI.

    :param x:
     List or numpy array.
    :param y:
     List or numpy array.
    :param normalized:
     Normalize the inputs. Defaults to False.
    :param base:
     The log base used in the MI calculation. Defaults to 2.
    :return:
     Float: the mutual information between x and y.
    """
    x = hlp.check_numpy_array(x)
    y = hlp.check_numpy_array(y)
    numobs = len(x)
    if numobs != len(y):
        raise Exception("Inputs not of matching length")
    mutual_info = 0.0
    uniq_x = set(x)
    uniq_y = set(y)
    for _x in uniq_x:
        for _y in uniq_y:
            px = shape(where(x == _x))[1] / numobs
            py = shape(where(y == _y))[1] / numobs
            pxy = len(where(in1d(where(x == _x)[0],
                                 where(y == _y)[0]) == True)[0]) / numobs
            if pxy > 0.0:
                mutual_info += pxy * math.log((pxy / (px * py)), base)
    if normalized: mutual_info = mutual_info / np.log2(numobs)
    return mutual_info


# Variation of information
def information_variation(x, y):
    """

    :param x:
     List or numpy array
    :param y:
     List or numpy array
    :return:
     Float: the information variation of x and y
    """
    x = hlp.check_numpy_array(x)
    y = hlp.check_numpy_array(y)
    return entropy(x) + entropy(y) - (2 * mutual_information(x, y))


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


def lag(x, y, lag_points=1):
    """
    Example:

    x, y = lag(x, y, 5)

    Note: this will reduce the length of the sequence by the number of lag points; in the above case, 5.

    :param x:
     numpy array
    :param y:
     numpy array
    :param lag_points:
     Integer. Number of places to shift by. Defaults to 1.
    :return:
     Tuple: (numpy array, numpy_array)
    """
    for i in range(lag_points):
        x.pop(0)
        y.pop()
    return x, y


def lagged_mutual_information(x, y, lag_points=1):
    """
    Convenience method to lag and do mutual information in one shot.

    :param x:
     numpy array
    :param y:
     numpy array
    :param lag_points:
     Integer. Number of places to shift by. Defaults to 1.
    :return:
     Float: the mutual information of x and y, after lag is applied.
    """
    x, y = lag(x, y, lag_points)
    return mutual_information(x, y)


def tsallis_entropy(X, entropic_index=0.99):
    """
    :param X:
        Python list or Numpy array. Should sum to 1.
    :param entropic_index:
        Real number [0, 1]. Defaults to 0.99.
    :return:
        Returns the real-valued Tsallis entropy.
    :raises:
        Raises generic ITTKException if 1 is passed, since this would result in a division by zero.
    """
    if int(entropic_index) == 1:
        raise ITTKException("Entropic index, 'q', cannot be 1 for tsallis entropy. Must be on real interval (0, 1).")
    return (1.0/(entropic_index-1)) * (1.0 - sum([p ** entropic_index for p in X]))
