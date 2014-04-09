'''
Ittk: Information Theory Toolkit.
2013 Maxwell Rebo.
MIT license.
'''

#All default to log base 2.  Modify to your purposes.
from __future__ import division
import math
import numpy as np
import ittk_helpers as hlp
from numpy import array, shape, where, in1d


def entropy(X):
    X = probs(X)
    total = 0
    for x in X:
        if x==0: continue
        total -= x*np.log2(x)
    return total

def probs(X):
    n = len(X)
    c = np.bincount(X)
    P = c / float(n)
    return P

def mutual_information(X, Y):
    #Expects numpy arrays.  Will not work on regular lists
    numobs = len(X)
    base = 2
    assert numobs == len(Y), "Not matching length"
    mutual_info = 0.0
    uniq_x = set(X)
    uniq_y = set(Y)
    for x in uniq_x:
        for y in uniq_y:
            px = shape(where(X==x))[1] / numobs
            py = shape(where(Y==y))[1] / numobs
            pxy = len(where(in1d(where(X==x)[0], 
                            where(Y==y)[0])==True)[0]) / numobs
            if pxy > 0.0:
                mutual_info += pxy * math.log((pxy / (px*py)), base)
    return mutual_info

#Variation of information
def information_variation(X, Y):
    return entropy(X) + entropy(Y) - 2*mutual_information(X, Y)

def kldiv(X, Y):
    p = probs(X)
    q = probs(Y)
    p, q = hlp.match_arrays(p, q)
    logpq = np.array([])
    for i in range(len(p)):
        if q[i]==0 or p[i]==0: logpq = np.append(logpq, 0)
        else: logpq = np.append(logpq, np.log2(p[i]/q[i]))
    kldivergence = np.dot( p, logpq )
    return kldivergence