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
from hlp import probs
from numpy import array, shape, where, in1d


def entropy(X):
    X = probs(X)
    total = 0
    for x in X:
        if x==0: continue
        total -= x*np.log2(x)
    return total

def mutualInformation(X, Y, normalized=False, base=2):
    #Accepts lists or numpy arrays; will make X and Y into numpy arrays if they're not already
    if (type(X).__module__ !='numpy'):
        X = array(X)
    if (type(Y).__module__ !='numpy'):
        X = array(Y)
    numobs = len(X)
    if numobs != len(Y):
        raise Exception("Not matching length")
        return None
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
    if normalized: mutual_info = mutual_info / np.log2(numobs) 
    return mutual_info

#Variation of information
def informationVariation(X, Y):
    return entropy(X) + entropy(Y) - 2*mutualInformation(X, Y)

def kldiv(X, Y, isprobs=False):
    if isprobs==False:
        p = probs(X)
        q = probs(Y)
    else:
        p = X
        q = Y
    p, q = hlp.match_arrays(p, q)
    logpq = np.array([])
    for i in range(len(p)):
        if q[i]==0 or p[i]==0: logpq = np.append(logpq, 0)
        else: logpq = np.append(logpq, np.log2(p[i]/q[i]))
    kldivergence = np.dot( p, logpq )
    return kldivergence

#Note: this will reduce the length of the sequence by the number of lag points
#X: numpy array
#Y: numpy array
#lag: integer.  defaults to 1
def laggedMutualInformation(X, Y, lag=1):
    for i in range(lag):
        X.pop(0)
        Y.pop()
    return mutualInformation(X, Y)


