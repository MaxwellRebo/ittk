import numpy as np


def probs(X):
    n = len(X)
    c = np.bincount(X)
    p = c / float(n)
    return p


def cond_probs(obs):
    if type(obs) is str:
        obs = list(obs)
    obs = [str(ob) for ob in obs]
    syms = set(obs)
    counts = {}
    probs_dict = {}
    totals = {}
    for sym in syms:
        totals[sym] = obs.count(sym)
        counts[sym] = {}
        probs_dict[sym] = {}
        for other_sym in syms:
            counts[sym][other_sym] = 0
            probs_dict[sym][other_sym] = 0
    for i in range(len(obs) - 1):
        if obs[i + 1] in counts[obs[i]]:
            counts[obs[i]][obs[i + 1]] += 1
            continue
        counts[obs[i]][obs[i + 1]] = 1
    for sym in syms:
        div = 1 if totals[sym] is 0 else totals[sym]
        for other_sym in syms:
            probs_dict[sym][other_sym] = counts[sym][other_sym] / float(div)
    return probs_dict


def match_arrays(X, Y):
    if len(X) > len(Y):
        for i in range(len(X) - len(Y)):
            Y = np.append(Y, 0)
    elif len(Y) > len(X):
        for i in range(len(Y) - len(X)):
            X = np.append(X, 0)
    return (X, Y)


def check_numpy_array(x):
    if type(x).__module__ != 'numpy':
        x = np.array(x)
    return x


# Number of unique symbols in list/array
def num_unique(X):
    return len(set(X))


# Checks that number of unique symbols in arrays is the same
def bin_match(X, Y):
    return num_unique(X) == num_unique(Y)


# Verifies that observation variables of X and Y are the same, i.e. they take values in the same probability space
def variable_match(X, Y):
    return set(X) == set(Y)

