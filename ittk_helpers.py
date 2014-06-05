import numpy as np

def matchArrays(X, Y):
    if len(X) > len(Y):
        for i in range(len(X) - len(Y)):
            Y = np.append(Y, 0)
    elif len(Y) > len(X):
        for i in range(len(Y) - len(X)):
            X = np.append(X, 0)
    return (X, Y)

# Number of unique symbols in list/array
def numUnique(X):
	return len(set(X))

# Checks that number of unique symbols in arrays is the same
def binMatch(X, Y):
	return numUnique(X) == numUnique(Y)

# Verifies that observation variables of X and Y are the same, i.e. they take values in the same probability space
def variableMatch(X, Y):
	return set(X) == set(Y)

