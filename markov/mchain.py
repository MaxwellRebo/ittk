import numpy as np

import ittk_exceptions


class MarkovChain(object):
    """
    Super-simple Markov Chain class, intended to be subclassed and extended.
    """

    def __init__(self,
                 num_states,
                 t_matrix,
                 starting_state=0,
                 state_labels=[]):
        """
        :param num_states:
         The number of states in this chain. (While this can be inferred from the dimensions of t_matrix, it is better
         to be explicit.)
        :param t_matrix:
         Either a list of lists or a numpy matrix. Will convert to a numpy matrix if list of lists.
        :param state_labels:
         Optional labels for each of the states.
        :return:
        """
        self.num_states = num_states
        self.current_state = starting_state
        self.t_matrix = np.matrix(t_matrix)
        self.state_labels = state_labels

        if len(t_matrix.A) != self.num_states and len(t_matrix.A[0]) != self.num_states:
            raise ittk_exceptions.AsymmetricMatrixException(
                "Asymmetric matrix or number of states incorrectly specified. "
                "Check that the dimensions of the matrix match the number of "
                "states.")

    def run(self, num_times=1):
        for _ in xrange(num_times):
            next_state = np.random.choice(self.num_states, 1, p=self.t_matrix.A[self.current_state])[0]
        self.current_state = next_state

    def print_current_state(self):
        if self.state_labels:
            print self.state_labels[self.current_state]
        else:
            print self.current_state

    def get_eigenvalues(self):
        return np.linalg.eig(self.t_matrix.A)

    def get_trace(self):
        return np.trace(self.t_matrix.A)

    def get_determinant(self):
        return np.linalg.det(self.t_matrix.A)
