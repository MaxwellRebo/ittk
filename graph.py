# Base graph class for ittk

import numpy as np

class Graph:
	def __init__(self, num_nodes):
		self.num_nodes = num_nodes
		self.edges = np.zeros((num_nodes, num_nodes))
		self.states = np.zeros(num_nodes)