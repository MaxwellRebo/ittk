# Phi: Measure of Integrated Information of a system
# Inspired from Tononi's Integrated Information Theory
# See: http://www.architalbiol.org/aib/article/view/15056
# All credit for the Integrated Information Theory of Consciousness goes to Dr. Giulio Tononi.

import numpy as np
import ittk
import graph as gr

#Class to represent the graph for which we want to calculate its integrated information
class PhiSystem:
	def __init__(self, num_nodes):
		self.graph = gr.Graph(num_nodes)
		self.phi = 0

	#Calculate the complete integrated information (phi-value) of this system
	def integratedInformation(self):
		pass

	#Generate complete Cause-Effect repertoire of system given 'node' is in 'state'
	def causeEffectRepertoire(self, node, state):
		pass

	#Calculate effective information of state
	def effectiveInformation(self, node, state):
		pass

	#Generate possible past/future states of system given 'node' is in 'state'
	def generatePossibleStates(self, node, state):
		pass

	#Given a list of complete probabilities of states, calculate divergence from uniform distribution
	def divergenceFromUniform(self, state_probs):
		uniform_prob = 1.0/float(self.num_nodes)
		uniform = np.zeros(self.num_nodes)
		for i in range(len(uniform)):
			uniform[i] = uniform_prob
		return ittk.kldiv(uniform, state_probs, True)
