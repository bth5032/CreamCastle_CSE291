import numpy as np


def probability(x, y, w):
	'''Probability of class y given x.  Sigmoid function'''
	return 1.0/(1.0 + np.exp(- y * np.dot(w,x) ))

def loss(X, Y, w):
	'''Loss Function'''
	val = 0.0
	for i in range(0, len(X)):
		val += np.log(1.0 + np.exp(-Y[i]*np.dot(w,X[i])))
	return val

def gradient(X, Y, w):
	'''Gradient of Loss Function'''	
	pass

def stochastic_gradient(X, Y, w):
	'''Gradient of Loss Function'''
	pass