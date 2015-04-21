import numpy as np


def sigmoid_probability(x, y, w):
	'''Probability of class y given x.  Sigmoid function'''
	return 1.0/(1.0 + np.exp(- y * np.dot(w,x) ))

def sigmoid_loss(X, Y, w):
	'''Sigmoid Loss Function'''
	val = 0.0
	for i in range(0, len(X)):
		val += np.log(1.0 + np.exp(-Y[i]*np.dot(w,X[i])))
	return val

def sigmoid_gradient(X, Y, w):
	'''Gradient of Sigmoid Loss Function'''	
	pass

def gradient_descent(X, Y, w):
	'''Gradient descent of Loss Function'''
	pass

def stochastic_gradient_descent(X, Y, w):
	'''Stochastic gradient descent of Loss Function'''
	pass