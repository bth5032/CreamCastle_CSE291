import numpy as np

'''Probability of class y given x.  Sigmoid function'''
def probability(x, y, w):
	return 1.0/(1.0 + np.exp(- y * np.dot(w,x) ))

'''Loss Function'''
def loss(X, Y, w):
	val = 0.0
	for i in range(0, len(X)):
		val += np.log(1.0 + np.exp(-Y[i]*np.dot(w,X[i])))
	return val

'''Gradient of Loss Function'''
def gradient(X, Y, w):
	pass