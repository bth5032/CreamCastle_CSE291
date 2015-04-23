import bigfloat
import numpy as np

def sigmoid_loss(X, Y, w):
	'''Sigmoid Loss Function'''	
	val = 0.0
	for i in range(0, len(X)):
		val += np.log(1.0 + np.exp(-Y[i]*np.dot(w,X[i])))
	return val


def sigmoid_gradient(X, Y, w):
	'''Gradient of Sigmoid Loss Function
	Input:
		0.  Training Examples Matrix, X.  
		1.  Training Labels Vector,   Y
		2.  Initalized Weight Vector, w
	Output:
		Gradient of loss function at w'''	
	val = np.zeros(len(w))

	for i in range(0, len(w)):
		val += -X[i]*Y[i]/(1.0 + np.exp(Y[i] * np.dot(w, X[i])))
	return val


def gradient_descent(X, Y, w, M):
	'''Gradient descent of Loss Function using backtracking
	Input: 
		0.  Training Examples Matrix, X.  
		1.  Training Labels Vector,   Y
		2.  Initalized Weight Vector, w
		3.  Max Number of Iterations, M
	Output:
		Optimized Weight Vector,      w
	Further information:  
	# Backtracking:  http://users.ece.utexas.edu/~cmcaram/EE381V_2012F/Lecture_4_Scribe_Notes.final.pdf	'''
	
	alpha = 0.15
	beta  = 0.5
	
	for i in range(0, M):
		eta = 1.0

		grad = sigmoid_gradient(X,Y,w)

		while sigmoid_loss(X, Y, (w - eta * grad)) >= (sigmoid_loss(X,Y,w) - alpha * eta *np.linalg.norm(grad)):
			eta = beta * eta

			if eta < 10E-5:
				break

		w = w - eta * sigmoid_loss(X, Y, w)
		print ("Log-loss: ", sigmoid_loss(X,Y,w), " at i = ", i)
	return w


def stochastic_gradient_descent(X, Y, w):
	'''Stochastic gradient descent of Loss Function'''
	pass