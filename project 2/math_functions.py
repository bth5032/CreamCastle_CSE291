import numpy as np
import random, math
import random
from scipy.misc import logsumexp


def sigmoid(x, w):
	'''Sigmoid function'''
	return 1.0/(1.0+np.exp(-1.0*np.dot(x, w)))	


def loss_logistic(X, Y, w):
	'''Loss Function'''
	val = 0.0
	# Overflow issue solution:
	# http://lingpipe-blog.com/2012/02/16/howprevent-overflow-underflow-logistic-regression/
	for i in range(0, len(X)):
		val += -1.0 * (Y[i]*-1.0*logsumexp([0, -1.0*np.dot(X[i], w)]))
		val += -1.0 * ((1.0-Y[i])*-1.0*logsumexp([0, np.dot(X[i], w)]))
	return val


def gradient(X, Y, w):
	'''Gradient Loss Function calculated from all training examples
	Input:
		0.  Training Examples Matrix, X.
		1.  Training Labels Vector,   Y
		2.  Initalized Weight Vector, w
	
	Output:
		Gradient of loss function at w'''
	val = np.zeros(len(w))

	for i in range(0, len(X)):
		val += X[i]*(sigmoid(X[i], w)-Y[i])
	return val


def gradient_batch(X, Y, w, n):
	'''Gradient Loss Function calculated from a batch of training examples
	Input:
		0.  Training Examples Matrix, X.
		1.  Training Labels Vector,   Y
		2.  Initalized Weight Vector, w
		3.  Batch size of number of examples to consider, n
	
	Output:
		Gradient of loss function at w
	'''
	val = np.zeros(len(w))

	# Retrieve a random subset of samples
	indices = random.sample(xrange(X), n)
	X_rand  = X[indices]
	Y_rand  = Y[indices]

	for i in range(0, len(X_rand)):
		val += X[i]*(sigmoid(X_rand[i], w)-Y_rand[i])
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
		print 'Iteration ', i
		eta = 1.0
		grad = gradient(X, Y, w)
		loss = loss_logistic(X, Y, w)

		while loss_logistic(X, Y, (w-eta*grad)) >= (loss - alpha*eta*np.linalg.norm(grad)):
			eta = beta * eta
			if eta < 10E-20:
				break
		print ' eta: ', eta
		w = w - eta * grad
		print ' loss: ', loss
	return w


def stochastic_gradient_descent(X, Y, w, M, n):
	'''Stochastic gradient descent of Loss Function using backtracking
	Input:
		0.  Training Examples Matrix, X.
		1.  Training Labels Vector,   Y
		2.  Initalized Weight Vector, w
		3.  Max Number of Iterations, M
		4.  Batch size of number of examples to consider, n
	
	Output:
		Optimized Weight Vector,      w
	Further information:
	http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/ '''
	eta   = 1E-5

	for i in range(0, M):
		w = w - eta*gradient_batch(X, Y, w, n)
	print 'loss: ', loss_logistic(X, Y, w)
	return w


def identity(y, k):
	'''Identiy function that checks if integers match'''
	if y == k:
		return 1
	else:
		return 0


def softmax(W, x):
	'''Computes the softmax probability for a W-matrix'''
	prob  = np.dot(W, x)
	prob  = np.exp(prob)
	denom = np.sum(prob)
	return prob/denom


def loss_softmax(W, X, Y):
	'''Loss function for softmax regression'''
	val = 0.0
	for i in range(0, len(X)):
		for k in range(0, 10):
			vec  = np.dot(W, X[i]) 
			val -= identity(Y[i], k)*(np.dot(W[:,k], X[i]) - logsumexp(vec))
	return val


def gradient_softmax_batch(X, Y, W, n):
	'''Gradient Loss Function calculated from a batch of training examples
	Input:
		0.  Training Examples Matrix, X.
		1.  Training Labels Vector,   Y
		2.  Initalized Weight Vector, w
		3.  Batch size of number of examples to consider, n
	
	Output:
		Gradient of loss function at w
	'''
	val = np.zeros(W.shape)

	# Retrieve a random subset of samples
	indices = random.sample(xrange(X), n)
	X_rand  = X[indices]
	Y_rand  = Y[indices]

	for i in range(0, len(X_rand)):
		for k in range(0, 10):
			val -= X_rand[i]*identity(Y_rand[i], k)-softmax(W, X_rand[i])
	return val


def stochastic_gradient_descent_softmax(X, Y, W, M, n):
	'''Gradient descent of using backtracking
	Input:
		0.  Training Examples Matrix (m, , X.
		1.  Training Labels Vector (m, 1),   Y
		2.  Initalized Weight Matrix (n, k), W
		3.  Max Number of Iterations, M
	
	Output:
		Optimized Weight Matrix (n, k),      W
	Further information:
	# Backtracking:  http://users.ece.utexas.edu/~cmcaram/EE381V_2012F/Lecture_4_Scribe_Notes.final.pdf	'''

	eta = 1.0

	for i in range(0, M):
		# Update the parameter matrix
		W = W - eta*gradient_softmax_batch(X, Y, W, n)
	return W
	

def value_difference(x, epsilon, f, df):
	'''computes the difference between the numerical gradient of f at x and df at x.'''

	y = np.array(x) #create y as a new vector
	gradientDiff = [] #stores return vals
	analyticGrad = df(x) #computed gradient from test function

	for i in xrange(0,len(x)):
		#add epsilon to the ith component
		y[i] += epsilon
		#print "taking (%f - %f)/%.2f - %.2f" % (f(x), f(y), epsilon, analyticGrad[i])
		gradientDiff.append(abs(abs(f(x) - f(y))/epsilon - analyticGrad[i]))
		#reset y
		y[i] -= epsilon

	mag = lambda v: math.sqrt(sum(i**2 for i in v))
	return mag(gradientDiff)


def gradient_check(f, df, numargs, stochastic=True, numcheck=10, x=None, epsilon=.001, domain=10):
	'''Takes in a vector valued function 'f,' and another function 'df,' which is the supposed
	gradient of f. 'numargs' is the number of arguments that are taken in by f. By default
	'stochastic' is True, which means the function will generate 'numcheck' points for f to ingest.
	Otherwise, x, which is a list of points, must be provided. 'epsilon' is the step size which
	will be used to compute the numerical gradient of f. 'domain' is the max value which can be
	assigned to the elements of the generated points.

	Returns a tuple of the form (min, avg, max), which is the minimum, maximum, and average difference
	between the numerical gradient and df for the points'''

	if stochastic:
		#generate points to check the value of f
		x = []
		if not numargs:
			raise ValueError("If you want points generated, you must define the number of arguments for f")
		for i in xrange(0, numcheck):
			x.append([domain*random.random() for j in xrange(1,numargs+1)])

	elif x==None:
		raise ValueError("If you don't want points generated, you must define the points x")

	min = None
	max = 0
	average = 0

	for i in xrange(0,numcheck):
		diff = value_difference(x[i],epsilon,f,df)
		average += diff/numcheck
		if diff > max:
			max = diff
		if (diff < min) or min == None:
			min = diff

	return (min, average, max)






###TESTING########################################################################
def sumSquare(x):
	'''x1^2 + x2^2'''
	return (x[0])*x[0] + (x[1])*x[1]

def dSumSquare(x):
	'''derivative of sum square'''
	return (2*x[0], 2*x[1])

if __name__=='__main__':
	#gradient_check(1,1,5)
	#gradient_check(1,2,3, False)
	#value_difference([2,3], .01, sumSquare, dSumSquare)
	print gradient_check(sumSquare,dSumSquare,2)
