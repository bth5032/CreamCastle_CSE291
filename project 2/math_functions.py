import bigfloat
import numpy as np
import random, math
from scipy.misc import logsumexp


def sigmoid(x, w):
	'''Sigmoid function'''
	return 1.0/(1.0+np.exp(-1.0*np.dot(x, w)))	


def loss_function(X, Y, w):
	'''Loss Function'''
	val = 0.0
	# Overflow issue solution:
	# http://lingpipe-blog.com/2012/02/16/howprevent-overflow-underflow-logistic-regression/
	for i in range(0, len(X)):
		val += -1.0 * (Y[i]*-1.0*logsumexp([0, -1.0*np.dot(X[i], w)]))
		val += -1.0 * ((1.0-Y[i])*-1.0*logsumexp([0, np.dot(X[i], w)]))
	return val


def gradient(X, Y, w):
	'''Gradient Loss Function
	Input:
		0.  Training Examples Matrix, X.
		1.  Training Labels Vector,   Y
		2.  Initalized Weight Vector, w
	Output:
		Gradient of loss function at w'''
	val = np.zeros(len(w))

	for i in range(0, len(w)):
		val += X[i]*(sigmoid(X[i], w)-Y[i])
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
		loss = loss_function(X, Y, w)

		while loss_function(X, Y, (w-eta*grad)) >= (loss - alpha*eta*np.linalg.norm(grad)):
			eta = beta * eta
			if eta < 10E-5:
				break
		w = w - eta * grad
		print " loss: ", loss
	return w


def stochastic_gradient_descent(X, Y, w):
	'''Stochastic gradient descent of Loss Function'''
	pass


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
