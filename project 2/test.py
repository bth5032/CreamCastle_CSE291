'''Logistic Regression for 0/1 in MNIST dataset'''
import numpy as np
import helper_functions as fn
import math_functions as mf
from logisticregression import LogisticRegression

	
# Load dataset from MNIST
full_trainarray = np.load('data/numpy/trainarray.npy')
full_trainlabel = np.load('data/numpy/trainlabel.npy')
full_testarray  = np.load('data/numpy/testarray.npy' )
full_testlabel  = np.load('data/numpy/testlabel.npy' )

X_train, Y_train = fn.preprocess_data(full_trainarray, full_trainlabel)
X_test, Y_test   = fn.preprocess_data(full_testarray, full_testlabel)

#---------------------#
# Logistic Regression #
#---------------------#
# 0.  Sklearn logistic regression
logreg = LogisticRegression(X_train, Y_train, X_test, Y_test)
logreg.fit()
p = logreg.predict()
print logreg.incorrect
w = logreg.w
print w.shape
print X_test
p1 = mf.predict_logistic(X_test, w.T)

for i in range(0, len(Y_test)):
	if p1[i] != Y_test[i]:
		print i



# 1.  Batch gradient descent logistic regression
w = np.zeros(X_train.shape[1])
w = mf.gradient_descent(X_train, Y_train, w, 25)
p = mf.predict(X_train, Y_train, w)


# 2.  Stochastic gradient descent logistic regression
w = np.zeros(X_train.shape[1])
w = mf.stochastic_gradient_descent(X_train, Y_train, w, 1000, 100)
print w

#--------------------#
# Softmax Regression #
#--------------------#
	


