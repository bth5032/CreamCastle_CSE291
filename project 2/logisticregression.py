'''Logistic Regression for 0/1 in MNIST dataset'''
import numpy as np
import helper_functions as fn
	
# Load dataset from MNIST
full_trainarray = np.load('data/numpy/trainarray.npy')
full_trainlabel = np.load('data/numpy/trainlabel.npy')
full_testarray  = np.load('data/numpy/testarray.npy' )
full_testlabel  = np.load('data/numpy/testlabel.npy' )

X_train, Y_train = fn.preprocess_data(full_trainarray, full_trainlabel)
X_test, Y_test = fn.preprocess_data(full_testarray, full_testlabel)

print X_train.shape
print Y_train.shape

# Logistic regression via sklearn
w = fn.logistic_regression_package(X_train, Y_train, regularization = 1.0)

print w


	


