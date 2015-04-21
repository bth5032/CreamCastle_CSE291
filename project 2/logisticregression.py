'''Logistic Regression for 0/1 in MNIST dataset'''
import numpy as np
import helper_functions as fn

if __name__ == "__main__":
	
	# Load dataset from MNIST
	full_trainarray = np.load('data/numpy/trainarray.npy')
	full_trainlabel = np.load('data/numpy/trainlabel.npy')
	full_testarray  = np.load('data/numpy/testarray.npy' )
	full_testlabel  = np.load('data/numpy/testlabel.npy' )

	X, Y = fn.preprocess_data(full_trainarray, full_trainlabel)

	


