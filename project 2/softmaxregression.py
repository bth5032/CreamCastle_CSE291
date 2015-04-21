'''SoftMax Regression for Digits in MNIST dataset'''
import numpy as np

if __name__ == "__main__":
	
	# Load dataset from MNIST
	full_trainarray = np.load('data/numpy/trainarray.npy')
	full_trainlabel = np.load('data/numpy/trainlabel.npy')
	full_testarray  = np.load('data/numpy/testarray.npy' )
	full_testlabel  = np.load('data/numpy/testlabel.npy' )