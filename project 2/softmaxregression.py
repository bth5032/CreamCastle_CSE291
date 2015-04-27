import numpy as np
import helper_functions as fn
from sklearn import linear_model


class SoftmaxRegression:
	'''sklearn based multiclass logistic (softmax) regression'''
	def __init__(self, X_train, Y_train, X_test, Y_test):
		self.model     = None
		self.W         = None
		self.incorrect = []
		self.X_train   = X_train
		self.X_test    = X_test
		self.Y_train   = Y_train
		self.Y_test    = Y_test

