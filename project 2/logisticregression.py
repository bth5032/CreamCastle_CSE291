import numpy as np
import os
import helper_functions as fn
from sklearn import linear_model


class SklearnLogisticRegression:
	'''sklearn based logistic regression wrapper'''
	def __init__(self):
		self.model     = linear_model.LogisticRegression()
		self.w         = None
		
	def train(self, X_train, Y_train):
		self.model.fit(X_train, Y_train)
		self.w = self.model.coef_
		return self.w

	def predict(self, X_test):
		predict = self.model.predict(X_test)
		return predict

#w = fn.logistic_regression_package(X, Y, regularization = 1.0)
	full_trainarray = np.load(os.path.join('data','numpy','trainarray.npy'))
	full_trainlabel = np.load(os.path.join('data','numpy','trainlabel.npy'))
	full_testarray  = np.load(os.path.join('data','numpy','testarray.npy' ))
	full_testlabel  = np.load(os.path.join('data','numpy','testlabel.npy' ))