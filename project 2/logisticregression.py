import numpy as np
import os
import helper_functions as fn
from sklearn import linear_model


class LogisticRegression:
	'''sklearn based logistic regression wrapper'''
	def __init__(self, X_train, Y_train, X_test, Y_test):
		self.model     = None
		self.w         = None
		self.incorrect = []
		self.X_train   = X_train
		self.X_test    = X_test
		self.Y_train   = Y_train
		self.Y_test    = Y_test

	def fit(self):
		logreg = linear_model.LogisticRegression()
		logreg.fit(self.X_train, self.Y_train)
		self.model = logreg
		self.w     = logreg.coef_

	def predict(self):
		predict = self.model.predict(self.X_test)

		for i in range(0, len(predict)):
			if predict[i] != self.Y_test[i]:
				self.incorrect.append(i)

		return predict

#Logistic regression via sklearn
#w = fn.logistic_regression_package(X, Y, regularization = 1.0)
if __name__=='__main__':
	# Load dataset from MNIST
	full_trainarray = np.load(os.path.join('data','numpy','trainarray.npy'))
	full_trainlabel = np.load(os.path.join('data','numpy','trainlabel.npy'))
	full_testarray  = np.load(os.path.join('data','numpy','testarray.npy' ))
	full_testlabel  = np.load(os.path.join('data','numpy','testlabel.npy' ))

	X_train, Y_train = fn.preprocess_data(full_trainarray, full_trainlabel)
	X_test, Y_test = fn.preprocess_data(full_testarray, full_testlabel)

