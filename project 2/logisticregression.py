import numpy as np
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

