import numpy as np
from sklearn import linear_model

class LogisticRegression:
	'''sklearn based logistic regression wrapper'''
	def __init__(self, X_train, Y_train, X_test, Y_test):
		self.w       = None
		self.X_train = X_train
		self.X_test  = X_test
		self.Y_train = Y_train
		self.Y_test  = Y_test

	def fit(self):
		logreg = linear_model.LogisticRegression()
		logreg.fit(self.X_train, self.Y_train)
		self.w = logreg.coef_

	#def predict():