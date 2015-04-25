import numpy as np
import os
import helper_functions as fn

class GradientDescent:
    def __init__(self, X_train, Y_train, X_test, Y_test):
		self.model     = None
		self.w         = None
		self.incorrect = []
		self.X_train   = X_train
		self.X_test    = X_test
		self.Y_train   = Y_train
		self.Y_test    = Y_test

    def batch_gd(self):
        pass
        
    def stoch_gd(self):
        pass

    def plot_error(self):
        pass

    def comparemethods(self):
        pass


