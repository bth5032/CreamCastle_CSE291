import numpy as np
import os
import helper_functions as fn

#For logistic regression 
class GradientDescent:      

    def __init__(self, X_train, Y_train, X_test, Y_test):
		self.model     = None
		self.w         = None
		self.incorrect = []
		self.X_train   = X_train
		self.X_test    = X_test
		self.Y_train   = Y_train
		self.Y_test    = Y_test
		
        #Keep the state of optimization   
                self.theta=np.random.rand(X_train.shape[1])
                self.history=[] 

    def batch_gd(self):
        stepsz=1e-4
        w -= stepsz*self.sigmoid_gradient()
        
        
    def stoch_gd(self):
        w = np.random.rand(X_train.shape[1])
        pass

    def plot_convergence(self):
        pass


if __name__=='__main__':
	# Load dataset from MNIST
	full_trainarray = np.load(os.path.join('data','numpy','trainarray.npy'))
	full_trainlabel = np.load(os.path.join('data','numpy','trainlabel.npy'))
	full_testarray  = np.load(os.path.join('data','numpy','testarray.npy' ))
	full_testlabel  = np.load(os.path.join('data','numpy','testlabel.npy' ))

	X_train, Y_train = fn.preprocess_data(full_trainarray, full_trainlabel)
	X_test, Y_test = fn.preprocess_data(full_testarray, full_testlabel)
