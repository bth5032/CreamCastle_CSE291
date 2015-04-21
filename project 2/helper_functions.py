import numpy as np
import matplotlib.pyplot as plt


def display_image(dataset, labels, index):
	'''Display a particular digit to screen'''	
	print "Image label: ", labels[index]
	imgplot = plt.imshow(dataset[index])
	plt.show()


def preprocess_data(dataset, labels):
	''' Preprocessing code
	0.  Extract 0 and 1 digits only from Test/Training
	1.  Append '1' feature to dataset for intercept term'''
	X_list = []
	Y_list = []

	for i in range(0, len(dataset)):
		if labels[i] == 0 or labels[i] == 1: 
			X_list.append([np.append(1.0, dataset[i])])
			Y_list.append(labels[i])

	X = np.asarray(X_list)
	Y = np.asarray(Y_list)

	return X, Y

def gradient_descent():
	pass

def stochastic_gradient_descent():
	pass
