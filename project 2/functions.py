import numpy as np
import matplotlib.pyplot as plt


def displayImage(dataset, label, index):
	'''Display a particular digit to screen'''
	print "Image label: ", label[index]
	imgplot = plt.imshow(dataset[index])
	plt.show()

def preprocess_data():
	pass
	''' Preprocessing code
	0.  Extract 0 and 1 digits only from Test/Training
	1.  Append '1' feature to dataset for intercept term'''

def gradient_descent():
	pass

def stochastic_gradient_descent():
	pass