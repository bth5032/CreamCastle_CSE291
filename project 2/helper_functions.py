import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


def display_image(dataset, labels, index):
	'''Display a particular digit to screen'''
	print "Image label: ", labels[index]
	imgplot = plt.imshow(dataset[index])
	plt.show()


def preprocess_data(dataset, labels):
	''' Preprocessing code
	0.  Normalize the pixel intensities to zero mean, unit variance
	1.  Extract 0 and 1 digits only from Test/Training
	2.  Append '1' feature to dataset for intercept term'''
	X_list = []
	Y_list = []

	for i in range(0, len(dataset)):
		if labels[i] == 0 or labels[i] == 1:
			mean = dataset[i].mean()
			std  = dataset[i].std()

			x = (dataset[i].flatten() - mean)/std

			X_list.append(np.append(1.0, x))
			
			if labels[i] == 0:
				Y_list.append(0)
			elif labels[i] == 1:
				Y_list.append(1)
			
	X = np.asarray(X_list)
	Y = np.asarray(Y_list)

	return X, Y
