#### Description
Motivated by the [Stanford Deep Learning Tutorial](http://ufldl.stanford.edu/tutorial/), we train a neural networks for the two tasks

* Identification of individuals from the NimStim dataset
* Identification of emotions from the POFA dataset

The primary code is in multilayer_supervised, but functionality borrowed from the tutorial, specifically, minFunc is utilized from common.  

#### Code Base
Images are preprocessed using prepare_data.m.  It is necessary that this code is run first in order to train the neural network, however, this only needs to be run for one session as long as the data remains in the current workspace.  

In run_train, select either the NimStim or POFA dataset.  Then, to train and test the neural network, from MATLAB,  

```
prepare_data
run_train
```

There are two optimizations available, minFunc and Stochastic Gradient Descent.  
