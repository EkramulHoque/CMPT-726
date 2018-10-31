Required Files:
===============

1. imagenet_finetune.py
2. html_output.py

Required Packages:
==================

1.torch
2.torchvision
3.torchvision.models
4.torch.nn 
5.torchvision.datasets 
6.torchvision.transforms as transforms
7.torch.optim as optim
8.html_output 
9.numpy 
10.torch.utils.data import Sampler, SubsetRandomSampler

Description:
============

imagenet_finetune.py

For every epoch, the code validates the CIFAR10 test dataset and stores the testing error along
with the classification scores for each test images. Generates an HTML output after the end of training 
and validating a given dataset.

	Function:- train()

	a) Downloads the CIFR10 data.
	b) Data is resized o 224 X 224 pixels image, converted into tensors, and normalize each 
	channel of the image (RGB - Red,Green,Blue) with mean 0.5 and standard deviation of 0.5. 
	All these tranformations are applied both on the traintset and testset.
	c) A subset of the dataset is taken using the function "sampler()" that takes a dataset
	and shuffle=boolean as inputs. Then the data is loaded with SubsetRandomSampler() from torch.utils.data
	d) Both the training and testing data are taken as a subset using the 'sampler' function.
	e) Check if CUDA is available and sets the images,labels and model to cuda
	f) pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      	True if using GPU
	f) On each epoch/iteration, the model is trained and tested;
	g) Results of the precision/testing error are stored per iterations along
	with the classsification scores for each test images.
	
	Function:- test()

	a) takes a trained model and test data set as an input
	b) The outputs are energies for the 10 classes.Higher the energy for a class, the more the network 
	thinks that the image is of the particular class.
	c) Takes the index of the highest energy
	d) Calculates the number of correct predictions
	e) Returns testing error, images used as test and predictions of object type tensors, classification scores
	
	Function:- sampler()
	
	a) takes dataset and shuffle=boolean as inputs
	b) takes the lenght of the dataset, divides it with a given parameter i.e the variable 'divide_data'
	c) splits the indices of the dataset with a ratio given valid_size; variable 'valid_size'
	d) shuffles the data if true
	e) returns the indices for test and train set 


html_output.py
	
	a) Takes the classification scores, testing error generated from imagenet_finetune.py along with number of samples used.
	b) Creates table with classification scores with each test image along with the prediction. A table for testing error is created as well
	c) Generates an HTML as an output in the working directory of the program.	


Run:
====
1. Open cmd/terminal and set the current working directory to the directory where the python file is present.
2. Type python imagenet.py


