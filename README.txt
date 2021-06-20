# CMPE597 Project 3 - VAE with an LSTM encoder and transpose CNN decoder on MNIST data
# Description:
This is a Python 3 implementation using Numpy for necessary operations, Matplotlib for visualization, Pytorch for loading data, creating and training the neural network architecture and saving/loading learned model parameters.

# Files:
model.py file builds the network architecture and implements a function to generate samples from latent space.

main.py file loads the dataset, trains the model and evaluates it on validation set during training, saves the model parameters to model.pk file and outputs losses per epoch and graphs of losses over epochs.

eval.py file loads the learned model parameters and generates samples from random vectors.

model.pk file contains learned model parameters.

# Instructions:
You can run the model by typing these commands in IPython:

In [1]: %cd "CURRENT DIRECTORY"

In [2]: %run main.py

In [3]: %run eval.py

While running main.py, you will see 16 input and output images to observe the performance of the network after each epoch of training.