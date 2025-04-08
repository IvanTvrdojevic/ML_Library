#ML_Library

ML Library from scratch, using Python for prototyping and C for the full implementation

DataLoader_Prototype: python code for a data loader that returns data depending on the directory that is passed can load data in a data array and then split it in train, test, validation set or it can be passed a directory that contains train test validation subdirectories from which it will extract train test and validation data

DataLoader: C implementation of the Python prototype

Both DataLoaders are unit tested in DataLoadersTests/TestDataLoader.py using Python Unittest and Ctypes libraries

FNN_Prototype: SimpleFNN.py: simple model used to test understanding behind the math of machine learning (linear algebra, vector calculus) only one hidden layer, predefined activations and initializations made to work on the MNIST dataset

FNN_Prototype.py: ML_Library prototype that will be used for the C implementation activations, initializations and number of hidden layers all customizable

FNN_PrototypeTest.py: using MNIST dataset trained a model to compare it with the results of SimpleFNN

Currently working on Convolutional layers