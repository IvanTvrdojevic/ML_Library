import numpy as np
import os
from tqdm import tqdm

import DataLoader_Prototype.MNISTDataLoader as dl

# -------------------------------------------------------------------------------------------------------
# This is a prototype for a machine learning library, mostly to confirm the math behind neural networks

def main():
    # Get data
    input_path = '../DataMNIST'
    training_images_filepath = os.path.join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = os.path.join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = os.path.join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = os.path.join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    (trainX, trainY), (testX, testY) = dl.load_data(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

    # Set parameters
    imgSize = 28*28
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    testX = np.array(testX)
    trainX = np.array(trainX)
    inLayerNeurons = imgSize
    hiddenLayerNeurons = 400
    outLayerNeurons = len(labels)

    # Initialize layers
    x, z, a, o, y, w, b = initializeLayers(inLayerNeurons, outLayerNeurons, hiddenLayerNeurons)

    # The training loop
    numOfImagesTrain = len(trainX)
    for i in tqdm(range(numOfImagesTrain)):
        actual = oneHotEncoding(trainY[i], labels)
        # Calculate hidden and output layer
        z, a, o, y = forwardStep(trainX, actual, i, z, a, o, y, w, b, False)
        # Adjust weights and biases 
        w, b = backwardStep(actual, x, a, y, w, b)

    # The test loop
    numOfImagesTest = len(testX)
    correctGuess = 0
    for i in range (numOfImagesTest):
        actual = oneHotEncoding(testY[i], labels)
        actualNumber, predictedNumber = forwardStep(testX, actual, i, z, a, o, y, w, b, True)
        if(actualNumber == predictedNumber):
            correctGuess += 1

    print('Model accuracy: ', (correctGuess/numOfImagesTest)*100)

def initializeLayers(inLayerNeurons, outLayerNeurons, hiddenLayerNeurons):
    # Init input layer
    x = np.empty(inLayerNeurons, dtype=float)
    # Init pre activation hidden layer
    z = np.empty(hiddenLayerNeurons, dtype=float)
    # Init activation hidden layer
    a = np.empty(hiddenLayerNeurons, dtype=float)
    # Init pre activation output layer 
    o = np.empty(outLayerNeurons, dtype=float)
    # Init output of the model
    y = np.empty(outLayerNeurons, dtype=float)
    
    # Init weight matrix
    limit = np.sqrt(2 / float(inLayerNeurons + hiddenLayerNeurons))
    w = []
    w.append(np.random.normal(0.0, limit, size=(hiddenLayerNeurons, inLayerNeurons)))
    w.append(np.random.normal(0.0, limit, size=(outLayerNeurons, hiddenLayerNeurons)))
    
    # Init bias
    b = []
    b.append(np.zeros(hiddenLayerNeurons))
    b.append(np.zeros(outLayerNeurons))
    b[0] = b[0].reshape(-1, 1)
    b[1] = b[1].reshape(-1, 1)

    return x, z , a, o, y, w, b

def forwardStep(trainX, actual, i, z, a, o, y, w, b, printResult):
    x = trainX[i].flatten()
    # Data normalization 
    x = x/255
    x = x.reshape(-1, 1)
    # Hidden layer
    z = (w[0]@x) + b[0]
    a = activationHiddenLayer(z)
    a = a.reshape(-1, 1)
    # Output layer  
    o = (w[1]@a) + b[1]
    # Using softmax we get an array of possibilites as output from the model
    y = activationOutputLayer(o)

    if printResult:
        actualNumber = getLabelFromArray(actual)
        predictedNumber = getLabelFromArray(y)
        print('----------------------------------------------')
        print('Actual number: ', actualNumber)
        print('Predicted number: ', predictedNumber)
        print('++++++++++++++++++++++++++++++++++++++++++++++')

        return actualNumber, predictedNumber

    return z, a, o, y

def backwardStep(actual, x, a, y, w, b):
    actual = actual.reshape(-1, 1)
    # Derivative of loss w.r.t. model output
    dLdY = y - actual
    # Jacobian matrix of derivatives
    I = np.eye(y.shape[0])
    dYdZ1 =  y * (I - y.T)
    # Error on the output layer
    outLError = dYdZ1.T @ dLdY 
    # Error on the hidden layer
    activationDeriv = np.where(a > 0, 1, 0)
    hiddenLError = (np.dot(w[1].T, outLError)) * activationDeriv
    learningRate = 0.1
    # Update bias 
    # dLdB is the error on the layer
    b[1] = b[1] - learningRate*outLError
    b[0] = b[0] - learningRate*hiddenLError
    # Update weights
    dLdW1 = outLError * a.T
    dLdW0 = hiddenLError * x.T
    w[1] = w[1] - learningRate*dLdW1
    w[0] = w[0] - learningRate*dLdW0

    return w, b

def activationHiddenLayer(x):
    return np.maximum(0, x)

def activationOutputLayer(x):
    x = np.array(x)  
    e_x = np.exp(x)
    return e_x / e_x.sum()

def oneHotEncoding(y, labels):
    oneHotArray = np.empty(0)
    for label in labels:
        label = int(label)
        if y != label:
            oneHotArray = np.append(oneHotArray, 0)
        else:
            oneHotArray = np.append(oneHotArray, 1)
    return oneHotArray

def getMeanSquaredLoss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    loss = np.mean((y_true - y_pred) ** 2)
    return loss

def getLabelFromArray(array):
    x = 0
    index = 0
    for i in range(len(array)):
        if array[i] > x:
            x = array[i]
            index = i
    return index

if __name__ == "__main__":
    main()




 