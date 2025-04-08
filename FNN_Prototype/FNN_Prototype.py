import sys
import os
import numpy as np
from dataclasses import dataclass
import cv2
import DataLoader_Prototype.MNISTDataLoader as dl

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from DataLoader_Prototype.DataLoader import *

# layer -> struct: z (len = size)
#                  a (len = size, after activation)
#                  string for actiavtion function
#                  string for cost function
#                  string for type (in, hidden, out)
#
# weights -> matrix
#
# biases -> array
#
# model -> struct: array of layers
#                  array of weights
#                  array of biases
#                  loss


# in c include sizes of layer.array, weights, bias since everything will be passes as pointers
# in c include number of layers in model, from that number of weights and biases can be calculated

@dataclass
class Layer:
    z: np.ndarray[int, float]
    a: np.ndarray[int, float]
    activation: str
    cost: str
    type: str

@dataclass
class Model:
    layers: np.ndarray[int, Layer]
    weights: np.ndarray[int, np.ndarray[float]]
    biases: np.ndarray[int, np.ndarray[float]]
    cost: float

def addInputLayer(layers, size):
    # add input layer to model.layers
    # flatten layers from convolution
    layer = Layer(np.zeros(size), np.zeros(size), "", "", "in")
    layer.z = layer.z.reshape(-1, 1)
    layer.a = layer.a.reshape(-1, 1)
    layers.append(layer)
    
def addHiddenLayer(layers, size, activation):
    # add hidden layer to model.layers
    layer = Layer(np.zeros(size), np.zeros(size), activation, "", "hidden")
    layer.z = layer.z.reshape(-1, 1)
    layer.a = layer.a.reshape(-1, 1)
    layers.append(layer)

def addOutputLayer(layers, size, activation, costFunction):
    # add output layer to model.layers
    layer = Layer(np.zeros(size), np.zeros(size), activation, costFunction, "out")
    layer.z = layer.z.reshape(-1, 1)
    layer.a = layer.a.reshape(-1, 1)
    layers.append(layer)

def initModel(layers, weightInitType):
    # add layers to model
    # initialize and add wights
    # initialize and addbias
    # init and add loss = 0, doesnt matter since the first step is forward step and it will write over 
    # the initial value
    model = Model(layers, initWeights(layers, weightInitType), initBiases(layers), 0)
    return model

def initWeights(layers, weightInitType):
    weights = []
    numOfLayers = len(layers)
    for i in range(numOfLayers - 1):
        if(weightInitType == "Xavier"):
            weights.append(xavierInit(len(layers[i+1].a.flatten()), len(layers[i].a.flatten())))

    return weights

def xavierInit(fanIn, fanOut):
    scale = np.sqrt(6.0 / (fanIn + fanOut))
    return np.random.uniform(-scale, scale, size=(fanIn, fanOut))

def initBiases(layers):
    biases = []
    numOfLayers = len(layers)
    for i in range(numOfLayers - 1):
        biases.append(np.zeros(len(layers[i+1].a)))
        biases[i] = biases[i].reshape(-1, 1)
    return biases

def train(model, data, labels):
    #dataLen = len(data)
    #print("num of data: ", dataLen)
    for labeledImage in data:
        forwardStep(model, labeledImage, labels, False)
        backwardStep(model, labeledImage[0], labels)

def test(model, data, labels):
    numOfCorrect = 0
    for labeledImage in data:
        numOfCorrect += forwardStep(model, labeledImage, labels, True)
    print("Model accuracy: ", (numOfCorrect / len(data)) * 100)

def forwardStep(model, labeledImage, labels, test):
    model.layers[0].a = labeledImage[1].flatten() / 255
    model.layers[0].a = model.layers[0].a.reshape(-1, 1)

    #img = labeledImage[1]
    #img_resized = cv2.resize(img, (24, 24), interpolation=cv2.INTER_AREA)
    #model.layers[0].a = img_resized.flatten() / 255

    #print("Normalised: ")
    #print(model.layers[0].a)
    numOfLayers = len(model.layers)
    for i in range(1, numOfLayers):
        model.layers[i].z = (model.weights[i-1]@model.layers[i-1].a) + model.biases[i - 1]
        model.layers[i].z = model.layers[i].z.reshape(-1, 1)
        #print("-------------------------------------------")
        #print("pre actiavation: ", i, "            \n", model.layers[i].z)
        if(model.layers[i].activation == "ReLu"):
            #print("relu")
            model.layers[i].a = ReLu(model.layers[i].z)
            #print(i)
            #print("post activation", model.layers[i].a)
        elif(model.layers[i].activation == "softmax"):   
            #print("softmax")
            #print("pre activatin softmax", model.layers[i].z)
            model.layers[i].a = softmax(model.layers[i].z)
            #print("softmax")
            #print("post softmax", model.layers[i].a)

    oneHotArray = oneHotEncoding(labeledImage[0], labels)
    if(model.layers[numOfLayers - 1].cost == "MSE"):
        #print("MSE")
        #print("onehot shape ", oneHotArray.shape)
        #print("result shape ", model.layers[numOfLayers - 1].a.shape)
        model.cost = MSE(oneHotArray, model.layers[numOfLayers - 1].a)
    elif(model.layers[numOfLayers - 1].cost == "categoricalCrossentropy"):
        model.cost = categoricalCrossentropy(oneHotArray, model.layers[numOfLayers - 1].a)

    # print("---------------------------------")
    # print("output")
    # print(model.layers[numOfLayers - 1].a)
    # print("actual")
    # print(oneHotArray)
    
    if test:
        actual = getLabelFromArray(oneHotArray, labels)
        predicted = getLabelFromArray(model.layers[numOfLayers - 1].a, labels)
        # print("---------------------------")
        # print("predicted label: ", predicted)
        # print("actual label: ", actual)
        # print("===========================")
        if(actual == predicted):
            return 1
        return 0
    else:
        print("cost")
        print(model.cost)

def getLabelFromArray(array, labels):
    labelIndex = 0
    for i in range(len(array)):
        if array[i] > array[labelIndex]:
            labelIndex = i
    return labels[labelIndex]

def infere(model, image):
    forwardStep(model, image)
    return softmax(model.layers[-1].neurons)

def softmax(input):
    eX = np.exp(input - np.max(input))
    return eX / np.sum(eX, axis=0)

def MSE(yTrue, yPred):
    yTrue = np.array(yTrue)
    yPred = np.array(yPred)
    cost = np.mean((yTrue - yPred) ** 2)
    return cost

def categoricalCrossentropy(yTrue, yPred):
    cost = -np.sum(yTrue * np.log(yPred)) / yTrue.shape[0]
    return cost

def oneHotEncoding(currentClass, labels):
    oneHotArray = np.empty(0)
    for label in labels:
        if currentClass != label:
            oneHotArray = np.append(oneHotArray, 0)
        else:
            oneHotArray = np.append(oneHotArray, 1)
    return oneHotArray.reshape(-1, 1)

def ReLu(input):
    return np.maximum(0, input)

def backwardStep(model, label, labels):
    numOfLayers = len(model.layers)
    oneHotArray = oneHotEncoding(label, labels)
    yPred = model.layers[numOfLayers - 1].a
    errorsOnLayers = []
    if(model.layers[numOfLayers - 1].cost == "categoricalCrossentropy"):
        dLdY = categoricalCrossentropyDerivative(oneHotArray, yPred)

    if(model.layers[numOfLayers - 1].activation == "softmax"):
        # dervivative of activaton w.r.t. pre activation (all z and a are interconnected because of the sum used to calculate softmax)
        jacobian = softmaxDerivative(yPred) 
        errorOnLayer = jacobian.T @ dLdY
        errorsOnLayers.insert(0, errorOnLayer) 
    
    for i in range(numOfLayers - 3, -1, -1):
        if(model.layers[i + 1].activation == "ReLu"):
            gradient = ReLuDerivative(model.layers[i + 1].a) 
            errorOnLayer = (model.weights[i + 1].T@errorsOnLayers[0]) * gradient 
            errorsOnLayers.insert(0, errorOnLayer)

    for i in range(numOfLayers - 2, 0, -1):
        dLdW = errorsOnLayers[i] * model.layers[i].a.T
        model.weights[i] = model.weights[i] - 0.01 * dLdW
        model.biases[i] = 0.01 * errorOnLayer[i]

def categoricalCrossentropyDerivative(yTrue, yPred):
    return yPred - yTrue

def MSEDerivative(yTrue, yPred):
    return  2 * (yPred - yTrue) / yTrue.shape[0]

def softmaxDerivative(yPred):
    I = np.eye(yPred.shape[0])
    return yPred * (I - yPred.T)

def ReLuDerivative(yPred):
    return np.where(yPred > 0, 1, 0)


