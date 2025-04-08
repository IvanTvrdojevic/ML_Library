import sys
import os
import numpy as np
import FNN_Prototype as fnn

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import DataLoader_Prototype.DataLoader as dl
import DataLoader_Prototype.MNISTDataLoader as mnistdl

def main():
    # Testing using MNIST dataset by getting it with the mnist loader and converting it to the format of data
    # that FNN_Prototype works with
    input_path = '../DataMNIST'
    training_images_filepath = os.path.join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = os.path.join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = os.path.join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = os.path.join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    (trainX, trainY), (testX, testY) = mnistdl.load_data(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # Converting to needed data format
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    trainY = trainY.astype(str)
    trainData = list(zip(trainY, trainX))
    # Now able to use DataLoader_Prototype functions along with FNN_Prototype
    dl.plotData(trainData, 10)
    
    testX = np.array(testX)
    testY = np.array(testY)
    testY = testY.astype(str)
    testData = list(zip(testY, testX))

    layers = []
    imgSize = 28*28
    fnn.addInputLayer(layers, imgSize)
    fnn.addHiddenLayer(layers, 400, "ReLu")
    fnn.addOutputLayer(layers, 10, "softmax", "categoricalCrossentropy")
    
    print("Layer shapes:")
    for layer in layers:
        print(layer.a.shape)
    print("--------------------------")

    print("Weight shapes:")
    model = fnn.initModel(layers, "Xavier")
    for weights in model.weights:
        print(weights.shape)
    print("--------------------------")

    print("Bias shapes:")
    for bias in model.biases:
        print(bias.shape)
    print("--------------------------")

    fnn.train(model, trainData, labels)
    fnn.test(model, testData, labels)
    #fnn.infere(model, image, label)


    # data = getSplitData("../../Data/Animals/")

    # labels = ["cats", "dogs", "snakes"]
    # trainData = data[0]
    # testData = data[1]
    # validationData = data[2]
    # plotData(trainData, 10)
if __name__ == "__main__":
    main()