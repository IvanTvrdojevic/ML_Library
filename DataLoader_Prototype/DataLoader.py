import os
import random
from PIL import Image
import numpy
import matplotlib.pyplot as plt
   
def getData(dataPath):
    data = []
    for className in os.scandir(dataPath):
        pathToImages = dataPath + className.name
        data += getDataFromDir(pathToImages, className.name)
    random.shuffle(data)
    
    return data
    
def getDataFromSplit(dataPath):
    data = []
    splits = ["train", "test", "validation"]
    for split in splits:
        dirData = []
        for className in os.scandir(dataPath + split):
            pathToImages = dataPath  + split + "/" + className.name
            dirData += (getDataFromDir(pathToImages, className.name))
            random.shuffle(dirData)
        data.append(dirData)
    
    return data

def getSplitData(dataPath):
    data = getData(dataPath)
    numOfImages = getNumOfImages(dataPath)
    numOfTrainImages = int(0.7 * numOfImages)
    numOfTestImages = int(0.2 * numOfImages)
    numOfValidationImages = numOfImages - (numOfTrainImages + numOfTestImages)
    usedIndexes = []
    trainData, usedIndexes = getPartOfData(data, numOfTrainImages, numOfImages, usedIndexes)
    testData, usedIndexes = getPartOfData(data, numOfTestImages, numOfImages, usedIndexes)
    validationData, usedIndexes = getPartOfData(data, numOfValidationImages, numOfImages, usedIndexes)

    return [trainData, testData, validationData]
    
def getDataFromDir(pathToImages, className, grayScale=True):
    data = []
    for imageName in os.scandir(pathToImages):
        fullPath = pathToImages + "/" + imageName.name
        classImage = (numpy.array(Image.open(fullPath)))
        if(grayScale):
            classImage = numpy.dot(classImage[..., :3], [0.2989, 0.5870, 0.1140]).astype(numpy.uint8)
        labeledImage = (className, classImage)
        data.append(labeledImage)
    return data
    
def getNumOfClasses(dataPath):
    cnt = 0
    for dataClass in os.scandir(dataPath):
        if dataClass.is_dir():
            cnt += 1

    return cnt

def getNumOfImages(dataPath):
    cnt = 0
    for dataClass in os.scandir(dataPath):
        if dataClass.is_dir():
            for _ in os.scandir(dataClass):
                cnt += 1

    return cnt

def getPartOfData(data, numOfImagesPart, numOfImages, usedIndexes):
    dataPart = []
    i = 0
    while(i < numOfImagesPart):
        randomIndex = random.randint(0, numOfImages - 1)
        if randomIndex not in usedIndexes:
            usedIndexes.append(randomIndex)
            dataPart.append(data[i])
            i += 1
    random.shuffle(dataPart)
    
    return dataPart, usedIndexes

def plotData(data, amountToPlot):
    plt.figure(figsize=(15, 3))
    for i in range(amountToPlot):
        label, image = data[i]
        plt.subplot(1, amountToPlot, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


