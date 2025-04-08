import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from DataLoader_Prototype.DataLoader import *

# Testing python prototype
class LoadDataPythonTestCase(unittest.TestCase):
    def test_0_getDataFromDir(self):
        data = getDataFromDir("UnitTestData/DataByClass/Class1/", "Class1")
        self.assertEqual(len(data), 2)
        for labeledImage in data:
            self.assertEqual(labeledImage[0], "Class1")

    def test_1_getDataFromDir(self):
        data = getDataFromDir("UnitTestData/SplitData/Test/Class2/", "Class2")
        self.assertEqual(len(data), 2)
        for labeledImage in data:
            self.assertEqual(labeledImage[0], "Class2")

    def test_2_getNumOfClasses(self):
        self.assertEqual(getNumOfClasses("UnitTestData/DataByClass"), 3)

    def test_3_getNumOfImages(self):
        self.assertEqual(getNumOfImages("UnitTestData/SplitData/Train"), 9)

    def test_4_getData(self):
        data = getData("UnitTestData/DataByClass/")
        self.assertEqual(len(data), 8)

    def test_5_getDataFromSplit(self):
        data = getDataFromSplit("UnitTestData/SplitData/")
        self.assertEqual(len(data), 3)
        self.assertEqual(len(data[0]), 9)
        self.assertEqual(len(data[1]), 6)
        self.assertEqual(len(data[2]), 4)

    def test_6_getSplitData(self):
        # Hard to test on a small dataset because of percentages
        data = getSplitData("UnitTestData/DataByClass/")
        self.assertEqual(len(data), 3)
        self.assertEqual(len(data[0]), 5)
        self.assertEqual(len(data[1]), 1)
        self.assertEqual(len(data[2]), 2)
        
    def test_7_getSplitData(self):
        # Using real dataset with 3k images
        data = getSplitData("../Data/Animals/")
        self.assertEqual(len(data), 3)
        self.assertEqual(len(data[0]), 2103)
        self.assertEqual(len(data[1]), 601)
        self.assertEqual(len(data[2]), 301)

# Testing C implementation using ctypes
from ctypes import *

class Image(Structure):
    _fields_ = [('width', c_int),
                ('height', c_int),
                ('channels', c_int),
                ('name', c_char * 50),
                ('label', c_char * 50),
                ('values', POINTER(c_ubyte))]

class LoadDataCTestCase(unittest.TestCase):
    def setUp(self):
        # Setup for c implementation testing
        currentDir = os.path.dirname(os.path.abspath(__file__))
        dllPath = os.path.join(currentDir, "bin/DataLoader.dll")
        self.libc = CDLL(dllPath)

        return super().setUp()

    def test_0_getImagesFromDir(self):
        self.libc.getImagesFromDir.argtypes = [c_char_p, c_char_p, POINTER(c_int), POINTER(Image), c_bool]
        self.libc.getImagesFromDir.restype = None

        index = c_int(0)
        images = (Image * 2)()

        self.libc.getImagesFromDir(b"UnitTestData/DataByClass/", b"Class1", byref(index), images, c_bool(False))

        self.assertEqual(images[0].label.decode(), "Class1")
        self.assertEqual(images[1].label.decode(), "Class1")

    def test_1_getNumberOfImagesC(self):
        self.libc.getNumberOfImages.argtypes = [c_char_p]
        self.libc.getNumberOfImages.restype = c_int
        numOfImages = self.libc.getNumberOfImages(b"UnitTestData/DataByClass/")

        self.assertEqual(numOfImages, 8)
        
    def test_2_loadDataFromClasses(self):
        self.libc.loadDataFromClasses.argtypes = [c_char_p, POINTER(Image), c_bool, c_bool]
        self.libc.loadDataFromClasses.restype = None

        images = (Image * 8)()

        self.libc.loadDataFromClasses(b"UnitTestData/DataByClass/", images, c_bool(False), c_bool(False))

        correctLabels = ["Class1", "Class1", "Class2", "Class2", "Class2", "Class3", "Class3", "Class3", ]
        for i in range(8):
            self.assertEqual(images[i].label.decode(), correctLabels[i])
    
    def test_3_loadNumOfImagesForSplits(self):
        self.libc.loadNumOfImagesForSplits.argtypes = [POINTER(c_int),
                                                       POINTER(c_int), 
                                                       POINTER(c_int), 
                                                       POINTER(c_float), 
                                                       c_int]
        self.libc.loadNumOfImagesForSplits.restype = None

        numOfTrainImages = c_int(0)
        numOfTestImages = c_int(0)
        numOfValidationImages = c_int(0)
        percentages = (c_float * 3)(70, 20, 10)
        numOfImages = c_int(3005)

        self.libc.loadNumOfImagesForSplits(byref(numOfTrainImages), 
                                           byref(numOfTestImages), 
                                           byref(numOfValidationImages), 
                                           percentages, 
                                           numOfImages)
        
        self.assertEqual(numOfTrainImages.value, 2103)
        self.assertEqual(numOfTestImages.value, 601)
        self.assertEqual(numOfValidationImages.value, 301)

    def test_4_splitData(self):
        self.libc.loadDataFromClasses.argtypes = [POINTER(Image), c_int, 
                                                  POINTER(Image), c_int,
                                                  POINTER(Image), c_int,
                                                  POINTER(Image), c_int,
                                                  POINTER(c_float)]
        self.libc.loadNumOfImagesForSplits.restype = None

        self.libc.loadDataFromClasses.argtypes = [c_char_p, POINTER(Image), c_bool, c_bool]
        self.libc.loadDataFromClasses.restype = None
        images = (Image * 8)()
        self.libc.loadDataFromClasses(b"UnitTestData/DataByClass/", images, c_bool(False), c_bool(False))
        trainImages = (Image * 5)()
        testImages = (Image * 2)()
        validationImages = (Image * 1)()
        percentages = (c_float * 3)(70, 20, 10)

        self.libc.splitData(images, 8,
                            trainImages, 5,
                            testImages, 2,
                            validationImages, 1,
                            percentages)

        correctLabelsTrain = ["Class1", "Class1", "Class2", "Class2", "Class2"]
        correctLabelsTest = ["Class3", "Class3"]
        correctLabelValidation = "Class3"
        for i in range(5):
            self.assertEqual(trainImages[i].label.decode(), correctLabelsTrain[i])
        for i in range(2):
            self.assertEqual(testImages[i].label.decode(), correctLabelsTest[i])
        self.assertEqual(validationImages[0].label.decode(), correctLabelValidation)

    def test_5_checkPercentages(self):
        self.libc.checkPercentages.argtypes = [POINTER(c_float)]
        self.libc.checkPercentages.restype = None

        percentages = (c_float * 3)(70, 20, 10)        
        
        self.libc.checkPercentages(percentages)

        sum = percentages[0] + percentages[1] + percentages[2]
        self.assertEqual(sum, 100)
        self.assertNotEqual(percentages[0], 0)
        self.assertNotEqual(percentages[1], 0)
        self.assertNotEqual(percentages[2], 0)

    def test_6_swap(self):
        self.libc.swap.argtypes = [POINTER(Image), POINTER(Image)]
        self.libc.swap.restype = None

        ImageA = Image(500)
        ImageB = Image(100)

        self.libc.swap(byref(ImageA), (ImageB))

        self.assertEqual(ImageA.width, 100)
        self.assertEqual(ImageB.width, 500)

        self.libc.swap(byref(ImageA), (ImageB))

        self.assertEqual(ImageA.width, 500)
        self.assertEqual(ImageB.width, 100)

    def test_7_shuffleImages(self):
        self.libc.shuffleImages.argtypes = [POINTER(Image), c_int]
        self.libc.shuffleImages.restype = None

        self.libc.loadDataFromClasses.argtypes = [c_char_p, POINTER(Image), c_bool, c_bool]
        self.libc.loadDataFromClasses.restype = None
        images = (Image * 8)()
        self.libc.loadDataFromClasses(b"UnitTestData/DataByClass/", images, c_bool(False), c_bool(False))

        self.libc.shuffleImages(images, 8)

        labelsBeforeShuffle = ["Class1", "Class1", "Class2", "Class2", "Class2", "Class3", "Class3", "Class3", ]
        labelsAfterShuffle = []
        for i in range (8):
            labelsAfterShuffle = images[i].label.decode()
        self.assertNotEqual(labelsAfterShuffle, labelsBeforeShuffle)


if __name__ == '__main__':
    unittest.main()