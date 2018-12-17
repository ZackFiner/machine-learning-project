'''
Created on May 29, 2018

@author: Zackary Finer
'''
import os
import numpy as np
from PIL import Image
from mnist import MNIST
from Layer import layer
from numpy.lib.function_base import gradient
class Network(object):
    '''
    There is something VERY WRONG with the code below, and the code in layer. All values produced by the get weighted product are large
    , and as such when they're run through sigmoid, they produce 1 EVERY TIME. I have clearly mis-understood something (or multiple things)
    along the way up to now, so i'd start by making sure our assumptions about sigmoid were correct, and then work out for there.
    '''
    '''
    The objective here is to apply backpropegation to train this network to perform some task
    In order to do this, we must determine the form of some cost function, which can be used to determine
    error from training sessions, and optimize the weights and biases for each neruon in order to reduce these errors
    '''
    m_layers = list()

    m_inputLayer = None
    m_outputLayer = None
    m_trainingExamples = None
    m_trainingSolutions = None
    m_numberOfLayers = 3
    m_currentError = [None]*m_numberOfLayers

    TRAININGTIME = 50000
    def __init__(self):
        '''
        Constructor
        '''
        HIDDEN_NEURON_NUM = 450
        INPUT_IMAGE_SIZE = 784

        #self.m_layers.put(0, self.m_outputLayer)
        self.m_inputLayer = layer(INPUT_IMAGE_SIZE, None)
        self.m_layers.insert(0, self.m_inputLayer )
        self.m_layers.insert(1, layer(HIDDEN_NEURON_NUM, self.m_layers[0]))
        #self.m_layers.insert(2, layer(HIDDEN_NEURON_NUM, self.m_layers[1]))
        self.m_outputLayer = layer(10, self.m_layers[1]) # output layer for image
        self.m_layers.insert(2, self.m_outputLayer)


    def getCost(self, expectedValue):
        '''
        Returns quadratic cost for a given input
        '''
        dif = expectedValue - self.m_outputLayer.m_activation
        return 0.5*(np.dot(dif, dif))

    def getErrorOutput(self, expectedValue):
        '''
        returns the error for a given input
        '''
        diff = self.m_outputLayer.m_activation - expectedValue
        v = self.m_outputLayer.getPrimeWeighted()
        return np.multiply(diff, self.m_outputLayer.getPrimeWeighted())

    def calcError(self, expectedValue):
        localLastError = self.getErrorOutput(expectedValue)
        numOfLayers = self.m_numberOfLayers
        self.m_currentError.insert(numOfLayers-1, localLastError) # insert the first error ~(expected - what we got for output)
        # iteratively determine the error of a specific layer, moving backward from the given error
        for index in range(numOfLayers-2, 0, -1): #loop backwards from the second to last layer to the second layer (we don't update the input layer!)
            leftSide = np.dot(self.m_layers[index+1].m_weights.T, localLastError) # index + 1 will be the layer in front of this one
            rightSide = self.m_layers[index].getPrimeWeighted() # we get the activations for the current layer, and return the prime weighted value
            localLastError = np.multiply(leftSide, rightSide) # hadamard product between the prime weighted and
            self.m_currentError[index] = localLastError

        return localLastError
    def runInput(self, array):
        self.m_inputLayer.setActivation(array)
        for x in range(1, self.m_numberOfLayers):
            self.m_layers[x].feedforward()

    def trainNetwork(self):
        '''
        Perform multiple training epoch's
        each epoch consists of running every training example through the network
        and then calculating cost, then correcting weights and biases with the given costs

        i'm still learning the terminology hear, but i believe this implementation uses
        stoischratic gradient descent (because it updates the network after each test)
        instead of batch gradient descent (where we would compile the data from each test, and then process it)
        '''
        for epochIndex in range(0, 1): # for each example in the number of training epochs
            i = 0
            p=0
            print('epoch'+str(epochIndex))
            for trainingImage in self.m_trainingExamples: # for each training example
                if 0==(i % 1000):
                    if (p!=0):
                        print("solution for 7")
                        print(self.getSolution(self.m_trainingExamples[0]))
                    print(str(p*10)+'%')
                    p+=1
                imgdata = np.array(trainingImage)
                self.runInput(imgdata)
                #gradientConstant = self.TRAININGTIME / len(self.m_trainingExamples)
                gradientConstant = 1.0
                self.calcError(toNumpyArray(self.m_trainingSolutions[i], 10)) # calculate the error for each layer
                i += 1
                for x in range(1,self.m_numberOfLayers):
                    layerError = self.m_currentError[x]
                    self.m_layers[x].m_bias = self.m_layers[x].m_bias - gradientConstant*layerError
                    rightSide = np.dot(layerError[np.newaxis], self.m_layers[x-1].m_activation[np.newaxis].T) # unfortunatley, we need to do some weird axis splitting here to get the behavior we want (column vector * row vector)
                    leftSide = gradientConstant*self.m_layers[x].m_weights
                    finalsub = np.multiply(leftSide, rightSide)
                    self.m_layers[x].m_weights = self.m_layers[x].m_weights - finalsub

            print("solution for 7")
            print(self.getMax(self.getSolution(self.m_trainingExamples[0]))[7]==1)
    def getMax(self, array):

        vMax = -1
        for x in array:
            if (x>vMax):
                vMax = x

        def filter(x):
            if (x/vMax != 1):
                return 0
            return 1
        return np.asarray(np.vectorize(filter)(array))
    def getSolution(self, image):
        self.m_inputLayer.setActivation(image)
        for x in range(1, self.m_numberOfLayers):
            self.m_layers[x].feedforward()
        return self.m_outputLayer.m_activation
    def testReults(self):
        successes = 0

        for x in range(0, self.m_trainingExamples.shape[0]):
            img = self.m_trainingExamples[x]
            self.runInput(img)
            correct = self.m_trainingSolutions[x]
            if (self.getMax(self.getSolution(self.m_trainingExamples[x]))[correct]==1):
                successes+=1

        print("Neural Network can correctly identify: ")
        print(successes/100)
        print("percent of training examples")


def toNumpyArray(value, max):
    returnVal = np.zeros(max)
    returnVal.flat[value] = 1
    return returnVal

def printImg(array):
    for x in range(0, 28):
        string = ""
        for y in range(0,28):
            if (array[x*28+y] > 50):
                string +="_"
            else:
                string+= " "
        print(string)
def makeColumnVec(arraylike):
    return np.asmatrix(arraylike).reshape(len(arraylike),1)
mndata = MNIST('samples')
images, labels = mndata.load_testing()
d = Network()
d.m_trainingExamples = np.asarray(np.vectorize(lambda x: x/255)(images))
printImg(images[1])
print(labels[1])
d.m_trainingSolutions = labels
d.trainNetwork()
d.testReults()
