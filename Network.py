'''
Created on May 29, 2018

@author: Zackary Finer
'''
import numpy as np
from PIL import Image
from NN.Layer import layer
from numpy.lib.function_base import gradient
class Network(object):
    '''
    The objective here is to apply backpropegation to train this network to perform some task
    In order to do this, we must determine the form of some cost function, which can be used to determine
    error from training sessions, and optimize the weights and biases for each neruon in order to reduce these errors
    '''
    m_layers = np.array
    
    m_inputLayer = None
    m_outputLayer = None
    m_trainingExamples = np.array
    m_trainingSolutions = np.array
    m_currentError = np.array
    m_numberOfLayers = 4
    TRAININGTIME = 2.5
    def __init__(self):
        '''
        Constructor
        '''
        HIDDEN_NEURON_NUM = 450
        INPUT_IMAGE_SIZE = 729
        
        #self.m_layers.put(0, self.m_outputLayer)
        self.m_inputLayer = layer(INPUT_IMAGE_SIZE)
        self.m_layers.flat[0] = self.m_inputLayer
        self.m_layers.flat[1] = layer(HIDDEN_NEURON_NUM, self.m_layers.flat[0])
        self.m_layers.flat[2] = layer(HIDDEN_NEURON_NUM, self.m_layers.flat[1])
        self.m_outputLayer = layer(10, self.m_layers.flat[2]) # output layer for image
        self.m_layers.flat[3] = self.m_outputLayer

    
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
        return np.multiply(self.m_outputVec - expectedValue, self.m_outputLayer.getPrimeWeighted())
    def calcError(self, expectedValue):
        localLastError = self.getErrorOutput(expectedValue)
        numOfLayers = self.m_numberOfLayers
        self.m_currentError.put(numOfLayers-1, localLastError)
        # iteratively determine the error of a specific layer, moving backward from the given error
        for index in range(1, numOfLayers):
            leftSide = np.matmul(self.m_layers.flat[numOfLayers-index].m_weights.transpose(),localLastError)
            rightSide = self.m_layers.flat[numOfLayers-1-index].getPrimeWeighted()
            localLastError = np.multiply(leftSide, rightSide)
            self.m_currentError.put(numOfLayers-1-index, localLastError)
        return localLastError
    
    def trainNetwork(self):
        '''
        Perform multiple training epoch's
        each epoch consists of running every training example through the network
        and then calculating cost, then correcting weights and biases with the given costs
        
        i'm still learning the terminology hear, but i believe this implementation uses
        stoischratic gradient descent (because it updates the network after each test)
        instead of batch gradient descent (where we would compile the data from each test, and then process it)
        '''
        for epochIndex in range(0, 20): # for each example in the number of training epochs
            for trainingImage in self.m_trainingExamples: # for each training example
                self.m_inputLayer.setActivation(trainingImage)
                for x in range(1, self.m_numberOfLayers):
                    self.m_layers.flat[x].feedforward() 
                gradientConstant = self.TRAININGTIME / self.m_trainingExamples.size
                self.calcError(self.m_trainingSolutions) # calculate the error for each layer
                for x in range(1, self.m_numberOfLayers):
                    layerError = self.m_currentError.flat[x]
                    self.m_layers.flat[x].m_bias =np.subtract(self.m_layers.flat[x].m_bias,  np.multiply(gradientConstant,layerError))
                    rightSide = np.matmul(layerError, self.m_layers.flat[x-1].m_activation.transpose()) # this should be a matrix, not a scalar/vector!
                    leftSide = np.multiply(gradientConstant,self.m_layers.flat[x].m_weights)
                    finalsub = np.multiply(leftSide, rightSide)
                    self.m_layers.flat[x].m_weights = np.subtract(self.m_layers.flat[x].m_weights, finalsub)
        
    
