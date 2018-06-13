'''
Created on May 27, 2018

@author: Zackary Finer
'''
import numpy as np
NUM_HIDDEN = 512
NUM_INPUT = 784
def sigmoid(nValue):
    return 1/(1+np.exp(-nValue))

def sigmoidPrime(nValue):
    d = sigmoid(nValue)
    return np.exp(-nValue)*d*d

class layer(object):
    '''
    The layer class represents a layer of neruons in the neural network
    each layer has an associated column vector for it's own activation values,
    a matrix containing the weights for the connections from the last layer,
    and another column vector for the biases of the layer.
    
    nL = sigmoid(m_weights*m_activationL + m_bias)
    
    where nL is the activation column vector for this layer, and m_activationL are the activation values from the last layer
    On the other hand, if this layer is an input layer then activation values will simply be fed in from whatever we are analyzing
    
    (*) nxm*mx1 -> nx1 -> m_bias: nx1 and nL is nx1, so if nL must be smaller than the first, this must be reflected by n
    where nL is the activations for the next layer of the network
    
    '''
    m_weights = np.matrix # matrix with the associated weights from each neuron in
    m_bias = np.array # a column vector containing the biases
    m_activation = np.array # activations from the last network call, this will change every time a new value is input
    m_isInput = False #initially false
    m_size = 0 #initially empty
    m_lastLayer = None
    def __init__(self, size, lastLayer):
        '''
        Initialize this layer so that it has the specified number of neurons, and reference to the last layer.
        All weights will be initially randomized, and the biases will be set to 1
        '''
        
        self.m_size = size #set the number of neurons in this layer
        if lastLayer is None: # if nothing was passed for the 
            self.m_isInput = True # mark this layer as an input
            return #terminate, we don't need to set the weights or bias as the activation of this layer will be input
        
         
        self.m_weights = np.random.rand(self.m_size, lastLayer.m_size) # randomize the weights to the last layer (*)
        self.m_bias = np.ones(self.m_size) # allocate a column vector of biases, initially all of 1 (i chose this arbitrarily)
        self.m_lastLayer = lastLayer
    #end of __init__
    
    def weightedInput(self):
        return np.add(np.matmul(self.m_weights, np.asmatrix(self.m_lastLayer.m_activation)), np.asmatrix(self.m_bias))#calculate weighted input
    #end weightedInput
    
    def feedforward(self):    
        rawM = self.weightedInput() #calculate weighted input
        mapFunc = np.vectorize(sigmoid) # use the vectorize interface to get a mapping function with sigmoid
        self.m_activation = np.asarray(mapFunc(rawM)) # map all values using the sigmoid function, and assign it to the activation vector for this layer
    #end of feedforward
    
    def getPrimeWeighted(self):
        rawM = self.weightedInput()
        mapFunc = np.vectorize(sigmoidPrime)
        return np.asarray(mapFunc(rawM))
    #end of getPrimeWeighted
    
    def setActivation(self, data):
        if self.m_isInput:
            self.m_activation = data
        #end if
    #end of setActivation
    
    def getMaxIndex(self):
        return np.argmax(self.m_activation)
