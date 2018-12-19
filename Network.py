import numpy as np
from PIL import Image
from mnist import MNIST
import random
import math
class Network:
    '''
    A neural network has layers: every layer has weights, biases, and a node
    weights are coefficients for connections to other nodes
    biases are a scalar addative to nodes
    and the node is a value
    we can conceptualize these things as vectors and matrices,
    '''
    def __init__(self):
        self.numlayers = 3
        self.layerShape = [784, 250, 10] # we have one hidden layer of 30 neurons, an input of 784, and an output of 10
        #weight dimensions = thisLayerSze x lastLayerSize
        self.weights = list()
        self.bias = list()
        self.weights.insert(0, np.zeros((784, 1)))
        self.bias.insert(0, np.zeros((784, 1)))

        for x in range(1,self.numlayers):#we start at 1 because input doesn't need weights (we're putting activation values directly in)
            self.weights.insert(x, 2*np.random.rand(self.layerShape[x], self.layerShape[x-1])-1)
            self.bias.insert(x, 2*np.random.rand(self.layerShape[x], 1)-1)
        self.activations = [np.zeros((a, 1)) for a in self.layerShape]# we needn't set this, as it will change every time we run the network
        self.rawActives = [np.zeros((a, 1)) for a in self.layerShape] # we also may need this
        '''
                Note: It's important (as i had to learn the hard way) that you ensure the shape
                of your arrays is correct. We will be working with quite a few of these things and it can be
                a real headache to try and figure out why one vector is not able to be added to another.
        '''

    def feedforward(self, inputActives):
        self.activations[0] = inputActives/255
        for x in range(1, self.numlayers):
            self.rawActives[x] = np.dot(self.weights[x], self.activations[x-1])+self.bias[x]
            self.activations[x] = sigmoid(self.rawActives[x])


    def backprop(self,input, expected):
        '''
        :param expected: what output you expect for this example
        :param input: the input activations
        :return:
        3 numpy objects
        errWeight: numpy matrices of errors on each layer's weights
        errBiass: numpy vector of errors on each layer's biases
        activesDs: numpy vector of the activations for each layer
        '''
        errWeights = [np.zeros(w.shape) for w in self.weights]
        errBiass = [np.zeros(b.shape) for b in self.bias]
        self.feedforward(input)
        activesDs = self.activations.copy()
        error = self.costFunction(self.activations[-1], expected) * sigmoidprime(self.rawActives[-1])
        errWeights[self.numlayers-1] = np.dot(error, self.activations[self.numlayers-2].T)
        errBiass[self.numlayers-1] = error
        '''remember, we are multiplying with the layer behind us'''
        #Note: as we'll see below, it is important that our activations are a column vector
        for x in range(self.numlayers-2, 0, -1):  # we work our way backwards from the second to last layer
            error = np.dot(self.activations[x+1].T, error) * sigmoidprime(self.rawActives[x]) # calculate this layer's error using the transpose of weights infront
            errWeights[x] = np.dot(error, self.activations[x-1].T)  # Note that we transpose activations, the result should be a matrix, not vector
            errBiass[x] = error

        return errWeights, errBiass, activesDs

    def printEfficiency(self, trainingExamples, trainingSolutions):
        d = zip(trainingExamples, trainingSolutions)
        c = 0
        for x in d:
            self.feedforward(x[0])
            if getDecision(self.activations[-1])[x[1]]==1:
                c += 1
        print(getDecision(self.activations[-1]))
        print("Correct Identifications: ")
        print(c)
        print("On Target Percentage: ")
        print((c/len(trainingSolutions))*100)

    def trainNetwork(self, trainingExamples, trainingSolutions, numEpochs):
        trainingSpeed = 3.0
        examplesAndSolutions = list(zip(trainingExamples, trainingSolutions))
        for x in range(numEpochs):
            printImg(examplesAndSolutions[0][0])
            self.printEfficiency(trainingExamples, trainingSolutions)
            random.shuffle(examplesAndSolutions) # rotate our data set
            batch_count = 10
            for x in range(int(len(trainingSolutions)/batch_count)):
                slice = x * batch_count
                mini_batch = examplesAndSolutions[slice:slice+batch_count]
                epoch_w_e = [np.zeros(w.shape) for w in self.weights]  # i did steal this from the tut because it's beautifully efficent syntax
                epoch_b_e = [np.zeros(b.shape) for b in self.bias]
                for test in mini_batch:
                    image, expectedNum = test
                    w_e, b_e, a_d = self.backprop(image, self.numToArray(expectedNum))
                    epoch_w_e = [w_cur + w_new for w_cur, w_new in zip(epoch_w_e, w_e)]
                    epoch_b_e = [b_cur + b_new for b_cur, b_new in zip(epoch_b_e, b_e)]

                self.weights = [w - ((trainingSpeed / len(mini_batch)) * err) for w, err in zip(self.weights, epoch_w_e)]
                self.bias = [b - ((trainingSpeed / len(mini_batch)) * err) for b, err in zip(self.bias, epoch_b_e)]
            #above, we are summing together all the found errors for these training examples
            #now, we simply use these epoch values to subtract from the weights



        #thus, after each epoch, our weights should change to better match what we want the network to detect

    def saveNetToFile(self, filepath):
        for i in range(len(self.weights)):
            x = self.weights[i]
            np.save(filepath+"/weights"+str(i), x)
        for i in range(len(self.bias)):
            x = self.bias[i]
            np.save(filepath+"/biases" + str(i), x)
    def loadNetFromFile(self, filepath):
        for i in range(len(self.weights)):
            self.weights[i] = np.load(filepath+"/weights"+str(i)+".npy")
        for i in range(len(self.bias)):
            self.bias[i] = np.load(filepath+"/biases"+str(i)+".npy")
    def numToArray(self, value):
        r = np.zeros((10, 1))
        r[value] = 1.0
        return r
    def costFunction(self, activated, expected):
        return activated - expected

    def getSolution(self, img):
        self.feedforward(img)
        b = getDecision(self.activations[-1])
        for i in range(10):
            if b[i]==1.0:
                return i

def getDecision(array):
    v = array/np.max(array)
    return v



def loadTrainingExampels(filepath):
    mndata = MNIST(filepath)
    img, lbl = mndata.load_training()
    imageArray = [np.reshape(a, (784, 1)) for a in img]
    solutions = lbl
    return imageArray, solutions

def loadTestingExample(filepath):
    mndata = MNIST(filepath)
    img, lbl = mndata.load_testing()
    imageArray = [np.reshape(a, (784, 1)) for a in img]
    solutions = lbl
    return imageArray, solutions

def sigmoid(x):
    v = np.clip(x, -500,500)
    return 1/(1+np.exp(-v))
def sigmoidprime(x):
    v = np.clip(x, -500, 500)
    d=sigmoid(v)
    return d*(1-d)
def printImg(array):
    for x in range(0, 28):
        string = ""
        for y in range(0,28):
            if array[x*28 + y] >50:
                string += "B"
            else:
                string +=" "
        print(string)

def loadfrompng(filepath):
    def filter(x):
        bscale = (x[0]-255 + x[1]-255 + x[2]-255)/3
        return math.fabs(bscale)
    img = Image.open(filepath)
    idata = np.array(img)
    a = np.zeros((784, 1))
    for x in range(28):
        for y in range(28):
            a[x*28 + y] = filter(idata[x][y])
    return a


#printImg(a)
#d = Network()
#d.loadNetFromFile('')
#d.feedforward(a)
#print(getDecision(d.activations[-1]))

#print(getDecision(d.activations[-1]))
#print(d.getSolution(a))

d = Network()
'''imgs, sols = loadTrainingExampels('')
timgs, tsols = loadTestingExample('')
d.trainNetwork(imgs, sols, 5)
d.saveNetToFile("test2_250")
d.printEfficiency(imgs, sols)
d.printEfficiency(timgs, tsols)'''
d.loadNetFromFile("test2_250")
a = loadfrompng("0.bmp")
print(d.getSolution(a))