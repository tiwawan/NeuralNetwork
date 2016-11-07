import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numpy.linalg as alg
from nnutils import *

class rnn:

    def __init__(self, size, a, mode):
        self.W = np.ones([size, size]) - np.eye(size);
        self.x = np.zeros([size, 1])
        self.t = np.ones([size,1])
        self.a = a
        self.mode = mode
        #mode 0:non-stochastic 1:stochastic
        self.size = size
        print("RNN created")
        self.nextupdate = 0

    def printStatus(self):
        print("W:" + str(self.W))
        print("x:" + str(self.x))

    def printValue(self):
        print(self.x.T)

    def getValue(self):
        return self.x

    def setValue(self, newx):
        self.x = newx
        
    def setWeight(self, newW):
        self.W = newW

    def setThreshold(self, newt):
        self.t = newt

    def update(self):
        u = self.nextupdate
        self.nextupdate = self.nextupdate + 1
        if self.nextupdate == self.size:
            self.nextupdate = 0
        self.x[u] = 0
        s = np.dot(self.W[u], self.x)
        p = sigmoid(self.a, s-self.t[u])
        if self.mode == 0:
            self.x[u] = threshold(p)
        elif self.mode == 1:
            self.x[u] = zeroOrOne(p)
        else:
            print("invalid mode:" + str(self.mode))



