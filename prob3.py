import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numpy.linalg as alg
from nnutils import *
from rnn import *

if __name__ == "__main__":
    A = np.array([[1, -1, 1, -1],[2, 1, -2, 1],[-1, -1, -2, 1],[1, -2, 1, 1]])
    b = np.matrix([0, -1, -3, -1]).T
    
    nn = rnn(4, 1, 0)
    W = np.zeros([4,4])
    W[0,3] = -6
    W[3,0] = -6
    W[1,2] = 6
    W[2,1] = 6
    W[1,3] = -2
    W[3,1] = -2

    t = np.array([7, -1, -4, 2])
    x = np.matrix([0, 1, 0, 1]).T
    
    nn.setWeight(W)
    nn.setThreshold(t)
    nn.setValue(x)

    for i in range(0, 20):
        nn.printValue()
        x_now = nn.getValue()
        axb = np.dot(A, x_now) - b
        E = axb.T * axb
        print(E)
        nn.update()
