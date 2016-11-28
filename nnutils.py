import numpy as np
from random import *

def sigmoid(a, x):
    return 1.0/(1.0+np.exp(-a*x))

def threshold(p):
    if p>=0.5:
        return 1
    else:
        return 0

def zeroOrOne(p):
    r = random()
    if p>=r:
        return 1
    else:
        return 0

def randomBinaryVec(n):
    v = np.random.random([n,1])
    for i in range(0,n):
        v[i] = threshold(v[i])
    return v
    
