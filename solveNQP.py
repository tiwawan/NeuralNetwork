"""Module to solve n-Hisha Problem"""

import numpy as np
import numpy.linalg as alg
from rnn import *




def energyNHP(x):
    n = findSqrt(x.size, 100000)
    if n == -1:
        return
    X = np.reshape(x, [n, n])
    constant = -1 * np.ones([n,1])

    M1 = np.concatenate((X, constant), axis=1)
    M2 = np.concatenate((X.T, constant), axis=1)

    e_row = alg.norm(np.sum(M1, 1), 2)
    e_col = alg.norm(np.sum(M2, 1), 2)
    
    return e_row + e_col

def findSqrt(nsq, maxn):
    for i in range(0, maxn):
        if i*i == nsq:
            return i
    return -1

def expandEnergy(efunc, n):
    x_zeros = np.zeros([n, 1])
    c = efunc(x_zeros)

    x_eye = np.eye(n)
    t = np.zeros([n,1])

    for i in range(0,n):
        t[i] = efunc(x_eye[:,i]) - c

    W = np.zeros([n,n])
    for i in range(0, n):
        for j in range(0, n):
            if i==j:
                continue
            W[i,j] = -2 * (efunc(x_eye[:,i] + x_eye[:,j]) - t[i] - t[j] - c)

    d = {};
    d["c"] = c
    d["t"] = t
    d["W"] = W
    
    return d;


if __name__ == "__main__":
    x = np.ones([1, 2]);
    en = energyNHP(x)
    d = expandEnergy(energyNHP, 4);

    nhp = rnn(4, 1, 1)
    nhp.setThreshold(d["t"]);
    nhp.setWeight(d["W"]);
    nhp.setValue(np.array([0,1,0,1]))

    for i in range(0,64):
        nhp.update()
        nhp.printValue()
        print(energyNHP(nhp.getValue()))
        
    


    
