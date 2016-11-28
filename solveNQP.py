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
    
    return e_row * e_row + e_col * e_col

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
            W[i,j] = -1 * (efunc(x_eye[:,i] + x_eye[:,j]) - t[i] - t[j] - c)

    d = {};
    d["c"] = c
    d["t"] = t
    d["W"] = W
    
    return d;


if __name__ == "__main__":

    n = 4
    n2 = n*n

    d = expandEnergy(energyNHP, n2);

    nhp = rnn(n2, 1, 0)
    nhp.setThreshold(d["t"]);
    nhp.setWeight(d["W"]);

    num_loop = 500

    solutions = [0] * num_loop
    subs = [0] * num_loop
    count = 0

    for i in range(0,num_loop):
        nhp.setValue(randomBinaryVec(n2))
        for j in range(0,num_loop):
            nhp.update()
            val = nhp.getValue()
            board = np.reshape(val, [n,n])
            
            if energyNHP(val) < 0.001:
                sub = tuple(np.where(val>0.5)[0])
                if sub in subs:
                    pass
                else:
                    solutions[count] = np.matrix(board)
                    subs[count] = sub
                    count = count + 1
                    print(board)
                break
    
        
    


    
