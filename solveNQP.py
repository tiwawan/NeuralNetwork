"""Module to solve n-Hisha Problem"""

import numpy as np
import scipy.linalg as sp




def energyNHP(x):
    
    X = np.reshape(x, [n, n])
    constant = -1 * np.ones([n,1])
    M1 = np.concatenate((X, constant), axis=1)
    M2 = np.concatenate((X.T, constant), axis=1)

    E = np.trace(M1 * M1.T) + np.trace( M2 * M2.T)
    
    return E

def expandEnergy(efunc):
    return 0;


    
