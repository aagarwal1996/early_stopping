
   
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import sys

'''
This script is used for the generation of simulation data. Specifically, 
'''


def sample_boolean_X(n,d):
    X = np.random.randint(0,2.0,(n,d))
    return X

def spike_model(X,s,beta,sigma):
    
    '''
    This method is used to crete responses from a spike_model
    Parameters:
    X: X matrix
    s: sparsity 
    beta: coefficient of spike
    sigma: s.d. of added noise 
    Returns: 
    numpy array of shape (n)        
    '''
    
    def create_y(x,s,beta):
        spike_term = 0
        spike_satisfied = True
        for i in range(s):
            if(x[i] == 0):
                spike_satisfied = False
        if (spike_satisfied):
            spike_term = beta
        else:
            spike_term = 0
        return spike_term
    y_train = np.array([create_y(X[i, :],s,beta) for i in range(len(X))])
    y_train = y_train + sigma * np.random.randn((len(X)))
    return y_train

def pyramid_model(X,s,beta,sigma):
    
    '''
    This method is used to crete responses from a sum of squares model with hard sparsity
    Parameters:
    X: X matrix
    s: sparsity 
    beta: coefficient vector. If beta not a vector, then assumed that 
    sigma: s.d. of added noise 
    Returns: 
    numpy array of shape (n)        
    '''
    
    def create_y(x,s,beta):
        if(len(beta)!=s):
            beta = [beta[0]]*s
        pyramid_term = 0
        for i in range(s):
            if(x[i] == 1):
                pyramid_term += x[i]*beta
            else:
                break
        return pyramid_term
    y_train = np.array([create_y(X[i, :],s,beta) for i in range(len(X))])
    y_train = y_train + sigma * np.random.randn((len(X)))
    return y_train
