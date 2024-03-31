'''Actovation Functions'''
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def MSE(x, y):
    return 0.5*np.sum( (x-y)**2)/len(x) 

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    return (np.exp(2*x)-1)/(np.exp(2*x)+1)

def tanh_prime(x):
    return 1-tanh(x)**2

def ReLU(x):
    return np.maximum(0, x)

def ReLU_prime(x):
    return np.where(x>=0, 1, 0)

def SiLU(x):
    return x*sigmoid(x)

def SiLU_prime(x):
    return x*sigmoid_prime(x) + sigmoid(x)



if __name__ == "__main__":
    plt.style.use('rose-pine-moon')
    
    X = np.linspace(-3, 3, num=100)
    
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot( X, sigmoid(X), c='blue' )
    ax[0, 1].plot( X, tanh(X), c='red' )
    
    ax[1, 1].plot( X, ReLU(X), c='green', linewidth=3 )
    ax[1, 1].plot( X, ReLU_prime(X), ls='--' )
    
    ax[1, 0].plot( X, SiLU(X), c='orange', linewidth=3 )
    ax[1, 0].set_ylim([-0.5, 3])
    ax[1, 0].plot( X, SiLU_prime(X), ls='--' )
    plt.grid(True)
    
    
