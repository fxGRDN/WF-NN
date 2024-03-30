'''Actovation Functions'''
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def MSE(x, y):
    return 0.5*np.sum( (x-y)**2)/len(x) 

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)/(1-sigmoid(x))

def tanh(x):
    return (np.exp(2*x)-1)/(np.exp(2*x)+1)

def tanh_prime(x):
    return 1-tanh(x)**2

def ReLU(x):
    return np.maximum(0, x)

def ReLU_prime(x):
    return np.where(x>=0, 1, 0)


plt.style.use('rose-pine-moon')

X = np.linspace(-2, 2, num=100)
plt.plot( X, sigmoid(X) )
plt.plot( X, tanh(X) )
plt.plot( X, ReLU(X) )
plt.plot( X, ReLU_prime(X) )
plt.grid(True)

# x = np.array((2, ))
# y = np.array((-1, ))
# print(MSE(x, y))

