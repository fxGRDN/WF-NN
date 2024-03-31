import numpy as np
import matplotlib.pyplot as plt
from activations import sigmoid, sigmoid_prime, MSE, ReLU


class NN():
    def __init__(self, nn_shape, activation, activation_prime):
        self.nn_shape = nn_shape
        self.activation = activation
        self.activation_prime = activation_prime
        self.reset_params()
    
    def reset_params(self):
        self.W = [ np.random.randn( self.nn_shape[L+1], self.nn_shape[L]) 
                          for L in range(len(self.nn_shape)-1) ] 
        self.b = [ np.random.randn( L, 1) for L in self.nn_shape[1:] ]
        
    def forward(self, X):
        self.a = [X]
        self.z = []
        
        for w_i, b_i in zip(self.W, self.b):
            z_i = w_i@self.a[-1] + b_i
            self.z.append(z_i)
            
            a_i = self.activation(self.z[-1])
            self.a.append(a_i)            
            
        return a_i  
    
    def calculate_err_gradient(self, X, Y):
        ## NN forward pass:
        for x, y in zip(X, Y):
            self.forward(x) 
            
            ## for (0.5*MSE) cost function: (0.5*2*(y-y_hat))
            delta_L = (y-self.a[-1])*self.activation_prime(self.z[-1])
            deltas_list = [delta_L]
            
            for l in reversed( range(len(self.nn_shape)-2) ):
                ## matrix mult, Hadamard mult :
                delta_i = (self.W[l+1].T @ deltas_list[-1]) * self.activation_prime(self.z[l])
                deltas_list.append(delta_i)
            
            self.deltas_list = deltas_list
            self.deltas_list.reverse()
            
            ## backpropagation
            for i in range(len(self.W)):
                self.W[i] = self.W[i] + (self.deltas_list[i] @ self.a[i].T)
                self.b[i] = self.b[i] + (self.deltas_list[i] )

        return self.deltas_list
    
    def backpropagation(self, X, Y):
        self.calculate_err_gradient(X, Y)
        
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - (self.deltas_list[i] @ self.a[i].T)
            self.b[i] = self.b[i] - (self.deltas_list[i] )

if __name__ == "__main__":
    
    basic_nn = NN( (2, 4, 2, 1), sigmoid, sigmoid_prime )

    X = np.array([(1, 1), (0, 0), (1, 0), (0, 1)]).reshape(4, 2, 1)
    Y = np.array([[0], [0], [1], [1]]).reshape(4, 1, 1)
    
    err = 0
    for x, y in zip(X, Y):
        basic_nn.forward(x) 
        err += MSE(basic_nn.forward(x), y)
    
    print(err/4)
    err_history = [err/4]
    
    for _ in range(10**3):
        basic_nn.calculate_err_gradient(X, Y)
        
        err = 0
        for x, y in zip(X, Y):
            basic_nn.forward(x) 
            err += MSE(basic_nn.forward(x), y)
        err_history.append(err/4)
        
    plt.plot(err_history)
    print(err_history[-1])
    
    for x, y in zip(X, Y):
        nn_output = basic_nn.forward(x) 
        print(nn_output, y)
    
    
    
    
    
    
    








