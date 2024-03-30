import numpy as np
from activations import sigmoid, sigmoid_prime, MSE, ReLU


class NN():
    def __init__(self, nn_shape, activation):
        self.nn_shape = nn_shape
        self.activation = activation
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
    
    def calculate_gradient(self, X, Y):
        y_hat = self.forward(X) 
        delta_L = (self.a[-1]-Y)*activation_prime(self.z[-1])
        ## to do: delta_l 'recursive' calculation:
            ## delta_l-1 = activation_prime_l-1 x (W_l).T * delta_l
        
basic_nn = NN( (3, 2, 1), sigmoid )
basic_nn.W = [
    np.array([2]*6).reshape(2, 3),
    np.array([1, 1]).reshape(1, 2)
    ]
basic_nn.b = [
    np.array([0]*2).reshape(2, 1),
    np.array([0]).reshape(1, 1)
    ]
X = np.array([0, 1, 2]).reshape(3, 1)
Y = np.array([1]).reshape(1, 1)

print(basic_nn.forward(X))
print(basic_nn.a)

















