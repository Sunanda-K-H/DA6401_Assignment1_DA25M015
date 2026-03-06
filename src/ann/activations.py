"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(h):
    return h*(1-h)

def tanh(x):
    return np.tanh(x)

def tanh_der(h):
    return 1-h**2

def relu(x):
    return np.maximum(0, x)

def relu_der(z):
    return (z > 0).astype(float)

if __name__=="__main__":
    x = np.array([[-1, 0, 1],
              [2, -3, 4]])

    print(relu(x))
    print(sigmoid(x))
    print(tanh(x))