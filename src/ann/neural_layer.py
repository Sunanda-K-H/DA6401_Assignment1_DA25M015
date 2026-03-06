"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from src.ann.activations import sigmoid, sigmoid_der, tanh, tanh_der, relu, relu_der


class Layer:
    def __init__(self, f_in, f_out, act=None, w_init="xavier"):
        self.f_in = f_in
        self.f_out = f_out
        self.act = act
        self.w_init = w_init

        if self.w_init == "random":
            self.W = np.random.uniform(-0.05, 0.05, size=(self.f_in, self.f_out))
            self.b = np.zeros((1, self.f_out), dtype=np.float64)

        elif self.w_init == "xavier":
            lim = np.sqrt(6.0 / (self.f_in + self.f_out))
            self.W = np.random.uniform(-lim, lim, size=(self.f_in, self.f_out))
            self.b = np.zeros((1, self.f_out), dtype=np.float64)
        
        elif self.w_init == "zeros":
            self.W = np.zeros((self.f_in, self.f_out), dtype=np.float64)
            self.b = np.zeros((1, self.f_out), dtype=np.float64)
        
        else:
            raise ValueError("Unknown initialization method")

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        self.x = None
        self.z = None
        self.a = None

    def forward(self, x):
        self.x = x
        self.z = self.x @ self.W + self.b

        if self.act is None:
            self.a = self.z
        elif self.act == "sigmoid":
            self.a = sigmoid(self.z)
        elif self.act == "tanh":
            self.a = tanh(self.z)
        elif self.act == "relu":
            self.a = relu(self.z)
        else:
            raise ValueError("Unknown activation function")

        return self.a

    def backward(self, grad_output):
        if self.act is None:
            grad_z = grad_output
        elif self.act == "sigmoid":
            grad_z = grad_output * sigmoid_der(self.a)
        elif self.act == "tanh":
            grad_z = grad_output * tanh_der(self.a)
        elif self.act == "relu":
            grad_z = grad_output * relu_der(self.z)
        else:
            raise ValueError("Unknown activation function")

        self.grad_W = self.x.T @ grad_z
        self.grad_b = np.sum(grad_z, axis=0, keepdims=True)
        grad_input = grad_z @ self.W.T

        return grad_input
    
    
if __name__ == "__main__":
    x = np.random.randn(4, 3)
    layer = Layer(3, 2, act="relu", w_init="xavier")

    out = layer.forward(x)
    print("out shape:", out.shape)

    grad_output = np.random.randn(4, 2)
    grad_input = layer.backward(grad_output)

    print("grad_input shape:", grad_input.shape)
    print("grad_W shape:", layer.grad_W.shape)
    print("grad_b shape:", layer.grad_b.shape)