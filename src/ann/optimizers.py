"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp
"""

import numpy as np


class Optimizer:
    def __init__(self, name="sgd", lr=0.01, weight_decay=0.0, gamma=0.9, beta=0.9, eps=1e-8):
        self.name = name.lower()
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma      # for momentum / nag
        self.beta = beta        # for rmsprop
        self.eps = eps
        self.state = {}

    def step(self, layers):
        if self.name == "sgd":
            self.sgd(layers)
        elif self.name == "momentum":
            self.momentum(layers)
        elif self.name == "nag":
            self.nag(layers)
        elif self.name == "rmsprop":
            self.rmsprop(layers)
        else:
            raise ValueError(f"Unknown optimizer: {self.name}")

    def sgd(self, layers):
        for layer in layers:
            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b
            layer.W -= self.lr * grad_W
            layer.b -= self.lr * grad_b

    def momentum(self, layers):
        if "vW" not in self.state:
            self.state["vW"] = [np.zeros_like(layer.W) for layer in layers]
            self.state["vb"] = [np.zeros_like(layer.b) for layer in layers]

        for i, layer in enumerate(layers):
            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b

            self.state["vW"][i] = self.gamma * self.state["vW"][i] + self.lr * grad_W
            self.state["vb"][i] = self.gamma * self.state["vb"][i] + self.lr * grad_b

            layer.W -= self.state["vW"][i]
            layer.b -= self.state["vb"][i]

    def nag(self, layers):
        if "vW" not in self.state:
            self.state["vW"] = [np.zeros_like(layer.W) for layer in layers]
            self.state["vb"] = [np.zeros_like(layer.b) for layer in layers]

        for i, layer in enumerate(layers):
            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b

            vW_prev = self.state["vW"][i].copy()
            vb_prev = self.state["vb"][i].copy()

            self.state["vW"][i] = self.gamma * self.state["vW"][i] + self.lr * grad_W
            self.state["vb"][i] = self.gamma * self.state["vb"][i] + self.lr * grad_b

            layer.W -= (-self.gamma * vW_prev + (1 + self.gamma) * self.state["vW"][i])
            layer.b -= (-self.gamma * vb_prev + (1 + self.gamma) * self.state["vb"][i])

    def rmsprop(self, layers):
        if "sW" not in self.state:
            self.state["sW"] = [np.zeros_like(layer.W) for layer in layers]
            self.state["sb"] = [np.zeros_like(layer.b) for layer in layers]

        for i, layer in enumerate(layers):
            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b

            self.state["sW"][i] = self.beta * self.state["sW"][i] + (1 - self.beta) * (grad_W ** 2)
            self.state["sb"][i] = self.beta * self.state["sb"][i] + (1 - self.beta) * (grad_b ** 2)

            layer.W -= self.lr * grad_W / (np.sqrt(self.state["sW"][i]) + self.eps)
            layer.b -= self.lr * grad_b / (np.sqrt(self.state["sb"][i]) + self.eps)