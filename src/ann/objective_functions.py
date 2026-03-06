"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np

def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=1, keepdims=True)
    softmax = numerator / denominator
    return softmax

def ce_loss(y_true, z_pred):
    probs = softmax(z_pred)
    epsilon = 1e-15
    y_pred = np.clip(probs, epsilon, 1. - epsilon)
    loss = -np.mean(np.sum(y_true * np.log(probs), axis=1))
    return loss

def ce_der(y_true, z_pred):
    y_pred_prob = softmax(z_pred)
    batch_size = y_true.shape[0]
    gradient = (y_pred_prob - y_true)/batch_size
    return gradient

def mse_loss(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

def mse_der(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot
