import numpy as np
"""
数学工具类
"""

def softmax(tensor, eps=1e-8):
    shifted_tensor = tensor - np.max(tensor, axis=-1, keepdims=True)
    exps = np.exp(shifted_tensor)
    return exps / (np.sum(exps, axis=-1, keepdims=True) + eps)

def Z_score(tensor, ddof=0, eps=1e-8):
    tensor = np.asarray(tensor, dtype=np.float32)
    mu = tensor.mean(axis=0, keepdims=True)
    sigma = np.std(tensor, axis=0, keepdims=True, ddof=ddof)
    return (tensor - mu) / (sigma + eps)

def ReLU(tensor):
    return np.maximum(0, tensor)

def ReLU_derivative(Z):
    return (Z > 0).astype(float)

def cross_entropy(y_true, y_pred, eps=1e-8):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + eps)) / m
    return loss