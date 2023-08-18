import math as m
import numpy as np
import copy as cp
import pandas as pd

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


def zscore_normalization(x_train):
    x_train_copy = cp.deepcopy(x_train)

    if (type(x_train) == pd.core.frame.DataFrame): x_train_copy = x_train_copy.to_numpy()

    # y_train_copy = cp.deepcopy(y_train)
    # y_train_norm = np.zeros(y_train.shape[0])
    x_train_norm = np.zeros([x_train.shape[0], x_train.shape[1]])

    x_mean = []
    x_std = []

    # Normalization per feature
    for i in range(x_train.shape[1]): # Each feature
        x_mean.append(x_train_copy[:,i].mean())
        x_std.append(x_train_copy[:,i].std())
        for j in range(x_train.shape[0]): # Each line
            x_train_norm[j][i] = (x_train_copy[j][i] - x_mean[i]) / x_std[i]

    # for i in range(y_train.shape[0]):
    #     y_train_norm[i] = (y_train_copy[i] - y_train_copy.mean()) / y_train_copy.std()

    # scaler = StandardScaler()
    # y_train_norm = scaler.fit_transform(y_train.reshape(-1, 1))

    return x_train_norm, x_mean, x_std

def linear_compute_cost(x_train, y_train, w, b, lambda_ = 0):
    f_wb = np.dot(x_train,w) + b 
    loss = (f_wb - y_train) ** 2
    J_wb = (sum(loss) + lambda_ * sum(w**2)) / (2*x_train.shape[0])

    return J_wb

def logistic_compute_cost(x_train, y_train, w, b, lambda_ = 0):
    f_wb = sigmoid(np.dot(x_train, w) + b)
    loss = - y_train * np.log(f_wb) - (1 - y_train) * np.log(1 - f_wb)
    J_wb = sum(loss) / x_train.shape[0]
    J_wb += (lambda_ / (2 * x_train.shape[0])) * sum(w**2)

    return J_wb 

def linear_compute_gradient(x_train, y_train, w, b):
    f_wb = np.dot(x_train, w) + b 
    dJdb = sum(f_wb - y_train) / x_train.shape[0]  
    dJdw = np.dot((f_wb - y_train), x_train) / x_train.shape[0] 

    return dJdb, dJdw

def logistic_compute_gradient(x_train, y_train, w, b):
    f_wb = sigmoid(np.dot(x_train, w) + b) 
    dJdb = sum(f_wb - y_train) / x_train.shape[0]  
    dJdw = np.dot(f_wb - y_train, x_train) / x_train.shape[0]

    return dJdb, dJdw

def gradient_descent(x_train, y_train, initial_w, initial_b, gradient_function, cost_function, n_iters, alpha, lambda_ = 0, verbose = False):
    w = cp.deepcopy(initial_w)
    b = cp.deepcopy(initial_b)

    J_history = []

    for i in range(n_iters):
        dJdb, dJdw = gradient_function(x_train, y_train, w, b)
        dJdw += (lambda_ / x_train.shape[0]) * w # Add regularization 
        w = w - alpha * dJdw
        b = b - alpha * dJdb

        cost = cost_function(x_train, y_train, w, b, lambda_)

        if i<100000:      # prevent resource exhaustion 
            J_history.append(cost)

        if verbose and (i % m.ceil(n_iters/10) == 0):
            print(f"Iteration {i}: Cost {float(J_history[-1]):8.2f}")
    
    return w, b, J_history

def linear_prediction_normalizedX(x_test, w, b):
    p = np.dot(x_test, w) + b

    return p

def sklearn_regression(x_train, y_train):
    scaler = StandardScaler()
    x_sk_norm = scaler.fit_transform(x_train)    

    sgdr = SGDRegressor(max_iter=1000)
    sgdr.fit(x_sk_norm, y_train)
    b_sk_norm = sgdr.intercept_
    w_sk_norm = sgdr.coef_

    p_sk = np.dot(x_sk_norm, w_sk_norm) + b_sk_norm
    
    return w_sk_norm, b_sk_norm, p_sk

def predict_new_value(x_test, w, b, x_mean, x_std):
    x_norm = np.zeros(len(x_test))
    for i in range(len(x_test)):
        x_norm[i] = (x_test[i] - x_mean[i]) / x_std[i] 

    p = np.dot(x_norm, w) + b

    return p

def sigmoid(z):
    g = 1 / (1 + np.exp(-1 * z))
    return g

def _test():
    return

if __name__ == "__main__":
    _test()
