import math as m
import numpy as np
import copy as cp

# from sklearn.linear_model import SGDRegressor
# from sklearn.preprocessing import StandardScaler


def zscore_normalization(x_train, y_train):
    x_train_copy = cp.deepcopy(x_train)
    y_train_copy = cp.deepcopy(y_train)
    y_train_norm = np.zeros(y_train.shape[0])
    x_train_norm = np.zeros([x_train.shape[0], x_train.shape[1]])

    # Normalization per feature
    for i in range(x_train.shape[1]): # Each feature
        for j in range(x_train.shape[0]): # Each line
            x_train_norm[j][i] = (x_train_copy[j][i] - x_train_copy[:,i].mean()) / x_train_copy[:,i].std()

    for i in range(y_train.shape[0]):
        y_train_norm[i] = (y_train_copy[i] - y_train_copy.mean()) / y_train_copy.std()

    # scaler = StandardScaler()
    # y_train_norm = scaler.fit_transform(y_train.reshape(-1, 1))

    return x_train_norm, y_train_norm


def _test():
    return


if __name__ == "__main__":
    _test()
