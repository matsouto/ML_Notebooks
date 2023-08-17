import matplotlib.pyplot as plt

def logistic_plot_features(X_train, Y_train):
    features = X_train.columns
    X_train = X_train.to_numpy()
    Y_train = Y_train.to_numpy()

    n_features = X_train.shape[1]
    fig, ax = plt.subplots(1,n_features, figsize=(14,2), sharey=True)
    for i in range(n_features):
        ax[i].scatter(X_train[:,i], Y_train, marker='x', color='tab:blue', s=12)
        ax[i].set_xlabel(features[i])

    ax[0].set_ylabel('Class')
    ax[0].legend(['target'], loc='upper right')


    
