import numpy as np

def rmse(y_true, y_pred):
    from tensorflow.keras import backend as K
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def normal(X):
    """Normalise signal by the max recorded value"""
    # Xc = X - np.mean(X_train, axis=1, keepdims=True)
    return X / X.max(axis=1, keepdims=True)
