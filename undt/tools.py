import numpy as np

def rmse(y_true, y_pred):
    from tensorflow.keras import backend as K
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def normal(X):
    """Normalise signal by the max recorded value"""
    return X / X.max(axis=1, keepdims=True)

def add_noise(X, level=0.05):
    """ Add white noise as a percentage of the peak signal. """
    noise = np.random.normal(size=X.shape) * 0.05
    X_noise = X + np.max(X, axis=1)[:, :, None] * noise
    return X_noise

def noise_augment(X, y: list, levels=[0, 0.01, 0.02, 0.04]):
    """ Adugment dataset with white noise of varying levels (wrt.
     the peak signal.) """
    X_aug = []
    y_aug = [np.vstack([i]*len(levels)) for i in y]

    for level in levels:
        X_aug.append(add_noise(X, level))
    return [np.vstack(X_aug)] + y_aug


