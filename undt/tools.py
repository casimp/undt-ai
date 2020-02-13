import numpy as np

def rmse(y_true, y_pred):
    from tensorflow.keras import backend as K
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def normal(X):
    """Normalise signal by the max recorded value"""
    # Xc = X - np.mean(X_train, axis=1, keepdims=True)
    return X / X.max(axis=1, keepdims=True)

def add_noise(X, level=0.05):
    """ Add white noise as a percentage of the peak signal. """
    noise = np.random.normal(size=X.shape) * 0.05
    X_noise = X + np.max(X, axis=1)[:, :, None] * noise
    return X_noise

def noise_augment(X, y: list, levels=[0.01, 0.025, 0.05], no_noise=True):
    """ Adugment dataset with white noise of varying levels (wrt.
     the peak signal.) """
    X_aug = [X] if no_noise else []
    y_aug = [np.vstack([i]*(len(levels)+no_noise)) for i in y]

    for level in levels:
        X_aug.append(add_noise(X, level))
    return [np.vstack(X_aug)] + y_aug


