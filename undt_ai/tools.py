import numpy as np
import tensorflow as tf

def rmse(y_true, y_pred):
    from tensorflow.keras import backend as K
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def normal(X, start_idx=0, axis=1):
    """Normalise signal by the max recorded value past the defined start_idx.
    Define the axis to normalise across. """
    return X / X[:, start_idx:].max(axis=axis, keepdims=True)

def add_noise(X, level=0.05):
    """ Add white noise as a percentage of the peak signal. """
    noise = np.random.normal(size=X.shape) * level
    X_noise = X + np.max(X, axis=1)[:, :, None] * noise
    return X_noise

def noise_augment(x, y: list, levels=[0.02, ]):
    """ Augment dataset with white noise of varying levels (wrt.
     the peak signal.) """
    x_aug = []
    y_aug = [np.vstack([i]*len(levels)) for i in y]

    for level in levels:
        x_aug.append(add_noise(x, level))
    return [np.vstack(x_aug)] + y_aug

def reset_weights(model):
  for layer in model.layers: 
    if isinstance(layer, tf.keras.Model):
      reset_weights(layer)
      continue
    for k, initializer in layer.__dict__.items():
      if "initializer" not in k:
        continue
      # find the corresponding variable
      var = getattr(layer, k.replace("_initializer", ""))
      var.assign(initializer(var.shape, var.dtype))
