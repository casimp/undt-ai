# TensorFlow and tf.keras
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPool1D, Dropout, Flatten, BatchNormalization
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
import logging
import itertools
import pandas as pd

# Helper libraries
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Library
from undt_ai.tools import rmse


def layer_create(inputs, filters=64, kernel_size=5, batch_norm=False, 
                 drop=True, drop_fraction=0.2, pool=True):
    """
    Creates a single layer for a 1D CNN. Takes another layer as input (i.e. 
    another CNN layer or an input layer).The number of filters and kernel size 
    can be specified. The layer can be followed by batch normalisation, 
    dropout and/or max_pooling where specified. When dropout (drop=True) is 
    included, the dropout percentage (drop_fraction) must be defined.
    """
    ki=glorot_normal(seed=np.random.randint(10000)) # force random weights
    conv = Conv1D(filters=filters, kernel_size=int(kernel_size), 
                  activation='relu',kernel_initializer=ki)(inputs)
    if batch_norm:
        conv = BatchNormalization(momentum=0.9)(conv)
    layer = MaxPool1D(pool_size=2)(conv) if pool else conv
    if drop:
        return Dropout(drop_fraction)(layer)
    else:
        return layer

def n_layer_create(inputs, n_layers=11, filters=128, kernel_size=5, 
                   batch_norm=True, drop_layers=[-1], 
                   drop_fraction=0.5, pool_step=0, current_layer=1):
    """
    Create and stack N convolutional layers, taking and 'input layer' as 
    input. The number of filters and kernel size can be specified. This is
    constant across all layers. Each layer can be followed by batch 
    normalisation (batch_norm=True) and/or dropout or max_pooling. 
    For dropout a list of layers (drop_layers) to include this on must be
    defined and the dropout percentage (drop_fraction) must be specified.
    For max_pooling, the spacing between pooling layers (pool_step) should be 
    defined (i.e. for every other layer pool_step=2). If pool_step=0 then max
    pooling is not used. 
    """
    drop_layers = [drop_layers] if isinstance(drop_layers, (int, np.integer)) else drop_layers
    drop_final = True if -1 in drop_layers and (n_layers - current_layer) == 0 else False
    drop = True if current_layer in drop_layers else True if drop_final else False
    pool = False if pool_step == 0 else True if current_layer % pool_step == 0 else False
    layer = layer_create(inputs, filters, kernel_size, batch_norm, drop, drop_fraction, pool)
    if (n_layers - current_layer) > 0:
        current_layer += 1
        return n_layer_create(layer, n_layers, filters, kernel_size, batch_norm, 
                              drop_layers, drop_fraction, pool_step, current_layer)
    else:
        return layer

def n_layer_model(n_layers=3, filters=64, kernel_size=71, input_shape=(255,1),
                  batch_norm=True, drop_layers=[-1], drop_fraction=0.2, 
                  pool_step=0, optimizer=Adam(),
                  n_predictions=1, loss='mse', metrics=rmse):

    """
    Create an n-layered 1D CNN model for data of a defined shape(input_shape).
    The number of filters and kernel size for each layer can be specified. This 
    is constant across all layers. Each layer can be followed by batch 
    normalisation (batch_norm=True) and/or dropout or max_pooling. 
    For dropout a list of layers (drop_layers) to include this on must be
    defined and the dropout percentage (drop_fraction) must be specified.
    For max_pooling, the spacing between pooling layers (pool_step) should be 
    defined (i.e. for every other layer pool_step=2). If pool_step=0 then max
    pooling is not used. 

    ReLu activation is used after each convolutional layer with linear activation
    for the final dense layer. The number of predictions/classes
    (n_predictions) can be specified. For a regression model n_predictions=1, 
    whereas for a classification problem n_predictions=n_classes. Finally, the
    optimizer (optimizer), loss function(loss) and metrics (metrics) for the model
    can be specified. For a regression model the loss is mean squared error (mse)
    and rmse is often a tracked metric.
    """
    inputs = Input(shape=input_shape)
    outputs = n_layer_create(inputs, n_layers, filters, kernel_size, batch_norm, 
                             drop_layers, drop_fraction, pool_step)
    flatten = Flatten()(outputs)
    predictions = Dense(n_predictions, activation='linear')(flatten)
    model = Model(inputs=inputs, outputs=[predictions])
    model.compile(optimizer=optimizer, loss=[loss],  metrics=[metrics])
    return model


def create_params(param_dict):
    """ Accepts a dictionary of parameters (ie. param:value pairs) and iterates
    through them, storing each unique combination. Returns the parameter keys and
    an array of parameters (list of lists of all combinations). These are also 
    stored in a pandas dataframe. The dataframe is also populated with spaces
    for train/val/test loss and rmse and additional metrics by which to track the
    associated models."""
    iter_params = itertools.product(*param_dict.values())
    params = []
    keys = list(param_dict)
    for values in iter_params:
        params.append(values)
    params = np.vstack(params)

    df = pd.DataFrame(params, columns=keys)
    df['train_loss'], df['train_rmse'] = '', ''
    df['val_loss'], df['val_rmse'] = '', ''
    df['test_loss'], df['test_rmse'] = '', ''
    df['run_time'] , df['batch_size'] , df['best_epoch'], df['epochs'] = '', '', '', ''
    df['trainable_params'] = ''
    df = df[(df['n_layers'] - df['pool_step']) >= 0]
    df.reset_index(drop=True, inplace=True)

    return keys, params, df

def n_layer_optimal(input_shape: tuple, n_layers=3, filters=64, drop=False):
    """
    For a defined input_shape creates an optimised 1D CNN network based on 
    the principle of maximising the receptive field size. The number of layers
    must be greater than 1. Btach normalisation is applied after each layer, 
    which will generally provide adequate regularisation and normalisation,
    although there is an optiion to include dropout on the final layer (p=0.2).
    """
    if len(input_shape) == 1:
        input_shape = input_shape + (1,)
    assert n_layers > 1, 'Poor performance on very shallow model (set N > =1)'
    kernel_size = input_shape[0] // n_layers

    model = n_layer_model(n_layers=n_layers, filters=filters, kernel_size=kernel_size, 
                          input_shape=input_shape, batch_norm=True, 
                          drop_layers=[-1 if drop else 0], drop_fraction=0.2, 
                          pool_step=0, optimizer=Adam(),
                          n_predictions=1, loss='mse', metrics=rmse)

    return model