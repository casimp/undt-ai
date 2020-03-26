"""
Grid search of the model hyper-parameter space for the raw, synthetic data. 
The number of layer, kernel size and pooling are sampled. Checks
are made prior to running a model to ensure that the model hasn't previously
been evaluated. Training and testing run on the mean profile thickness.
"""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import logging

# from tensorflow import errors
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, Callback
from tensorflow.keras. models import load_model

from undt_ai.synthetic_build import n_layer_model, create_params
from undt_ai.tools import rmse, normal
from undt_ai.load import merge_load_pipeline

import time

class ModelStuck(Callback):
	def on_epoch_end(self, epoch, logs={}):
		if (logs.get('val_rmse') > 2) and (epoch > 30):
			print(f"\nVal_rmse greater than 2 after {epoch} epochs, so stopping training!!")
			self.model.stop_training = True

# Model parameters
overwrite = False
repeat_bad= True
epochs = 600
batch_size = 128

param_dict = {'n_layers':[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 21],
        'kernel_size' : [3, 5, 7, 11, 15, 21, 31, 51, 71, 91],
        'pool_step' : [1, 2, 0],
        'drop_layers' : [0, -1],
        'batch_norm': [False, True]}

root = Path(__file__).resolve().parents[2]
fpath_s = root / 'data/synth_data.npz'
fpath_sf = root / 'data/synth_flat.npz'
fpath_save = root / 'data/synth_model_matrix_adam.pkl'
fpath_ckpnt = root / 'data/checkpoints/ckpnt.h5'
fpath_log = root / 'data/checkpoints/training.log'
fpath_log_store = root / 'data/results/synth_model_matrix_adam_log.pkl'
fpath_ckpnt = fpath_ckpnt.as_posix()


# Load synthetic data
data = merge_load_pipeline([fpath_s, fpath_sf], levels=[0,], split=[0.8, 0.8])
t, data = data[:3], data[3:]
X_train, X_val, X_test = data[:3]
y_train_D, y_val_D, y_test_D = data[6:9]

# Reload and continue analysis versus overwrite/create
if Path(fpath_save).is_file() and not overwrite:
    df = pd.read_pickle(fpath_save)
    keys = ['n_layers', 'kernel_size', 'pool_step', 'drop_layers', 'batch_norm']
    params = np.array(df[keys])
else:
    keys, params, df = create_params(param_dict)

df_log_store = pd.read_pickle(fpath_log_store)

# ... and run.
for idx, values in enumerate(params):

    ps = dict(zip(keys, values))
    if ps['n_layers'] <= 3:
        param_loc = (df[list(ps)] == pd.Series(ps)).all(axis=1)

        successful = False
        while not successful:
            # Check whether this run is already complete and successful

            if np.any(df.loc[param_loc, 'val_rmse'].values != '') and not overwrite:
                test_r = df.loc[param_loc, "test_rmse"].values
                if test_r > 1.5 and repeat_bad:
                    print(f'{values}: Not successful. Repeating (test_rmse = {test_r[0]:.2f}).')
                else:
                    print(f'{values}: Run already complete. Skipping (test_rmse = {test_r[0]:.2f}).')
                    successful = True
                    continue

            try:
                print(f'{values}: Starting run')
                model = n_layer_model(**ps)
                checkpoint = ModelCheckpoint(filepath=fpath_ckpnt, monitor='val_loss',save_weight_only=False,
                                verbose=0, save_best_only=True)
                csv_logger = CSVLogger(fpath_log, separator=',', append=False)

                start_time = time.time()
                # print(ps)
            except:
                print('Invalid combination of parameters')
                # Save current state of pickle (in case of failure)
                df.loc[param_loc, ('train_loss', 'train_rmse')] = np.nan
                df.loc[param_loc, ('val_loss', 'val_rmse', 'test_loss', 'test_rmse', 'run_time', 'batch_size', 'epochs', 'trainable_params', 'best_epoch')] = np.nan
                df.to_pickle(fpath_save)
                successful = True

            else:
                model.fit(normal(X_train), [1000 * y_train_D], epochs=epochs,
                        batch_size=batch_size, validation_data=[normal(X_val), 1000 * y_val_D],
                        callbacks=[checkpoint, csv_logger, ModelStuck()], shuffle=True)
                end_time = time.time()

                # Reload best model and evaluate
                model.load_weights(fpath_ckpnt)
                train = model.evaluate(x=normal(X_train), y=1e3*y_train_D, verbose=0)
                val = model.evaluate(x=normal(X_val), y=1e3*y_val_D, verbose=0)
                test = model.evaluate(x=normal(X_test), y=1e3*y_test_D, verbose=0)

                if test[1] > 1.5 and repeat_bad:
                    print(f'{values}: Not successful. Restart and repeating (test_rmse = {test_r[0]:.2f}).')
                    successful = False
                    quit()

                else:
                    # Save current state of pickle (in case of failure)
                    log_data = pd.read_csv(fpath_log, sep=',', index_col=0)
                    for i in ['rmse', 'val_rmse']:
                        vr = i == 'val_rmse'
                        log_ps = ps.copy()
                        log_ps['val_rmse'] = vr
                        log_loc = (df_log_store[list(log_ps)] == pd.Series(log_ps)).all(axis=1)
                        df_log_store.loc[log_loc, np.arange(600)] = log_data[i].values
                        df_log_store.to_pickle(fpath_log_store)
                    best_epoch = log_data.sort_values('val_loss').index.values[0]
                    df.loc[param_loc, ('train_loss', 'train_rmse')] = train
                    df.loc[param_loc, ('val_loss', 'val_rmse', 'test_loss', 'test_rmse')] = val + test
                    df.loc[param_loc, ('run_time', 'batch_size', 'trainable_params')] = end_time - start_time, batch_size, model.count_params()
                    df.loc[param_loc, ('best_epoch', 'epochs')] = best_epoch, epochs
                    df.to_pickle(fpath_save)
                    successful = True

