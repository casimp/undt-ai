"""
Grid search of the importance of noise augmentation and frequency correction on
real-world, experimental test accuracy. Each data instance is re-run 5 times with 
freshly initialised weights and randomly seeded noise. Data is pickled. Checks
are made prior to running a model to ensure that the datset hasn't previously
been run. Training and testing run on the minimum profile thickness.
"""
# TensorFlow and tf.keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger

# Helper libraries
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

# Load own code
from undt_ai.load import load_single, load_val_split, merge_load, load_pipeline, merge_load_pipeline
from undt_ai.tools import rmse, normal, noise_augment, reset_weights
from undt_ai.synthetic_build import n_layer_model


# Synthetic Data
fpath_s = Path.home().joinpath('Dropbox/Projects/undt-ai/Data/synth_data.npz')
fpath_sv = Path.home().joinpath('Dropbox/Projects/undt-ai/Data/synth_data_vel.npz')
fpath_sf = Path.home().joinpath('Dropbox/Projects/undt-ai/Data/synth_data_fft.npz')
fpath_sfv = Path.home().joinpath('Dropbox/Projects/undt-ai/Data/synth_data_fft_vel.npz')
fpath_scv = Path.home().joinpath('Dropbox/Projects/undt-ai/Data/synth_data_comb_vel.npz')

# Synthetic Flat
fpath_s_flat = Path.home().joinpath('Dropbox/Projects/undt-ai/Data/synth_flat.npz')
fpath_sv_flat = Path.home().joinpath('Dropbox/Projects/undt-ai/Data/synth_flat_vel.npz')
fpath_sf_flat = Path.home().joinpath('Dropbox/Projects/undt-ai/Data/synth_flat_fft.npz')
fpath_sfv_flat = Path.home().joinpath('Dropbox/Projects/undt-ai/Data/synth_flat_fft_vel.npz')


# Experimental
fpath_e_coup = Path.home().joinpath('Dropbox/Projects/undt-ai/Data/exp_coupons.npz')
fpath_e_plate = Path.home().joinpath('Dropbox/Projects/undt-ai/Data/exp_plate.npz')

# Load coupon data
t_coup, signal_coup, Dmin_coup, D_coup, rms_coup = load_single(fpath_e_coup)
signal_coup[:, :20] = 0
Dmin_coup = Dmin_coup * 1000

# Create initial model (save/reload weights)
model = n_layer_model(n_layers=3, filters=64, kernel_size=71, 
                      drop_fraction=0.2, pool_step=0, batch_norm=True,
                      drop_layers=[-1])
ipath = Path.home().joinpath('Dropbox/Projects/undt-ai/Data/models/L3_K71_BN_D20_initial.h5')
model.save(ipath)

spath = Path.home() / 'Dropbox/Projects/undt-ai/Data/models/'
pkl_spath = spath.parent / f'results/coupon_models_Dmin.pkl'

columns = ['ID', 'Input_Data', 'Noise', 'Subset', 'Epochs', 
           'Run', 'train_rmse', 'val_rmse', 'val_rmse_noise', 
           'test_rmse', 'test_rmse_noise', 'coupon', 'coupon_rmse', 
           'coupon_noise', 'coupon_rmse_noise']

try:
    df = pd.read_pickle(pkl_spath)
except:
    df = pd.DataFrame(columns=columns)

# Iteratre (clumsily) across combinations of enpochs, noise levels
for epochs in [10, 25, 50, 100, 200, 500]:

    for fpath in [fpath_sv, fpath_sfv, fpath_scv]:


        for noise in [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:

            # Note subsetting not used but allowed dataset to be split
            for s_idx, subset in enumerate([None, ]):
                
                sub_int = 0 if subset is None else 1
                sub_str = False if subset is None else s_idx
                stage_ID = f'L3_K71_BN_D20_E{epochs}_{fpath.stem}_N{noise}_S{sub_int}'

                stage_complete = np.all([f'{stage_ID}_{i:02d}' in df['ID'].values for i in range(5)])

                if not stage_complete:
                    
                    data = merge_load_pipeline([fpath, ], levels=[0.0,], split=[0.8, 0.8], crop=45, subset=subset)
                    t, data = data[:3], data[3:]
                    x_train, x_val, x_test = data[:3]
                    y_train_Dmin, y_val_Dmin, y_test_Dmin = data[3:6]
                    y_train_D, y_val_D, y_test_D = data[6:9]
                    y_train_rms, y_val_rms, y_test_rms = data[9:]

                    # Create noisy data
                    x_train_noise, _ = noise_augment(x_train, [y_train_Dmin], levels=[noise])
                    x_val_noise, _ = noise_augment(x_val, [y_val_Dmin], levels=[noise])
                    x_test_noise, _ = noise_augment(x_test, [y_test_Dmin], levels=[noise])

                    for i in range(5):

                        while True:
                            try:
                                model = load_model(ipath, custom_objects={'rmse': rmse})
                                run_ID = f'{stage_ID}_{i:02d}'
                                if run_ID not in df['ID'].values:

                                    print(f'\nStarting {run_ID}\n')

                                    reduce_lr = ReduceLROnPlateau(patience=50, min_delta=0, factor=0.5)
                                    ckpnt_spath = spath / f'{run_ID}_Dmin.h5'
                                    model_checkpoint = ModelCheckpoint(filepath=ckpnt_spath.as_posix(), monitor='val_loss',
                                                                    save_best_only=True)
                                    csv_spath = spath / f'{run_ID}_Dmin.csv'
                                    csv_logger = CSVLogger(csv_spath)
                                    callbacks = [reduce_lr, model_checkpoint, csv_logger]

                                    # Run
                                    model.fit(normal(x_train_noise), [1000 * y_train_Dmin],
                                                    epochs=epochs, batch_size=128, 
                                                    validation_data=[normal(x_val), 1000 * y_val_Dmin], 
                                                    callbacks=callbacks)

                                    # Reload Best
                                    model.load_weights(ckpnt_spath.as_posix())

                                    # Evaluate
                                    train_rmse = model.evaluate(normal(x_train_noise), [1000 * y_train_Dmin], verbose=0)[1]
                                    val_rmse = model.evaluate(normal(x_val), [1000 * y_val_Dmin], verbose=0)[1]
                                    val_rmse_noise = model.evaluate(normal(x_val_noise), [1000 * y_val_Dmin], verbose=0)[1]
                                    test_rmse = model.evaluate(normal(x_test), [1000 * y_test_Dmin], verbose=0)[1]
                                    test_rmse_noise = model.evaluate(normal(x_test_noise), [1000 * y_test_Dmin], verbose=0)[1]

                                    # Evaluate on coupon data
                                    signal_coup_noise, _ = noise_augment(signal_coup, [Dmin_coup], [noise])
                                    
                                    yhat = model.predict(normal(signal_coup))
                                    yhat_noise = model.predict(normal(signal_coup_noise))

                                    coup_rmse = model.evaluate(normal(signal_coup), [Dmin_coup], verbose=0)[1]
                                    coup_rmse_noise = model.evaluate(normal(signal_coup_noise), [Dmin_coup], verbose=0)[1]


                                    data = np.array([run_ID, fpath.stem, noise, sub_str, epochs, i, train_rmse, 
                                            val_rmse, val_rmse_noise, test_rmse, test_rmse_noise, list(yhat.flatten()), coup_rmse, list(yhat_noise.flatten()), coup_rmse_noise])[None, :]
                                    df_b = pd.DataFrame(data=data, columns=columns)
                                    df = df.append(df_b, ignore_index=True)
                                    
                                    df.to_pickle(pkl_spath)

                                else:
                                    print(f'Run: {run_ID} complete. Skipping.')

                            except OSError:
                                print('Something went wrong (OSErro) - Will retry.')
                                continue
                            break
                else:
                    print(f'Stage: {stage_ID} complete. Skipping.')





