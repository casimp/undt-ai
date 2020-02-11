import numpy as np

def load_single(fpath, crop=45):
    """ Load single data file containing synthetic/experimental
    UNDT (dpeth) data. This should have been pre-processed, such
    that the signal array is 300 pnts long."""
    data_dict = np.load(fpath)
    D, Dmin = data_dict['D'], data_dict['Dmin']
    rms = data_dict['rms']
    signal = data_dict['signal'][:, crop:]

    return Dmin, D, rms, signal

def merge_load(fpaths: list, crop=45):
    """ Iteratively call load_single, merging data stored within data files.
    This data should have been pre-processed, such that the signal array is
    300 pnts long."""
    data = []
    for fpath in fpaths:
        data.append(load_single(fpath, crop))
    Dmin, D, rms, signal = np.column_stack(data)

    return Dmin, D, rms, signal


