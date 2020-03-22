import numpy as np
from undt_ai.tools import noise_augment
from sklearn .model_selection import train_test_split

def load_raw_syn(fpath, t_i=np.linspace(0, 12/1e6, 300), right=0):
    """ Loads a single raw synthetic (FE generated) data file, crops the
    final portion of the signal."""
    t, signal = np.load(fpath)
    signal_i = np.interp(t_i, t, signal, right=right)
    return t_i, signal_i


def load_single(fpath, crop=45):
    """ Load single data file containing synthetic/experimental UNDT (dpeth) 
    data. This should have been pre-processed, such that the signal array is 
    300 pnts long. Option to crop the first portion of the signal to remove
    cross-talk. This version includes time data. """
    data_dict = np.load(fpath)
    t = data_dict['t'][:, crop:]
    signal = data_dict['signal'][:, crop:]
    d = data_dict['d']
    try:
        dmin = data_dict['dmin']
    except KeyError:
        dmin = np.nan * np.ones_like(d)
    try:
        rms = data_dict['rms']
    except KeyError:
        rms = np.nan * np.ones_like(d)

    return t, signal, dmin, d, rms

def merge_data(data: list):
    """ Merge list of list of arrays. """
    return [np.vstack(i) for i in zip(*data)]


def merge_load(fpaths: list, crop=45):
    """ Iteratively call load_single, merging data stored within data files.
    This data should have been pre-processed, such that the signal array is
    300 pnts long. . Option to crop the first portion of the signal to remove
    cross-talk."""
    data = []
    for fpath in fpaths:
        data.append(load_single(fpath, crop))
    return merge_data(data)


def train_val_test_split(data: list, split=[0.75, 0.9], random_state=1):
    """Takes list of data and performs a train-test-val split on the using 
    sklearn train_test_split function. This single split is carried 
    out twice, with the split variable defining the train-test split and then 
    the splitting of the training data into train-al data sets."""

    # Split data into 'train', test
    data_train_test = train_test_split(*data, train_size=split[0], 
                                       random_state=random_state)
    data_train = data_train_test[::2]
    data_test = data_train_test[1::2]

    # Take train data and resplit into train, val
    data_train_val = train_test_split(*data_train, train_size=split[1], 
                                      random_state=random_state)
    data_train = data_train_val[::2]
    data_val = data_train_val[1::2]

    # Interleave and return
    return [v for pair in zip(data_train, data_val, data_test) for v in pair]


def load_val_split(fpath, crop=45, split=[0.75, 0.9], random_state=1):
    """Load single data set and carry out a train-test and then a
    train-validation split. These two splits can be specified and there is an 
    option to crop the first portion of the signal to remove cross-talk. 
    Set random_split to ensure repeatability of splits. """
    data  = load_single(fpath, crop)
    data_s = train_val_test_split(data, split=split, random_state=random_state)
    if data_s[0].ndim == 2:
        data_s[:3] = [i[:, :, None] for i in data_s[:3]]
    return data_s


def load_pipeline(fpath, levels=[0, 0.01, 0.02, 0.04], crop=45, 
                  split=[0.75, 0.9], random_state=1):
    """Load single data set and carry out a train-test and then a
    train-validation split. These two splits can be specified and there is an 
    option to crop the first portion of the signal to remove cross-talk. 
    Set random_split to ensure repeatability of splits. The training data is 
    then augmented by gaussian noise of a specified level (normalised wrt. max 
    peak signal). """
    data = load_val_split(fpath, crop, split, random_state)
    data_train = data[::3].copy()
    data_train = noise_augment(data_train[0], data_train[1:], levels)
    data[::3] = data_train
    return data



def merge_load_pipeline(fpaths: list, levels=[0, 0.01, 0.02, 0.04], crop=45, 
                        split=[0.75, 0.9], random_state=1):
    """Load and merge multiple data set and carry out a train-test and then a
    train-validation split. These two splits can be specified and there is an 
    option to crop the first portion of the signal to remove cross-talk. 
    Set random_split to ensure repeatability of splits. The training data is 
    then augmented by gaussian noise of a specified level (normalised wrt. max 
    peak signal). """
    data = []
    for fpath in fpaths:
        d = load_pipeline(fpath, levels, crop, split, random_state)
        assert len(d) == 15, f'{len(d)}'
        data.append(d)
    return merge_data(data)


def merge_load_split_pipeline(fpaths_list: list, levels_list: list, crop=45, 
                              split=[0.75, 0.9], random_state=1):
    """Load and merge multiple data set and carry out a train-test and then a
    train-validation split. These two splits can be specified and there is an 
    option to crop the first portion of the signal to remove cross-talk. 
    Set random_split to ensure repeatability of splits. The training data is 
    then augmented by gaussian noise of a specified level (normalised wrt. max 
    peak signal). Multiple lists of filepaths can associated noise levels can
    be specified for split augmentation."""
    data = []
    for fpaths, levels in zip(fpaths_list, levels_list):
        d = merge_load_pipeline(fpaths, levels, crop, split, random_state)
        data.append(d)
    return merge_data(data)


