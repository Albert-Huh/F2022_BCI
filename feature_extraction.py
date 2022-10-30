import os
import mne
import numpy as np

def eeg_power_band(epochs, mean = False):
    """
    EEG relative power band feature extraction.

    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.

    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # specific frequency bands
    FREQ_BANDS = {"delta": [1.0, 4.0],
                  "theta": [4.0, 8.0],
                  "alpha": [8.0, 13.0],
                  "beta_low": [13.0, 16.0],
                  "beta_high": [16.0, 30.0]}

    psds, freqs = mne.time_frequency.psd_welch(epochs, picks='eeg', fmin=1.0, fmax=30)
    
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)
    print(psds.shape)
    X = []
    # Raw of Evoked type
    if len(psds.shape) == 2:
        for fmin, fmax in FREQ_BANDS.values():
            psds_band = psds[:, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
            X.append(psds_band.reshape(len(psds), -1))
        return np.concatenate(X, axis=-1)
    # Epochs type
    elif len(psds.shape) == 3:
        for fmin, fmax in FREQ_BANDS.values():
            psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
            print(psds_band.shape)
            if mean == False:
                X.append(psds_band.reshape(len(psds), -1))
            elif mean == True:
                psds_band_average = psds_band.mean(axis=1)
                X.append(psds_band_average.reshape(len(psds), -1))
        # for n in range(psds.shape[0]):
        #     for m in range(psds.shape[1]):
        #         for fmin, fmax in FREQ_BANDS.values():
        #             psds_band = psds[n,:, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        #             print(psds_band.shape)
        #         X.append(psds_band.reshape(psds.shape, -1))
        return np.concatenate(X, axis=-1)\

def create_train_test_sets(X, Y, test_portion):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    train_idx = []
    test_idx = []   
    # num_sample = len(Y)
    # num_test = np.floor(num_sample*test_portion)
    event_types = np.unique(Y)
    # print(event_types)
    for event in event_types:
        # search_idx = np.where(Y==event)
        event_idx = np.where(Y==event)[0] # 1D index array
        num_test_for_event = int(np.floor(len(event_idx)*test_portion))
        test_idx.append(np.random.choice(event_idx, num_test_for_event, replace=False))
    test_idx = np.concatenate(test_idx)
    X_test= X[test_idx, :]
    Y_test = Y[test_idx]
    train_idx = np.delete(np.arange(len(Y)), test_idx, axis=0)
    # print(train_idx)
    # print(len(train_idx))
    # print(test_idx)
    # print(len(test_idx))
    # print(np.intersect1d(train_idx, test_idx))
    X_train= X[train_idx, :]
    Y_train = Y[train_idx]
    return X_train, Y_train, X_test, Y_test

