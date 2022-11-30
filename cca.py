import mne
import setup
import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA

def canonical_correlation_analysis(run_paths, montage, preprocessing_param, car_on=False, show_components=True):
    # epoch initialization
    epochs = []
    epochs_corr = []
    epochs_err = []

    for run in run_paths:
        #import raw data
        raw = setup.load_raw_data(path=run, montage=montage)

        # apply notch (60, 120 Hz) and bandpass filter (1-40 Hz)
        filters = preprocessing.Filtering(raw, l_freq=preprocessing_param['low_f_c'], h_freq=preprocessing_param['high_f_c'])
        raw = filters.external_artifact_rejection()

        if car_on == True:
            # common average reference (CAR)
            raw = raw.set_eeg_reference('average')

        # create epochs from MNE events
        events, event_dict = setup.create_events(raw)
        epoc = mne.Epochs(raw, events, event_id=event_dict, tmin=preprocessing_param['epoch_tmin'], tmax=preprocessing_param['epoch_tmax'], baseline=(preprocessing_param['epoch_tmin'], 0), preload=True, picks='eeg')

        # creat epochs for only correct and error trials
        correct = epoc['Correct trial & Target reached', 'Correct trial']
        error = epoc['Error trial']
        # append epoch lists
        epochs.append(epoc)
        epochs_corr.append(correct)
        epochs_err.append(error)

    # concatenate epoch from differnt runs into a single epoch (for plotting grand avg stuff)
    all_epochs = mne.concatenate_epochs(epochs)
    all_correct = mne.concatenate_epochs(epochs_corr)
    all_error = mne.concatenate_epochs(epochs_err)

    # average evoked potential over all trials
    grand_avg_corr = all_correct.average()
    grand_avg_err = all_error.average()

    # get input X (13 ch, (n epochs*n sample))
    X_correct = np.concatenate(all_correct.get_data(),axis=1)
    X_error = np.concatenate(all_error.get_data(),axis=1)
    X = np.append(X_correct, X_error, axis=1).T

    # get desired ref Y (5 central ch, (n epochs*n sample))
    Y_correct = np.tile(grand_avg_corr.get_data(picks=['FCz', 'FC1', 'FC2', 'Cz', 'Fz']),(1,len(all_correct.get_data())))
    Y_error = np.tile(grand_avg_err.get_data(picks=['FCz', 'FC1', 'FC2', 'Cz', 'Fz']),(1,len(all_error.get_data())))
    Y = np.append(Y_correct, Y_error, axis=1).T

    # apply CCA
    cca = CCA(n_components=preprocessing_param['n_cca_comp'])
    cca.fit(X, Y)

    # get CCA spatial filters
    W_s = cca.x_rotations_
    print(W_s.shape)
    print(W_s)

    if show_components == True:
        # visulaize CCA components
        fig, axs = plt.subplots(nrows=1, ncols=preprocessing_param['n_cca_comp'])
        for i in range(preprocessing_param['n_cca_comp']-1):
                mne.viz.plot_topomap(W_s.T[i], raw.info, axes=axs[i], size=3,vlim=(0, 1), show=False)
        im, cn = mne.viz.plot_topomap(W_s.T[preprocessing_param['n_cca_comp']-1], raw.info, axes=axs[preprocessing_param['n_cca_comp']-1], size=3, vlim=(-1, 1), show=False)
        cbar = plt.colorbar(im, ax=axs)
        tick_font_size = 22
        cbar.ax.tick_params(labelsize=tick_font_size)
        cbar.set_label('Weight (A.U.)',fontsize=22)
        plt.rcParams.update({'font.size': 20})
        fig.suptitle('CCA Components',fontsize=40)
        plt.show()

    return W_s