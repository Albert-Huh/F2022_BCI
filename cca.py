import mne
import setup
import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA

def canonical_correlation_analysis(run_paths, montage, preprocessing_param, ch_picks=['FCz', 'FC1', 'FC2', 'Cz', 'Fz'], drop=None, electrode_type='Gel', reg_base_on=False, car_on=False, combine=False,show_components=True):
    '''
    run_paths = list of file pathes (str) for raw EEG data (Brain Vision)
    montage = MNE montage file
    preprocessing_param = dict {'low_f_c':low_f_c, 'high_f_c':high_f_c, 'epoch_tmin':epoch_tmin, 'epoch_tmax':epoch_tmax, 'n_cca_comp':n_cca_comp}
    '''
    
    # epoch initialization
    epochs = []
    epochs_corr = []
    epochs_err = []
    event_list = []

    for run in run_paths:
        #import raw data
        raw = setup.load_raw_data(path=run, montage=montage, electrode_type=electrode_type)
        if drop != None:
            raw.drop_channels(drop)
        # apply notch (60, 120 Hz) and bandpass filter (1-40 Hz)
        filters = preprocessing.Filtering(raw, l_freq=preprocessing_param['low_f_c'], h_freq=preprocessing_param['high_f_c'])
        raw = filters.external_artifact_rejection(phase='zero')

        if car_on == True:
            # common average reference (CAR)
            raw = raw.set_eeg_reference('average')

        # create epochs from MNE events
        events, event_dict = setup.create_events(raw)
        epoc = mne.Epochs(raw, events, event_id=event_dict, tmin=preprocessing_param['epoch_tmin'], tmax=preprocessing_param['epoch_tmax'], baseline=None, preload=True, picks='eeg')

        # creat epochs for only correct and error trials
        correct = epoc['Correct trial & Target reached', 'Correct trial']
        error = epoc['Error trial']
        # append epoch lists
        epochs.append(epoc)
        epochs_corr.append(correct)
        epochs_err.append(error)
        event_list.append(events)
        
    # concatenate epoch from differnt runs into a single epoch (for plotting grand avg stuff)
    all_epochs = mne.concatenate_epochs(epochs)
    all_correct = mne.concatenate_epochs(epochs_corr)
    all_error = mne.concatenate_epochs(epochs_err)
    
    # creat epochs for only correct and error trials
    baseline = (preprocessing_param['epoch_tmin'], 0)
    if reg_base_on == False:
        all_correct = all_correct.apply_baseline(baseline)
        all_error = all_error.apply_baseline(baseline)
    else:
        corr_predictor = all_epochs.events[:, 2] != all_epochs.event_id['Error trial']
        err_predictor = all_epochs.events[:, 2] == all_epochs.event_id['Error trial']

        baseline_predictor = (all_epochs.copy().crop(*baseline)
                .pick_channels(ch_picks)
                .get_data()     # convert to NumPy array
                .mean(axis=-1)  # average across timepoints
                .squeeze()      # only 1 channel, so remove singleton dimension
                .mean(axis=-1)  # average across channels
                .squeeze()      # only 1 channel, so remove singleton dimension
        )
        baseline_predictor *= 1e6  # convert V ??? ??V

        design_matrix = np.vstack([corr_predictor,
                        err_predictor,
                        baseline_predictor,
                        baseline_predictor * corr_predictor]).T
        reg_model = mne.stats.linear_regression(all_epochs, design_matrix,
                                    names=["Correct", "ErrP",
                                            "baseline",
                                            "baseline:Correct"])
        W_beta = reg_model['baseline'].beta.get_data() 
        reg_all_epochs_list = []
        for i in range(len(event_list)):
            np_reg_epoc = epochs[i].get_data() * W_beta
            reg_epoc = mne.EpochsArray(np_reg_epoc,raw.info, events=event_list[i], tmin=preprocessing_param['epoch_tmin'], event_id=event_dict)
            reg_all_epochs_list.append(reg_epoc)
        reg_all_epochs = mne.concatenate_epochs(reg_all_epochs_list)
        all_correct = reg_all_epochs['Correct trial & Target reached', 'Correct trial']
        all_error = reg_all_epochs['Error trial']

    # average evoked potential over all trials
    grand_avg_corr = all_correct.average()
    grand_avg_err = all_error.average()

    # get input X (13 ch, (n epochs*n sample))
    X_correct = np.concatenate(all_correct.get_data(),axis=1)
    X_error = np.concatenate(all_error.get_data(),axis=1)
    X = np.append(X_correct, X_error, axis=1).T

    # get desired ref Y (5 central ch, (n epochs*n sample))
    Y_correct = np.tile(grand_avg_corr.get_data(picks=ch_picks),(1,len(all_correct.get_data())))
    Y_error = np.tile(grand_avg_err.get_data(picks=ch_picks),(1,len(all_error.get_data())))
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
        vmin = W_s.min()
        vmax = W_s.max()
            
        fig, axs = plt.subplots(nrows=1, ncols=preprocessing_param['n_cca_comp'])
        if preprocessing_param['n_cca_comp'] > 1:
            for i in range(preprocessing_param['n_cca_comp']-1):
                mne.viz.plot_topomap(W_s.T[i], raw.info, axes=axs[i], size=3,vlim=(-1, 1), vmin=vmin, vmax=vmax, show=False)
            im, cn = mne.viz.plot_topomap(W_s.T[preprocessing_param['n_cca_comp']-1], raw.info, axes=axs[preprocessing_param['n_cca_comp']-1], size=3, vlim=(-1, 1), vmin=vmin, vmax=vmax, show=False)
            cbar = plt.colorbar(im, ax=axs)
            tick_font_size = 22
            cbar.ax.tick_params(labelsize=tick_font_size)
            cbar.set_label('Weight (A.U.)',fontsize=22)
            plt.rcParams.update({'font.size': 20})
            fig.suptitle('CCA Components',fontsize=30)
            plt.show()
        else:
            im, cn = mne.viz.plot_topomap(W_s.T[0], raw.info, axes=axs, size=3,vlim=(-1, 1), vmin=vmin, vmax=vmax, show=False)
            cbar = plt.colorbar(im, ax=axs)
            tick_font_size = 22
            cbar.ax.tick_params(labelsize=tick_font_size)
            cbar.set_label('Weight (A.U.)',fontsize=22)
            plt.rcParams.update({'font.size': 20})
            fig.suptitle('CCA Components',fontsize=30)
            plt.show()


    if combine==True:
        # choose representing CCA component as a sapatial filter
        cca_num = list(input("Choose CCA component(s): ").split(','))
        cca_num = [int(x) for x in cca_num]
        W_cca = combine_cca_components(W_s, cca_num, raw.info)
        print(W_cca.shape)
        print(W_cca) # check the filter weights
        W_s = W_cca

    return W_s

def combine_cca_components(Ws, comp_list, raw_info):
    '''
    Ws = CCA rotation matrix of X (data) [n_components, n_channels]

    comp_list = cca component list
    '''
    W_cca = np.zeros(Ws.T[0].shape)
    for ind in comp_list:
        W_cca += Ws.T[ind]

    fig, axs = plt.subplots(nrows=1, ncols=1)
    im, cn = mne.viz.plot_topomap(W_cca, raw_info, axes=axs, size=3, vlim=(-1, 1), show=False)
    cbar = plt.colorbar(im, ax=axs)
    tick_font_size = 22
    cbar.ax.tick_params(labelsize=tick_font_size)
    cbar.set_label('Weight (A.U.)',fontsize=22)
    plt.rcParams.update({'font.size': 20})
    fig.suptitle('Combined CCA Component',fontsize=40)
    plt.show()
    return W_cca

    


