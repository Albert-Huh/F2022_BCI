'''
F2022 BCI Course Term-Project
ErrP Group 3
Authors: Heeyong Huh, Hyonyoung Shin, Susmita Gangopadhyay
'''

import os
import numpy as np 
import mne 
import setup
import preprocessing
import cca
import numpy.matlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

###### 1 Initialization ######
# define data directory
data_dir = os.path.join(os.getcwd(), 'ErrP_data')
n_subject = 3
n_electrode_type = 2

# get file paths (nested list: [subj] [electode_type]) 
offline_files, online1_files, online2_files, online_files = setup.get_file_paths(data_dir, n_subject, n_electrode_type)

# get international 10-20 montage
montage = mne.channels.make_standard_montage('standard_1020')

plt.rcParams.update({'font.size': 20})

###### 2 Offline Analysis ######
print('Start offline analysis: ')

print(offline_files[0][0])
for subj_ind in range(n_subject):
    subject_num = subj_ind+6
    print('Subject number: ' + str(subject_num))

    # TODO: Implement POLiTAG import
    for electrode_type_ind in range(1): # 11/29/2022 using gel only for now 
        electrode_type = ['Gel','POLiTAG'][electrode_type_ind]
        print('Electrode type: ' + str(electrode_type))

        n_run = len(offline_files[subj_ind][electrode_type_ind])
        runs = offline_files[subj_ind][electrode_type_ind]

        ### 2.1 get preprocessing setup ###
        # temporal filter parameters
        low_f_c = 2
        high_f_c = 10
        epoch_tmin = -0.5
        epoch_tmax = 0.5
        n_cca_comp = 3
        preprocessing_param = {'low_f_c':low_f_c, 'high_f_c':high_f_c, 'epoch_tmin':epoch_tmin, 'epoch_tmax':epoch_tmax, 'n_cca_comp':n_cca_comp}

        # spatial filter parameters
        spat_filt = int(input("Spatial filtering setups (None [0], CAR [1], CCA [2], CAR+CCA [3]): "))
        if spat_filt==1 or spat_filt==3:
            car_on = True
        else:
            car_on = False
        if spat_filt==2 or spat_filt==3:
            cca_on = True
            # build CCA using all runs
            W_s = cca.canonical_correlation_analysis(runs, montage, preprocessing_param, car_on=car_on, show_components=True)
            # choose representing CCA component as a sapatial filter
            cca_num = int(input("Choose CCA component: "))
            W_cca = W_s.T[cca_num]
            print(W_cca) # check the filter weights
        else:
            cca_on = False

        # epoch initialization
        epochs = []
        epochs_corr = []
        epochs_err = []

        for run in runs:
            ### 2.2 import raw data ###
            raw = setup.load_raw_data(path=run, montage=montage)

            ### 2.3 preprocessing ###
            # apply notch (60, 120 Hz) and bandpass filter (1-40 Hz)
            filters = preprocessing.Filtering(raw, l_freq=low_f_c, h_freq=high_f_c)
            raw = filters.external_artifact_rejection()

            if car_on == True:
                # common average reference (CAR)
                raw = raw.set_eeg_reference('average')
            if cca_on == True:
                # apply CCA based spatial filter
                raw_cca = (raw.get_data().T * W_cca).T
                raw = mne.io.RawArray(raw_cca,raw.info)

            ### 2.4 create epochs ###
            # create epochs from MNE events
            events, event_dict = setup.create_events(raw)
            epoc = mne.Epochs(raw, events, event_id=event_dict, tmin=epoch_tmin, tmax=epoch_tmax, baseline=(epoch_tmin, 0), preload=True, picks='eeg')

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

        ### 2.5 analysis visulaization ###
        time_unit = dict(time_unit="s")
        grand_avg_corr.plot_joint(title="Average Evoked for Correct", picks='eeg',ts_args=time_unit, topomap_args=time_unit)
        grand_avg_err.plot_joint(title="Average Evoked for ErrP", picks='eeg',ts_args=time_unit, topomap_args=time_unit)  # show difference wave
        evokeds = dict(corr=grand_avg_corr, err=grand_avg_err)
        mne.viz.plot_compare_evokeds(evokeds, picks='Fz', combine='mean')
        mne.viz.plot_compare_evokeds(evokeds, picks=['FCz', 'FC1', 'FC2', 'Cz', 'Fz'], combine='mean')
'''
        ### 2.6 create training data ###
        temp = []
        temp2 = []

        for test_fold in epochs:
            train_folds = [i for i in epochs if i != test_fold] # use all other runs to train 
            # print(test_run, train_runs)

            xs = []; Y_train = []
            for train_fold in train_folds: 
                x, y, f = vhdr2numpy(offline_files[subj][electrode_type][train_run], False, 'CCA', [-0.5, 0.5])
                # print(x.shape)
                # print(y)
                # print(len(y))
                xs.append(x) 
                Y_train = Y_train + y

            for run_epoch in epochs:
                    fold = run_epoch.get_data()
                    temp.append(fold)
                    case = run_epoch.events[:,2]
                    temp2.append(case)
            folds = np.array(temp)
            y = np.array(temp2)
            print(folds.shape) # (n_fold, n_epoch, n_chs, n_sample)
            print(y.shape) # (n_fold, n_epoch)   
        
        x = epoc.get_data()
        feature_names = []
        assert x.shape[1] == len(epoc.ch_names)
        for ch in range(x.shape[1]):
            for sample in range(x.shape[2]):
                feature_names.append(str(epoc.ch_names[ch]) + "_" + str(sample))

        x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))  # trials * (channels*samples)
        y = epoc.events[:, 2]
        y = [1 if yy in [9, 13] else 0 for yy in y]  # make labels binary
'''