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
# preprocessing parameters
low_f_c = 2
high_f_c = 10
epoch_tmin = -0.5
epoch_tmax = 0.5
n_cca_comp = 3
preprocessing_param = {'low_f_c':low_f_c, 'high_f_c':high_f_c, 'epoch_tmin':epoch_tmin, 'epoch_tmax':epoch_tmax, 'n_cca_comp':n_cca_comp}

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

        # epoch initialization
        epochs = []
        epochs_corr = []
        epochs_err = []

        # build CCA using all runs
        W_s = cca.canonical_correlation_analysis(runs, montage, preprocessing_param, car_on=False, show_components=True)

        for run in runs:
            ### 2.1 import raw data ###
            raw = setup.load_raw_data(path=run, montage=montage)

            ### 2.2 temporal filter ###
            # apply notch (60, 120 Hz) and bandpass filter (1-40 Hz)
            filters = preprocessing.Filtering(raw, l_freq=low_f_c, h_freq=high_f_c)
            raw = filters.external_artifact_rejection()

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
