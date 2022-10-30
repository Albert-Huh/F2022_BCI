'''
F2022 BCI Course Term-Project
ErrP Group 3
Authors: Heeyong Huh, Hyonyoung Shin, Susmita Gangopadhyay
'''

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import preprocessing
import feature_extraction

subject_num = input("Subject number: ")
electrode_type = ['Gel', 'POLiTAG'][int(input("Electrode type (Gel [0] or POLiTAG [1]): "))]
# session = input("Session (offline [0], online_S1 [1], online_S2 [2]): ")
raw_data_list = os.listdir(os.path.join(os.getcwd(), 'ErrP_data'))
data_name = subject_num + '_' + electrode_type
for data_folder in raw_data_list:
    if data_folder.startswith(data_name):
        raw_offline_folder = os.path.join(os.getcwd(), 'ErrP_data', data_folder, 'offline')
        raw_run_list = os.listdir(raw_offline_folder)

raw_path_list = []
for run in raw_run_list:
        raw_path_list.append(os.path.join(raw_offline_folder, run, run+'.vhdr'))

raw = mne.io.read_raw_brainvision(raw_path_list[0])
# raw = raw.filter(1,50)
raw.load_data()
raw.set_eeg_reference('average') 
filters = preprocessing.Filtering(raw, l_freq=1, h_freq=50)
raw = filters.external_artifact_rejection()

print(raw.info)
fig = raw.plot(block=True)
events, event_dict = mne.events_from_annotations(raw, event_id='auto')
fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info['sfreq'], first_samp=raw.first_samp)