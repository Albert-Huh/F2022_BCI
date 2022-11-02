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
# fig = raw.plot(block=True)
onset = raw.annotations.onset
duration = raw.annotations.duration
description = raw.annotations.description
print(description)
new_description = []
for i in range(len(description)):
        if description[i] == 'Stimulus/10':
                new_description.append('Error trial')
        elif description[i] == 'Stimulus/13':
                new_description.append('Correct trial & Target reached')
        elif description[i] == 'Stimulus/9':
                new_description.append('Correct trial')
        elif description[i] == 'Stimulus/16':
                new_description.append('Blank screen')
        elif description[i] == 'Stimulus/8':
                new_description.append('Create target')
        else:
                new_description.append(description[i])
print(new_description)
my_annot = mne.Annotations(onset=onset, duration=duration, description=new_description)
raw.set_annotations(my_annot)

custom_dict = {'Error trial': 10, 'Correct trial & Target reached': 13, 'Correct trial': 9}
events, event_dict = mne.events_from_annotations(raw, event_id=custom_dict)
print(events)
print(event_dict)
fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info['sfreq'], first_samp=raw.first_samp)

epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=0.5, preload=True, picks='eeg')
epochs_fcz = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=0.5, preload=True, picks='FCZ')
correct = epochs['Correct trial & Target reached', 'Correct trial'].average()
error = epochs['Error trial'].average()

correct_fcz = epochs['Correct trial & Target reached', 'Correct trial'].average()
error_fcz = epochs['Error trial'].average()

fig1 = correct_fcz.plot()
fig2 = error_fcz.plot(spatial_colors=True)

for evk in (correct, error):
        # global field power and spatial plot
        evk.plot(gfp=False, spatial_colors=True, ylim=dict(eeg=[-4, 4]))
evokeds = dict(corr=correct, err=error)
mne.viz.plot_compare_evokeds(evokeds, picks='FCZ', combine='mean')
        # spatial plot + topomap
        # evk.plot_joint()
# evokeds = dict(correct_epochs=list(correct), visual=list(error))
# mne.viz.plot_compare_evokeds(evokeds, legend='upper left', show_sensors='upper right')

# times = np.arange(0.05, 0.151, 0.02)
# evoked.plot_topomap(times, ch_type='mag')