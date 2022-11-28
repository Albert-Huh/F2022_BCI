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

# import data
subject_num = input("Subject number: ")
electrode_type = ['Gel', 'POLiTAG'][int(input("Electrode type (Gel [0] or POLiTAG [1]): "))]
session_type = input("Session (offline [0], online_S1 [1], online_S2 [2]): ")
if session_type == '0':
        session = 'offline'
elif session_type == '1':
        session = 'online_S1'
elif session_type == '2':
        session = 'online_S2'

raw_data_list = os.listdir(os.path.join(os.getcwd(), 'ErrP_data'))
data_name = subject_num + '_' + electrode_type
for data_folder in raw_data_list:
    if data_folder.startswith(data_name):
        raw_offline_folder = os.path.join(os.getcwd(), 'ErrP_data', data_folder, session)
        raw_run_list = os.listdir(raw_offline_folder)

montage = mne.channels.make_standard_montage('standard_1020')
raw_path_list = []
for run in raw_run_list:
        if run.endswith('training'):
                raw_path_list.append(os.path.join(raw_offline_folder, run, run+'.vhdr'))

correct_epochs = []
error_epochs = []
print(raw_path_list)



for file_name in raw_path_list:
        raw = mne.io.read_raw_brainvision(file_name)

        # raw = raw.filter(1,50)
        raw.load_data()
        new_names = dict(
                (ch_name,
                ch_name.replace('Z', 'z').replace('FP','Fp'))
                for ch_name in raw.ch_names)
        raw.rename_channels(new_names)
        raw.set_montage(montage)
        raw.set_eeg_reference('average') # CAR
        # fig = raw.plot_sensors(show_names=True)

        filters = preprocessing.Filtering(raw, l_freq=5, h_freq=15)
        raw = filters.external_artifact_rejection()
        print(raw.info)
        # fig = raw.plot(block=True)
        onset = raw.annotations.onset
        duration = raw.annotations.duration
        description = raw.annotations.description
        # print(description)
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
        # print(new_description)
        my_annot = mne.Annotations(onset=onset, duration=duration, description=new_description)
        raw.set_annotations(my_annot)

        custom_dict = {'Error trial': 10, 'Correct trial & Target reached': 13, 'Correct trial': 9}
        events, event_dict = mne.events_from_annotations(raw, event_id=custom_dict)
        # print(events)
        # print(event_dict)
        # fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info['sfreq'], first_samp=raw.first_samp)

        epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=0.5, preload=True, picks='eeg')
        correct = epochs['Correct trial & Target reached', 'Correct trial']
        error = epochs['Error trial']
        # correct = epochs['Correct trial & Target reached', 'Correct trial'].average()
        # error = epochs['Error trial'].average()
        correct_epochs.append(correct)
        error_epochs.append(error)
all_correct = mne.concatenate_epochs(correct_epochs)
print('len of epochs are', len(correct_epochs[0]) + len(correct_epochs[1]) + len(correct_epochs[2]))
print('len of epochs are', len(all_correct))

all_error = mne.concatenate_epochs(error_epochs)
correct_evoked = all_correct.average()
error_evoked = all_error.average()

X_correct = np.concatenate(all_correct.get_data(),axis=1)
X_error = np.concatenate(all_error.get_data(),axis=1)
'''
print(type(all_correct.get_data()), type(all_error.get_data()), type(correct_evoked.get_data()), type(error_evoked.get_data()))
print(len(all_correct.get_data()), len(all_error.get_data()), len(correct_evoked.get_data()), len(error_evoked.get_data()))
print(len(all_correct.get_data()))
print(len(all_correct.get_data()[0]))
print(len(all_correct.get_data()[0][0]))
print(len(correct_evoked.get_data()))
print(len(correct_evoked.get_data()[0]))

print(np.concatenate(all_correct.get_data(),axis=1).shape)
print(np.concatenate(all_correct.get_data(),axis=1)[0:4])
print(np.tile(correct_evoked.get_data(),(1,len(all_correct.get_data()))).shape)
print(np.tile(correct_evoked.get_data(),(1,len(all_correct.get_data())))[0:4])

print(np.concatenate(all_correct.get_data(),axis=1)[0][0])
print(np.concatenate(all_correct.get_data(),axis=1)[0][359])
print(np.concatenate(all_correct.get_data(),axis=1)[0][718])
print(np.tile(correct_evoked.get_data(),(1,len(all_correct.get_data())))[0][0])
print(np.tile(correct_evoked.get_data(),(1,len(all_correct.get_data())))[0][359])
print(np.tile(correct_evoked.get_data(),(1,len(all_correct.get_data())))[0][718])
'''
# for evk in (correct_evoked, error_evoked):
#         # global field power and spatial plot
#         evk.plot(gfp=False, spatial_colors=True, ylim=dict(eeg=[-4, 4]))
time_unit = dict(time_unit="s")
correct_evoked.plot_joint(title="Average Evoked for Correct", picks='eeg',ts_args=time_unit, topomap_args=time_unit)
error_evoked.plot_joint(title="Average Evoked for ErrP", picks='eeg',ts_args=time_unit, topomap_args=time_unit)  # show difference wave
evokeds = dict(corr=correct_evoked, err=error_evoked)
mne.viz.plot_compare_evokeds(evokeds, picks='Fz', combine='mean')
mne.viz.plot_compare_evokeds(evokeds, picks=['FCz', 'FC1', 'FC2', 'Cz', 'Fz'], combine='mean')


'''

'''