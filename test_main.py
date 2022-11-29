'''
F2022 BCI Course Term-Project
ErrP Group 3
Authors: Heeyong Huh, Hyonyoung Shin, Susmita Gangopadhyay
'''

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
import preprocessing
import feature_extraction

from sklearn.cross_decomposition import CCA

###### 1 Initialization ######
# get subject number
subject_num = input("Subject number: ")
# get electrode type
electrode_type = ['Gel', 'POLiTAG'][int(input("Electrode type (Gel [0] or POLiTAG [1]): "))]
# get session type
session_type = input("Session (offline [0], online_S1 [1], online_S2 [2]): ")
if session_type == '0':
        session = 'offline'
elif session_type == '1':
        session = 'online_S1'
elif session_type == '2':
        session = 'online_S2'

# get list of file pathes
raw_data_list = os.listdir(os.path.join(os.getcwd(), 'ErrP_data'))
data_name = subject_num + '_' + electrode_type
for data_folder in raw_data_list:
    if data_folder.startswith(data_name):
        raw_offline_folder = os.path.join(os.getcwd(), 'ErrP_data', data_folder, session)
        raw_run_list = os.listdir(raw_offline_folder)
raw_path_list = []
for run in raw_run_list:
        if run.endswith('training'):
                raw_path_list.append(os.path.join(raw_offline_folder, run, run+'.vhdr'))
# print(raw_path_list) # debugging pathes

# get international 10-20 montage
montage = mne.channels.make_standard_montage('standard_1020')

plt.rcParams.update({'font.size': 20})

###### 2 Load Data ######
low_f_c = 2
high_f_c = 10
epoch_tmin = -0.5
epoch_tmax = 0.5
n_comp = 3
# epoch initialization
epochs = []
epochs_corr = []
epochs_err = []
epochs_car = []
epochs_car_corr = []
epochs_car_err = []

for file_name in raw_path_list:
        ### 2.1 import raw data ###
        # load raw data
        raw = mne.io.read_raw_brainvision(file_name,preload=True)
        print(raw.info) # check raw data info

        # raw = raw.filter(1,50) # in case you want to see a filtered raw
        # raw.load_data() # display raw run data
        # raw.plot(block=True)
        
        # replace some channel names for MNE functions
        new_names = dict((ch_name, ch_name.replace('Z', 'z').replace('FP','Fp'))for ch_name in raw.ch_names)
        raw.rename_channels(new_names)

        # set montage
        raw.set_montage(montage)
        # fig = raw.plot_sensors(show_names=True) # display existing ch loc on topomap

        ### 2.2 temporal filter ###
        # apply notch (60, 120 Hz) and bandpass filter (1-40 Hz)
        filters = preprocessing.Filtering(raw, l_freq=low_f_c, h_freq=high_f_c)
        raw = filters.external_artifact_rejection()

        ### 2.3 CAR spatial filter ###
        # common average reference (CAR)
        raw_car = raw.copy().set_eeg_reference('average')

        # fig = raw.plot(block=True)
        # print(raw.info) # check preprocessed raw info

        ### 2.4 epoch creation ###
        # get MNE annotations from raw data
        onset = raw.annotations.onset
        duration = raw.annotations.duration
        description = raw.annotations.description
        # print(description) # check existing discriptions

        # renaming decriptions
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
        # print(new_description) # check new discriptions
        
        # create new annotations
        my_annot = mne.Annotations(onset=onset, duration=duration, description=new_description)
        # set new annotations
        raw.set_annotations(my_annot)
        raw_car.set_annotations(my_annot)

        # create custom MNE events dictionary
        custom_dict = {'Error trial': 10, 'Correct trial & Target reached': 13, 'Correct trial': 9}
        # choose needed annotations and convert them to events
        events, event_dict = mne.events_from_annotations(raw, event_id=custom_dict)

        # print(events) # check events np array
        # print(event_dict) # check custum events dict
        # fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info['sfreq'], first_samp=raw.first_samp) # visualize events

        # create epochs from MNE raw and events
        epoc = mne.Epochs(raw, events, event_id=event_dict, tmin=epoch_tmin, tmax=epoch_tmax, baseline=(epoch_tmin, 0), preload=True, picks='eeg')
        epoc_car = mne.Epochs(raw_car, events, event_id=event_dict, tmin=epoch_tmin, tmax=epoch_tmax, baseline=(epoch_tmin, 0), preload=True, picks='eeg')

        # creat epochs for only correct and error trials
        correct = epoc['Correct trial & Target reached', 'Correct trial']
        error = epoc['Error trial']
        correct_car = epoc_car['Correct trial & Target reached', 'Correct trial']
        error_car = epoc_car['Error trial']

        # append epoch lists
        epochs.append(epoc)
        epochs_corr.append(correct)
        epochs_err.append(error)
        epochs_car.append(epoc_car)
        epochs_car_corr.append(correct_car)
        epochs_car_err.append(error_car)


# concatenate epoch from differnt runs into a single epoch (for plotting grand avg stuff)
all_epochs = mne.concatenate_epochs(epochs)
all_correct = mne.concatenate_epochs(epochs_corr)
all_error = mne.concatenate_epochs(epochs_err)
all_epochs_car = mne.concatenate_epochs(epochs_car)
all_car_correct = mne.concatenate_epochs(epochs_car_corr)
all_car_error = mne.concatenate_epochs(epochs_car_err)

# average evoked potential over all trials
grand_avg_corr = all_correct.average()
grand_avg_err = all_error.average()
grand_avg_car_corr = all_car_correct.average()
grand_avg_car_err = all_car_error.average()

### 2.5 canonical correlation analysis (CCA) ###
# get input X (13 ch, (n epochs*n sample))
X_correct = np.concatenate(all_correct.get_data(),axis=1)
X_error = np.concatenate(all_error.get_data(),axis=1)
X = np.append(X_correct, X_error, axis=1).T

# get desired ref Y (5 central ch, (n epochs*n sample))
Y_correct = np.tile(grand_avg_corr.get_data(picks=['FCz', 'FC1', 'FC2', 'Cz', 'Fz']),(1,len(all_correct.get_data())))
Y_error = np.tile(grand_avg_err.get_data(picks=['FCz', 'FC1', 'FC2', 'Cz', 'Fz']),(1,len(all_error.get_data())))
Y = np.append(Y_correct, Y_error, axis=1).T

# apply CCA
cca = CCA(n_components=n_comp)
cca.fit(X, Y)

# get CCA spatial filters
W_s = cca.x_rotations_
print(W_s.shape)
print(W_s)

# visulaize CCA components
fig, axs = plt.subplots(nrows=1, ncols=n_comp)
for i in range(n_comp-1):
        mne.viz.plot_topomap(W_s.T[i], raw.info, axes=axs[i], size=3,vlim=(0, 1), show=False)
im, cn = mne.viz.plot_topomap(W_s.T[n_comp-1], raw.info, axes=axs[n_comp-1], size=3, vlim=(-1, 1), show=False)
cbar = plt.colorbar(im, ax=axs)
tick_font_size = 22
cbar.ax.tick_params(labelsize=tick_font_size)
cbar.set_label('Weight (A.U.)',fontsize=22)
plt.rcParams.update({'font.size': 20})
fig.suptitle('CCA Components',fontsize=40)
plt.show()

### 2.5 CAR + CCA ###
# get input X (13 ch, (n epochs*n sample))
X_correct = np.concatenate(all_car_correct.get_data(),axis=1)
X_error = np.concatenate(all_car_error.get_data(),axis=1)
X = np.append(X_correct, X_error, axis=1).T

# get desired ref Y (5 central ch, (n epochs*n sample))
Y_correct = np.tile(grand_avg_car_corr.get_data(picks=['FCz', 'FC1', 'FC2', 'Cz', 'Fz']),(1,len(all_car_correct.get_data())))
Y_error = np.tile(grand_avg_car_err.get_data(picks=['FCz', 'FC1', 'FC2', 'Cz', 'Fz']),(1,len(all_car_error.get_data())))
Y = np.append(Y_correct, Y_error, axis=1).T

# apply CCA
cca_car = CCA(n_components=n_comp)
cca_car.fit(X, Y)

# get CCA spatial filters
W_s_car = cca_car.x_rotations_
print(W_s_car.shape)
print(W_s_car)

# visulaize CCA components
fig, axs = plt.subplots(nrows=1, ncols=n_comp)
for i in range(n_comp-1):
        mne.viz.plot_topomap(W_s_car.T[i], raw_car.info, axes=axs[i], size=3,vlim=(0, 1), show=False)
im, cn = mne.viz.plot_topomap(W_s_car.T[n_comp-1], raw_car.info, axes=axs[n_comp-1], size=3, vlim=(-1, 1), show=False)
cbar = plt.colorbar(im, ax=axs)
tick_font_size = 22
cbar.ax.tick_params(labelsize=tick_font_size)
cbar.set_label('Weight (A.U.)',fontsize=22)
plt.rcParams.update({'font.size': 20})
fig.suptitle('CAR + CCA Components',fontsize=40)
plt.show()

###### 3 Feature Selection ######
# get preprocessing setup
spat_filt = int(input("Spatial filtering setups (None [0], CAR [1], CCA [2], CAR+CCA [3]): "))
low_f_c = 2
high_f_c = 10
epoch_tmin = -0.5
epoch_tmax = 0.5
epochs = []
epochs_corr = []
epochs_err = []

if spat_filt==2:
        # visulaize CCA components
        fig, axs = plt.subplots(nrows=1, ncols=n_comp)
        for i in range(n_comp-1):
                mne.viz.plot_topomap(W_s.T[i], raw_car.info, axes=axs[i], size=3,vlim=(0, 1), show=False)
        im, cn = mne.viz.plot_topomap(W_s.T[n_comp-1], raw_car.info, axes=axs[n_comp-1], size=3, vlim=(-1, 1), show=False)
        cbar = plt.colorbar(im, ax=axs)
        tick_font_size = 22
        cbar.ax.tick_params(labelsize=tick_font_size)
        cbar.set_label('Weight (A.U.)',fontsize=22)
        plt.rcParams.update({'font.size': 20})
        fig.suptitle('CCA Components',fontsize=40)
        plt.show()
        
        # choose representing CCA component as a sapatial filter
        cca_num = int(input("Choose CCA component: "))
        W = W_s.T[cca_num]
        print(W) # check the filter weights
elif spat_filt==3:
        # visulaize CCA components
        fig, axs = plt.subplots(nrows=1, ncols=n_comp)
        for i in range(n_comp-1):
                mne.viz.plot_topomap(W_s_car.T[i], raw_car.info, axes=axs[i], size=3,vlim=(0, 1), show=False)
        im, cn = mne.viz.plot_topomap(W_s_car.T[n_comp-1], raw_car.info, axes=axs[n_comp-1], size=3, vlim=(-1, 1), show=False)
        cbar = plt.colorbar(im, ax=axs)
        tick_font_size = 22
        cbar.ax.tick_params(labelsize=tick_font_size)
        cbar.set_label('Weight (A.U.)',fontsize=22)
        plt.rcParams.update({'font.size': 20})
        fig.suptitle('CAR + CCA Components',fontsize=40)
        plt.show()
        
        # choose representing CCA component as a sapatial filter
        cca_num = int(input("Choose CCA component: "))
        W = W_s_car.T[cca_num]
        print(W) # check the filter weights

# reimport raw data with feature selection
for file_name in raw_path_list:
        raw = mne.io.read_raw_brainvision(file_name)
        print(raw.info)
        new_names = dict((ch_name, ch_name.replace('Z', 'z').replace('FP','Fp'))for ch_name in raw.ch_names)
        raw.rename_channels(new_names)
        raw.set_montage(montage)
        filters = preprocessing.Filtering(raw, l_freq=low_f_c, h_freq=high_f_c)
        raw = filters.external_artifact_rejection()
        if spat_filt==1 or spat_filt==3:
                # CAR before applying CCA
                raw = raw.set_eeg_reference('average')
        onset = raw.annotations.onset
        duration = raw.annotations.duration
        description = raw.annotations.description
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
        my_annot = mne.Annotations(onset=onset, duration=duration, description=new_description)
        raw.set_annotations(my_annot)
        custom_dict = {'Error trial': 10, 'Correct trial & Target reached': 13, 'Correct trial': 9}
        events, event_dict = mne.events_from_annotations(raw, event_id=custom_dict)

        if spat_filt==2 or spat_filt==3:
        # apply CCA based spatial filter
                raw_cca = (raw.get_data().T * W).T
                raw = mne.io.RawArray(raw_cca,raw.info)
                print(raw.info)

        epoc = mne.Epochs(raw, events, event_id=event_dict, tmin=epoch_tmin, tmax=epoch_tmax, baseline=(epoch_tmin, 0), preload=True, picks='eeg')
        correct = epoc['Correct trial & Target reached', 'Correct trial']
        error = epoc['Error trial']
        epochs.append(epoc)
        epochs_corr.append(correct)
        epochs_err.append(error)

# concatenate epoch from differnt runs into a single epoch 
all_epochs = mne.concatenate_epochs(epochs)
all_correct = mne.concatenate_epochs(epochs_corr)
all_error = mne.concatenate_epochs(epochs_err)

# average evoked potential over all trials
grand_avg_corr = all_correct.average()
grand_avg_err = all_error.average()


time_unit = dict(time_unit="s")
grand_avg_corr.plot_joint(title="Average Evoked for Correct", picks='eeg',ts_args=time_unit, topomap_args=time_unit)
grand_avg_err.plot_joint(title="Average Evoked for ErrP", picks='eeg',ts_args=time_unit, topomap_args=time_unit)  # show difference wave
evokeds = dict(corr=grand_avg_corr, err=grand_avg_err)
mne.viz.plot_compare_evokeds(evokeds, picks='Fz', combine='mean')
mne.viz.plot_compare_evokeds(evokeds, picks=['FCz', 'FC1', 'FC2', 'Cz', 'Fz'], combine='mean')

###### 4 Feature Selection ######
### 4.1 3-fold cross-validation ###
temp = []
temp2 = []
for run in epochs:
        fold = run.get_data()
        temp.append(fold)
        case = run.events[:,2]
        temp2.append(case)
folds = np.array(temp)
y = np.array(temp2)
print(folds.shape) # (n_fold, n_epoch, n_chs, n_sample)
print(y.shape) # (n_fold, n_epoch)