import os
import mne
import numpy as np
import preprocessing
import cca
from math import ceil


def get_file_paths(data_dir: str, n_subject: int, n_electrode_type: int):
    # nested list: [subj] [electode_type] 
    offline_files = [ [[] for i in range(n_electrode_type)] for i in range(n_subject) ]
    online1_files = [ [[] for i in range(n_electrode_type)] for i in range(n_subject) ]
    online2_files = [ [[] for i in range(n_electrode_type)] for i in range(n_subject) ]
    online_files = [ [[] for i in range(n_electrode_type)] for i in range(n_subject) ]

    # Categorize input files 
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".vhdr"):
                subj_ind = int(file.split("_")[0]) - 6

                if "_rest" in root:
                    continue
                if 'Gel' in root:
                    electrode_type_ind = 0
                elif 'POLiTAG' in root:
                    electrode_type_ind = 1
                assert electrode_type_ind in [0, 1]
                
                if 'offline' in root:
                    offline_files[subj_ind][electrode_type_ind].append(os.path.join(root, file))
                elif 'online_S1' in root:
                    online1_files[subj_ind][electrode_type_ind].append(os.path.join(root, file))
                elif 'online_S2' in root:
                    online2_files[subj_ind][electrode_type_ind].append(os.path.join(root, file))

    # Input parsing sanity check
    for subj_ind in range(n_subject):
        print("Subject number " + str(subj_ind+6))
        for electrode_type_ind in range(n_electrode_type):
            online_files[subj_ind][electrode_type_ind] = online1_files[subj_ind][electrode_type_ind] + online2_files[subj_ind][electrode_type_ind] # concatenate
        print(str(len(offline_files[subj_ind][0])) + " offline Gel files found")
        print(str(len(offline_files[subj_ind][1])) + " offline POLITAG files found")
        print(str(len(online_files[subj_ind][0])) + " online Gel files found")
        print(str(len(online_files[subj_ind][1])) + " online POLITAG files found")

    return offline_files, online1_files, online2_files, online_files

def load_raw_data(path, montage, electrode_type='Gel'):
    # load raw data
    raw = mne.io.read_raw_brainvision(path,preload=True)
    print(raw.info) # check raw data info

    # replace some channel names for MNE functions
    if electrode_type == 'Gel':
        new_names = dict((ch_name, ch_name.replace('Z', 'z').replace('FP','Fp'))for ch_name in raw.ch_names)
        raw.rename_channels(new_names)
        print(raw.ch_names)
    else:
        wrong_ch_names = raw.ch_names
        correct_ch_names = ['Fz', 'FC5', 'FC1', 'FC6', 'C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'FCz', 'FC2']
        # ['Fz', 'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'FCz']
        new_names = dict(zip(wrong_ch_names, correct_ch_names))
        raw.rename_channels(new_names)
        print(raw.ch_names)

    # set montage
    raw.drop_channels(['Status'], on_missing='ignore')
    raw.set_montage(montage)
    # fig = raw.plot_sensors(show_names=True) # display existing ch loc on topomap

    return raw

def create_events(raw):
    # get MNE annotations from raw data
    onset = raw.annotations.onset
    duration = raw.annotations.duration
    description = raw.annotations.description

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

            # for online datasets
            elif description[i] == 'Stimulus/condition 1':
                new_description.append('Correct trial')
            elif description[i] == 'Stimulus/condition 2':
                new_description.append('Error trial')
            else:
                new_description.append(description[i])
    
    # create new annotations
    my_annot = mne.Annotations(onset=onset, duration=duration, description=new_description)
    # set new annotations
    raw.set_annotations(my_annot)

    # create custom MNE events dictionary
    custom_dict = {'Error trial': 10, 'Correct trial & Target reached': 13, 'Correct trial': 9}
    # choose needed annotations and convert them to events
    events, event_dict = mne.events_from_annotations(raw, event_id=custom_dict)

    return events, event_dict


def vhdr2numpy(filename, montage, electrode_type, spatial_filter, t_baseline, epoch_window, spectral_window, downsampling_ratio, W_s=None, cca_num=None):
    assert t_baseline <= 0
    assert epoch_window[0] >= 0
    assert epoch_window[0] < epoch_window[1]
    assert len(epoch_window) == 2
    
    raw = load_raw_data(filename, montage, electrode_type)
    filters = preprocessing.Filtering(raw, l_freq=spectral_window[0], h_freq=spectral_window[1])
    raw = filters.external_artifact_rejection()
    events, event_dict = create_events(raw)

    if spatial_filter == 'CAR':
        raw = raw.set_eeg_reference('average')
    if spatial_filter == 'CCA':
        W_cca = cca.combine_cca_components(W_s, cca_num, raw.info)
        raw_cca = (raw.get_data().T * W_cca).T
        raw = mne.io.RawArray(raw_cca,raw.info)

    epoc = mne.Epochs(raw, events, event_id=event_dict, tmin=t_baseline, tmax=epoch_window[1], baseline=(t_baseline, 0), preload=True, picks='eeg')
    # create epochs for only correct and error trials
    correct = epoc['Correct trial & Target reached', 'Correct trial']
    error = epoc['Error trial']

    print(str(len(correct)) + " correct trials detected in this run.")
    print(str(len(error)) + " error trials detected in this run.") 

    x = epoc.get_data()
    assert x.shape[1] == len(epoc.ch_names)
    train_start_sample = ceil(abs(t_baseline) * 512) #  skip past negative t portion of epoch
    feature_names = []
    for ch in range(x.shape[1]):
        for sample in range(train_start_sample, x.shape[2], int(1/downsampling_ratio)):
            feature_names.append(str(epoc.ch_names[ch]) + "_" + str(sample - train_start_sample))
    print("feature_names")
    print(len(feature_names))

    print(x.shape)
    x = x[:, :, train_start_sample:x.shape[2]:int(1/downsampling_ratio)]  # downsample x
    print(x.shape)
    x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))  # trials * (channels*samples)
    print(x.shape)

    y = epoc.events[:, 2]
    y = [1 if yy in [9, 13] else 0 for yy in y]  # make labels binary

    print("y")

    print(len(y))
    return x, y, feature_names