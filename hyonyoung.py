'''
F2022 BCI Course Term-Project
ErrP Group 3
Authors: Heeyong Huh, Hyonyoung Shin, Susmita Gangopadhyay
'''
  
import os
import numpy as np 
import mne 
import preprocessing

# Define data directory
data_dir = 'C:/Users/mcvai/F2022_BCI/ErrP_data'

offline_files = [] 
online1_files = [] 
online2_files = []
all_files = [] 

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".vhdr"):
            all_files.append(os.path.join(root, file))
            
            if 'offline' in root:
                offline_files.append(os.path.join(root, file))

            elif 'online_S1' in root:
                online1_files.append(os.path.join(root, file))

            elif 'online_S2' in root:
                online2_files.append(os.path.join(root, file))
            
online_files = [online1_files, online2_files] # concatenate 
                       
print(str(len(offline_files)) + " offline files found: ")
print(offline_files)
print(str(len(online_files)) + " online files found: ")
print(online_files)

data = offline_files[1]

raw = mne.io.read_raw_brainvision(data, preload=True)
new_names = dict(
                (ch_name,
                ch_name.replace('Z', 'z').replace('FP','Fp'))
                for ch_name in raw.ch_names)

print("Channel name conversion: ")
print(raw.ch_names)
print("has been changed to ")
print(new_names)

raw.rename_channels(new_names)
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)
raw.set_eeg_reference('average') # CAR
# fig = raw.plot_sensors(show_names=True)

filters = preprocessing.Filtering(raw, l_freq=2, h_freq=10)
raw = filters.external_artifact_rejection()
print(raw.info)
# fig = raw.plot(block=True)
onset = raw.annotations.onset
duration = raw.annotations.duration
description = raw.annotations.description
print("onset:")
print(onset)
print("duration:")
print(duration)
print("description:")
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

    my_annot = mne.Annotations(onset=onset, duration=duration, description=new_description)
    raw.set_annotations(my_annot)

    custom_dict = {'Error trial': 10, 'Correct trial & Target reached': 13, 'Correct trial': 9}
    events, event_dict = mne.events_from_annotations(raw, event_id=custom_dict)
    # print(events)
    # print(event_dict)
    # fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info['sfreq'], first_samp=raw.first_samp)

    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.5, tmax=0.5, baseline=(-0.5, 0), preload=True, picks='eeg')
    correct = epochs['Correct trial & Target reached', 'Correct trial']
    error = epochs['Error trial']
    # correct = epochs['Correct trial & Target reached', 'Correct trial'].average()
    # error = epochs['Error trial'].average()
    correct_epochs.append(correct)
    error_epochs.append(error)


eeg_channels = mne.pick_types(raw.info, eeg=True)

events = mne.events_from_annotations(raw, event_id='auto')

# reference

raw = raw.copy().filter(l_freq=1, h_freq=50)
print(raw.info)
raw.plot(duration=60, order=eeg_channels, n_channels=len(eeg_channels),
         remove_dc=True, block=True)
spectrum = raw.compute_psd()
spectrum.plot(average=True, block=True)

# ############### IMPORT DATA & SIGNAL PROCESSING ###############
# for file_name in raw_data_list:
#     if file_name.endswith('.fif') and file_name.startswith('071422_Dual_Nback_Test_CG_PM_1'):
#         # files with .fif format are manually inspected filtered raw data
#         # import the annotations about BAD (artifact contaminated) channels and segments from .fif
#         raw_path_annot = os.path.join(os.path.join(os.getcwd(), 'data/raw_data'), file_name)
#         raw_annot = setup(raw_path_annot, montage_path, mode='Binary')
#         onset, duration, description = raw_annot.get_annotation_info()
#         print(description)

#         # import raw data from .vhdr file and montage from .bvef file
#         raw_path = os.path.join(os.path.join(os.getcwd(), 'data/raw_data'), file_name.replace('.fif','.vhdr'))
#         montage_path = os.path.join(os.getcwd(), 'data/Workspaces_Montages/active electrodes/actiCAP for LiveAmp 32 Channel','CLA-32.bvef')
#         raw = setup(raw_path, montage_path, mode='Dual')
#         # set custom annotations from .fif on the raw
#         raw.set_annotation(raw.raw, onset=onset, duration=duration, description=description)
        
#         # seperate fullhead EEG and forehead EEG raw and rerefernece them
#         raw.get_brainvision_raw()
#         raw.bv_raw.load_data()
#         raw.bv_raw.set_eeg_reference('average') # CAR
#         raw.get_e_tattoo_raw()
#         raw.et_raw.load_data()
#         raw.et_raw.set_eeg_reference(ref_channels=['A1', 'A2']) # behind of ears reference

#         # raw data preprocessing: downsampling, nf, bpf
#         bv_filters = preprocessing.Filtering(raw.bv_raw, l_freq=1, h_freq=50)
#         raw.bv_raw = bv_filters.external_artifact_rejection()
#         et_filters = preprocessing.Filtering(raw.et_raw, l_freq=1, h_freq=50)
#         raw.et_raw = et_filters.external_artifact_rejection()

#         # eye blink artifact rejection through ICA
#         bv_ica = preprocessing.Indepndent_Component_Analysis(raw.bv_raw, n_components=8)
#         et_ica = preprocessing.Indepndent_Component_Analysis(raw.et_raw, n_components=4)
#         bv_eog_evoked = bv_ica.create_physiological_evoked()
#         et_eog_evoked = et_ica.create_physiological_evoked()
#         bv_ica.perfrom_ICA()
#         et_ica.perfrom_ICA()

#         # plot filtered raw data
#         # fig = raw.bv_raw.plot()
#         # fig = raw.et_raw.plot()
#         # plt.show()

#         # create events from raw and nback game report timestamps
#         meas_date = str(raw.raw.info['meas_date']) # get raw data measurement timestamp
#         # text reformatting to compare later with nback event timestamps
#         recorder_meas_time = meas_date[0:4]+meas_date[5:7]+meas_date[8:10]+meas_date[11:19].replace(':','')
#         # list of paths of nback reports in local data folder
#         report_list = os.listdir(os.path.join(os.getcwd(), 'data/reports'))
#         for report_name in report_list:
#             report_path = os.path.join(os.path.join(os.getcwd(), 'data/reports'), report_name)
#             # get nback reportlog timestamp from file name
#             report_log_time = report_path.split('_',1)[1][0:15].replace('_', '')
#             # search the nback report file that logged when the raw is recorded by comparing timestamps
#             if abs(int(recorder_meas_time)-int(report_log_time)) < 60:
#                 # get events from the nback report
#                 resampled_freq = 200
#                 nback_events = raw.get_events_from_nback_report(report_path=report_path, fs=resampled_freq)
#         # custom event dictionary
#         event_dict = {'0-back': 0, '1-back': 1, '2-back': 2}
#         events_list.append(nback_events)
#         # fig = mne.viz.plot_events(nback_events, event_id=event_dict, sfreq=resampled_freq, first_samp=raw.bv_raw.first_samp)
        
#         '''
#         bv_theta = preprocessing.Filtering(raw.bv_raw, 4, 7)
#         bv_alpha = preprocessing.Filtering(raw.bv_raw, 8, 13)
#         bv_beta = preprocessing.Filtering(raw.bv_raw, 14, 30)
#         bv_theta_raw = bv_theta.bandpass()
#         bv_alpha_raw = bv_alpha.bandpass()
#         bv_beta_raw = bv_beta.bandpass()
#         fig = bv_theta_raw.plot()
#         fig = bv_alpha_raw.plot()
#         fig = bv_beta_raw.plot()
#         plt.show()

#         bv_theta_epochs = mne.Epochs(bv_theta_raw, events=nback_events, event_id=event_dict, tmin=-0.2, tmax=1.8, preload=True, picks='eeg')
#         bv_alpha_epochs = mne.Epochs(bv_alpha_raw, events=nback_events, event_id=event_dict, tmin=-0.2, tmax=1.8, preload=True, picks='eeg')
#         bv_beta_epochs = mne.Epochs(bv_beta_raw, events=nback_events, event_id=event_dict, tmin=-0.2, tmax=1.8, preload=True, picks='eeg')
#         fig = bv_alpha_epochs['0-back'].plot_image(picks='eeg',combine='mean')
#         fig = bv_alpha_epochs['1-back'].plot_image(picks='eeg',combine='mean')
#         fig = bv_alpha_epochs['2-back'].plot_image(picks='eeg',combine='mean')
#         fig = bv_beta_epochs['0-back'].plot_image(picks='eeg',combine='mean')
#         fig = bv_beta_epochs['1-back'].plot_image(picks='eeg',combine='mean')
#         fig = bv_beta_epochs['2-back'].plot_image(picks='eeg',combine='mean')
#         fig = bv_theta_epochs['0-back'].plot_image(picks='eeg',combine='mean')
#         fig = bv_theta_epochs['1-back'].plot_image(picks='eeg',combine='mean')
#         fig = bv_theta_epochs['2-back'].plot_image(picks='eeg',combine='mean')
#         '''
#         # create epochs from raw data with the events
#         bv_epochs = mne.Epochs(raw=raw.bv_raw, events=nback_events, event_id=event_dict, tmin=-0.2, tmax=1.8, preload=True, picks='eeg') # BAD epochs are automatically dropped
#         bv_epochs.equalize_event_counts() # equalize the num of epochs per event
#         bv_epochs_list.append(bv_epochs)
#         et_epochs = mne.Epochs(raw=raw.et_raw, events=nback_events, event_id=event_dict, tmin=-0.2, tmax=1.8, preload=True, picks=['Fp1','Fp2','F7','F8']) # BAD epochs are automatically dropped
#         et_epochs.equalize_event_counts() # equalize the num of epochs per event
#         et_epochs_list.append(et_epochs)

# # concatenate all epochs from different trials
# all_bv_epochs = mne.concatenate_epochs(bv_epochs_list)
# all_et_epochs = mne.concatenate_epochs(et_epochs_list)
# # print(len(all_bv_epochs))
# # print(len(all_et_epochs))

# ############### FEATURE EXTRACTION & SELECTION ###############
# bv_x = feature_extraction.eeg_power_band(all_bv_epochs, mean=False)
# et_x = feature_extraction.eeg_power_band(all_et_epochs,mean=False)
# print(bv_x.shape)
# print(et_x.shape)
# y = all_bv_epochs.events[:,2]
# print(y.shape)
# bv_X_train, Y_train, bv_X_test, Y_test = feature_extraction.create_train_test_sets(bv_x, y, 0.2)
# et_X_train, Y_train, et_X_test, Y_test = feature_extraction.create_train_test_sets(et_x, y, 0.2)

# # print(len(et_X_train), len(Y_train), len(et_X_test), len(Y_test))


# ############### CLASSIFICATION ###############
# ############### Random forest classification
# # pipe_RF = make_pipeline(FunctionTransformer(feature_extraction.eeg_power_band, validate=False), RandomForestClassifier(n_estimators=100, random_state=42))
# pipe_RF = make_pipeline(RandomForestClassifier(n_estimators=50, random_state=42))
# pipe_RF.fit(bv_X_train, Y_train)

# # Test
# Y_pred = pipe_RF.predict(bv_X_test)
# # Assess the results
# acc = accuracy_score(Y_test, Y_pred)
# print('BV Random Forest Classifier')
# print('Accuracy score: {}'.format(acc))
# print('Confusion Matrix:')
# print(confusion_matrix(Y_test, Y_pred))
# print('Classification Report:')
# print(classification_report(Y_test, Y_pred, target_names=event_dict.keys()))

# # ET ver
# pipe_RF.fit(et_X_train, Y_train)

# # Test
# Y_pred = pipe_RF.predict(et_X_test)
# # Assess the results
# acc = accuracy_score(Y_test, Y_pred)
# print('ET Random Forest Classifier')
# print('Accuracy score: {}'.format(acc))
# print('Confusion Matrix:')
# print(confusion_matrix(Y_test, Y_pred))
# print('Classification Report:')
# print(classification_report(Y_test, Y_pred, target_names=event_dict.keys()))

# ############### Linear discriminant analysis
# oa = OAS(store_precision=False, assume_centered=False)
# # pipe_LDA = make_pipeline(FunctionTransformer(feature_extraction.eeg_power_band, validate=False), LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=oa))
# pipe_LDA = make_pipeline(LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=oa))
# pipe_LDA.fit(bv_X_train, Y_train)
# # Test
# Y_pred = pipe_LDA.predict(bv_X_test)
# # Assess the results
# acc = accuracy_score(Y_test, Y_pred)
# print('BV Linear Discriminant Analysis')
# print('Accuracy score: {}'.format(acc))
# print('Confusion Matrix:')
# print(confusion_matrix(Y_test, Y_pred))
# print('Classification Report:')
# print(classification_report(Y_test, Y_pred, target_names=event_dict.keys()))

# # ET
# pipe_LDA.fit(et_X_train, Y_train)
# # Test
# Y_pred = pipe_LDA.predict(et_X_test)
# # Assess the results
# acc = accuracy_score(Y_test, Y_pred)
# print('ET Linear Discriminant Analysis')
# print('Accuracy score: {}'.format(acc))
# print('Confusion Matrix:')
# print(confusion_matrix(Y_test, Y_pred))
# print('Classification Report:')
# print(classification_report(Y_test, Y_pred, target_names=event_dict.keys()))

# ###############  K-nearest neighbors classification
# k_range = np.arange(1,20)
# weight_fuc = ['uniform', 'distance']
# opt_KNN_param = []
# acc_max = 0
# opt_Y_pred = []
# for KNN_param in [(x, y) for x in k_range for y in weight_fuc]:
#     pipe_KNN = make_pipeline(KNeighborsClassifier(n_neighbors=KNN_param[0],weights=KNN_param[1]))
#     pipe_KNN.fit(bv_X_train, Y_train)
#     # Test
#     Y_pred = pipe_KNN.predict(bv_X_test)
#     # Assess the results
#     acc = accuracy_score(Y_test, Y_pred)
#     f1 = f1_score(Y_test, Y_pred, average='weighted', labels=np.unique(Y_pred))
#     if acc_max < acc and f1 > 0.0:
#         acc_max = acc
#         opt_KNN_param = KNN_param
#         opt_Y_pred = Y_pred

# print('BV KNN Classifier')
# print(opt_KNN_param)
# print('Accuracy score: {}'.format(acc))
# print('Confusion Matrix:')
# print(confusion_matrix(Y_test, opt_Y_pred))
# print('Classification Report:')
# print(classification_report(Y_test, opt_Y_pred, target_names=event_dict.keys()))

# # ET
# opt_KNN_param = []
# acc_max = 0
# opt_Y_pred = []
# for KNN_param in [(x, y) for x in k_range for y in weight_fuc]:
#     pipe_KNN = make_pipeline(KNeighborsClassifier(n_neighbors=KNN_param[0],weights=KNN_param[1]))
#     pipe_KNN.fit(et_X_train, Y_train)
#     # Test
#     Y_pred = pipe_KNN.predict(et_X_test)
#     # Assess the results
#     acc = accuracy_score(Y_test, Y_pred)
#     f1 = f1_score(Y_test, Y_pred, average='weighted', labels=np.unique(Y_pred))
#     if acc_max < acc and f1 > 0.0:
#         acc_max = acc
#         opt_KNN_param = KNN_param
#         opt_Y_pred = Y_pred

# print('ET KNN Classifier')
# print(opt_KNN_param)
# print('Accuracy score: {}'.format(acc))
# print('Confusion Matrix:')
# print(confusion_matrix(Y_test, opt_Y_pred))
# print('Classification Report:')
# print(classification_report(Y_test, opt_Y_pred, target_names=event_dict.keys()))

# ############### Support vector machine
# C_range = np.logspace(-2, 10, 13)
# gamma_range = np.logspace(-9, 3, 13)
# opt_C_gamma_param = []
# acc_max = 0
# opt_Y_pred = []
# for C_gamma_param in [(x, y) for x in C_range for y in gamma_range]:
#     pipe_SVM = make_pipeline(StandardScaler(), SVC(kernel='rbf',C=C_gamma_param[0],gamma=C_gamma_param[1]))
#     pipe_SVM.fit(bv_X_train, Y_train)
#     # Test
#     Y_pred = pipe_SVM.predict(bv_X_test)
#     acc = accuracy_score(Y_test, Y_pred)
#     f1 = f1_score(Y_test, Y_pred, average='weighted', labels=np.unique(Y_pred))
#     if acc_max < acc and f1 > 0.0:
#         acc_max = acc
#         opt_C_gamma_param = C_gamma_param
#         opt_Y_pred = Y_pred

# print('BV SVM Classifier')
# print(opt_C_gamma_param)
# print('Accuracy score: {}'.format(acc_max))
# print('Confusion Matrix:')
# print(confusion_matrix(Y_test, opt_Y_pred))
# print('Classification Report:')
# print(classification_report(Y_test, opt_Y_pred, target_names=event_dict.keys()))

# # ET
# opt_C_gamma_param = []
# acc_max = 0
# opt_Y_pred = []
# for C_gamma_param in [(x, y) for x in C_range for y in gamma_range]:
#     pipe_SVM = make_pipeline(StandardScaler(), SVC(kernel='rbf',C=C_gamma_param[0],gamma=C_gamma_param[1]))
#     pipe_SVM.fit(et_X_train, Y_train)
#     # Test
#     Y_pred = pipe_SVM.predict(et_X_test)
#     acc = accuracy_score(Y_test, Y_pred)
#     f1 = f1_score(Y_test, Y_pred, average='weighted', labels=np.unique(Y_pred))
#     if acc_max < acc and f1 > 0.0:
#         acc_max = acc
#         opt_C_gamma_param = C_gamma_param
#         opt_Y_pred = Y_pred

# print('ET SVM Classifier')
# print(opt_C_gamma_param)
# print('Accuracy score: {}'.format(acc_max))
# print('Confusion Matrix:')
# print(confusion_matrix(Y_test, opt_Y_pred))
# print('Classification Report:')
# print(classification_report(Y_test, opt_Y_pred, target_names=event_dict.keys()))
       

