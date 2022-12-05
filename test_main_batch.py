'''
F2022 BCI Course Term-Project
ErrP Group 3
Authors: Heeyong Huh, Hyonyoung Shin, Susmita Gangopadhyay
'''

import os
import numpy as np 
import mne 
import preprocessing
from sklearn.cross_decomposition import CCA
import numpy.matlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, auc 
from sklearn.metrics import plot_confusion_matrix

# Define data directory
data_dir = 'C:/Users/mcvai/F2022_BCI/ErrP_data'

# nested list: [subj] [electode_type] 
offline_files = [ [[] for i in range(2)] for i in range(3) ]
online1_files = [ [[] for i in range(2)] for i in range(3) ]
online2_files = [ [[] for i in range(2)] for i in range(3) ]
online_files = [ [[] for i in range(2)] for i in range(3) ]

# Categorize input files 
for root, dirs, files in os.walk(data_dir):
    electrode_type = 99 
    for file in files:
        if file.endswith(".vhdr"):
            subj = int(file.split("_")[0]) - 6

            if "_rest" in root:
                continue
            if 'Gel' in root:
                electrode_type = 0
            elif 'POLiTAG' in root:
                electrode_type = 1
            else: 
                electrode_type = 99  # should not reach 

            assert electrode_type in [0, 1]
            
            if 'offline' in root:
                offline_files[subj][electrode_type].append(os.path.join(root, file))

            elif 'online_S1' in root:
                online1_files[subj][electrode_type].append(os.path.join(root, file))

            elif 'online_S2' in root:
                online2_files[subj][electrode_type].append(os.path.join(root, file))
            

# Input parsing sanity check
for subj in range(0, 3):
    print("Subject number " + str(subj+6))
    for electrode_type in range(0, 2):
        online_files[subj][electrode_type] = online1_files[subj][electrode_type] + online2_files[subj][electrode_type] # concatenate

    print(str(len(offline_files[subj][0])) + " offline Gel files found")
    print(str(len(offline_files[subj][1])) + " offline POLITAG files found")
    print(str(len(online_files[subj][0])) + " online Gel files found")
    print(str(len(online_files[subj][1])) + " online POLITAG files found")

def vhdr2numpy(filename, politag: bool, spatial_filter: str, t_epoch):
    assert spatial_filter.upper() in ['CAR', 'CCA', 'CAR+CCA']

    # get international 10-20 montage
    montage = mne.channels.make_standard_montage('standard_1020')

    raw = mne.io.read_raw_brainvision(filename, preload=True)

    # replace some channel names for MNE functions
    if not politag: 
        new_names = dict((ch_name, ch_name.replace('Z', 'z').replace('FP','Fp'))for ch_name in raw.ch_names)
        raw.rename_channels(new_names)
    else:
        wrong_ch_names = raw.ch_names
        correct_ch_names = ['Fz', 'FC5', 'FC1', 'FC6', 'C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'FCz', 'FC2']
        # ['Fz', 'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'FCz']
        new_names = dict(zip(wrong_ch_names, correct_ch_names))
        raw.rename_channels(new_names)

    # set montage
    # raw.set_channel_types({'Status': 'misc'})
    raw.drop_channels(['Status'], on_missing='ignore')
    raw.set_montage(montage)
    
    # apply notch (60, 120 Hz) and bandpass filter (1-40 Hz)
    filters = preprocessing.Filtering(raw, l_freq=2, h_freq=12)
    raw = filters.external_artifact_rejection()

    # get MNE annotations from raw data
    onset = raw.annotations.onset
    duration = raw.annotations.duration
    description = raw.annotations.description

    # renaming decriptions
    new_description = []
    for i in range(len(description)):
        print(description[i])
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
    
    # create and set new annotations
    my_annot = mne.Annotations(onset=onset, duration=duration, description=new_description)
    raw.set_annotations(my_annot)

    # prepare common average reference (CAR)
    if spatial_filter == 'CAR' or spatial_filter == 'CAR+CCA':
        print("CAR performed!")
        raw = raw.copy().set_eeg_reference('average')

    # create custom MNE events dictionary using only events that we care about
    custom_dict = {'Error trial': 10, 'Correct trial & Target reached': 13, 'Correct trial': 9}
    # choose needed annotations and convert them to events
    events, event_dict = mne.events_from_annotations(raw, event_id=custom_dict)

    # create epochs from MNE raw and events
    epoc = mne.Epochs(raw, events, event_id=event_dict, tmin=t_epoch[0], tmax=t_epoch[1], baseline=(t_epoch[0], 0), preload=True, picks='eeg')

    # creat epochs for only correct and error trials
    correct = epoc['Correct trial & Target reached', 'Correct trial']
    error = epoc['Error trial']

    # epoching sanity check 
    print(str(len(correct)) + " correct trials detected in this run.")
    print(str(len(error)) + " error trials detected in this run.") 

    if spatial_filter == 'CCA' or spatial_filter == 'CAR+CCA': 
        n_comp = 5
        print("CCA performed!")
        # m samples * n channels * k trials 
        X_correct = correct.get_data()
        print(X_correct.shape)
        X_error = error.get_data()
        X = np.append(X_correct, X_error, axis=0).T

        # n * m * k
        X = np.transpose(X, [1, 0, 2]) 
        X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2])).T
        print("Dimension of X: " + str(X.shape))

        grand_avg_correct = correct.average()
        grand_avg_error = error.average() 

        # print("Dimension of grand_avg_correct: " + str(grand_avg_correct.get_data(picks=['FCz', 'FC1', 'FC2', 'Cz', 'Fz']).shape))  # correct
    
        Y_correct = np.matlib.repmat(grand_avg_correct.get_data(picks=['FCz', 'FC1', 'FC2', 'Cz', 'Fz']), 1, X_correct.shape[0])
        Y_error = np.matlib.repmat(grand_avg_error.get_data(picks=['FCz', 'FC1', 'FC2', 'Cz', 'Fz']), 1, X_error.shape[0])
        Y = np.append(Y_correct, Y_error, axis=1).T
        print("Dimension of Y: " + str(Y.shape))

        # apply CCA
        cca = CCA(n_components=n_comp)
        cca.fit(X, Y)

        W_s = cca.x_rotations_
        print("Dimension of W_s: " + str(W_s.shape))
    
        # Visualize and choose spatial filter
        # fig, axs = plt.subplots(nrows=1, ncols=5)
        # for i in range(n_comp):
        #         mne.viz.plot_topomap(W_s.T[i], raw.info, axes=axs[i], size=3,vlim=(0, 1), show=False)
        # im, cn = mne.viz.plot_topomap(W_s.T[n_comp], raw.info, axes=axs[n_comp], size=3, vlim=(-1, 1), show=False)
        
        # cbar = plt.colorbar(im, ax=axs)
        # tick_font_size = 22
        # cbar.ax.tick_params(labelsize=tick_font_size)
        # cbar.set_label('Weight (A.U.)',fontsize=22)
        
        # plt.rcParams.update({'font.size': 20})
        # fig.suptitle('CAR + CCA Components',fontsize=40)
        # plt.show()

        # cca_num = int(input("Choose CCA component - use 0 indexing!: "))
        cca_num = 0 
        W = W_s.T[cca_num]

        # Apply the spatial weights 
        raw_cca = (raw.get_data().T * W).T
        raw = mne.io.RawArray(raw_cca,raw.info)
        epoc = mne.Epochs(raw, events, event_id=event_dict, tmin=t_epoch[0], tmax=t_epoch[1], baseline=(t_epoch[0], 0), preload=True, picks='eeg')
        correct = epoc['Correct trial & Target reached', 'Correct trial']
        error = epoc['Error trial']

        # epoching sanity check (after CCA)
        print("After CCA complete: ")
        print(str(len(correct)) + " correct trials detected in this run.")
        print(str(len(error)) + " error trials detected in this run.") 

    print(epoc.ch_names)
    x = epoc.get_data()

    feature_names = []
    assert x.shape[1] == len(epoc.ch_names)
    
    for ch in range(x.shape[1]):
        for sample in range(256, x.shape[2]):
            feature_names.append(str(epoc.ch_names[ch]) + "_" + str(sample-256))

    x = x[:, :, 256:]
    x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))  # trials * (channels*samples)
    
    print(feature_names)

    y = epoc.events[:, 2]
    y = [1 if yy in [9, 13] else 0 for yy in y]  # make labels binary

    return x, y, feature_names
        


models = ['svm-rbf', 'lda', 'multilog', 'rf']
validate = False
if validate: 
    for subj in range(1,2 ):
        for electrode_type in range(1, 2): # 11/29/2022 using gel only for now 

            runs = range(len(offline_files[subj][electrode_type]))
            
            validation_accuracies = [[] for i in range(len(models))]
            validation_models = [[] for i in range(len(models))]

            for test_run in runs:
                
                train_runs = [i for i in runs if i != test_run] # use all other runs to train 
                # print(test_run, train_runs)

                xs = []; Y_train = []
                for train_run in train_runs: 

                    if electrode_type == 1:
                        politag = True
                    else:
                        politag = False

                    x, y, f = vhdr2numpy(offline_files[subj][electrode_type][train_run], politag, 'CAR', [-0.5, 0.5])
                    # print(x.shape)
                    # print(y)
                    # print(len(y))

                    scaler = StandardScaler() 
                    scaler.fit(x)
                    x = scaler.transform(x)
                    print(x)

                    xs.append(x) 
                    Y_train = Y_train + y
                
                X_train = np.vstack(xs)
                X_train = pd.DataFrame(data=X_train, columns=f)
                Y_train = pd.DataFrame(data=Y_train)

                X_test, Y_test, f = vhdr2numpy(offline_files[subj][electrode_type][test_run], False, 'CAR', [-0.5, 0.5])
                X_test = pd.DataFrame(data=X_test, columns=f)
                Y_test = pd.DataFrame(data=Y_test)

                for i, model in enumerate(models):

                    if model == 'svm-rbf': 
                        # scaler = StandardScaler()  # normalization: zero mean, unit variance
                        # scaler.fit(X_train)  # scaling factor determined from the training set
                        # X_train = scaler.transform(X_train)
                        # X_test = scaler.transform(X_test)  # apply the same scaling to the test set 

                        # SVM (support vector classifier SVC)
                        clf = SVC(kernel='rbf')  # default C=1
                        
                    elif model == 'lda':
                         
                        clf = LDA() # solver = svd is good for large number of features, eigenvector is optimal though

                    elif model == 'multilog':
                        # scaler = StandardScaler()  # normalization: zero mean, unit variance
                        # scaler.fit(X_train)  # scaling factor determined from the training set
                        # X_train = scaler.transform(X_train)
                        # X_test = scaler.transform(X_test)  # apply the same scaling to the test set 
                        
                        clf = LogisticRegression()

                    elif model == 'rf':
                        # scaler = StandardScaler()  # normalization: zero mean, unit variance
                        # scaler.fit(X_train)  # scaling factor determined from the training set
                        # X_train = scaler.transform(X_train)
                        # X_test = scaler.transform(X_test)  # apply the same scaling to the test set 
                        
                        clf = RandomForestClassifier()

                    # Train the model using the training sets
                    clf.fit(X_train, Y_train.values.ravel())
                    # Make predictions using the test set
                    Y_pred = clf.predict(X_test)
                    # Prediction accuracy on training data
                    print('Training accuracy =', clf.score(X_test, Y_test), '\n')

                    validation_accuracies[i].append(float(clf.score(X_test, Y_test)))
                    validation_models[i].append(clf) 

            print(str(len(runs)) + "-fold cross-validation accuracy: " + str(np.mean(validation_accuracies[i])))

            model_cv_accuracies = [sum(sub_list) / len(sub_list) for sub_list in validation_accuracies]
            plt.figure()
            plt.bar(models, model_cv_accuracies)
            plt.title("Subject " + str(subj) + [' Gel ', ' POLITAG '][electrode_type])
            plt.ylabel('Accuracy')
            plt.show()
            best_model = model_cv_accuracies.index(max(model_cv_accuracies))
            best_model = models[best_model]
            print("")

    print(validation_accuracies)

test = True
if test:
    ## After deciding on best model, train on all offline datasets 
    for subj in range(1,2):
        for electrode_type in range(1,2):
            if electrode_type == 1:
                politag = True
            else:
                politag = False

            train_runs = range(len(offline_files[subj][electrode_type]))

            xs = []; Y_train = []
            print(train_runs)
            for train_run in train_runs: 
                x, y, f = vhdr2numpy(offline_files[subj][electrode_type][train_run], politag, 'CAR', [-0.5, 0.5])

                scaler = StandardScaler() 
                scaler.fit(x)
                x = scaler.transform(x)

                xs.append(x) 

                Y_train = Y_train + y

            X_train = np.vstack(xs)
            X_train = pd.DataFrame(data=X_train, columns=f)
            Y_train = pd.DataFrame(data=Y_train)

            xs = []; Y_test = [] 
            test_runs = range(len(online_files[subj][electrode_type]))
            print(test_runs)
            for test_run in test_runs: 
                x, y, f = vhdr2numpy(online_files[subj][electrode_type][test_run], politag, 'CAR', [-0.5, 0.5])
                
                scaler = StandardScaler() 
                scaler.fit(x)
                x = scaler.transform(x)

                xs.append(x)
                Y_test = Y_test + y

            X_test = np.vstack(xs)
            X_test = pd.DataFrame(data=X_test, columns=f)
            Y_test = pd.DataFrame(data=Y_test)
            
            clf = RandomForestClassifier() 
            # clf = SVC(kernel='rbf')
            
            # Train the model using the training sets
            clf.fit(X_train, Y_train.values.ravel())
            # Make predictions using the test set
            Y_pred = clf.predict(X_test)
            prob = clf.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(Y_test, prob[:,1])
            auc = auc(fpr, tpr)  # area under ROC curve
            print('\nThresholds:')
            print(thresholds)
            print('\nAUC = ' + str(auc))
            print('\nThresholds:')
            print(thresholds)
            print('\nAUC = ' + str(auc))
            print('\nConfusion matrix:')
            print(confusion_matrix(Y_test, Y_pred))
            
            print("Accuracy")
            print(clf.score(X_test, Y_test))

            plt.figure()
            plot_confusion_matrix(clf, X_test, Y_test)
            plt.show()

            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.plot(fpr, tpr)
            plt.xlim([0, 1])
            plt.ylim([0, 1.05])
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.show()

            print(clf.score(X_test, Y_test))



            importances = clf.feature_importances_
            std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
        
            forest_importances = pd.Series(importances, index=X_train.columns)

            forest_importances.sort_values(ascending=False, inplace=True)
            # print(forest_importances.head(20))
            fig, ax = plt.subplots()
            forest_importances.plot.bar(yerr=std, ax=ax)
            ax.set_title("Feature importances using MDI")
            ax.set_ylabel("Mean decrease in impurity")
            fig.tight_layout()
            fig.show()

# X_test, Y_test, f = vhdr2numpy(online_files[0][0][0], False, 'CCA', [-0.5, 0.5])

# print("Testset accuracy: ")   
# print(clf.score(X_test, Y_test))

bonus = False 
if bonus:
    ## After deciding on best model, train on all offline datasets 
    for subj in range(1, 2):
        for electrode_type in range(1):
            if electrode_type == 1:
                politag = True
            else:
                politag = False

            train_runs = range(len(offline_files[subj][electrode_type]))

            xs = []; Y_train = []
            for train_run in train_runs: 
                x, y, f = vhdr2numpy(offline_files[subj][electrode_type][train_run], politag, 'CAR', [-0.5, 0.5])
                
                #scaler = StandardScaler() 
                #scaler.fit(x)
                #x = scaler.transform(x)
                
                xs.append(x) 
                Y_train = Y_train + y

            online1_runs = range(len(online1_files[subj][electrode_type]))
            for train_run in online1_runs:
                x, y, f = vhdr2numpy(online1_files[subj][electrode_type][train_run], politag, 'CAR', [-0.5, 0.5])
                xs.append(x) 
                Y_train = Y_train + y

            X_train = np.vstack(xs)
            X_train = pd.DataFrame(data=X_train, columns=f)
            Y_train = pd.DataFrame(data=Y_train)

            xs = []; Y_test = [] 
            test_runs = range(len(online2_files[subj][electrode_type]))
            print(test_runs)
            for test_run in test_runs: 
                x, y, f = vhdr2numpy(online2_files[subj][electrode_type][test_run], politag, 'CAR', [-0.5, 0.5])
                
                #scaler = StandardScaler() 
                #scaler.fit(x)
                #x = scaler.transform(x)
                
                xs.append(x)
                Y_test = Y_test + y

            X_test = np.vstack(xs)
            X_test = pd.DataFrame(data=X_test, columns=f)
            Y_test = pd.DataFrame(data=Y_test)
            
            clf = RandomForestClassifier() 
            # clf = SVC(kernel='rbf')
            
            # Train the model using the training sets
            clf.fit(X_train, Y_train.values.ravel())
            # Make predictions using the test set
            Y_pred = clf.predict(X_test)
            prob = clf.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(Y_test, prob[:,1])
            auc = auc(fpr, tpr)  # area under ROC curve
            print('\nThresholds:')
            print(thresholds)
            print('\nAUC = ' + str(auc))
            print('\nConfusion matrix:')
            print(confusion_matrix(Y_test, Y_pred))
            
            print("Accuracy")
            print(clf.score(X_test, Y_test))

            plt.figure()
            plot_confusion_matrix(clf, X_test, Y_test)
            plt.show() 

            print('\nClassification report:')
            print(classification_report(Y_test, Y_pred))
            
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.plot(fpr, tpr)
            plt.xlim([0, 1])
            plt.ylim([0, 1.05])
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.show()

            

            importances = clf.feature_importances_
            std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
             
        
            forest_importances = pd.Series(importances, index=X_train.columns)
            # times = [i.split('_')[1] for i in X_train.columns.tolist()] 
            # channels = [i.split('_')[0] for i in X_train.columns.tolist()] 

            # layout = mne.channels.read_layout("EEG1005")
            # print(layout)

            # picks = [] 
            # for channel in list(set(channels)): 
            #     picks.append(layout.pos[layout.names.index(channel)][0:2])
            # print("picks")
            # print(np.array(picks))

            # for i in list(set(times)): 
            #     ti = [ii for ii, x in enumerate(times) if x == i]
            #     channels_at_that_time = [channels[i] for i in ti]
            #     values_at_that_time = [forest_importances[i] for i in ti]
            #     mne.viz.plot_topomap(np.array(values_at_that_time).T, np.array(picks))

            forest_importances.sort_values(ascending=False, inplace=True)
            std = pd.Series(std).reindex(forest_importances)

            print(forest_importances.head(20))

            fig, ax = plt.subplots()
            forest_importances.head(15).plot.bar(yerr=std.head(15), ax=ax)
            ax.set_title("Feature importances using MDI")
            ax.set_ylabel("Mean decrease in impurity")
            # fig.tight_layout()
            plt.show()