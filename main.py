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
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns

# Classifier learning packages
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import OAS
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression 

###### 1 Initialization ######
# define data directory
data_dir = os.path.join(os.getcwd(), 'ErrP_data')
n_subject = 3
n_electrode_type = 2

# get file paths (nested list: [subj] [electode_type]) 
offline_files, online1_files, online2_files, online_files = setup.get_file_paths(data_dir, n_subject, n_electrode_type)

# get international 10-20 montage
montage = mne.channels.make_standard_montage('standard_1020')

# plt.rcParams.update({'font.size': 20})

###### 2 Offline Analysis ######
offline_analysis = False  # if we are not using CCA

if offline_analysis: 
    print('Start offline analysis: ')
    plotting = int(input("Ignore POLiTAG (yes [0], no [1]: "))
    if plotting == 0:
        n_subject = 3
        n_electrode_type = 1

    print(offline_files[0][0])
    for subj_ind in range(n_subject):
        subject_num = subj_ind+6
        print('Subject number: ' + str(subject_num))

        for electrode_type_ind in range(n_electrode_type): 
            if plotting == 1:
                electrode_type_ind = 1
            electrode_type = ['Gel','POLiTAG'][electrode_type_ind]
            print('Electrode type: ' + str(electrode_type))

            #### debugging
            # subj_ind = 0
            # electrode_type_ind = 1
            # electrode_type = ['Gel','POLiTAG'][electrode_type_ind]
            # ####
            n_run = len(offline_files[subj_ind][electrode_type_ind])
            runs = offline_files[subj_ind][electrode_type_ind]

            ### 2.1 get preprocessing setup ###
            # baseline correction paremeter
            baseline_option = int(input("Baseline correction setups (Traditional [0], Regression-based [1] : "))
            if baseline_option==0:
                reg_base_on = False
            else:
                reg_base_on = True
            
            # temporal filter parameters
            low_f_c = 4
            high_f_c = 12
            epoch_tmin = -0.3
            epoch_tmax = 0.5
            n_cca_comp = 5
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
                W_s = cca.canonical_correlation_analysis(runs, montage, preprocessing_param, electrode_type=electrode_type, reg_base_on=reg_base_on, car_on=car_on, show_components=True)
                # choose representing CCA component as a sapatial filter
                cca_num = list(input("Choose CCA components: ").split(','))
                cca_num = [int(x) for x in cca_num]
                # W_cca = W_s.T[cca_num]
            else:
                cca_on = False

            # epoch initialization
            epochs = []
            epochs_corr = []
            epochs_err = []
            event_list = []

            for run in runs:
                ### 2.2 import raw data ###
                raw = setup.load_raw_data(path=run, montage=montage, electrode_type=electrode_type)

                ### 2.3 preprocessing ###
                # apply notch (60, 120 Hz) and bandpass filter (1-30 Hz)
                filters = preprocessing.Filtering(raw, l_freq=low_f_c, h_freq=high_f_c)
                raw = filters.external_artifact_rejection()
                events, event_dict = setup.create_events(raw)

                if car_on == True:
                    # common average reference (CAR)
                    raw = raw.set_eeg_reference('average')
                if cca_on == True:
                    # apply CCA based spatial filter
                    W_cca = cca.combine_cca_components(W_s, cca_num, raw.info)
                    print(W_cca.shape)
                    print(W_cca) # check the filter weights
                    raw_cca = (raw.get_data().T * W_cca).T
                    raw = mne.io.RawArray(raw_cca,raw.info)

                ### 2.4 create epochs ###
                # create epochs from MNE events
                
                epoc = mne.Epochs(raw, events, event_id=event_dict, tmin=epoch_tmin, tmax=epoch_tmax, baseline=None, preload=True, picks='eeg')

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
            
            fc_chs = ['FCz', 'FC1', 'FC2', 'Cz', 'Fz']
            baseline = (epoch_tmin, 0)
            if reg_base_on == False:
                all_correct = all_correct.apply_baseline(baseline)
                all_error = all_error.apply_baseline(baseline)
            else:
                corr_predictor = all_epochs.events[:, 2] != all_epochs.event_id['Error trial']
                err_predictor = all_epochs.events[:, 2] == all_epochs.event_id['Error trial']
                baseline_predictor = (all_epochs.copy().crop(*baseline)
                        .pick_channels(fc_chs)
                        .get_data()     # convert to NumPy array
                        .mean(axis=-1)  # average across timepoints
                        .squeeze()      # only 1 channel, so remove singleton dimension
                        .mean(axis=-1)  # average across channels
                        .squeeze()      # only 1 channel, so remove singleton dimension
                )
                baseline_predictor *= 1e6  # convert V → μV

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
                    reg_epoc = mne.EpochsArray(np_reg_epoc,raw.info, events=event_list[i], tmin=epoch_tmin, event_id=event_dict)
                    reg_all_epochs_list.append(reg_epoc)
                reg_all_epochs = mne.concatenate_epochs(reg_all_epochs_list)
                all_correct = reg_all_epochs['Correct trial & Target reached', 'Correct trial']
                all_error = reg_all_epochs['Error trial']

            if reg_base_on == True:
                # traditional vs regression-based baselines
                trad_corr = all_epochs['Correct trial & Target reached', 'Correct trial'].average().apply_baseline(baseline)
                trad_err = all_epochs['Error trial'].average().apply_baseline(baseline)
                
                corr_predictor = all_epochs.events[:, 2] != all_epochs.event_id['Error trial']
                err_predictor = all_epochs.events[:, 2] == all_epochs.event_id['Error trial']
                baseline_predictor = (all_epochs.copy().crop(*baseline)
                        .pick_channels(fc_chs)
                        .get_data()     # convert to NumPy array
                        .mean(axis=-1)  # average across timepoints
                        .squeeze()      # only 1 channel, so remove singleton dimension
                        .mean(axis=-1)  # average across channelss
                        .squeeze()      # only 1 channel, so remove singleton dimension
                )
                baseline_predictor *= 1e6  # convert V → μV

                design_matrix = np.vstack([corr_predictor,
                                err_predictor,
                                baseline_predictor,
                                baseline_predictor * corr_predictor]).T
                reg_model = mne.stats.linear_regression(all_epochs, design_matrix,
                                            names=["Correct", "ErrP",
                                                    "baseline",
                                                    "baseline:Correct"])

                effect_of_baseline = reg_model['baseline'].beta
                effect_of_baseline.plot(picks=fc_chs, hline=[1.], units=dict(eeg=r'$\beta$ value'),
                            titles=dict(eeg=fc_chs), selectable=False)

                reg_corr = reg_model['Correct'].beta
                reg_err = reg_model['ErrP'].beta
                kwargs = dict(picks=fc_chs, show_sensors=False, truncate_yaxis=False)
                mne.viz.plot_compare_evokeds(dict(Correct=trad_corr, ErrP=trad_err),
                                            title="Traditional", **kwargs, combine='mean')
                mne.viz.plot_compare_evokeds(dict(Correct=reg_corr, ErrP=reg_err),
                                            title="Regression-based", **kwargs, combine='mean')

                diff_traditional = mne.combine_evoked([trad_corr, trad_err], weights=[1, -1])
                diff_regression = mne.combine_evoked([reg_corr, reg_err], weights=[1, -1])
                vmin = min(diff_traditional.get_data().min(),
                        diff_regression.get_data().min()) * 1e6
                vmax = max(diff_traditional.get_data().max(),
                        diff_regression.get_data().max()) * 1e6
                topo_kwargs = dict(vlim=(vmin, vmax), ch_type='eeg',
                                times=np.linspace(0.05, 0.45, 9))


                fig = plt.figure(constrained_layout=True)
                fig.suptitle("Comparison of Potentials with Different Baseline Correction",fontsize=30)
                # create 3x1 subfigs
                subfigs = fig.subfigures(nrows=2, ncols=1)
                subfig_titles = ["Traditional", "Regression-based"]
                for row, subfig in enumerate(subfigs):
                    subfig.suptitle(subfig_titles[row],fontsize=25)
                    # create 1x10 subplots per subfig
                    axs = subfig.subplots(nrows=1, ncols=10, gridspec_kw={'width_ratios': [4,4,4,4,4,4,4,4,4,1]})
                    if row ==0:
                        diff_traditional.plot_topomap(**topo_kwargs,axes=axs,show=False)
                    else:
                        diff_regression.plot_topomap(**topo_kwargs,axes=axs,show=False)
                plt.show()

                title = "Difference in Grand Average Potential (ErrP minus Correct)"
                fig = mne.viz.plot_compare_evokeds(dict(Traditional=diff_traditional,
                                                        Regression=diff_regression),
                                                title=title, **kwargs, combine='mean')    
            
            # average evoked potential over all trials
            grand_avg_corr = all_correct.average()
            grand_avg_err = all_error.average()

            ### 2.5 analysis visulaization ###
            # time-frequency plot
            if high_f_c >= 30:
                freqs = np.logspace(*np.log10([1, 30]), num=160)
                n_cycles = freqs / 2.  # different number of cycle per frequency
                power, itc = mne.time_frequency.tfr_morlet(all_correct, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=1, n_jobs=1, picks='eeg')
                power.plot(baseline=(epoch_tmin, 0), combine='mean', mode='logratio', title='Correct Epoch Average Frontal Central Power')
                power, itc = mne.time_frequency.tfr_morlet(all_error, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=1, n_jobs=1, picks='eeg')
                power.plot(baseline=(epoch_tmin, 0), combine='mean', mode='logratio', title='ErrP Epoch Average Frontal Central Power')

            # # grand average waveform
            # grand_avg_corr.plot(titles="Correct: Average Potentials", picks='eeg',time_unit='s',gfp=False)
            # grand_avg_err.plot(titles="Error: Average Potentials", picks='eeg',time_unit='s',gfp=False)  # show difference wave
            # grand average waveform + topoplot (all ch)
            time_unit = dict(time_unit="s")
            grand_avg_corr.plot_joint(title="Correct: Average Potentials", picks='eeg', ts_args=time_unit, topomap_args=time_unit)
            grand_avg_err.plot_joint(title="Error: Average Potentials", picks='eeg',ts_args=time_unit, topomap_args=time_unit)  # show difference wave
            # grand average waveform + topoplot (fc ch)
            time_unit = dict(time_unit="s")
            grand_avg_corr.plot_joint(title="Correct: Average Potentials at Frontal Central Channels", picks=fc_chs, ts_args=time_unit, topomap_args=time_unit)
            grand_avg_err.plot_joint(title="Error: Average Potentials at Frontal Central Channels", picks=fc_chs,ts_args=time_unit, topomap_args=time_unit)  # show difference wave

            
            
            vmin = min(grand_avg_corr.get_data().min(),
                    grand_avg_err.get_data().min()) * 1e6
            vmax = max(grand_avg_corr.get_data().max(),
                    grand_avg_err.get_data().max()) * 1e6
            topo_kwargs = dict(vlim=(vmin, vmax), ch_type='eeg',
                            times=np.linspace(0.05, 0.45, 9))

            # grand average topoplots
            fig = plt.figure(constrained_layout=True)
            fig.suptitle("Grand Average Potentials",fontsize=40)
            # create 3x1 subfigs
            subfigs = fig.subfigures(nrows=2, ncols=1)
            subfig_titles = ["Correct", "ErrP"]
            for row, subfig in enumerate(subfigs):
                subfig.suptitle(subfig_titles[row],fontsize=25)
                # create 1x10 subplots per subfig
                axs = subfig.subplots(nrows=1, ncols=10, gridspec_kw={'width_ratios': [4,4,4,4,4,4,4,4,4,1]})
                if row ==0:
                    grand_avg_corr.plot_topomap(**topo_kwargs,axes=axs,show=False)
                else:
                    grand_avg_err.plot_topomap(**topo_kwargs,axes=axs,show=False)
            plt.show()

            # evokeds = dict(Correct=grand_avg_corr, ErrP=grand_avg_err) # mean only
            evokeds = dict(Correct=list(all_correct.iter_evoked()), # mean and variance
                ErrP=list(all_error.iter_evoked()))
            mne.viz.plot_compare_evokeds(evokeds, picks='FCz', combine='mean')
            mne.viz.plot_compare_evokeds(evokeds, picks=fc_chs, combine='mean')

            '''
            ### 2.6 create training data ###
            # 3-fold cross validation
            for test_fold in epochs:
                train_epochs = [i for i in epochs if i != test_fold] # use all other runs to train
                # print(test_run, train_runs)

                # xs = []; Y_train = []

                train_folds = mne.concatenate_epochs(train_epochs) # use all other runs to train
                X_train = train_folds.get_data()
                X_train = X_train.reshape(X_train.shape[0], -1)
                print(X_train.shape)

                Y_train = train_folds.events[:, 2]
                Y_train[Y_train==10] = 0
                Y_train[Y_train!=10] = 1
                # y = [1 if yy in [9, 13] else 0 for yy in y]  # make labels binary 0: err, 1: corr
                print(Y_train.shape)

                X_test = test_fold.get_data()
                X_test = X_test.reshape(X_test.shape[0], -1)
                print(X_test.shape)

                Y_test = test_fold.events[:, 2]
                Y_test[Y_test==10] = 0
                Y_test[Y_test!=10] = 1
                # y = [1 if yy in [9, 13] else 0 for yy in y]  # make labels binary 0: err, 1: corr
                print(Y_test.shape)

                ############### Linear discriminant analysis
                oa = OAS(store_precision=False, assume_centered=False)
                # pipe_LDA = make_pipeline(FunctionTransformer(feature_extraction.eeg_power_band, validate=False), LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=oa))
                pipe_LDA = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=oa))
                pipe_LDA.fit(X_train, Y_train)

                # TODO: need some debugging
                
                # Train set performance 
                Y_pred_train = pipe_LDA.predict(X_train)
                train_acc = accuracy_score(Y_train, Y_pred_train)
                print('Train Set Performance: Linear Discriminant Analysis')
                print('Accuracy score: {}'.format(train_acc))
                print('Confusion Matrix:')
                print(confusion_matrix(Y_train, Y_pred_train))
                print('Classification Report:')
                print(classification_report(Y_train, Y_pred_train, target_names=event_dict.keys()))
                
                # Test set performance
                Y_pred_test = pipe_LDA.predict(X_test)
                test_acc = accuracy_score(Y_test, Y_pred_test)
                print('Test Set Performance: Linear Discriminant Analysis')
                print('Accuracy score: {}'.format(test_acc))
                print('Confusion Matrix:')
                print(confusion_matrix(Y_test, Y_pred_test))
                print('Classification Report:')
                print(classification_report(Y_test, Y_pred_test, target_names=event_dict.keys()))
            '''


def grouped_barplot(df, cat,subcat, val, err):
    u = df[cat].unique()
    x = np.arange(len(u))
    subx = df[subcat].unique()
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    width= np.diff(offsets).mean()
    for i,gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        plt.bar(x+offsets[i], dfg[val].values, width=width, 
                label="{} {}".format(subcat, gr), yerr=dfg[err].values, capsize=5)
    plt.xlabel(cat)
    plt.ylabel(val)
    plt.xticks(x, u)
    plt.legend()

###### 3 Classification ######
models = ['SVM (RBF kernel)', 'LDA', 'LogReg', 'RandomForest']
classify_mode = 'validate'
assert classify_mode in ['validate', 'test', 'bonus'] 
## validate: performs 3-fold cross-validation on offline datasets to output average cross-validated performance metric(s)
## test: train on all offline data, test on all online data (S1 + S2)
## bonus: train on all offline data + S1, then test on online S2 data 

n_electrode_type = 2

# nested list: [subj] [electode_type]
scores = [ [[] for i in range(n_electrode_type)] for i in range(n_subject) ]
errors = [ [[] for i in range(n_electrode_type)] for i in range(n_subject) ]

for electrode_type in range(0, n_electrode_type):
    for subj in range(n_subject):
        e = 'Gel' if electrode_type == 0 else 'Politag'
        
        if classify_mode == 'validate':
            available_runs = range(len(offline_files[subj][electrode_type]))
            validation_accuracies = [[] for i in range(len(models))]
            validation_models = [[] for i in range(len(models))]

            for test_run in available_runs:
                train_runs = [i for i in available_runs if i != test_run] # hold out currently picked fold 

                xs = []; y_train = []
                for train_run in train_runs: 
                    x, y, f = setup.vhdr2numpy(offline_files[subj][electrode_type][train_run], montage, electrode_type=e, spatial_filter='CAR', t_baseline=-0.3, epoch_window=[0.2, 0.5], spectral_window=[2,12])
                    xs.append(x)
                    y_train = y_train + y

                X_train = np.vstack(xs)
                X_train = pd.DataFrame(data=X_train, columns=f)
                y_train = pd.DataFrame(data=y_train)

                X_test, y_test, f = setup.vhdr2numpy(offline_files[subj][electrode_type][test_run], montage, electrode_type=e, spatial_filter='CAR', t_baseline=-0.3, epoch_window=[0.2, 0.5], spectral_window=[2,12])
                X_test = pd.DataFrame(data=X_test, columns=f)
                y_test = pd.DataFrame(data=y_test)

                scaler = StandardScaler()  # normalization: zero mean, unit variance
                scaler.fit(X_train)  # scaling factor determined from the training set
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)  # apply the same scaling to the test set 

                for i, model in enumerate(models):
                    if model == 'SVM (RBF kernel)':
                        clf = SVC(kernel='rbf')  # default C=1
                    if model == 'LDA':
                        clf = LDA() # solver = svd is good for large number of features, eigenvector is optimal though
                    if model == 'LogReg':
                        clf = LogisticRegression(max_iter=300, solver='saga') 
                    if model == 'RandomForest':
                        clf = RandomForestClassifier()
                
                    # Train the model using the training sets
                    clf.fit(X_train, y_train.values.ravel())
                    # Make predictions using the test set
                    y_pred = clf.predict(X_test)
                    # Prediction accuracy on training data
                    # print('Training accuracy =', clf.score(X_test, y_test), '\n')
                    validation_accuracies[i].append(float(clf.score(X_test, y_test)))
                    validation_models[i].append(clf) 

            model_cv_accuracies = [np.mean(sub_list) for sub_list in validation_accuracies]
            error = [np.std(sub_list) for sub_list in validation_accuracies]
            scores[subj][electrode_type] = model_cv_accuracies
            errors[subj][electrode_type] = error 
            # plt.figure()
            # plt.bar(models, model_cv_accuracies, yerr=error, capsize=8)
            # plt.title("Subject " + str(subj+6) + [' Gel ', ' POLITAG '][electrode_type])
            # plt.ylabel('Accuracy')
            # plt.show()

    my_df = [] 
    for i in range(0, n_subject):
        for mi, mm in enumerate(models): 
            d = {'subject' : i+6,  # some formula for obtaining values
                    'model' : mm,
                    'accuracy' : scores[i][electrode_type][mi],
                    'error' : errors[i][electrode_type][mi]
                    }
            my_df.append(d)

    my_df = pd.DataFrame(my_df)

    # sns.barplot(data=my_df, x='model', y='accuracy', hue='subject')
    grouped_barplot(df=my_df, cat='model', subcat='subject', val='accuracy', err='error')
    plt.title('3-fold cross-validation accuracies for ' + e)
    plt.ylim([0, 1])
    plt.show()     

   

        



'''
#SVM
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
opt_C_gamma_param = []
acc_max = 0
opt_Y_pred = []
for C_gamma_param in [(x, y) for x in C_range for y in gamma_range]:
    pipe_SVM = make_pipeline(StandardScaler(), SVC(kernel='rbf',C=C_gamma_param[0],gamma=C_gamma_param[1]))
    pipe_SVM.fit(bv_X_train, Y_train)
    # Test
    Y_pred = pipe_SVM.predict(bv_X_test)
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, average='weighted', labels=np.unique(Y_pred))
    if acc_max < acc and f1 > 0.0:
        acc_max = acc
        opt_C_gamma_param = C_gamma_param
        opt_Y_pred = Y_pred

print('BV SVM Classifier')
print(opt_C_gamma_param)
print('Accuracy score: {}'.format(acc_max))
print('Confusion Matrix:')
print(confusion_matrix(Y_test, opt_Y_pred))
print('Classification Report:')
print(classification_report(Y_test, opt_Y_pred, target_names=event_dict.keys()))

            X_train = np.vstack(xs)
            X_train = pd.DataFrame(data=X_train, columns=f)
            Y_train = pd.DataFrame(data=Y_train)

            X_test, Y_test, f = vhdr2numpy(offline_files[subj][electrode_type][test_run], False, 'CCA', [-0.5, 0.5])
            X_test = pd.DataFrame(data=X_test, columns=f)
            Y_test = pd.DataFrame(data=Y_test)

            scaler = StandardScaler()  # normalization: zero mean, unit variance
            scaler.fit(X_train)  # scaling factor determined from the training set
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)  # apply the same scaling to the test set 

'''