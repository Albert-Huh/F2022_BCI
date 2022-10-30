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
from setup import Setup as setup
import preprocessing
import feature_extraction

subject_num = input("Subject number: ")
electrode_type = input("Electrode tiype (Gel [0] or POLiTAG [1]): ")
# session = input("Session (offline [0], online_S1 [1], online_S2 [2]): ")

raw_data_list = os.listdir(os.path.join(os.getcwd(), 'ErrP_data'))
data_name = subject_num + '_' + electrode_type
for data_folder in raw_data_list:
    if data_folder.startswith(data_name):
        raw_folder = os.path.join(os.getcwd(), 'ErrP_data', data_folder, 'offline')
        
