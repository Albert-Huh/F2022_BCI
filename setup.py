import sys
import os
from sqlite3 import Row
import numpy as np
from datetime import datetime, date, time, timedelta
import matplotlib
import matplotlib.pyplot as plt
import read_nback_report as nback
import read_hrt_report as hrt

import mne
matplotlib.use('TkAgg')

class Setup:
    def __init__(self, raw_path=None, montage_path=None, mode=None):
        if mode == 'Binary':
            self.raw = mne.io.read_raw_fif(raw_path)
        else:
            self.raw = mne.io.read_raw_brainvision(raw_path)
        self.montage = mne.channels.read_custom_montage(montage_path)
        self.mode = mode # 'Binary', 'Brainvision' 'Dual'
        self.raw.set_channel_types({'EOG':'eog'})
        if mode == 'Dual':
            self.bv_raw = self.raw.copy().pick_channels(['Fp1','Fp2','Fz','F3','F4','F7','F8','Cz','C3','C4','T7','T8','Pz','P3','P4','P7','P8','O1','O2','EOG'])
            self.et_raw = self.raw.copy().pick_channels(['Fp1_ET','Fp2_ET','F7_ET','F8_ET','A1','A2','EOG'])
            new_names = dict(
                (ch_name,
                ch_name.replace('_ET', ''))
                for ch_name in self.et_raw.ch_names)
            self.et_raw.rename_channels(new_names)

            self.bv_raw.set_montage(self.montage)
            # fig = self.bv_raw.plot_sensors(show_names=True)
            # et_montage = self.montage.copy()
            # self.et_raw.set_montage(et_montage)
            # fig = self.et_raw.plot_sensors(show_names=True, block=True)
        elif mode == 'E-tattoo':
            self.et_raw = self.raw.copy().pick_channels(['Fp1','Fp2','F7','F8','A1','EOG'])
        elif mode == 'Brainvision':
            self.raw.set_montage(self.montage)
            # pass

    def get_brainvision_raw(self):
        self.bv_raw = self.raw.copy().pick_channels(['Fp1','Fp2','Fz','F3','F4','F7','F8','Cz','C3','C4','T7','T8','Pz','P3','P4','P7','P8','O1','O2','EOG'])
        self.bv_raw.set_montage(self.montage)
        # fig = self.bv_raw.plot_sensors(show_names=True)

    def get_e_tattoo_raw(self):
        self.et_raw = self.raw.copy().pick_channels(['Fp1_ET','Fp2_ET','F7_ET','F8_ET','A1','A2','EOG'])
        new_names = dict(
            (ch_name,
            ch_name.replace('_ET', ''))
            for ch_name in self.et_raw.ch_names)
        self.et_raw.rename_channels(new_names)
        # fig = self.et_raw.plot_sensors(show_names=True, block=True)
        
    def get_annotation_info(self):
        onset = self.raw.annotations.onset
        duration = self.raw.annotations.duration
        description = self.raw.annotations.description
        return onset, duration, description

    def set_annotation(self, raw, onset, duration, description):
        my_annot = mne.Annotations(onset=onset, duration=duration, description=description)
        raw.set_annotations(my_annot)

    def annotate_interactively(self):
        fig = self.raw.plot()
        fig.fake_keypress('a')
        plt.show()
        interactive_annot = self.raw.annotations
        return interactive_annot

    def get_events_from_annot(self, custom_mapping='auto'):
        events, event_dict = mne.events_from_annotations(self.raw, event_id=custom_mapping)
        print(event_dict)
        print(events)
        return events, event_dict

    def get_events_from_raw(self, stim_channel=None):
        events = mne.find_events(self.raw, stim_channel=stim_channel)
        return events

    def get_events_from_nback_report(self, report_path, fs):
        lines = nback.read_report_txt(report_path)
        key_string_list = nback.get_key_string()
        nback_report = nback.get_report_data(lines, key_string_list)
        meas_isodate = datetime.fromisoformat(str(self.raw.info['meas_date']))
        # fs = self.raw.info['sfreq']
        samplestamp_tdel = nback.get_stim_time_delta(nback_report, meas_isodate, fs)
        nback_event = nback.get_nback_event(nback_report, samplestamp_tdel, fs)
        return nback_event

    def get_events_from_hrt_report(self, report_path, fs):
        lines = hrt.read_report_txt(report_path)
        key_string_list = hrt.get_key_string()
        hrt_report = hrt.get_report_data(lines, key_string_list)
        meas_isodate = datetime.fromisoformat(str(self.raw.info['meas_date']))
        # fs = self.raw.info['sfreq']
        start_samplestamp_tdel, end_samplestamp_tdel = hrt.get_stim_time_delta(hrt_report, meas_isodate, fs)
        hrt_events = hrt.get_hrt_event(hrt_report, start_samplestamp_tdel, end_samplestamp_tdel, fs,5)
        return hrt_events

# event visiualization
# fig = mne.viz.plot_events(events_from_annot, event_id=event_dict, sfreq=raw.info['sfreq'], first_samp=raw.first_samp)