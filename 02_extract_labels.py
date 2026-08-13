# General imports
import os, sys, json
import warnings
import argparse
import glob
from datetime import datetime, timedelta
import time
import pytz

import numpy as np
import pandas as pd
import scipy

import sqlite3
from tqdm import tqdm
from joblib import Parallel, delayed
from pandas.errors import SettingWithCopyWarning

working_dir = os.path.abspath("")
raw_data_path = "edf"
database_path = "data"
participants_path = "participants.tsv"

sys.path.append(working_dir)
sys.path.append(os.path.join(working_dir,'funcs'))

from funcs.utils import *
from funcs.feat_funcs import *

def interval_overlap(win_start, win_stop, intervals):
    for start, end in intervals:
        if win_stop >= start and win_start <= end:
            return 1
    return 0

def get_feat_times(ts_array, win_size, win_stride):
    start_time, end_time = ts_array[0], ts_array[-1]
    window_starts = pd.date_range(start=start_time, end=end_time, freq=f"{win_stride}s")
    window_starts = np.asarray(window_starts)
    win_size = pd.Timedelta(seconds=win_size)
    ts_ns = ts_array.view('int64')
    ws_ns = window_starts.view('int64')
    we_ns = ws_ns + win_size.value
    # Find start and end indices for each window
    starts = np.searchsorted(ts_ns, ws_ns, side='left') # find indices where ts[i-1]<window_start<=ts[i]
    ends = np.searchsorted(ts_ns, we_ns, side='left')# find indices where ts[i-1]<window_end<=ts[i]

    window_times = []
    for s, e, we in zip(starts, ends, we_ns):
        if (e - s) != win_size.seconds:
            continue
        window_times.append(pd.Timestamp(we))
    return np.array(window_times)


FEAT_BY_RECORDING = {
    "Scalp": {"win": 10, "stride": 2},  # eeg
    "SEEG": {"win": 1, "stride": 1},    # ieeg
}

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description="Get labels")
    parser.add_argument("-p", "--patient", type=str, required=True, help="Patient to run on")
    args = parser.parse_args()

    participants = pd.read_csv(participants_path, sep="\t")
    feat_by_id = {
        str(r.PennEPIID): FEAT_BY_RECORDING[r.recording_type]
        for _, r in participants.iterrows()
    }
    rids = os.listdir(raw_data_path) if args.patient == "all" else args.patient.split(",")
    for rid in rids:
        feat_setting = feat_by_id[rid]
        conn = sqlite3.connect(f'{database_path}/{rid}.db')
        cursor = conn.cursor()
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_event_time ON {rid}(start_timestamp)")
        data_timestamps = pd.read_sql(f"SELECT start_timestamp FROM {rid} ORDER BY start_timestamp",conn)
        data_timestamps = data_timestamps['start_timestamp'].apply(lambda x: pd.Timestamp(x).tz_localize(None))
        data_timestamps = data_timestamps.apply(lambda x:x.replace(microsecond=0)+ timedelta(seconds=(x.microsecond >= 500000))).to_numpy()
        feat_timestamps = get_feat_times(data_timestamps, feat_setting['win'], feat_setting['stride'])#feat_data['timestamp'].dt.tz_localize(None).to_numpy()
        # true events
        events = pd.read_csv(f'annots/{rid}/seizure_events.csv', parse_dates=['Start','Stop'])
        data_labels = np.zeros(len(data_timestamps))
        feat_labels = np.zeros(len(feat_timestamps))
        if not events.empty:
            intervals = events[['Start','Stop']].to_numpy() 
            for e, t in enumerate(feat_timestamps):
                if interval_overlap(t - pd.Timedelta(seconds=feat_setting['stride']), t, intervals):
                    feat_labels[e] = 1
            for e, t in enumerate(data_timestamps):
                if interval_overlap(t, t+pd.Timedelta(seconds=1), intervals):
                    data_labels[e] = 1
        feat_labels = pd.DataFrame(feat_labels, index=feat_timestamps, columns=['label'])
        data_labels = pd.DataFrame(data_labels, index=data_timestamps, columns=['label'])
        feat_labels.to_pickle(f'labels/{rid}_feat_labels.pkl')
        data_labels.to_pickle(f'labels/{rid}_data_labels.pkl')
        # patient events and labels
        patient_events = pd.read_csv(f'annots/{rid}/patient_events.csv', parse_dates=['onset'])
        patient_events['Start'] = patient_events['onset']-pd.Timedelta(minutes=1)
        patient_events['Stop'] = patient_events['onset']+pd.Timedelta(minutes=2)
        data_labels = np.zeros(len(data_timestamps))
        feat_labels = np.zeros(len(feat_timestamps))
        if not patient_events.empty:
            intervals = patient_events[['Start','Stop']].to_numpy() 
            for e, t in enumerate(feat_timestamps):
                if interval_overlap(t - pd.Timedelta(seconds=feat_setting['stride']), t, intervals):
                    feat_labels[e] = 1
            for e, t in enumerate(data_timestamps):
                if interval_overlap(t, t+pd.Timedelta(seconds=1), intervals):
                    data_labels[e] = 1
        feat_labels = pd.DataFrame(feat_labels, index=feat_timestamps, columns=['label'])
        data_labels = pd.DataFrame(data_labels, index=data_timestamps, columns=['label'])
        feat_labels.to_pickle(f'labels/{rid}_patient_feat_labels.pkl')
        data_labels.to_pickle(f'labels/{rid}_patient_data_labels.pkl')

        # databricks annotations
        data_labels = np.zeros(len(data_timestamps))
        feat_labels = np.zeros(len(feat_timestamps))
        if not os.path.exists(f'annots/{rid}/databricks_events.csv'):
            continue
        events = pd.read_csv(f'annots/{rid}/databricks_events.csv', parse_dates=['start_time','stop_time'])
        events['start_time'] = events['start_time'].apply(pd.Timestamp).dt.tz_localize(None)
        events['stop_time'] = events['stop_time'].apply(pd.Timestamp).dt.tz_localize(None)
        events['classified_response'] = events['classified_response'].fillna('uncertain')
        yes_events = events[events['classified_response'] == 'yes']
        no_events = events[events['classified_response'] == 'no']
        uncertain_events = events[events['classified_response'] == 'uncertain']
        if not yes_events.empty:
            yes_intervals = yes_events[['start_time','stop_time']].to_numpy() 
            for e, t in enumerate(data_timestamps):
                if interval_overlap(t, t+pd.Timedelta(seconds=1), yes_intervals):
                    data_labels[e] = 1
            for e, t in enumerate(feat_timestamps):
                if interval_overlap(t - pd.Timedelta(seconds=feat_setting['stride']), t, yes_intervals):
                    feat_labels[e] = 1
        if not uncertain_events.empty:
            uncertain_intervals = uncertain_events[['start_time','stop_time']].to_numpy() 
            for e, t in enumerate(data_timestamps):
                if interval_overlap(t, t+pd.Timedelta(seconds=1), uncertain_intervals):
                    data_labels[e] = np.nan
            for e, t in enumerate(feat_timestamps):
                if interval_overlap(t, t-pd.Timedelta(seconds=feat_setting['stride']), uncertain_intervals):
                    data_labels[e] = np.nan
        feat_labels = pd.DataFrame(feat_labels, index=feat_timestamps, columns=['label'])
        data_labels = pd.DataFrame(data_labels, index=data_timestamps, columns=['label'])
        feat_labels.to_pickle(f'labels/{rid}_databricks_feat_labels.pkl')
        data_labels.to_pickle(f'labels/{rid}_databricks_data_labels.pkl')