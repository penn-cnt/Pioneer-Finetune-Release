# General imports
import os, sys, json, gc
import warnings
import argparse
import pickle
from datetime import datetime, timedelta
import pytz
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
import re

# edf relevant
import mne
mne.set_log_level('ERROR')
from edf_reader import EdfWrapper
import pyedflib
import sqlite3
from joblib import Parallel, delayed

# paths
working_dir = os.path.abspath("")
raw_data_path = "../rawdata"
sys.path.append(working_dir)
sys.path.append(os.path.join(working_dir,'feat_funcs'))

from funcs.utils import *
from funcs.feat_funcs import *

feat_settings = {
    'synchrony':{'win':int(60), 'stride':int(60),
                 'reref':'CAR','prewhite':False},
    'ad_ratio':{'win':int(60), 'stride':int(60),
                 'reref':'CAR','prewhite':False},
    'yasa':{'win':int(600),'stride':int(600),
                 'reref':'CAR','prewhite':False,
                 'resample':int(100)},
    'spikenet':{'win':int(2), 'stride':int(1),
                 'reref':'CAR','prewhite':False,
                 'resample':int(128)},
    'spike':{'win':int(60), 'stride':int(60),
                 'reref':'CAR','prewhite':False},
    'sparcnet':{'win':int(10),'stride':int(2),
                'reref':'BIPOLAR','prewhite':False},
    'wvnt':{'win':int(1),'stride':int(1),
            'reref':'BIPOLAR','prewhite':True,
            'resample':int(128)},
    'oncet':{'win':int(1),'stride':int(1),
            'reref':'BIPOLAR','prewhite':True},
}

prepro = Preprocessor()

def jsonify_df(df):
    for col, dtype in zip(df.columns,df.dtypes):
        if dtype == object and col != 'start_timestamp':
            df[col] = df[col].apply(json.dumps).astype("string") 
    return df

def jsondecode_df(df):
    for col, dtype in zip(df.columns,df.dtypes):
        if dtype == object:
            df[col] = df[col].apply(json.loads)
    return df

def process_clip(clip, i, t, info, split_n):
    if np.all(np.isnan(clip)):
        return
    start_timestamp = pd.to_datetime(t, unit='us')
    timestamps = pd.date_range(start=start_timestamp, periods=clip.shape[0], freq=pd.Timedelta(seconds=1/info['samplingFreqRaw']))
    clip = pd.DataFrame(clip, columns = info['channelNames'], index = timestamps)
    # clip_start_sample = int(i * info['samplingFreqRaw'] * split_n)
    # stamps = list(range(clip_start_sample,clip_start_sample+clip.shape[0],int(info['samplingFreqRaw'])))
    if info['samplingFreqRaw'] == 1024:
        clip = clip.iloc[::2,:]
        # stamps = stamps[::2]
        timestamps = timestamps[::2]
    timestamps = timestamps.astype(str).tolist()
    preprocessed = prepro.preprocess(clip)
    preprocessed_rows = []
    split_inds = np.arange(0,len(timestamps),info['samplingFreq']).astype(int)
    split_inds = np.r_[split_inds,[len(timestamps)]]
    for n in range(split_n):
        tmp_clip = preprocessed['BIPOLAR'].iloc[split_inds[n]:split_inds[n+1]].reset_index(drop=True)
        if np.any(np.isnan(tmp_clip)):
            continue
        preprocessed_dict = {'BIPOLAR': tmp_clip.round(4).to_dict('list')}
        preprocessed_df = pd.DataFrame.from_dict([preprocessed_dict])
        preprocessed_df['start_timestamp'] = timestamps[split_inds[n]]
        preprocessed_df['timestamps'] = [timestamps[split_inds[n]:split_inds[n+1]]]
        # preprocessed_df['stamps']  = [stamps[split_inds[n]:split_inds[n+1]]]
        # preprocessed_df['start_stamp'] = stamps[split_inds[n]]
        preprocessed_df['patient_id'] = info['patientGuid']
        preprocessed_rows.append(preprocessed_df)
    if len(preprocessed_rows) == 0:
        return
    preprocessed_df = pd.concat(preprocessed_rows)
    preprocessed_df = jsonify_df(preprocessed_df)
    # preprocessed_df.to_sql(sql_table, conn, if_exists='append', index=False)
    del clip
    gc.collect()
    return preprocessed_df
    # except Exception:
    #     return
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EEG data preprocessing.")

    parser.add_argument("-p", "--patient", type=str, required=True,help="Patient to run on")
    parser.add_argument("-f", "--force", action = 'store_true', help="Enforce re-run")

    # Parse the arguments
    args = parser.parse_args()
    # get patient ids to run on
    if args.patient == 'all':
        rids = os.listdir(raw_data_path)
    else:
        rids = args.patient.split(',')
    force = args.force

    # because only testing seizure relevant features, just getting bipolar data and bipolar bad channels, WVNT, etc. need additional prewhitening
    prepro_vars = ['BIPOLAR']
    for rid in rids:
        print(f'Processing {rid}')
        path = f'{raw_data_path}/{rid}'
        conn = sqlite3.connect(f'data/{rid}.db')
        files = os.listdir(path)
        files = [f for f in files if f.endswith('.edf')]
        # ____________________preprocess each file__________________________
        for e, f in enumerate(files):
            print(f)
            edf_file_path = os.path.join(path, f)
            # patient name
            tmp = re.split(r'[~ _.]',f)
            patient_name_first = tmp[2]
            patient_name_last = tmp[0]
            patient_guid = tmp[3]
            sql_table = rid
            # read data
            reader = EdfWrapper(edf_file_path)
            discont = len(reader.get_discontinuities()) > 0
            info = {'firstName':patient_name_first,
                    'lastName':patient_name_last,
                    'patientGuid':patient_guid, #is actually file_id
                    'studyType':'eeg',
                    'xlCreationTime':reader.header['start_datetime'].replace(tzinfo=None),
                    'numberOfChannels':len(reader.read_ts_channel_basic_info()),
                    'samplingFreq':reader.get_sampling_frequency(0),
                    'samplingFreqRaw':reader.get_sampling_frequency(0),
                    'channelNames':[ch['name'] for ch in reader.read_ts_channel_basic_info()]}
            info['stopTime'] = pd.to_datetime(reader.read_ts_channel_basic_info()[0]['end_time'],unit='us').to_pydatetime()
            preprocess_size = 120
            n_jobs = 10
            if info['samplingFreqRaw'] > 512:
                info['studyType'] = 'ieeg'
                info['samplingFreq'] = 512
                n_jobs = 5
            ### tmp fix
            # if not discont and info['studyType'] == 'eeg':
            #     print('Continuous scalp EEG file, skipping')
            #     continue
            # else:
            #     start_time = info['xlCreationTime']
            #     stop_time = info['stopTime']
            #     cursor = conn.cursor()
            #     cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table'AND name='{rid}'")
            #     table_exists = cursor.fetchone() is not None
            #     if table_exists:
            #         cursor.execute(f"DELETE FROM {rid} WHERE start_timestamp >= '{start_time}' AND start_timestamp < '{stop_time}'")
            #         conn.commit()
            ### end of tmp fix
            prepro.fit(info)
            # dura = (info['stopTime']-info['xlCreationTime']).total_seconds()
            
            # n_clips = int(np.floor(dura/preprocess_size))
            start_uutc = int(np.floor(reader.read_ts_channel_basic_info()[0]['start_time']/1e6)*1e6)
            stop_uutc = reader.read_ts_channel_basic_info()[0]['end_time']
            clip_start_utcs = np.arange(start_uutc, stop_uutc, preprocess_size*1e6)#[start_uutc + k*preprocess_size*1e6 for k in range(n_clips)]
            # clips = [reader.read_ts_channels_uutc(info['channelNames'],[start, min(start+preprocess_size*1e6,start+dura*1e6)]).T for start in clip_start_utcs]
            if info['studyType'] == 'ieeg':
                chunk_size = 50
                n_chunk = len(clip_start_utcs)//chunk_size+1
                for n in range(n_chunk):
                    sub_utcs = clip_start_utcs[n*chunk_size:min((n+1)*chunk_size,len(clip_start_utcs))]
                    chunks = Parallel(n_jobs=n_jobs)(delayed(process_clip)(reader.read_ts_channels_uutc(info['channelNames'],[t, t+preprocess_size*1e6]).T, e, t, info, preprocess_size) for e, t in tqdm(enumerate(sub_utcs), total=len(sub_utcs), desc="Processing clips"))
                    for c in chunks:
                        if c is not None:
                            c.to_sql(sql_table, conn, if_exists='append', index=False)
                    del chunks
                # for e,t in tqdm(enumerate(clip_start_utcs), total=len(clip_start_utcs), desc="Processing clips"):
                #     c = process_clip(reader.read_ts_channels_uutc(info['channelNames'],[t, t+preprocess_size*1e6]).T, e, t, info, preprocess_size)
                #     if c is not None:
                #         c.to_sql(sql_table, conn, if_exists='append', index=False)
                #     del c
            else:
                chunks = Parallel(n_jobs=n_jobs)(delayed(process_clip)(reader.read_ts_channels_uutc(info['channelNames'],[t, t+preprocess_size*1e6]).T, e, t, info, preprocess_size) for e, t in tqdm(enumerate(clip_start_utcs), total=len(clip_start_utcs), desc="Processing clips"))
                for c in chunks:
                    if c is not None:
                        c.to_sql(sql_table, conn, if_exists='append', index=False)
                del chunks
                # raw.close()
            reader.close()
            #, clips
            gc.collect()