# General imports
import numpy as np
import pandas as pd
import os, sys, json

import matplotlib.pyplot as plt
import scipy

import pickle
import mne
from tqdm import tqdm
from datetime import datetime, timedelta
import pytz
import re
import argparse
import sqlite3

mne.set_log_level('ERROR')

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
            'resample':int(128)}
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

if __name__ == '__main__':
    val_data_path = '../validation/clips/UEOEEC'
    data_list = os.listdir(val_data_path)
    data_list = sorted(data_list, key=lambda x: int(x[:-4]))
    os.makedirs(os.path.join(val_data_path,'preprocessed'),exist_ok=True)

    mode = 'sparcnet'
    for f in data_list:
        raw = mne.io.read_raw_edf(os.path.join(val_data_path,f), preload=False)
        df = raw.to_data_frame()
        df = df.iloc[:6*60*256]
        df = df.drop(columns = 'time')
        info = {'samplingFreq':raw.info['sfreq'],
                'samplingFreqRaw':raw.info['sfreq'],
                'studyType':'eeg',
                'numberOfChannels':raw.info['nchan'],
                'channelNames':raw.info['ch_names']} 
        prepro.fit(info)
        prepro_vars = ['BIPOLAR']
        preprocessed = prepro.preprocess(df)
        preprocessed_df = preprocessed['BIPOLAR']
        with open(os.path.join(val_data_path,'preprocessed',f.replace('.edf','.pkl')), 'wb') as f:
            pickle.dump(preprocessed_df, f)
