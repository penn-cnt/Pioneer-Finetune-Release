# General imports
import os, sys, json
import warnings
import argparse
import pickle
from datetime import datetime, timedelta
import time
import pytz
from tqdm import tqdm
import re
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend

# edf relevant imports
import mne
mne.set_log_level('ERROR')
import pyedflib
from edf_reader import EdfWrapper
from joblib import Parallel, delayed
from databricks import sql

# paths
working_dir = os.path.abspath("")
raw_data_path = "../rawdata"
sys.path.append(working_dir)
sys.path.append(os.path.join(working_dir,'funcs'))

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

        
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description="Annotation Extraction"
    )

    # Define arguments
    parser.add_argument("-p", "--patient", type=str, required=True,help="Patient to run on")
    parser.add_argument("--plot", action = 'store_true', help="Generate trending plot for each event type")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # annotations to export
    annot_types = ['spike','seizure','sleep','patient']
    annot_wording_dict = {'spike':['spike'],
                          'seizure':['ueo','eec','sz','seizure','off'],
                          'sleep':['n1','n2','n3','sleep','wake','rem','sleep','drowsy','arousal'],
                          'patient':['patient event']}
    if args.patient == 'all':
        rids = os.listdir(raw_data_path)
    else:
        rids = args.patient.split(',')#['Pxxx','Pxxx']
    do_plot = args.plot
    for rid in rids:
        print(f'Processing patient {rid}')
        os.makedirs(os.path.join('annots',rid), exist_ok=True)
        folder = os.path.join(raw_data_path,rid)
        files = os.listdir(folder)
        files = [f for f in files if f.endswith('.edf')]
        all_annots = []
        info = {}
        for e, f in enumerate(files):
            edf_file_path = os.path.join(folder,f)
            use_mne = True
            try:
                # original reading with mne
                raw = mne.io.read_raw_edf(edf_file_path, preload=False)
                annot_df = raw.annotations.to_data_frame()
            except:
                use_mne = False
                raw = pyedflib.EdfReader(edf_file_path)
                # _______________get annotations__________________
                (onset,duration,description) = raw.readAnnotations()
                annot_df = pd.DataFrame({'onset':onset, 'duration':duration,'description':description})
                start_time = raw.getStartdatetime()
                annot_df["onset"] = annot_df["onset"].apply(lambda x: start_time + timedelta(seconds=x))
            annot_path = f"annots/{rid}/{f.replace('.edf','.csv')}"
            annot_df.to_csv(annot_path, index=False)
            all_annots.append(annot_df)
            # _______________extract info__________________
            tmp = re.split(r'[~ _.]',f)
            patient_name_first = tmp[2]
            patient_name_last = tmp[0]
            patient_guid = tmp[3]
            if not info:
                if use_mne:
                    info = {'device_id':raw.info['device_info'],
                                'firstName':patient_name_first,
                                'lastName':patient_name_last,
                                'patientGuid':patient_guid, #is actually file_id
                                'fileName':edf_file_path,
                                'studyType':'eeg',
                                'xlCreationTime':raw.info['meas_date'].replace(tzinfo=None),#.tz_localize(None),
                                'numberOfChannels':raw.info['nchan'],
                                'samplingFreq':raw.info['sfreq'],
                                'samplingFreqRaw':raw.info['sfreq'],
                                'channelNames':raw.info['ch_names']}
                else:
                    tmp_dt = raw.getStartdatetime()
                    info = {'firstName':patient_name_first,
                            'lastName':patient_name_last,
                            'patientGuid':patient_guid, #is actually file_id
                            'fileName':edf_file_path,
                            'studyType':'eeg',
                            'xlCreationTime':tmp_dt.replace(microsecond=0) + timedelta(seconds=(tmp_dt.microsecond >= 500000)),
                            'numberOfChannels':raw.signals_in_file,
                            'samplingFreq':raw.getSampleFrequency(0),
                            'samplingFreqRaw':raw.getSampleFrequency(0),
                            'channelNames':raw.getSignalLabels()}
                if info['samplingFreqRaw'] > 512:
                    info['studyType'] = 'ieeg'
                    info['samplingFreq'] = 512
            else:
                if use_mne:
                    if raw.info['meas_date'].replace(tzinfo=None) <= info['xlCreationTime']:
                        info['xlCreationTime'] = raw.info['meas_date'].replace(tzinfo=None)
                else:
                    if raw.getStartdatetime() <= info['xlCreationTime']:
                        tmp_dt = raw.getStartdatetime()
                        info['xlCreationTime'] = tmp_dt.replace(microsecond=0) + timedelta(seconds=(tmp_dt.microsecond >= 500000))
                
            raw.close()
        with open(f'INFO/{rid}.pkl','wb') as f:
            pickle.dump(info,f)

        all_annots = pd.concat(all_annots).sort_values(by='onset')
        for t in annot_types:
            judge = all_annots['description'].str.lower().replace('speech','').apply(
                lambda x: any(word in x for word in annot_wording_dict[t])
            )
            sub_df = all_annots[judge]
            if sub_df.empty:
                print(f'No {t} annotations.')
                continue
            if t == 'seizure':
                #sub_df = pd.concat([sub_df, all_annots[all_annots['description'].str.lower().startswith('off')
                sub_df = sub_df[sub_df['description']!='Persyst SeizureDetection']
                sub_df = sub_df[sub_df['description']!='Video Recording OFF']
                sub_df = sub_df[sub_df['description']!='Photic: Off']
            sub_df['onset'] = sub_df['onset'].dt.tz_localize(None)
            sub_df.to_csv(f'annots/{rid}/{t}.csv',index=False)
            if do_plot:
                figure_path = f"annots/{rid}/figures"
                os.makedirs(figure_path, exist_ok = True)
                fig, ax = plt.subplots(figsize=(10, 3))
                for i, row in sub_df.iterrows():
                    ax.axvline(row['onset'], color='k')  # vertical line
                    if t != 'spike':
                        ax.text(row['onset'], 0.5, row['description'], rotation=90, fontsize=9)
                # Customize plot
                ax.set_ylim(0, 1.2)
                ax.set_yticks([])
                ax.set_xlabel("Time")
                ax.set_title(t)
                ax.xaxis_date()  # enable datetime formatting on x-axis
                fig.autofmt_xdate()  # rotate x-axis date labels
                plt.tight_layout()
                fig.savefig(f"annots/{rid}/figures/{f.replace('.edf','_'+t+'.png')}")
            if t == 'seizure':
                # get events first
                seizure_annots = sub_df
                events = []
                current = None
                pending_ts = []  # store keywords that appear before OFF
                if not seizure_annots.empty:
                    for _, row in seizure_annots.iterrows():
                        
                        comment = str(row["description"]).lower()
                        if 'ueo' in comment or 'eec' in comment:
                            pending_ts.append(row["onset"])
                            current = True
                    
                        # OFF found
                        last_comment = comment.split(':')[-1].strip(' ')
                        if current and last_comment.startswith("off"):
                            # check gap from first keyword to OFF
                            stop_ts = row['onset']
                            while True:
                                if (stop_ts - pending_ts[0]).total_seconds()/60 > 20:
                                    e = {"Start":pending_ts[0],"Stop":pending_ts[0]+timedelta(minutes=2),"Comment":'No OFF found in 20 minutes'}
                                    events.append(e)
                                    pending_ts = [t for t in pending_ts if t > e['Stop']]
                                    if not pending_ts:
                                        current = False
                                        break
                                    else:
                                        continue
                                else:
                                    e = {"Start":pending_ts[0],"Stop":stop_ts, "Comment":''}
                                    events.append(e)
                                    pending_ts = []
                                    current = False
                                    break
                events = pd.DataFrame(events)
                if events.empty:
                    events = pd.DataFrame(columns=["Start","Stop","Comment"])
                events = events[events['Comment']!='No OFF found in 20 minutes']
                events.to_csv(f'annots/{rid}/seizure_events.csv', index=False)
            if t == 'patient':
                patient_annots = sub_df
                events = []
                if not patient_annots.empty:
                    for _, row in patient_annots.iterrows():
                        events.append({"Start":row['onset']-timedelta(minutes=1),"Stop":row['onset']+timedelta(minutes=2),"Comment":"-1-1min from patient reporting"})
                events = pd.DataFrame(events)
                if events.empty:
                    events = pd.DataFrame(columns=["Start","Stop","Comment"])
                events.to_csv(f'annots/{rid}/patient_events.csv', index=False)