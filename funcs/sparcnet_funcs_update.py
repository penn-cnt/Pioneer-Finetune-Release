# General imports
import os, sys, json
import warnings
import argparse
import pickle
import glob
from datetime import datetime, timedelta
import time
import pytz
import logging
import importlib

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter
matplotlib.use('Agg')  # Non-GUI backend
import re

# import mne
# mne.set_log_level('ERROR')
import sqlite3
from tqdm import tqdm
from joblib import Parallel, delayed
import mlflow

# torch and scikit learn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split, ConcatDataset, WeightedRandomSampler
from mne.filter import filter_data, notch_filter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import balanced_accuracy_score, average_precision_score, roc_auc_score, brier_score_loss, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import CalibrationDisplay
from matplotlib.gridspec import GridSpec

# paths
working_dir = os.path.abspath("")
sys.path.append(working_dir)
# sys.path.append(os.path.join(working_dir,'funcs'))
code_v2_path = os.path.join(working_dir,'funcs/SPaRCNet')
sys.path.append(code_v2_path)

# custom utils
from funcs.utils import *
from funcs.feat_funcs import *
from funcs.context import pred_ctx, ctx
from funcs.metric_funcs import compute_metrics, compute_ece, metric_labels, get_optimal_auc

# sparcnet imports
from DenseNetClassifier import *
# from SPaRCNet.utils import *

# setup logging
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
# mne.set_log_level('ERROR')

feat_setting = {'win':int(10),
                'stride':int(2),
                'reref':'BIPOLAR',
                'prewhite':False}
# clip_loader = ClipLoader(feat_setting,aligned='right')

# model settings
device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")
unfreeze_order = [
        'classifier', 
        'features.densenet.denseblock7',
        'features.densenet.transition6',
        'features.densenet.denseblock6',
        'features.densenet.transition5',
        'features.densenet.denseblock5',
        'features.densenet.transition4',
        'features.densenet.denseblock4'
    ]

# generalized validation data
val_data_path = '../validation/clips/UEOEEC/preprocessed'
data_list = os.listdir(val_data_path)
data_list = sorted(data_list, key=lambda x: int(x[:-4]))
# train_data_list = data_list[]


def load_model(model_path, mode = 'train', unfreeze_stage = 0, dropout_rate = 0.2, drop_fc = 0.5, unfreeze_base = True):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = DenseNetClassifierwithTemperature(drop_rate = dropout_rate, drop_fc = drop_fc)
    model.load_state_dict(checkpoint, strict = False)
    model = model.to(device)
    if mode == 'train':
        unfreeze_modules = unfreeze_order[:unfreeze_stage+1]
        for name, module in model.named_modules():
            for param in module.parameters():
                param.requires_grad = False 
        for name, module in model.named_modules():
            if name in unfreeze_modules:
                for param in module.parameters():
                    param.requires_grad = True
        if unfreeze_base:
            for name, module in model.named_modules():
                if name == 'features.densenet.conv0':
                    for param in module.parameters():
                        param.requires_grad = True
        model.temperature.requires_grad = True
        model.train()
    else:
        model.eval()
    return model

# ________________________________Fintuning Funcs_________________________________

def write_state(state_dict):
    state = pd.DataFrame([state_dict])
    # state = state[["patient","batch","sequence","step",
    #                 'is_best','timestamp', 
    #                 'train_loss','validation_loss','balanced_accuracy','auprc','auroc', 'ece', 'brier']]
    state.to_csv(ctx.progress_tracker, index=False, mode = 'a', header=False)


def jsondecode_df(df):
    for col, dtype in zip(df.columns,df.dtypes):
        if dtype == object:
            try:
                df[col] = df[col].apply(json.loads)
            except:
                pass
    return df

class WeightedKLDivWithLogitsLoss(nn.KLDivLoss):
    def __init__(self, reduction='none', log_target=False, class_weights=None, sample_weights = None):
        super(WeightedKLDivWithLogitsLoss, self).__init__(reduction=reduction, log_target=log_target)
        
        if class_weights is not None:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            self.register_buffer('class_weights', class_weights_tensor)
        else:
            self.class_weights = None
        if sample_weights is not None:
            sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32).to(device)
            self.register_buffer('sample_weights', sample_weights_tensor)
        else:
            self.sample_weights = None

    def forward(self, logits, target):
        # Convert logits to log-probabilities
        log_prob = F.log_softmax(logits, dim=1)

        # Compute element-wise KL divergence (shape: [batch_size, num_classes])
        element_loss = super(WeightedKLDivWithLogitsLoss, self).forward(log_prob, target)

        # Sum over classes to get per-sample loss
        sample_loss = torch.sum(element_loss, dim=1)

        if self.class_weights is not None:
            # Compute per-sample weights based on target and class weights
            sample_weight = torch.sum(target * self.class_weights, dim=1)
            weighted_loss = sample_loss * sample_weight
        elif self.sample_weights is not None:
            weighted_loss = sample_loss * self.sample_weights
        else:
            weighted_loss = sample_loss

        # Return average loss over the batch
        return weighted_loss.mean()

class SPARCNetDataset(Dataset):
    def __init__(self, data_df, labels, fs, windows = None, device = device, revert = True): # should have a bad/valid here
        # takes in data_df (one row is one second), labels, 
        self.device = device
        self.data = data_df
        self.fs = fs
        if windows is not None:
            self.time_window = True
            self.windows = windows
        else:
            self.time_window = False
            win_starts = np.arange(0,len(data_df)-feat_setting['win']+1,feat_setting['stride'])
            self.windows = pd.DataFrame({'window_start':win_starts,'window_end':win_starts + feat_setting['win']})
        if len(labels) == len(self.windows):
            self.window_label = labels.flatten()
        elif len(labels) == len(data_df):
            self.data_label = labels
            self.window_label = np.array([np.any(self.data_label[row['window_start']:row['window_end']]==1) for _,row in self.windows.iterrows()])
        # nan_remove = np.array([np.all(np.isnan(self.labels[idx:idx+feat_setting['win']])) for idx in self.windows])
        nan_remove = np.isnan(self.window_label)
        self.windows = self.windows[~nan_remove]
        self.window_label = self.window_label[~nan_remove]
        if revert:
            self.revert = np.r_[np.zeros_like(self.window_label),np.ones_like(self.window_label)]
            self.windows = pd.concat([self.windows,self.windows],axis=0)
            self.window_label = np.r_[self.window_label,self.window_label]
        else:
            self.revert = np.zeros_like(self.window_label)

    def __len__(self):
        return len(self.windows)
    
    def __process_data__(self, idx):
        window = self.windows.iloc[idx]
        if self.time_window:
            data = self.data[(self.data['start_timestamp'] >= window['window_start']) & (self.data['start_timestamp'] < window['window_end'])]
        else:
            data = self.data.iloc[window['window_start']:window['window_end']]
        #tmp_df = jsondecode_df(data)
        data_df = pd.concat([pd.DataFrame(d['BIPOLAR'],index = d['timestamps']) for _, d in data.iterrows()]).sort_index()
        data_values = data_df[['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2',
                    'Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4','F4-C4','C4-P4','P4-O2']].values
        data_values = bandpass_filter(data_values, self.fs, lo=1, hi=40)
        data_values = downsample(data_values, self.fs, 200)
        data_values = np.clip(data_values, -500, 500) 
        return data_values
        
    def __getitem__(self, idx):
        data = self.__process_data__(idx)
        if self.revert[idx]:
            data = data[:,np.r_[4:8,:4,12:16,8:12]]
        label = self.window_label[idx]
        values = torch.from_numpy(data.astype(np.float32)).to(self.device)
        values = values.T
        labels = [0, 1, 0, 0, 0, 0] if label == 1 else [0, 0, 0, 0, 0, 1]
        labels = torch.from_numpy(np.array(labels).astype(np.float32)).to(self.device)
        return values, labels

def aggregate_chunk(chunk):
    data = {col: chunk[col].tolist() for col in chunk.columns}
    stamps = chunk.index.tolist()
    agg = pd.DataFrame([{'BIPOLAR':data,'timestamps':stamps}])
    return agg

def get_general_datasets(seed, patient_num = 10):
    fs = 256
    np.random.seed(seed)  # set seed
    use_data_list = np.random.choice(data_list, size=patient_num, replace=False)
    train_data = []
    val_data = []
    # use only first 4 minutes pre seizure to avoid differences in pre-ictal session
    # use the last 20 seconds (5 samples vs 40 train samples) of seizure for val, use the 4th minute for val
    label_base = np.r_[np.zeros(60),np.ones(60)].astype(int)
    # val_label_base = np.r_[np.zeros(30),np.ones(15)].astype(int)
    train_labels = np.tile(label_base, 8)
    val_labels = np.tile(label_base, 2)
    # train_idx = np.r_[np.arange(0,int(3.5*60)),np.arange(5*60, (5*60+45))].astype(int)
    # val_idx = np.r_[np.arange(int(3.5*60),4*60),np.arange((5*60+45), 6*60)].astype(int)
    idx = np.arange(int(4*60), int(6*60)).astype(int)
    for e, p in enumerate(use_data_list):
        with open(os.path.join(val_data_path,p),'rb') as f:
            df = pickle.load(f)
            df = df.reset_index(drop=True)
            df = df.groupby(np.arange(len(df)) // fs).apply(aggregate_chunk).reset_index(drop=True)
            df = df.iloc[idx]
            # val_df = df.iloc[val_idx]
            if e < 8:
                train_data.append(df)
            else:
                val_data.append(df)
    train_data = pd.concat(train_data)
    val_data = pd.concat(val_data)  
    train_dataset = SPARCNetDataset(train_data, train_labels, fs, revert=True)
    val_dataset =  SPARCNetDataset(val_data, val_labels, fs, revert=False)
    return train_dataset, val_dataset

def get_dataset(data_df, labels, fs, train_ratio = 0.85, stratify_label = None):
    # num_fp = np.sum(conf_ind != 'TN')
    # labels = labels[(conf_ind != 'TN')]
    windows = labels.copy().reset_index().rename(columns={'index': 'window_end'})
    windows['window_start'] = windows['window_end']-pd.Timedelta(seconds=feat_setting['win'])
    labels = labels.values
    if stratify_label is None:
        stratify_label = labels
    # dataset = SPARCNetDataset(data_df, labels.values, fs, windows = windows, revert = False)
    # windows = dataset.windows
    # window_label = dataset.window_label
    if np.any(stratify_label == 1):
        sz_idx = np.where(stratify_label == 1)[0]
        non_sz_idx = np.where(stratify_label!=1)[0]
        num_train_sz = int(len(sz_idx)*train_ratio)
        num_train_non_sz = int(len(non_sz_idx)*train_ratio)
        train_idx = np.concatenate([sz_idx[:num_train_sz],non_sz_idx[:num_train_non_sz]]).flatten()
        train_idx.sort()
        val_idx = np.concatenate([sz_idx[num_train_sz:],non_sz_idx[num_train_non_sz:]]).flatten()
        val_idx.sort()
        train_windows = windows.iloc[train_idx]
        val_windows = windows.iloc[val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
    else:
        num_train = int(len(windows)*train_ratio)
        train_windows = windows.iloc[:num_train]
        val_windows = windows.iloc[num_train:]
        train_labels = labels[:num_train]
        val_labels = labels[num_train:]
    train_dataset = SPARCNetDataset(data_df, train_labels, fs, windows = train_windows, revert = True)
    val_dataset = SPARCNetDataset(data_df, val_labels, fs, windows = val_windows, revert = False)
    return train_dataset, val_dataset
    
def get_buffer_inds(labels):
    labels = np.array(labels)  # convert to numpy for easy indexing
    pos_inds = np.where(labels == 1)[0]
    return pos_inds
    

def calibrate_temp(model, val_loader, max_iter=50, lr=0.01):
    """
    logits: torch.Tensor of shape (N,)
    labels: torch.Tensor of shape (N,)
    """
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            y_label = y_batch[:,1].unsqueeze(1)
            pos_weight = (y_label.shape[0]-torch.sum(y_label==1))/torch.sum(y_label==1).numpy()
            if np.isnan(pos_weight) or np.isinf(pos_weight):
                pos_weight = 1
            output, _ = model(x_batch)
            loss_fn = nn.BCELoss(weight=torch.where(y_label == 1, pos_weight, 1.0).to(device))
    optimizer = torch.optim.LBFGS([model.temperature], lr=lr, max_iter=max_iter)
    def eval_step():
        optimizer.zero_grad()
        logits = output/model.temperature
        prob = F.softmax(logits, dim=1)[:,1].unsqueeze(1)
        loss = loss_fn(prob, y_label)
        loss.backward()
        return loss
    
    optimizer.step(eval_step)
    return model

def plot_finetune_progress(true, prev_prob, prev_prob_cal, prob, prob_cal, out_path):
    # generate a 1x2 figure, left is auroc, right is auprc, 4 traces are overlay
    # generate a calibration figure
    data_dict = {'Prev Raw':prev_prob, 'Prev Cal':prev_prob_cal, 'Finetune Raw':prob, 'Finetune Cal': prob_cal}
    fig, axes = plt.subplots(1,2,figsize=(16, 8))
    colors = plt.get_cmap("Dark2")
    calibration_displays = {}
    markers = ["^", "v", "s", "o"]
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    baseline = np.mean(true)
    axes[1].hlines(baseline, xmin=0, xmax=1, colors="k", linestyles="--", label=f'Baseline={baseline:.3f}')
    for i, (name,data) in enumerate(data_dict.items()):
        fpr, tpr, thr = roc_curve(true, data)
        auc_score = auc(fpr, tpr)
        prec, recall, thr_prc = precision_recall_curve(true, data)
        prc_score = average_precision_score(true, data)
        ind = np.argmax(tpr-fpr)
        opt_thr, opt_tpr, opt_fpr = thr[ind], tpr[ind],fpr[ind]
        ind = np.argmax(2 * (prec * recall) / (prec + recall))
        opt_thr_prc, opt_prec, opt_recall = thr_prc[ind], prec[ind], recall[ind]
        axes[0].plot(fpr, tpr, label=f"{name} (AUROC = {auc_score:.3f}), opt thres {opt_thr:.3f}, tpr {opt_tpr:.3f}, fpr {opt_fpr:.3f}")
        axes[1].plot(recall, prec, label=f"{name} (AUPRC = {prc_score:.3f}) opt thres {opt_thr_prc:.3f}, recall {opt_recall:.3f}, prec {opt_prec:.3f}")
    axes[0].set_title("AUROC")
    axes[0].legend(loc='lower right')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[1].set_title("AUPRC")
    axes[1].legend(loc='upper right')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    fig.savefig(f'{out_path}.svg', dpi=300)
    plt.close(fig)

    # generate calibration figure
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 4)
    colors = plt.get_cmap("Dark2")
    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}
    markers = ["^", "v", "s", "o"]
    for i, (name,data) in enumerate(data_dict.items()):
        display = CalibrationDisplay.from_predictions(true, data, n_bins=10,name=name,ax=ax_calibration_curve,color=colors(i),marker=markers[i])
        calibration_displays[name] = display
    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots")
    ax_calibration_curve.legend(loc='upper left')
    # Add histogram
    grid_positions = [(0, 2), (0, 3), (1, 2), (1, 3)]
    for i, name in enumerate(data_dict.keys()):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])
        ax.hist(
            calibration_displays[name].y_prob,
            range=(0, 1),
            bins=10,
            label=name,
            color=colors(i),
        )
        ax.set(title=name)
        if i < len(data_dict)-1:
            ax.set_xticklabels([])
    ax.set(xlabel="Mean predicted probability")
    fig.savefig(f'{out_path}_calibration.svg', dpi=300)
    plt.close(fig)

def get_type_label(labels, prev_pred):
    TP = (labels == 1) & (prev_pred == 1)
    FP = (labels == 0) & (prev_pred == 1)
    TN = (labels == 0) & (prev_pred == 0)
    FN = (labels == 1) & (prev_pred == 0)
    type_df = pd.DataFrame({'TN':TN,'TP':TP,'FP': FP, 'FN': FN})
    type_label = type_df.idxmax(axis=1)
    return type_label

def get_sample_weights(labels, prev_pred, class_weights = {'TN':0.2, 'TP':1, 'FP':4, 'FN':6}):
    TP = (labels == 1) & (prev_pred == 1)
    FP = (labels == 0) & (prev_pred == 1)
    TN = (labels == 0) & (prev_pred == 0)
    FN = (labels == 1) & (prev_pred == 0)
    type_df = pd.DataFrame({'TN':TN,'TP':TP,'FP': FP, 'FN': FN})
    type_label = type_df.idxmax(axis=1)
    weights = type_label.map(lambda x: class_weights.get(x, 1))  # Default to 1 if class is missing in class_weights
    return weights

def ConfMatLoader(dataset, prev_pred, class_weights = {'TN':0.2, 'TP':1, 'FP':4, 'FN':6}, batch_size=32, pos_multiplier=5, shuffle = True):

    labels = torch.tensor([y[1] for _, y in dataset]).detach().numpy()
    if class_weights == 'auto':
        type_label = get_type_label(labels, prev_pred)
        unique, counts = np.unique(type_label.to_numpy(), return_counts=True)
        conf_perc = {k:counts/len(type_label) for k, counts in zip(unique, counts)}
        logger.info(f'Percentages {conf_perc}')
        class_weights = {k:np.min([1/v,60]) if v != 0 else 0 for k,v in conf_perc.items()}
        logger.info(f'Weight {class_weights}')
    sample_weights = type_label.map(lambda x: class_weights.get(x, 1))
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader

def SubsampleDataSetLoader(dataset, prev_pred, batch_size=32, shuffle=True):
    labels = torch.tensor([y[1] for _, y in dataset]).detach().numpy()
    type_label = get_type_label(labels, prev_pred).to_numpy()
    tn_inds = np.random.choice(np.where(type_label=='TN')[0], size=min(max(np.sum(type_label!='TN')*5,300),np.sum(type_label=='TN')), replace=False)
    inds = np.r_[tn_inds,np.where(type_label!='TN')[0]]
    unique, counts = np.unique(type_label[inds], return_counts=True)
    conf_perc = {k:counts for k, counts in zip(unique, counts)}
    logger.info(f'Percentages {conf_perc}')
    sub_dataset = Subset(dataset, inds)
    return DataLoader(sub_dataset, batch_size=batch_size, shuffle=shuffle)

def BalancedLoader(dataset, batch_size=32, pos_multiplier=5, shuffle = True):
    """
    pos_multiplier: how many times to oversample positives
                    (2–5 is common; don’t oversample too much)
    """
    labels = torch.tensor([y[1] if y is not None else None for _, y in dataset])
    # labels = torch.cat([torch.tensor(d.window_label) for d in dataset.datasets])


    # Count positive and negative
    pos_mask = labels == 1
    neg_mask = labels == 0

    # Assign weights
    weights = torch.ones(len(dataset))*0.5
    weights[pos_mask] *= pos_multiplier  # oversample positives

    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, collate_fn = skip_none_collate)

    return loader

def get_moving_max_prob(sz_probs,win_size=10,gap_size=2):
    perc = gap_size/win_size
    probs = [np.percentile(sz_probs[i-win_size:i],perc*100) for i in range(win_size,len(sz_probs))]
    return np.max(probs)       

def finetune():
    fs = ctx.sampling_freq
    batch_ind = ctx.batch_ind
    base_model = ctx.base_model
    base_model = load_model(base_model, mode='eval')
    out_model = ctx.out_model
    use_no_sz_setting = False
    use_teacher = True

    prev_pred = ctx.prev_pred_batch['smoothed_pred'].values
    if ctx.mode == 'pioneer':
        logger.info(f'Doing pioneer version of finetuning.')
        if np.all(prev_pred==0):
            logger.info(f'BATCH {batch_ind}: Skip finetuning due to no seizure detection during this session.')
            return
    df, labels = ctx.get_batch_data()
    if df is None:
        logger.info(f'BATCH {batch_ind}: Skipping due to no data.')
        state = {"patient":ctx.current_patient, "batch": ctx.batch_ind, "sequence": ctx.num_epoch+3,
                'step':f'skipping finetuning', 'is_best': True, 
                "timestamp": datetime.now()}
        write_state(state)
        return
    if np.all(prev_pred==0) and np.all(labels == 0):
        logger.info(f'BATCH {batch_ind}: Skip finetuning due to no seizure detection and no seizure events during this session.')
        state = {"patient":ctx.current_patient, "batch": ctx.batch_ind, "sequence": ctx.num_epoch+3,
                'step':f'skipping finetuning', 'is_best': True, 
                "timestamp": datetime.now()}
        write_state(state)
        return
    elif np.all(prev_pred == 0) and np.any(labels == 1):
        logger.info(f'BATCH {batch_ind}: No seizure detection but seizure events during this session.')
    elif np.any(prev_pred == 1) and np.all(labels == 0):
        logger.info(f'BATCH {batch_ind}: Has seizure detection but no seizure events during this session.')
        if len(extract_seiz_ranges(prev_pred)) < 2 or np.sum(prev_pred)/len(prev_pred) < 0.001:
            logger.info(f'BATCH {batch_ind}: Skip finetuning due to few false positive during this session.')
            return
    elif np.any(prev_pred == 1) and np.any(labels == 1):
        logger.info(f'BATCH {batch_ind}: Has seizure detection and seizure events during this session.')
    
    prev_prob = labels.merge(ctx.prev_pred_batch[['timestamp','sz_prob']], left_index=True, right_on='timestamp', how='left')['sz_prob'].values
    prev_pred = labels.merge(ctx.prev_pred_batch[['timestamp','smoothed_pred']], left_index=True, right_on='timestamp', how='left')['smoothed_pred'].values
    # labels.loc[prev_prob >= 0.95,'label'] = 1
    type_label = get_type_label(labels['label'].values, prev_pred)
    train_dataset, val_dataset = get_dataset(df, labels, fs)#, stratify_label=(type_label!='TN').astype(int))
    prev_pred_train = train_dataset.windows.merge(ctx.prev_pred_batch[['timestamp','smoothed_pred']], left_on='window_end', right_on='timestamp', how='left')['smoothed_pred'].values
    prev_pred_val = val_dataset.windows.merge(ctx.prev_pred_batch[['timestamp','smoothed_pred']], left_on='window_end', right_on='timestamp', how='left')['smoothed_pred'].values

    logger.info(f'BATCH {batch_ind}: Initial train dataset size {len(train_dataset)}, validation dataset size {len(val_dataset)}')
    prev_sz_data = glob.glob(f"{ctx.sz_data_dir}/finetune_*")
    train_list = []
    val_list = []
    prev_pred_train_add = []
    prev_pred_val_add = []
    if prev_sz_data:
        logger.info(f'BATCH {batch_ind}: Previous seizure data included')
        for f in prev_sz_data:
            tmp = torch.load(f, weights_only=False)
            if tmp['train']:
                train_list.append(tmp['train'])
                with torch.no_grad():
                    output, _ = base_model(torch.stack([x for x,_ in tmp['train']]))
                    prob = F.softmax(output,dim=1)[:,1].unsqueeze(1).cpu().numpy().flatten()
                    pred = get_prediction(prob)
                    # _, pred = smooth_pred_and_extend(prob, thres_high = ctx.thres_high, thres_low = ctx.thres_low, return_pred = True)#prob >= ctx.use_thres
                    prev_pred_train_add.append(pred)
            if tmp['val']:
                val_list.append(tmp['val'])
                with torch.no_grad():
                    output, _ = base_model(torch.stack([x for x,_ in tmp['val']]))
                    prob = F.softmax(output,dim=1)[:,1].unsqueeze(1).cpu().numpy().flatten()
                    pred = get_prediction(prob)
                    # _, pred = smooth_pred_and_extend(prob, thres_high = ctx.thres_high, thres_low = ctx.thres_low, return_pred = True)#prob >= ctx.use_thres
                    prev_pred_val_add.append(pred)
    else:
        # use teacher model to avoid drastic forgetting?
        if not np.any(labels == 1):
            use_no_sz_setting = True
        pass
    if np.any(labels == 1):
        train_sz_inds = get_buffer_inds(train_dataset.window_label)
        val_sz_inds = get_buffer_inds(val_dataset.window_label)
        subset_train = [train_dataset[i] for i in train_sz_inds]
        subset_val = [val_dataset[i] for i in val_sz_inds]
        torch.save({'train': subset_train, 
                    'val': subset_val}, os.path.join(ctx.sz_data_dir, f"finetune_{ctx.batch_ind}.pt"))
    train_dataset = ConcatDataset([train_dataset] + train_list) #if train_list else train_dataset
    val_dataset = ConcatDataset([val_dataset] + val_list) #if val_list else val_dataset
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    prev_pred_train = np.concatenate([prev_pred_train]+prev_pred_train_add)
    prev_pred_val = np.concatenate([prev_pred_val]+prev_pred_val_add)
    # train_loader = DataLoader(train_dataset, batch_size=ctx.batch_size, shuffle=True)
    train_loader = ConfMatLoader(train_dataset, prev_pred_train, class_weights = 'auto', batch_size=ctx.batch_size)
    # train_loader = SubsampleDataSetLoader(train_dataset, prev_pred_train, batch_size=ctx.batch_size, shuffle=True)
    logger.info(f'BATCH {batch_ind}: Final train dataset size {len(train_dataset)}, validation dataset size {len(val_dataset)}')
    if ctx.do_epochs:
        if ctx.original:
            model = load_model('models/base_model.pt', unfreeze_stage = ctx.unfreeze_stage)
            logger.info(f'BATCH {batch_ind}: Loading model models/base_model.pt at unfreeze stage {unfreeze_order[ctx.unfreeze_stage]} and temperature {model.temperature.cpu().item()}')
        else:
            model = load_model(ctx.base_model, unfreeze_stage = ctx.unfreeze_stage) #, dropout_rate=ctx.drop_rate, drop_fc = ctx.drop_fc
            logger.info(f'BATCH {batch_ind}: Loading model {ctx.base_model} at unfreeze stage {ctx.unfreeze_stage} and temperature {model.temperature.cpu().item()}')
        # if use_no_sz_setting:
        #     alpha = 0.1
        # else:
        alpha = 0.5
        teacher_model = load_model('models/base_model.pt', unfreeze_stage = ctx.unfreeze_stage)
        
        y_label_train = np.array([y[1] if y is not None else None for _, y in train_dataset])
        # type_label_train = get_type_label(y_label_train,prev_pred_train).values
        # y_label = (type_label_train!='TN').astype(int)
        # pos_weight = (np.sum(y_label != 1)/np.sum(y_label == 1))
        pos_weight = (np.sum(y_label_train != 1)/np.sum(y_label_train == 1))
        # if np.isnan(pos_weight) or np.isinf(pos_weight) or pos_weight == 0:
        #     pos_weight = 1
        # logger.info(f'BATCH {batch_ind}: Train data loss function pos_weight {pos_weight}')
        # mlflow.log_param('pos_weight', pos_weight)
        # sample_weight = np.array([pos_weight if x==1 else 1 for x in y_label])
        # pos_weight = np.array([1, pos_weight, 1, 1, 1, 1])
        loss_fn = WeightedKLDivWithLogitsLoss()#class_weights=pos_weight)
        # for validation data, add some positive samples to avoid predicting everything as negative, add pos_weight to add weights for rare positive samples
        y_label_val = np.array([y[1] if y is not None else None for _, y in val_dataset])
        # type_label_val = get_type_label(y_label_val,prev_pred_val).values
        # y_label = (type_label_val!='TN').astype(int)
        pos_weight = (np.sum(y_label_val != 1)/np.sum(y_label_val == 1))
        if np.isnan(pos_weight) or np.isinf(pos_weight) or pos_weight == 0:
            pos_weight = 1
        mlflow.log_param('val_pos_weight', pos_weight)
        # sample_weight = np.array([pos_weight if x==1 else 1 for x in y_label])
        pos_weight = np.array([1, pos_weight, 1, 1, 1, 1])
        logger.info(f'BATCH {batch_ind}: Validation data loss function pos_weight {pos_weight}')
        val_loss_fn = WeightedKLDivWithLogitsLoss(class_weights=pos_weight)
        # val_loss_fn = WeightedKLDivWithLogitsLoss()
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam(model.parameters(), lr=ctx.lr, weight_decay=ctx.wd, betas = (0.9,0.999),eps = 1.0*1e-8)
        logger.info(f'BATCH {batch_ind}: Loss function pos_weight {pos_weight}.')
        logger.info(f"BATCH {batch_ind}: Learning rate of {ctx.lr} used.")
        logger.info(f"BATCH {batch_ind}: Total {ctx.num_epoch} training epochs.")
        logger.info(f"BATCH {batch_ind}: Optimizer initialized with learning rate {ctx.lr} and weight_decay {ctx.wd}")
        # get current best balanced accuracy and validation loss
        if ctx.batch_max_step >= 0:
            best_metric = ctx.last_state[['validation_loss', 'auroc','auprc','recall', 'fp','precision','f1','mcc', 'ece', 'brier']].to_dict()
        else:
            model.eval()
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    output, _ = model(x_batch)
                    if use_teacher:
                        teacher_logits, _ = teacher_model(x_batch)
                        teacher_probs = F.softmax(teacher_logits, dim=1)
                        student_log_probs = F.log_softmax(output, dim=1)
                        kl = kl_loss(student_log_probs, teacher_probs)
                        ce = val_loss_fn(output, y_batch)
                        print(kl,ce)
                        val_loss = alpha * ce + (1 - alpha) * kl
                    else:
                        val_loss = val_loss_fn(output, y_batch)
                    prob = F.softmax(output,dim=1)[:,1].unsqueeze(1).cpu().numpy().flatten()
                    pred = get_prediction(prob)
                    # _, pred = smooth_pred_and_extend(prob, thres_high = ctx.thres_high, thres_low = ctx.thres_low, return_pred = True)#prob >= ctx.use_thres
                    y_true = y_batch[:,1].unsqueeze(1).cpu().numpy().flatten()
                    # metric = get_optimal_auc(y_true, prob, opt_method='yodenj')
                    metric = compute_metrics(y_true, pred, prob)
                    best_metric = {'auroc':metric['auroc_sample'],
                              'auprc':metric['auprc_sample'],
                              'recall':metric['recall_event'],
                              'fp':metric['fp'],
                              'precision':metric['precision_event'],
                              'f1':metric['f1_event'],
                              'mcc':metric['mcc']}
                    # best_metric = get_optimal_auc(y_true, prob, opt_method='yodenj')
                    best_metric.update({'validation_loss':val_loss.item(),
                            'ece':compute_ece(y_true, prob),
                            'brier':brier_score_loss(y_true, prob)})
            # mlflow logging
            mlflow.log_metrics(best_metric, step = 0)
            for k, v in best_metric.items():
                logger.info(f'BATCH {batch_ind}: Initial best {k} set to {v:.4f}')
            # logger.info(f'BATCH {batch_ind}: Initial best balanced accuracy set to {best_metric["balanced_accuracy"]:.4f}')
            # logger.info(f'BATCH {batch_ind}: Initial best AUPRC set to {best_metric["auprc"]:.4f}')
            # logger.info(f'BATCH {batch_ind}: Initial best AUROC set to {best_metric["auroc"]:.4f}')
            # logger.info(f'BATCH {batch_ind}: Initial best ECE set to {best_metric["ece"]:.4f}')
            # logger.info(f'BATCH {batch_ind}: Initial best Brier Loss set to {best_metric["brier"]:.4f}')
            # write state
            state = {"patient":ctx.current_patient, "batch": ctx.batch_ind, "sequence": 0,
                    'step':'base_model_performance', 'is_best': True, 
                    "timestamp": datetime.now(),'train_loss':None}
            state.update(best_metric)
            write_state(state)
        if best_metric['auroc'] > 0.98 or best_metric['auprc'] > 0.95:#or best_metric['auroc'] > 0.98:
            logger.info(f'BATCH {batch_ind}: Model showed perfect performance, skip finetuning')
            return
        if best_metric['auroc'] < 0.5:#or best_metric['auroc'] > 0.98:
            logger.info(f'BATCH {batch_ind}: Best AUROC < 0.5, seem to be a wrong annotation, skipping.')
            try:
                os.remove(os.path.join(ctx.sz_data_dir, f"finetune_{ctx.batch_ind}.pt"))
            except:
                pass
            return
        
        # training epochs
        counter = 0 # counter of not improving steps
        unfreeze_stage = ctx.unfreeze_stage
        for epoch in range(1, ctx.num_epoch+1):
            if ctx.batch_max_step >= epoch:
                continue
            if use_no_sz_setting and epoch > 10:
                continue
            logger.info(f'Processing epoch {epoch}')
            
            # if epoch <= 8:
            #     train_loader = ConfMatLoader(train_dataset, cleaned_prev_pred, class_weights = 'auto', batch_size=ctx.batch_size)
            # else:
            #     train_loader = ConfMatLoader(train_dataset, cleaned_prev_pred, class_weights = {'TN':1, 'TP':2, 'FP':4, 'FN':4}, batch_size=ctx.batch_size)
            
            logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Train set size {len(train_dataset)}')
            # train
            train_losses = []
            model.train()
            for x_batch, y_batch in tqdm(train_loader):
                optimizer.zero_grad()
                output, _ = model(x_batch)
                if use_teacher:
                    teacher_logits, _ = teacher_model(x_batch)
                    teacher_probs = F.softmax(teacher_logits, dim=1)
                    student_log_probs = F.log_softmax(output, dim=1)
                    kl = kl_loss(student_log_probs, teacher_probs)
                    ce = loss_fn(output, y_batch)
                    loss = alpha * ce + (1 - alpha) * kl
                else:
                    loss = loss_fn(output, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            train_loss = np.mean(train_losses)
            mlflow.log_metric('train_loss', train_loss, step = epoch)
            logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Train loss {train_loss:.4f}')
            
            # evaluation
            model.eval()
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    output, _ = model(x_batch)
                    if use_teacher:
                        teacher_logits, _ = teacher_model(x_batch)
                        teacher_probs = F.softmax(teacher_logits, dim=1)
                        student_log_probs = F.log_softmax(output, dim=1)
                        kl = kl_loss(student_log_probs, teacher_probs)
                        ce = val_loss_fn(output, y_batch)
                        print(kl,ce)
                        val_loss = alpha * ce + (1 - alpha) * kl
                    else:
                        val_loss = val_loss_fn(output, y_batch)
                    prob = F.softmax(output,dim=1)[:,1].unsqueeze(1).cpu().numpy().flatten()
                    pred = get_prediction(prob)#prob >= ctx.use_thres
                    y_true = y_batch[:,1].unsqueeze(1).cpu().numpy().flatten()
                    metric = compute_metrics(y_true, pred, prob)
                    metric = {'auroc':metric['auroc_sample'],
                              'auprc':metric['auprc_sample'],
                              'recall':metric['recall_event'],
                              'fp':metric['fp'],
                              'precision':metric['precision_event'],
                              'f1':metric['f1_event'],
                              'mcc':metric['mcc']}
                    metric.update({'validation_loss':val_loss.item(),
                            'ece':compute_ece(y_true, prob),
                            'brier':brier_score_loss(y_true, prob)})
            mlflow.log_metrics(metric, step = epoch)
            for k, v in metric.items():
                logger.info(f"BATCH {batch_ind}: {k} is {v:.4f}, current best {k} is {best_metric[k]:.4f}")
            # logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Validation loss {metric["validation_loss"]:.4f}, current best validation loss {best_metric["validation_loss"]:.4f}')
            # logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Balanced accuracy {metric["balanced_accuracy"]:.4f}, current best balanced accuracy {best_metric["balanced_accuracy"]:.4f}')
            # logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Balanced AUPRC {metric["auprc"]:.4f}, current best AUPRC {best_metric["auprc"]:.4f}')
            # logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Balanced AUROC {metric["auroc"]:.4f}, current best AUROC {best_metric["auroc"]:.4f}')
            # logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Balanced ECE {metric["ece"]:.4f}, current best ECE {best_metric["ece"]:.4f}')
            # logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Balanced Brier Loss {metric["brier"]:.4f}, current best Brier Loss {best_metric["brier"]:.4f}')

            # if metric['validation_loss'] < best_metric['validation_loss']-0.01:
            if metric['auroc'] is not np.nan:
                update = (metric['auroc'] > best_metric['auroc'] and metric['auprc'] > best_metric['auprc']) #metric['validation_loss'] < best_metric['validation_loss'] or 
            else:
                update = metric['validation_loss'] < best_metric['validation_loss']-0.01
            if update:
                logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Performance improved.')
                best_metric = metric
                counter = 0
                checkpoint = model.state_dict()
                torch.save(checkpoint, out_model)
                logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Patience counter set to {counter}.')
                logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Checkpoint saved.')
                state = {"patient":ctx.current_patient, "batch": ctx.batch_ind, "sequence": epoch,
                    'step':f'finetune_epoch_{epoch}', 'is_best': True, 
                    "timestamp": datetime.now(),'train_loss':train_loss}
                state.update(metric)
                write_state(state)
                if best_metric['auroc'] > 0.98 or best_metric['auprc'] > 0.95: #or best_metric['auroc'] > 0.98:
                    logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Early stopping due to perfect performance')
                    break 
            else:
                logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Performance not improved.')
                counter += 1
                logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Patience counter increase to {counter}.')
                state = {"patient":ctx.current_patient, "batch": ctx.batch_ind, "sequence": epoch,
                    'step':f'finetune_epoch_{epoch}', 'is_best': False, 
                    "timestamp": datetime.now(),'train_loss':train_loss}
                state.update(metric)
                write_state(state)
                if counter >= ctx.patience:
                    logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Counter greater than patience {ctx.patience}.')
                    logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Early stopping')
                    state = {"patient":ctx.current_patient, "batch": ctx.batch_ind, "sequence": ctx.num_epoch,
                    'step':f'early_stopping', 'is_best': True, 
                    "timestamp": datetime.now(),'train_loss':train_loss}
                    write_state(state)
                    break
                elif counter % ctx.unfreeze_step == 0 and ctx.unfreeze and unfreeze_stage < len(unfreeze_order)-1:
                    logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Increasing unfreezing level')
                    unfreeze_stage += 1
                    for name, module in model.named_modules():
                        if name == unfreeze_order[unfreeze_stage]:
                            for param in module.parameters():
                                param.requires_grad = True
                    logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Model unfreezing stage increase to {unfreeze_order[unfreeze_stage]}')

    # model calibration
    if ctx.do_cal:
        if not os.path.exists(out_model):
            logger.info(f'BATCH {batch_ind}: No improved model found. Skipping Calibration.')
        else:
            model = load_model(out_model)
            logger.info(f'BATCH {batch_ind}: Loading model finetuned_{batch_ind}')
            model = calibrate_temp(model, val_loader)
            logger.info(f'BATCH {batch_ind}: Temperature calibrated to be {model.temperature.cpu().item()}')
            checkpoint = model.state_dict()
            torch.save(checkpoint, out_model) 
        state = {"patient":ctx.current_patient, "batch": ctx.batch_ind, "sequence": ctx.num_epoch+1,
                'step':f'calibration', 'is_best': True, 
                "timestamp": datetime.now()}
        write_state(state)


# __________________Predict Probability__________________________
def sparcnet_single(model, data_df, fs):
    data = data_df.drop(columns=['Fz-Cz','Cz-Pz'])  # Remove last column (assumed label or timestamp)
    data_values = data.values
    data_values = bandpass_filter(data_values, fs, lo=1, hi=40)
    data_values = downsample(data_values, fs, 200)
    data_values = np.clip(data_values, -500, 500) 
    data = torch.from_numpy(data_values.astype(np.float32))
    data = data.T.unsqueeze(0).to(device)
    output, _ = model(data)
    prob = F.softmax(output, 1).detach().cpu().numpy().flatten()
    prob_temp = F.softmax(output/model.temperature, 1).detach().cpu().numpy().flatten()
    return prob, prob_temp

def process_window(model, df, fs):
    tmp_df = jsondecode_df(df)
    data_df = pd.concat([pd.DataFrame(d['BIPOLAR'],index = d['timestamps']) for _, d in tmp_df.iterrows()]).sort_index()
    col_labels = ['SZ','LPD','GPD','LRDA','GRDA','OTHER']
    prob, prob_temp = sparcnet_single(model, data_df, fs)
    return pd.DataFrame([np.concatenate([prob, prob_temp])], columns=col_labels+[c+'_temp' for c in col_labels])

def process_batch(batch_id, model, pred_ctx):
    out_path = f'{pred_ctx.tmp_dir}/{batch_id}.csv'
    if not pred_ctx.force and os.path.exists(out_path):
        return
    else:
        warnings.filterwarnings('ignore')
        conn = sqlite3.connect(f'data/{pred_ctx.current_patient}.db')
        batch_start = pred_ctx.batch_start[batch_id]
        batch_end = batch_start + pd.Timedelta(hours = pred_ctx.batch_hour)
        # offset = 0 if batch_id == 0 else batch_id*pred_ctx.batch_dura-8
        # query = f"SELECT * FROM {pred_ctx.current_patient} ORDER BY start_timestamp LIMIT {pred_ctx.batch_dura} OFFSET {offset}"
        query = f"SELECT * FROM {pred_ctx.current_patient} WHERE start_timestamp >= '{batch_start}' AND start_timestamp < '{batch_end}'"
        df = pd.read_sql(query,conn)
        if df.empty:
            return
        df['start_timestamp'] = pd.to_datetime(df["start_timestamp"]).dt.tz_localize(None).apply(lambda x:x.replace(microsecond=0) + timedelta(seconds=(x.microsecond >= 500000)))
        with open(f'labels/{pred_ctx.current_patient}_feat_labels.pkl','rb') as f:
            label_df = pickle.load(f)
        label_df = label_df.reset_index().rename(columns={'index': 'timestamp'})
        label_df['window_starts'] = label_df['timestamp'] - pd.Timedelta(seconds=feat_setting['win'])
        label_df = label_df[(label_df['window_starts'] >= batch_start)&(label_df['timestamp'] < batch_end)]
        # indices = range(0,len(df)-feat_setting['win']+1,feat_setting['stride'])
        feat_list = []
        for _, row in tqdm(label_df.iterrows(), total = len(label_df), desc = 'Window'):
            tmp = df[(df['start_timestamp'] >= row['window_starts'])&(df['start_timestamp'] < row['timestamp'])]
            feat_list.append(process_window(model, tmp, pred_ctx.sampling_freqs[pred_ctx.current_patient]))
        feat_list = pd.concat(feat_list)
        feat_list['timestamp'] = label_df['timestamp'].values
        feat_list['label'] = label_df['label'].values
        # feat_list = pd.concat([feat_list, label_df], axis=1, ignore_index=True)
        feat_list['batch_id'] = batch_id
        feat_list = feat_list.sort_values('timestamp')
        feat_list.to_csv(out_path,index=False)

def predict_proba():
    n_batch = pred_ctx.n_batch
    model = load_model(pred_ctx.eval_model, mode = 'eval')
    with tqdm(total=n_batch,desc = 'Processing block'):
        results = Parallel(n_jobs=5)(delayed(process_batch)(i, model, pred_ctx) for i in range(n_batch))
    # concate results
    all_files = glob.glob(f'{pred_ctx.tmp_dir}/*.csv')
    feat_df = pd.concat([pd.read_csv(f,parse_dates=['timestamp']) for f in all_files],ignore_index=False).sort_values(by='timestamp').reset_index(drop=True)
    # feat_df['timestamp'] = feat_df['timestamp'].dt.round('1s')
    feat_df.to_csv(pred_ctx.prob_file,index=False)
    for file in all_files:
        os.remove(file)

# __________________Binary prediction__________________________
def binary_prediction():
    # get prediction
    prob_df = pd.read_csv(pred_ctx.prob_file)
    prob = prob_df.iloc[:,:6].values
    prob_temp = prob_df.iloc[:,6:12].values # temperature calibrated
    # func = lambda x: ((x[:,1] >= pred_ctx.thres).astype(int), x[:,1])
    sz_prob = prob[:,1]
    sz_prob_temp = prob_temp[:,1]
    pred_df = pd.DataFrame(np.vstack([sz_prob, sz_prob_temp]).T,columns=['sz_prob','sz_prob_temp'])
    pred = get_prediction(sz_prob)
    pred_temp = get_prediction(sz_prob_temp)
    pred_df['smoothed_pred'] = pred
    pred_df['smoothed_pred_temp'] = pred_temp
    # pred_df['smoothed_pred'] = smooth_pred(pred_df['pred'].values)
    # pred_df['smoothed_pred_temp'] = smooth_pred(pred_df['pred_temp'].values)
    # _, pred_df['smoothed_pred'] = smooth_pred_and_extend(sz_prob, thres_high = pred_ctx.thres_high, thres_low = pred_ctx.thres_low, return_pred=True)
    # _, pred_df['smoothed_pred_temp'] = smooth_pred_and_extend(sz_prob_temp, thres_high = pred_ctx.thres_high, thres_low = pred_ctx.thres_low,return_pred=True)
    # pred_df['smoothed_pred_filtered'] = smooth_pred_filtering(pred_df['smoothed_pred'].values, sz_prob)
    # pred_df['smoothed_pred_temp_filtered'] = smooth_pred_filtering(pred_df['smoothed_pred_temp'].values, sz_prob_temp)
    # if 'base_model' not in pred_ctx.prob_file and pred_ctx.adjusted_thres is not None:
    #     pred_df['smoothed_pred_temp_adjth']= smooth_pred((pred_df['sz_prob_temp'].values >= pred_ctx.adjusted_thres).astype(int))
    #     pred_df['smoothed_pred_temp_adjth_filtered']= smooth_pred_filtering(pred_df['smoothed_pred_temp_adjth'].values, prob_temp)
    pred_df = pd.concat([pred_df, prob_df.iloc[:,12:]],axis=1)
    pred_df.to_csv(pred_ctx.pred_file,index=False)

#_________________________Metric Calculation___________________________

def calc_metric_batch(feat_data):
    suffix = ['','_temp']#,'_filtered','_temp_filtered']#,'_temp_adjth','_temp_adjth_filtered'
    row_label = ['Finetune','Finetune+Calibrate']#, 'Finetune+Filtering','FC+Filtering']#,'FC+AdjustThres','FC+AdjThres+Filtering'
    labels = feat_data['label'].astype(int).to_numpy()
    all_metrics = []
    for e, suf in enumerate(suffix):
        if 'smoothed_pred'+suf in feat_data.columns:
            pred = feat_data['smoothed_pred'+suf].astype(int).to_numpy()
            if 'temp' in suffix:
                prob = feat_data['sz_prob_temp'].to_numpy()
            else:
                prob = feat_data['sz_prob'].to_numpy()
            metrics = compute_metrics(labels, pred, prob)
            metrics = pd.DataFrame([metrics], index=[row_label[e]])
            all_metrics.append(metrics)
    all_metrics = pd.concat(all_metrics).reset_index(names='setting')
    all_metrics['batch_id'] = feat_data['batch_id'].iloc[0]
    return all_metrics

def calculate_metrics():
    feat_data = pd.read_csv(pred_ctx.pred_file, parse_dates=['timestamp'])
    # labels = pd.read_pickle(f'labels/{pred_ctx.current_patient}_feat_labels.pkl')
    # feat_data['timestamp'] = feat_data['timestamp'].tz_localize(None)
    # feat_data = feat_data.merge(labels, how = 'left', left_on='timestamp', right_index = True)
    metrics = feat_data.groupby('batch_id').apply(calc_metric_batch).reset_index(drop=True).set_index('batch_id')
    full_metrics = calc_metric_batch(feat_data).reset_index(drop=True).drop(columns='batch_id')
    metrics.to_csv(pred_ctx.metric_file)
    full_metrics.to_csv(pred_ctx.metric_file.replace('.csv','_full.csv'),index=False)


# _____________________________________Plot Functions_____________________________________________

def plot_trajectory(overlay = False):
    plot_vars = pred_ctx.plot_vars
    metrics_df = pd.read_csv(pred_ctx.metric_file, index_col=0)
    col = 'b'
    if overlay:
        baseline_metrics_df = pd.read_csv(pred_ctx.baseline_metric_file, index_col=0)
        col = 'r'
        overlay_col = 'b'
    for setting in ['Finetune','Finetune+Calibrate']:
        metrics = metrics_df[metrics_df['setting']==setting]
        if overlay:
            baseline_metrics = baseline_metrics_df[baseline_metrics_df['setting'] == setting]
        # 1. plot a trajectory of the model over time
        fig, axes = plt.subplots(len(plot_vars), 1, figsize=(10, 10), sharex=True)
        for e, var in enumerate(plot_vars):
            if overlay:
                axes[e].plot(baseline_metrics.index.astype(int), baseline_metrics[var].values,label='baseline',color=overlay_col, marker='o', markerfacecolor='white')        
            axes[e].plot(metrics.index, metrics[var].values, color = col, marker='o', markerfacecolor='white')
            axes[e].set_ylabel(metric_labels[var])
            axes[e].legend()    
        axes[e].set_xlabel("Batch")
        axes[e].xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
        plt.tight_layout()
        # fig.savefig(pred_ctx.trajectory_file+'.png', dpi=300)
        if overlay:
            fig.savefig(f'{pred_ctx.overlay_file}_{setting}.svg', dpi=300)
        else:
            fig.savefig(f'{pred_ctx.trajectory_file}_{setting}.svg', dpi=300)
        plt.close(fig)

def plot_paired_line(df1, df2, variables=None, figsize=(15, 6), save_path=None):
    if variables is None:
        variables = df1.columns.tolist()
    fig, axes = plt.subplots(1, len(variables), figsize=figsize, sharex=False)
    if len(variables) == 1:
        axes = [axes]  # make iterable
    for ax, var in zip(axes, variables):
        # Plot individual subjects
        for i in range(len(df1)):
            ax.plot([0, 1], [df1[var].iloc[i], df2[var].iloc[i]], 
                    marker="o", color="gray", alpha=0.6)
        ax.plot([0, 1], 
                [df1[var].mean(), df2[var].mean()], 
                marker="o", color="red", linewidth=2)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_title(metric_labels[var])
    plt.tight_layout()
    if save_path:
        # plt.savefig(save_path+'.png', dpi=300)
        plt.savefig(save_path+'.svg', dpi=300)
        plt.close(fig)

def plot_calibration(overlay=False):
    
    prob_df = pd.read_csv(pred_ctx.prob_file)
    true = prob_df['label'].values
    prob = prob_df['LPD'].values
    prob_cal = prob_df['LPD_temp'].values
    data_dict = {'Finetuned':prob,'Finetuned+Calibration':prob_cal}
    if overlay:
        baseline_prob_df = pd.read_csv(pred_ctx.baseline_prob_file)
        baseline_prob = baseline_prob_df['LPD'].values
        data_dict['Baseline'] = baseline_prob
    
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(3, 2)
    colors = plt.get_cmap("Dark2")
    ax_calibration_curve = fig.add_subplot(gs[:3, :1])
    calibration_displays = {}
    markers = ["^", "v", "s", "o"]
    for i, (name,data) in enumerate(data_dict.items()):
        display = CalibrationDisplay.from_predictions(true, data, n_bins=10,name=name,ax=ax_calibration_curve,color=colors(i),marker=markers[i])
        calibration_displays[name] = display
    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots")
    ax_calibration_curve.legend(loc='upper left')
    # Add histogram
    grid_positions = [(0, 1), (1, 1), (2, 1)]
    for i, name in enumerate(data_dict.keys()):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])
        ax.hist(
            calibration_displays[name].y_prob,
            range=(0, 1),
            bins=10,
            label=name,
            color=colors(i),
        )
        ax.set(title=name)
        if i < len(data_dict)-1:
            ax.set_xticklabels([])
    ax.set(xlabel="Mean predicted probability")
    plt.show()
    if overlay:
        fig.savefig(f'{pred_ctx.cal_overlay_file}.svg', dpi=300)
    else:
        fig.savefig(f'{pred_ctx.cal_file}.svg', dpi=300)
    plt.close(fig)

def plot_auroc(overlay=False):
    prob_df = pd.read_csv(pred_ctx.prob_file)
    true = prob_df['label'].values
    prob = prob_df['LPD'].values
    prob_cal = prob_df['LPD_temp'].values
    if len(np.unique(true)) == 1:
        return
    data_dict = {'Finetuned':prob,'Finetuned+Calibration':prob_cal}
    if overlay:
        baseline_prob_df = pd.read_csv(pred_ctx.baseline_prob_file)
        baseline_prob = baseline_prob_df['LPD'].values
        data_dict['Baseline'] = baseline_prob
    fig, axes = plt.subplots(1,2,figsize=(16, 8))
    colors = plt.get_cmap("Dark2")
    calibration_displays = {}
    markers = ["^", "v", "s", "o"]
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    baseline = np.mean(true)
    axes[1].hlines(baseline, xmin=0, xmax=1, colors="k", linestyles="--")
    for i, (name,data) in enumerate(data_dict.items()):
        fpr, tpr, _ = roc_curve(true, data)
        auc_score = auc(fpr, tpr)
        prec, recall, _ = precision_recall_curve(true, data)
        prc_score = average_precision_score(true, data)
        axes[0].plot(fpr, tpr, label=f"{name} (AUROC = {auc_score:.3f})")
        axes[1].plot(recall, prec, label=f"{name} (AUPRC = {prc_score:.3f})")
    axes[0].set_title("AUROC plots")
    axes[0].legend(loc='lower right')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[1].set_title("APRC plots")
    axes[1].legend(loc='upper right')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    plt.show()
    if overlay:
        fig.savefig(f'{pred_ctx.auroc_overlay_file}.svg', dpi=300)
    else:
        fig.savefig(f'{pred_ctx.auroc_file}.svg', dpi=300)
    plt.close(fig)

def agg_metrics(batch_inds, feat_file = None, setting = None):
    if feat_file is not None:
        feat_data = pd.read_csv(feat_file, parse_dates=['timestamp'])
    else:
        feat_data = pd.read_csv(pred_ctx.pred_file, parse_dates=['timestamp'])
    feat_data = feat_data[feat_data['batch_id'].isin(batch_inds)]
    metrics = calc_metric_batch(feat_data).reset_index(drop=True)
    if setting is not None:
        if not isinstance(setting, list):
            setting = [setting]
        metrics = metrics[metrics['setting'].isin(setting)]
    return metrics

def plot_paired():
    plot_vars = pred_ctx.plot_vars
    baseline_metrics_df = pd.read_csv(pred_ctx.baseline_metric_file, index_col=0)
    metric_files = glob.glob(f'{pred_ctx.metric_dir}/*.csv')
    metric_files = [f for f in metric_files if 'full' not in f]
    metric_files_df = pd.DataFrame({'filename':metric_files})
    metric_files_df['batch_ind'] = metric_files_df['filename'].apply(lambda x:int(x.split('/')[-1].split('_')[1]))
    metric_files_df['adjth'] = metric_files_df['filename'].apply(lambda x: 1 if 'adjth' in x else 0)
    metric_files_df = metric_files_df.sort_values(['batch_ind', 'adjth'])
    if pred_ctx.adjth:
        metric_files_df = metric_files_df.drop_duplicates('batch_ind', keep='last')
    else:
        metric_files_df = metric_files_df.drop_duplicates('batch_ind', keep='first')

    # metric_files = sorted([f for f in metric_files if 'full' not in f and 'adjth' not in f], key = lambda x:int(x.split('/')[-1].split('_')[1]))
    metric_files = metric_files_df['filename'].to_list()
    batch_inds = metric_files_df['batch_ind'].to_list()#[int(f.split('/')[-1].split('_')[1]) for f in metric_files]
    prev_finetune_files = [pred_ctx.baseline_metric_file]+metric_files[:-1]

    out_prev = []
    out_finetune = []
    out_baseline = []
    for setting in ['Finetune','Finetune+Calibrate']:
        prev_test = []
        finetuned_test = []
        baseline_test = []
        baseline_metrics = baseline_metrics_df[baseline_metrics_df['setting'] == setting] # 0-21
        for e, (f, prev_f) in enumerate(zip(metric_files,prev_finetune_files)): # last one should be n_batch - 1
            batch_ind = batch_inds[e]
            metrics = pd.read_csv(f, index_col=0)
            metrics = metrics[metrics['setting']==setting]
            prev_metrics = pd.read_csv(prev_f, index_col=0)
            prev_metrics = prev_metrics[prev_metrics['setting']==setting]
            to_eval_batch_ind = batch_ind+1
            # to_compare_with_next_ind = batch_ind+2
            finetuned_test.append(agg_metrics(list(range(to_eval_batch_ind, len(metrics))), feat_file=f.replace('metrics','pred'), setting = setting))
            # print(list(range(to_compare_with_next_ind,len(baseline_metrics))))
            prev_test.append(agg_metrics(list(range(to_eval_batch_ind,len(prev_metrics))), feat_file = prev_f.replace('metrics','pred'), setting = setting))
            baseline_test.append(agg_metrics(list(range(to_eval_batch_ind,len(baseline_metrics))), feat_file = pred_ctx.baseline_feat_file, setting = setting))
        prev_test = pd.concat(prev_test) 
        finetuned_test = pd.concat(finetuned_test) 
        baseline_test = pd.concat(baseline_test) 

        out_prev.append(prev_test)
        out_finetune.append(finetuned_test)
        out_baseline.append(baseline_test)
        
        plot_paired_line(prev_test, finetuned_test, variables = plot_vars, save_path = pred_ctx.paired_line_test_prev+'_'+setting)
        plot_paired_line(baseline_test, finetuned_test, variables = plot_vars, save_path = pred_ctx.paired_line_test_base+'_'+setting)

    if pred_ctx.adjth:
        pd.concat(out_prev).to_csv(f'{pred_ctx.agg_dir}/prev_finetune_adjth.csv',index=False)
        pd.concat(out_finetune).to_csv(f'{pred_ctx.agg_dir}/finetune_adjth.csv',index=False)
        pd.concat(out_baseline).to_csv(f'{pred_ctx.agg_dir}/baseline_adjth.csv',index=False) 
    else:
        pd.concat(out_prev).to_csv(f'{pred_ctx.agg_dir}/prev_finetune.csv',index=False)
        pd.concat(out_finetune).to_csv(f'{pred_ctx.agg_dir}/finetune.csv',index=False)
        pd.concat(out_baseline).to_csv(f'{pred_ctx.agg_dir}/baseline.csv',index=False) 


def generate_plots():
    plot_trajectory()
    plot_auroc()
    plot_calibration()
    if pred_ctx.plot_overlay:
        plot_trajectory(overlay = True)
        plot_auroc(overlay = True)
        plot_calibration(overlay = True)
    if pred_ctx.plot_paired_line:
        plot_paired()