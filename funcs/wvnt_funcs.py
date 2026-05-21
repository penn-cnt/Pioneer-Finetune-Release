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

import mne
mne.set_log_level('ERROR')
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
code_v2_path = os.path.join(working_dir,'funcs/WVNT')
sys.path.append(code_v2_path)

# custom utils
from funcs.utils import *
from funcs.feat_funcs import *
from funcs.context import pred_ctx, ctx
from funcs.metric_funcs import compute_metrics, compute_ece, metric_labels, get_optimal_auc

# wvnt imports
from WVNT.model import WVNT
from WVNT.WaveModel import *
# from SPaRCNet.utils import *

# setup logging
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

feat_setting = {'win':int(1),
                'stride':int(1),
                'reref':'BIPOLAR',
                'prewhite':True}
# clip_loader = ClipLoader(feat_setting,aligned='right')

# model settings
device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")
unfreeze_order = [
        'fc2', 
        'fc1',
        'conv3',
        'conv2',
        'conv1'
    ]

def load_model(model_path, mode = 'train', unfreeze_stage = 1):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = WaveModel()
    model.load_state_dict(checkpoint)
    model.to(device)
    if mode == 'train':
        unfreeze_modules = unfreeze_order[:unfreeze_stage+1]
        for name, module in model.named_modules():
            for param in module.parameters():
                param.requires_grad = False 
        for name, module in model.named_modules():
            if name in unfreeze_modules:
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

class WaveNetDataset(Dataset):
    def __init__(self, data_df, labels, fs, windows = None, device = device): # should have a bad/valid here
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

    def __len__(self):
        return len(self.windows)

    
    def __process_data__(self, idx):
        window = self.windows.iloc[idx]
        if self.time_window:
            data = self.data[(self.data['start_timestamp'] >= window['window_start']) & (self.data['start_timestamp'] < window['window_end'])]
        else:
            data = self.data.iloc[window['window_start']:window['window_end']]#tmp_df = jsondecode_df(data)
        data_df = pd.concat([pd.DataFrame(d['BIPOLAR'],index = d['timestamps']) for _, d in data.iterrows()]).sort_index()
        data_df.iloc[1:,:] = ar_one(data_df.values)
        data_df = pd.DataFrame(ctx.scaler.transform(data_df), columns=data_df.columns, index=data_df.index)
        data_values = downsample(data_df, ctx.sampling_freq, 128).values
        return data_values
        
    def __getitem__(self, idx):
        data = self.__process_data__(idx)
        label = self.window_label[idx]
        values = torch.from_numpy(data.T).float().unsqueeze(-1).to(self.device)
        labels = torch.from_numpy(np.array(label)).float().to(self.device)
        return values, labels

def aggregate_chunk(chunk):
    data = {col: chunk[col].tolist() for col in chunk.columns}
    stamps = chunk.index.tolist()
    agg = pd.DataFrame([{'BIPOLAR':data,'timestamps':stamps}])
    return agg

def get_dataset(data_df, labels, fs):
    # num_fp = np.sum(conf_ind != 'TN')
    # labels = labels[(conf_ind != 'TN')]
    windows = labels.copy().reset_index().rename(columns={'index': 'window_end'})
    windows['window_start'] = windows['window_end']-pd.Timedelta(seconds=feat_setting['win'])
    labels = labels.values
    if np.any(labels == 1):
        sz_idx = np.where(labels == 1)[0]
        non_sz_idx = np.where(labels!=1)[0]
        num_train_sz = int(len(sz_idx)*0.85)
        num_train_non_sz = int(len(non_sz_idx)*0.85)
        train_idx = np.concatenate([sz_idx[:num_train_sz],non_sz_idx[:num_train_non_sz]]).flatten()
        train_idx.sort()
        val_idx = np.concatenate([sz_idx[num_train_sz:],non_sz_idx[num_train_non_sz:]]).flatten()
        val_idx.sort()
        train_windows = windows.iloc[train_idx]
        val_windows = windows.iloc[val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
    else:
        num_train = int(len(windows)*0.85)
        train_windows = windows.iloc[:num_train]
        val_windows = windows.iloc[num_train:]
        train_labels = labels[:num_train]
        val_labels = labels[num_train:]
    train_dataset = WaveNetDataset(data_df, train_labels, fs, windows = train_windows)
    val_dataset = WaveNetDataset(data_df, val_labels, fs, windows = val_windows)
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
            y_label = y_batch.view(-1)
            pos_weight = (y_label.shape[0]-torch.sum(y_label==1))/torch.sum(y_label==1).numpy()
            if np.isnan(pos_weight) or np.isinf(pos_weight):
                pos_weight = 1
            raw_logits = model(x_batch, return_type='raw')
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = torch.optim.LBFGS([model.model.temperature], lr=lr, max_iter=max_iter)
    def eval_step():
        optimizer.zero_grad()
        # logits = (raw_logits/model.model.temperature).view(x_batch.shape[0], x_batch.shape[-1])          # reshape back
        probs = F.softmax(raw_logits/model.model.temperature,dim=1)[:,1].view(x_batch.shape[0], x_batch.shape[1])#.cpu().numpy().flatten()
        probs = torch.quantile(probs, q=0.8, dim=1)
        loss = loss_fn(probs, y_label)
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
        fpr, tpr, _ = roc_curve(true, data)
        auc_score = auc(fpr, tpr)
        prec, recall, _ = precision_recall_curve(true, data)
        prc_score = average_precision_score(true, data)
        axes[0].plot(fpr, tpr, label=f"{name} (AUROC = {auc_score:.3f})")
        axes[1].plot(recall, prec, label=f"{name} (AUPRC = {prc_score:.3f})")
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

class ChannelModelWrapper(torch.nn.Module):
    def __init__(self, channel_model, pool = '80th'):
        super().__init__()
        self.model = channel_model   # must accept [1,1,T] and output 1 logit
        self.model.eval()
        self.pool = pool

    def mil_pool_lse(self, logits: torch.Tensor, tau: float = 1.0):
        # logits: [B, C]
        # returns bag-level logits [B]
        # scaled LSE: logsumexp(logits / tau) * tau  (equivalently tau * LSE(logits/tau))
        return torch.logsumexp(logits / tau, dim=1) * tau

    def mil_pool_topk_mean(self, logits: torch.Tensor, k: int):
        # logits: [B, C]
        if k <= 0:
            raise ValueError("k must be >=1")
        # get top-k values along channel dim
        topk_vals, _ = torch.topk(logits, k=min(k, logits.size(1)), dim=1)
        return topk_vals.mean(dim=1)
    
    def forward(self, x, temp = False, return_type = 'bag'):
        """
        x: [B, C, 1, T]
        return: logits per channel [B, C]
        return_type, bag, reshape, raw
        """
        x = x.permute(0, 1, 3, 2)
        B, C, _, _ = x.shape
        x = x.view(B * C, 1, x.shape[-1])   # feed channels separately
        raw_logits = self.model(x)
        if temp:
            raw_logits = raw_logits/self.model.temperature
        logits = raw_logits[:,1].view(B, C)          # reshape back
        probs = F.softmax(raw_logits,dim=1)[:,1].view(B,C)#.cpu().numpy().flatten()
        if self.pool == 'topk':
            k = int(max(1, int(C*ctx.perc)))
            bag_logits = self.mil_pool_topk_mean(logits, k)
            bag_probs = self.mil_pool_topk_mean(probs, k)
        elif self.pool == 'lse':
            tau = 0.3
            bag_logits = self.mil_pool_lse(logits, tau=tau)
            bag_probs = torch.sigmoid(bag_logits)
        elif self.pool == 'max':
            bag_logits = logits.max(dim=1)
            bag_probs = probs.max(dim=1)
        elif self.pool == '95th':
            bag_logits = torch.quantile(logits, q=0.95, dim=1)
            bag_probs = torch.quantile(probs, q=0.95, dim=1)
        elif self.pool == '90th':
            bag_logits = torch.quantile(logits, q=0.9, dim=1)
            bag_probs = torch.quantile(probs, q=0.9, dim=1)
        elif self.pool == '80th':
            bag_logits = torch.quantile(logits, q=0.8, dim=1)
            bag_probs = torch.quantile(probs, q=0.8, dim=1)
        if return_type == 'raw':
            return raw_logits
        elif return_type == 'bag':
            return bag_logits, bag_probs
        elif return_type == 'reshape':
            return logits, probs
        elif return_type == 'all':
            return raw_logits, bag_logits, bag_probs

def get_type_label(labels, prev_pred):
    TP = (labels == 1) & (prev_pred == 1)
    FP = (labels == 0) & (prev_pred == 1)
    TN = (labels == 0) & (prev_pred == 0)
    FN = (labels == 1) & (prev_pred == 0)
    type_df = pd.DataFrame({'TN':TN,'TP':TP,'FP': FP, 'FN': FN})
    type_label = type_df.idxmax(axis=1)
    return type_label

def get_sample_weights(labels, prev_pred, class_weights = {'TN':0.2, 'TP':1, 'FP':4, 'FN':6}):
    type_label = get_type_label(labels, prev_pred)
    weights = type_label.map(lambda x: class_weights.get(x, 1))  # Default to 1 if class is missing in class_weights
    return weights

def ConfMatLoader(dataset, prev_pred, class_weights = {'TN':0.2, 'TP':1, 'FP':4, 'FN':6}, batch_size=32, pos_multiplier=5, shuffle = True):

    labels = torch.tensor([y for _, y in dataset]).detach().numpy()
    type_label = get_type_label(labels, prev_pred)
    if class_weights == 'auto':  
        unique, counts = np.unique(type_label.to_numpy(), return_counts=True)
        conf_perc = {k:counts/len(type_label) for k, counts in zip(unique, counts)}
        logger.info(f'Percentages {conf_perc}')
        class_weights = {k:np.min([1/v,60]) if v != 0 else 0 for k,v in conf_perc.items()}
        logger.info(f'Weight {class_weights}')
    sample_weights = type_label.map(lambda x: class_weights.get(x, 1))
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader

def BalancedLoader(dataset, batch_size=32, pos_multiplier=5, shuffle = True):
    """
    pos_multiplier: how many times to oversample positives
                    (2–5 is common; don’t oversample too much)
    """
    labels = torch.tensor([y for _, y in dataset])
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


def finetune():
    fs = ctx.sampling_freq
    batch_ind = ctx.batch_ind
    base_model = ctx.base_model
    base_model_ins = ChannelModelWrapper(load_model(base_model, mode='eval'))
    out_model = ctx.out_model
    use_no_sz_setting = False
    use_teacher = True

    pred = ctx.prev_pred_batch['smoothed_pred_temp'].values
    if ctx.mode == 'pioneer':
        if np.all(pred==0):
            logger.info(f'BATCH {batch_ind}: Skip finetuning due to no seizure detection during this session.')
            return
    elif ctx.mode == 'HITL':
        pass
    else:
        df, labels = ctx.get_batch_data()
        if np.all(pred==0) and np.all(labels == 0):
            logger.info(f'BATCH {batch_ind}: Skip finetuning due to no seizure detection and no seizure events during this session.')
            state = {"patient":ctx.current_patient, "batch": ctx.batch_ind, "sequence": ctx.num_epoch+3,
                    'step':f'skipping finetuning', 'is_best': True, 
                    "timestamp": datetime.now()}
            write_state(state)
            return
        elif np.all(pred == 0) and np.any(labels == 1):
            logger.info(f'BATCH {batch_ind}: No seizure detection but seizure events during this session.')
        elif np.any(pred == 1) and np.all(labels == 0):
            logger.info(f'BATCH {batch_ind}: Has seizure detection but no seizure events during this session.')
        elif np.any(pred == 1) and np.any(labels == 1):
            logger.info(f'BATCH {batch_ind}: Has seizure detection and seizure events during this session.')
    train_dataset, val_dataset = get_dataset(df, labels, fs)
    prev_pred_train = train_dataset.windows.merge(ctx.prev_pred_batch[['timestamp','smoothed_pred_temp']], left_on='window_end', right_on='timestamp', how='left')['smoothed_pred_temp'].values
    prev_pred_val = val_dataset.windows.merge(ctx.prev_pred_batch[['timestamp','smoothed_pred_temp']], left_on='window_end', right_on='timestamp', how='left')['smoothed_pred_temp'].values
    

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
            train_list.append(tmp['train'])
            with torch.no_grad():
                _, probs = base_model_ins(torch.stack([x for x,_ in tmp['train']]), return_type='reshape')
                pred = ieeg_get_pred(probs, threshold = ctx.thres, perc_chan_thres = ctx.perc)
                prev_pred_train_add.append(pred)
            val_list.append(tmp['val'])
            with torch.no_grad():
                _, probs = base_model_ins(torch.stack([x for x,_ in tmp['val']]), return_type='reshape')
                pred = ieeg_get_pred(probs, threshold = ctx.thres, perc_chan_thres = ctx.perc)
                prev_pred_val_add.append(pred)
    else:
        if not np.any(labels == 1):
            use_no_sz_setting = True
    if np.any(labels == 1):
        train_sz_inds = get_buffer_inds(train_dataset.window_label)
        val_sz_inds = get_buffer_inds(val_dataset.window_label)
        subset_train = [train_dataset[i] for i in train_sz_inds]
        subset_val = [val_dataset[i] for i in val_sz_inds]
        torch.save({'train': subset_train, 
                    'val': subset_val}, os.path.join(ctx.sz_data_dir, f"finetune_{ctx.batch_ind}.pt"))
    train_dataset = ConcatDataset([train_dataset] + train_list)
    val_dataset = ConcatDataset([val_dataset] + val_list)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    prev_pred_train = np.concatenate([prev_pred_train]+prev_pred_train_add)
    prev_pred_val = np.concatenate([prev_pred_val]+prev_pred_val_add)
    logger.info(f'BATCH {batch_ind}: Final train dataset size {len(train_dataset)}, validation dataset size {len(val_dataset)}')
    train_loader = ConfMatLoader(train_dataset, prev_pred_train, class_weights = 'auto', batch_size=ctx.batch_size)
    
    if ctx.do_epochs:
        # if use_no_sz_setting:
        #     model = ChannelModelWrapper(load_model(base_model, unfreeze_stage = 0)) #, dropout_rate=ctx.drop_rate, drop_fc = ctx.drop_fc
        #     alpha = 1.0
        #     teacher_model = ChannelModelWrapper(load_model(f'models/{ctx.ieeg_model}_base_model.pt', unfreeze_stage = 0))
        # else:
        model = ChannelModelWrapper(load_model(base_model, unfreeze_stage = ctx.unfreeze_stage)) #, dropout_rate=ctx.drop_rate, drop_fc = ctx.drop_fc
        alpha = 0.5
        teacher_model = ChannelModelWrapper(load_model(f'models/{ctx.ieeg_model}_base_model.pt',unfreeze_stage = ctx.unfreeze_stage))
    
        logger.info(f'BATCH {batch_ind}: Loading model {base_model} at unfreeze stage {unfreeze_order[ctx.unfreeze_stage]} and temperature {model.model.temperature.cpu().item()}')
        y_label = torch.tensor([y for _, y in val_dataset])
        pos_weight = (torch.sum(y_label != 1)/torch.sum(y_label == 1)).detach().numpy()
        if np.isnan(pos_weight) or np.isinf(pos_weight):
            pos_weight = 1
        mlflow.log_param('pos_weight', pos_weight)
        loss_fn = nn.BCEWithLogitsLoss()
        val_loss_fn = nn.BCEWithLogitsLoss(pos_weight = torch.from_numpy(np.array([pos_weight])).to(device))
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam(model.parameters(), lr=ctx.lr, weight_decay=ctx.wd, betas = (0.9,0.999),eps = 1.0*1e-8)
        logger.info(f'BATCH {batch_ind}: Batch contains seizure events.') 
        logger.info(f'BATCH {batch_ind}: Loss function pos_weight {pos_weight}.')
        logger.info(f"BATCH {batch_ind}: Learning rate of {ctx.lr} used.")
        logger.info(f"BATCH {batch_ind}: Total {ctx.num_epoch} training epochs.")
        logger.info(f"BATCH {batch_ind}: Optimizer initialized with learning rate {ctx.lr} and weight_decay {ctx.wd}")
        # get current best balanced accuracy and validation loss
        if ctx.batch_max_step >= 0:
            best_metric = ctx.last_state[['auprc','auroc','opt_thr','opt_tpr', 'opt_fpr','opt_precision','opt_f1','opt_mcc', 'validation_loss', 'ece', 'brier']].to_dict()
        else:
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    raw_logits, bag_logits, bag_probs = model(x_batch, return_type = 'all')
                    # if use_teacher:
                    #     teacher_logits = teacher_model(x_batch, return_type = 'raw')
                    #     teacher_probs = F.softmax(teacher_logits, dim=1)
                    #     student_log_probs = F.log_softmax(raw_logits, dim=1)
                    #     kl = kl_loss(student_log_probs, teacher_probs)
                    #     ce = val_loss_fn(bag_logits, y_batch)
                    #     print(ce, kl)
                    #     val_loss = alpha * ce + (1 - alpha) * kl
                    # else:
                    val_loss = val_loss_fn(bag_logits, y_batch)
                    bag_probs = bag_probs.cpu().numpy().flatten()
                    # pred = ieeg_get_pred(prob, threshold = ctx.thres, perc_chan_thres = ctx.perc)
                    y_true = y_batch.cpu().numpy().flatten()
                    best_metric = get_optimal_auc(y_true, bag_probs, opt_method='yodenj')
                    best_metric.update({'validation_loss':val_loss.item(),
                            'ece':compute_ece(y_true, bag_probs),
                            'brier':brier_score_loss(y_true, bag_probs)})
            # mlflow logging
            mlflow.log_metrics(best_metric, step = 0)
            for k, v in best_metric.items():
                logger.info(f'BATCH {batch_ind}: Initial best {k} set to {v:.4f}')
            # write state
            state = {"patient":ctx.current_patient, "batch": ctx.batch_ind, "sequence": 0,
                    'step':'base_model_performance', 'is_best': True, 
                    "timestamp": datetime.now(),'train_loss':None}
            state.update(best_metric)
            write_state(state)
            if best_metric['auroc'] > 0.98 or best_metric['auprc'] > 0.95:#or best_metric['auroc'] > 0.98:
                logger.info(f'BATCH {batch_ind}: Model showed perfect performance, skip finetuning')
                return
        
        # training epochs
        counter = 0 # counter of not improving steps
        unfreeze_stage = ctx.unfreeze_stage
        for epoch in range(1, ctx.num_epoch+1):
            if ctx.batch_max_step >= epoch:
                continue
            # if use_no_sz_setting and epoch > ctx.no_sz_settings['num_epoch']:
            #     continue
            logger.info(f'Processing epoch {epoch}')
            # if epoch <= 8:
            
            logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Train set size {len(train_dataset)}')
            
            # train
            train_losses = []
            model.train()
            for x_batch, y_batch in tqdm(train_loader):
                optimizer.zero_grad()
                raw_logits, bag_logits, bag_probs = model(x_batch, return_type = 'all')
                if use_teacher:
                    teacher_logits = teacher_model(x_batch, return_type = 'raw')
                    teacher_probs = F.softmax(teacher_logits, dim=1)
                    student_log_probs = F.log_softmax(raw_logits, dim=1)
                    kl = kl_loss(student_log_probs, teacher_probs)
                    ce = loss_fn(bag_logits, y_batch)
                    loss = alpha * ce + (1 - alpha) * kl
                else:
                    loss = loss_fn(bag_logits, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            train_loss = np.mean(train_losses)
            mlflow.log_metric('train_loss', train_loss, step = epoch)
            logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Train loss {train_loss:.4f}')
            
            # evaluation
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    raw_logits, bag_logits, prob = model(x_batch, return_type = 'all')
                    # if use_teacher:
                    #     teacher_logits = teacher_model(x_batch, return_type = 'raw')
                    #     teacher_probs = F.softmax(teacher_logits, dim=1)
                    #     student_log_probs = F.log_softmax(raw_logits, dim=1)
                    #     kl = kl_loss(student_log_probs, teacher_probs)
                    #     ce = val_loss_fn(bag_logits, y_batch)
                    #     print(ce, kl)
                    #     val_loss = alpha * ce + (1 - alpha) * kl
                    # else:
                    val_loss = val_loss_fn(bag_logits, y_batch)
                    prob = prob.cpu().numpy().flatten()
                    y_true = y_batch.cpu().numpy().flatten()
                    metric = get_optimal_auc(y_true, prob, opt_method='yodenj')
                    metric.update({'validation_loss':val_loss.item(),
                            'ece':compute_ece(y_true, prob),
                            'brier':brier_score_loss(y_true, prob)})
            mlflow.log_metrics(metric, step = epoch)
            for k, v in metric.items():
                logger.info(f"BATCH {batch_ind}: {k} is {v:.4f}, current best {k} is {best_metric[k]:.4f}")

            if metric['validation_loss'] < best_metric['validation_loss']:
                logger.info(f'BATCH {batch_ind} EPOCH {epoch}: Performance improved.')
                best_metric = metric
                counter = 0
                checkpoint = model.model.state_dict()
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
        state = {"patient":ctx.current_patient, "batch": ctx.batch_ind, "sequence": ctx.num_epoch+1,
                'step':f'calibration', 'is_best': True, 
                "timestamp": datetime.now()}
        write_state(state)

# __________________Predict Probability__________________________
def wvnt_single(model, data_df, fs):
    data = data_df.values
    n_ch = data.shape[1]
    # x should be samples x channels
    x = torch.tensor(data).float().unsqueeze(-1).permute(1,2,0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = F.softmax(logits, dim=1).cpu().numpy()[:,1].flatten()
        prob_temp = F.softmax(logits/model.temperature, dim=1).cpu().numpy()[:,1].flatten()
    return prob, prob_temp

def process_window(model, df, fs, pred_ctx):
    tmp_df = jsondecode_df(df)
    data_df = pd.concat([pd.DataFrame(d['BIPOLAR'],index = d['timestamps']) for _, d in tmp_df.iterrows()]).sort_index()
    data_df.iloc[1:,:] = ar_one(data_df.values)
    data_df = pd.DataFrame(pred_ctx.scaler.transform(data_df), columns=data_df.columns, index=data_df.index)
    data_df = downsample(data_df, fs, 128)
    prob, prob_temp = wvnt_single(model, data_df, fs)
    return pd.DataFrame([np.concatenate([prob, prob_temp])], columns=list(data_df.columns)+[c+'_temp' for c in data_df.columns])

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
        df['start_timestamp'] = df["start_timestamp"].apply(pd.Timestamp).dt.tz_localize(None).apply(lambda x:x.replace(microsecond=0) + timedelta(seconds=(x.microsecond >= 500000)))
        with open(f'labels/{pred_ctx.current_patient}_feat_labels.pkl','rb') as f:
            label_df = pickle.load(f)
        label_df = label_df.reset_index().rename(columns={'index': 'timestamp'})
        label_df['window_starts'] = label_df['timestamp'] - pd.Timedelta(seconds=feat_setting['win'])
        label_df = label_df[(label_df['window_starts'] >= batch_start)&(label_df['timestamp'] < batch_end)]
        # indices = range(0,len(df)-feat_setting['win']+1,feat_setting['stride'])
        feat_list = []
        for _, row in tqdm(label_df.iterrows(), total = len(label_df), desc = 'Window'):
            tmp = df[(df['start_timestamp'] >= row['window_starts'])&(df['start_timestamp'] < row['timestamp'])]
            feat_list.append(process_window(model, tmp, pred_ctx.sampling_freqs[pred_ctx.current_patient], pred_ctx))
        feat_list = pd.concat(feat_list)
        feat_list['timestamp'] = label_df['timestamp'].values
        feat_list['label'] = label_df['label'].values
        feat_list['batch_id'] = batch_id
        feat_list = feat_list.sort_values('timestamp')
        feat_list.to_csv(out_path,index=False)

def predict_proba():
    n_batch = pred_ctx.n_batch
    model = load_model(pred_ctx.eval_model, mode = 'eval')
    with tqdm(total=n_batch,desc = 'Processing block'):
        results = Parallel(n_jobs=1)(delayed(process_batch)(i, model, pred_ctx) for i in range(n_batch))
    # concate results
    all_files = glob.glob(f'{pred_ctx.tmp_dir}/*.csv')
    feat_df = pd.concat([pd.read_csv(f,parse_dates=['timestamp']) for f in all_files],ignore_index=False).sort_values(by='timestamp').reset_index(drop=True)
    # feat_df['timestamp'] = feat_df['timestamp'].dt.round('1s')
    feat_df.to_csv(pred_ctx.prob_file,index=False)
    for file in all_files:
        os.remove(file)

# __________________Binary prediction__________________________
def get_pred(sz_prob, threshold, w_size=1, w_stride=0.5, filter_w=10, rwin_size=5, rwin_req=4, num_chan = 1, ret_mat = False):
    filter_w_idx = int(np.floor((filter_w - w_size) / w_stride)) + 1
    rwin_size_idx = int(np.floor((rwin_size - w_size) / w_stride)) + 1
    rwin_req_idx = int(np.floor((rwin_req - w_size) / w_stride)) + 1

    sz_clf = sz_prob > threshold
    sz_clf = scipy.ndimage.median_filter(sz_clf, size=filter_w_idx, mode='nearest',axes=0,origin=0)
    
    rolling_sums = np.round(np.abs(scipy.signal.fftconvolve(sz_clf, np.ones([rwin_size_idx,1]), mode='full',axes=0)))
    rolling_sums[:rwin_size_idx-1,:] = rolling_sums[rwin_size_idx-1,:]
    rolling_sums = rolling_sums[:sz_clf.shape[0],:]
    sz_spread_idxs_all = rolling_sums >= rwin_req_idx
    
    pred = (np.sum(sz_spread_idxs_all, axis=1) >= num_chan).astype(int)
    if ret_mat:
        return pred, sz_spread_idxs_all
    else:
        return pred

def binary_prediction():
    # get prediction
    prob_df = pd.read_csv(pred_ctx.prob_file, parse_dates=['timestamp'])
    prob_df = prob_df.set_index(['timestamp'])
    remain_df = prob_df[['batch_id','label']]
    prob_df = prob_df.drop(columns=['batch_id','label'])
    n_ch = int(prob_df.shape[1]//2)
    prob = prob_df.iloc[:,:n_ch]
    prob_temp = prob_df.iloc[:,n_ch:] # temperature calibrated
    out = []
    for df in [prob, prob_temp]:
        _, smoothed_pred = get_pred(df.values, pred_ctx.thres, ret_mat = True)
        smoothed_pred_df = pd.DataFrame(smoothed_pred, columns = df.columns, index=df.index)
        perc_channel = smoothed_pred_df.sum(axis=1)/df.shape[1]
        pred = (perc_channel > pred_ctx.perc).astype(int)
        if 'auto' in pred_ctx.setting:
            auto_perc = float(pred_ctx.setting.split('_')[-1])
            moving_thres = perc_channel.rolling(window='30min', min_periods=60).quantile(auto_perc / 100)
            moving_thres = moving_thres.fillna(pred_ctx.perc)
            moving_thres = moving_thres.clip(lower=0.1)
            pred = (perc_channel > moving_thres).astype(int)
        _, smoothed_pred = get_events(pred, gap_num = 2, min_event_num=10, return_pred=True)
        out.append(pd.DataFrame({'avg_prob': df.mean(axis=1), 'perc': perc_channel,'pred': pred, 'smoothed_pred':smoothed_pred}, index=df.index))
    out[1] = out[1].rename(columns={'avg_prob':'avg_prob_temp','perc':'perc_temp','pred':'pred_temp','smoothed_pred':'smoothed_pred_temp'})
    final_df = pd.concat([out[0],out[1],remain_df],axis=1)
    final_df = final_df.reset_index().rename({'index':'timestamp'})
    final_df.to_csv(pred_ctx.pred_file, index=False)

#_________________________Metric Calculation___________________________

def calc_metric_batch(feat_data):
    columns = ['pred','pred_temp','smoothed_pred','smoothed_pred_temp']#,'_temp_adjth','_temp_adjth_filtered'
    row_label = ['Finetune','Finetune+Calibrate', 'Finetune+Filtering','FC+Filtering']#,'FC+AdjustThres','FC+AdjThres+Filtering'
    labels = feat_data['label'].astype(int).to_numpy()
    all_metrics = []
    for e, col in enumerate(columns):
        pred = feat_data[col].astype(int).to_numpy()
        if 'temp' in columns:
            prob = feat_data['avg_prob_temp'].to_numpy()
        else:
            prob = feat_data['avg_prob'].to_numpy()
        metrics = compute_metrics(labels, pred, prob, stride = feat_setting['stride'])
        metrics = pd.DataFrame([metrics], index=[row_label[e]])
        all_metrics.append(metrics)
    all_metrics = pd.concat(all_metrics).reset_index(names='setting')
    all_metrics['batch_id'] = feat_data['batch_id'].iloc[0]
    return all_metrics

def calculate_metrics():
    feat_data = pd.read_csv(pred_ctx.pred_file, parse_dates=['timestamp'])
    metrics = feat_data.groupby('batch_id').apply(calc_metric_batch).reset_index(drop=True).set_index('batch_id')
    full_metrics = calc_metric_batch(feat_data).reset_index(drop=True).drop(columns='batch_id')
    metrics.to_csv(pred_ctx.metric_file)
    full_metrics.to_csv(pred_ctx.metric_file.replace('.csv','_full.csv'),index=False)

def plot_trajectory(overlay = False):
    plot_vars = pred_ctx.plot_vars
    metrics = pd.read_csv(pred_ctx.metric_file, index_col=0)
    col = 'b'
    if overlay:
        baseline_metrics = pd.read_csv(pred_ctx.baseline_metric_file, index_col=0)
        col = 'r'
        overlay_col = 'b'
    for setting in ['Finetune','Finetune+Calibrate']:
        metrics = metrics[metrics['setting']==setting]
        if overlay:
            baseline_metrics = baseline_metrics[baseline_metrics['setting'] == setting]
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
    prob_df = pd.read_csv(pred_ctx.prob_file, parse_dates=['timestamp'])
    prob_df = prob_df.set_index(['timestamp'])
    true = prob_df['label'].values
    prob_df = prob_df.drop(columns=['batch_id','label'])
    n_ch = int(prob_df.shape[1]//2)
    prob = prob_df.iloc[:,:n_ch].mean(axis=1).values
    prob_cal = prob_df.iloc[:,n_ch:].mean(axis=1).values # temperature calibrated
    data_dict = {'Finetuned':prob,'Finetuned+Calibration':prob_cal}
    if overlay:
        baseline_prob_df = pd.read_csv(pred_ctx.baseline_prob_file, parse_dates=['timestamp'])
        baseline_prob = baseline_prob_df.iloc[:,:n_ch].mean(axis=1).values
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
    prob_df = pd.read_csv(pred_ctx.prob_file, parse_dates=['timestamp'])
    prob_df = prob_df.set_index(['timestamp'])
    true = prob_df['label'].values
    if len(np.unique(true)) == 1:
        return
    prob_df = prob_df.drop(columns=['batch_id','label'])
    n_ch = int(prob_df.shape[1]//2)
    prob = prob_df.iloc[:,:n_ch].mean(axis=1).values
    prob_cal = prob_df.iloc[:,n_ch:].mean(axis=1).values # temperature calibrated
    data_dict = {'Finetuned':prob,'Finetuned+Calibration':prob_cal}
    if overlay:
        baseline_prob_df = pd.read_csv(pred_ctx.baseline_prob_file, parse_dates=['timestamp'])
        baseline_prob = baseline_prob_df.iloc[:,:n_ch].mean(axis=1).values
        data_dict['Baseline'] = baseline_prob
    fig, axes = plt.subplots(1,2,figsize=(16, 8))
    colors = plt.get_cmap("Dark2")
    calibration_displays = {}
    markers = ["^", "v", "s", "o"]
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    baseline = np.mean(true)
    axes[1].hlines(baseline, xmin=0, xmax=1, colors="k", linestyles="--", linewidth=1)
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
    axes[1].set_ylim([0,0.2])
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
    metric_files = sorted([f for f in metric_files if 'full' not in f], key = lambda x:int(x.split('/')[-1].split('_')[1]))
    for setting in ['Finetune','Finetune+Calibrate']:
        prev_test = []
        finetuned_test = []
        baseline_test = []
        baseline_metrics = baseline_metrics_df[baseline_metrics_df['setting'] == setting]
        prev_test.append(agg_metrics(list(range(1,len(baseline_metrics))), feat_file = pred_ctx.baseline_feat_file, setting = setting))
        for e, f in enumerate(metric_files): # last one should be n_batch - 1
            batch_ind = int(f.split('/')[-1].split('_')[1])
            metrics = pd.read_csv(f, index_col=0)
            metrics = metrics[metrics['setting']==setting]
            to_eval_batch_ind = batch_ind+1
            to_compare_with_next_ind = batch_ind+2
            finetuned_test.append(agg_metrics(list(range(to_eval_batch_ind, len(metrics))), setting = setting))
            if e < len(metric_files)-1:
                prev_test.append(agg_metrics(list(range(to_compare_with_next_ind,len(baseline_metrics))), feat_file = pred_ctx.baseline_feat_file, setting = setting))
            baseline_test.append(agg_metrics(list(range(to_eval_batch_ind,len(baseline_metrics))), feat_file = pred_ctx.baseline_feat_file, setting = setting))
        prev_test = pd.concat(prev_test) 
        finetuned_test = pd.concat(finetuned_test) 
        baseline_test = pd.concat(baseline_test) 

        plot_paired_line(prev_test, finetuned_test, variables = plot_vars, save_path = pred_ctx.paired_line_test_prev+'_'+setting)
        plot_paired_line(baseline_test, finetuned_test, variables = plot_vars, save_path = pred_ctx.paired_line_test_base+'_'+setting)


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
