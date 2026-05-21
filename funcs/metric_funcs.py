# General imports
import os, sys
import numpy as np
from sklearn.metrics import balanced_accuracy_score, average_precision_score, roc_auc_score, roc_curve, matthews_corrcoef, precision_recall_curve, auc, precision_score, f1_score
from timescoring.annotations import Annotation
from timescoring import scoring 
from timescoring import visualization

# paths
working_dir = os.path.abspath("")
sys.path.append(working_dir)

# custom utils
from funcs.utils import *
from funcs.feat_funcs import *

# __________________Metrics Functions_________________________

metric_labels = {'total_dura':'Total Duration, hour',
           'tn_hour':'Data Reduction, hour', 
           'tn':'Specificity, %', 
           'fp_min':'False Positives, min', 
           'total_sz_dura':'Total Seizure Duration, min',
           'avg_sz_dura':'Average Seizure Duration, min',
           'num_sz': 'Number of Seizure',
           'auroc_sample': 'AUROC',
           'auprc_sample': 'AUPRC',
           'recall_event':'Seizure Event Detected, %', # 1-miss
           'recall_sample':'Seizure Sample Detected, %', # total_sz_dura-miss_min
           'precision_event':'Precision Event, %',
           'precision_sample':'Precision Sample, %',
           'fp':'False Alarm per hour',
           'f1_event':'F1 Event',
           'f1_sample':'F1 Sample',
           'balanced_acc':'Balanced Accuracy',
           'mcc':'MCC',
           'ece':'Expected Calibration Error'} 


def compute_ece(labels, prob, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        probabilities: array of predicted probabilities (confidence scores)
        labels: array of true binary labels (0 or 1)
        n_bins: number of bins (default 10)
        
    Returns:
        ece: Expected Calibration Error (float)
    """
    prob = np.asarray(prob)
    new_prob = np.max(np.c_[prob,1-prob],axis=1)
    pred = np.argmax(np.c_[prob,1-prob],axis=1)
    labels = np.asarray(labels)
    
    bin_boundaries = np.linspace(0.5, 1, n_bins + 1)
    ece = 0.0
    n = len(prob)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Get indices in the current bin
        in_bin = (new_prob > bin_lower) & (new_prob <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_confidence = np.nanmean(new_prob[in_bin])
            sub_pred = pred[in_bin]
            sub_label = labels[in_bin]
            bin_accuracy = np.nanmean([(np.sum((sub_label == 1) & (sub_pred == sub_label)) / np.sum(sub_label==1),np.sum((sub_label == 0) & (sub_pred == sub_label)) / np.sum(sub_label==0))])
            ece += (bin_size / n) * abs(bin_accuracy - bin_confidence)
    
    return ece 

def compute_metrics(true, pred, prob, stride = 2):
    """Compute metrics for seizure detection based on true labels and predicted labels.
    pred will be filtered that only >=5 consecutive positve predictions are kept.
    Seizure events counted as detected if more than 10% of the event duration or 10 samples are detected. A margin of 2 samples are allowed. 
    """
    true = np.squeeze(true)
    pred = np.squeeze(pred)
    tp = np.sum((pred == 1) & (true == 1))
    tn = np.sum((pred == 0) & (true == 0))
    fp = np.sum((pred == 1) & (true == 0))
    seiz_ranges = extract_seiz_ranges(true)
    pred_seiz_ranges = extract_seiz_ranges(pred)
    
    metrics = {}
    metrics['total_dura']=len(true) * stride / 3600 # in hours
    metrics['tn_hour']=tn * stride / 3600 # in hours
    metrics['tn']=tn / np.sum(true == 0)
    metrics['fp_min']= fp * stride / 60 # in minutes
    metrics['num_sz'] = len(seiz_ranges)

    labels = Annotation(true, 1/stride)
    preds = Annotation(pred, 1/stride)
    param = scoring.EventScoring.Parameters(
        toleranceStart=30,
        toleranceEnd=60,
        minOverlap=0,
        maxEventDuration=5 * 60,
        minDurationBetweenEvents=90)
    scores = scoring.EventScoring(labels, preds, param)
    scores_sample = scoring.SampleScoring(labels, preds)

    if np.any(true == 1):
        metrics['total_sz_dura']=np.sum(true) * stride / 60
        metrics['avg_sz_dura']=np.mean([(end-start) * stride / 60 for start, end in seiz_ranges])
        metrics['auprc_sample'] = average_precision_score(true, prob)
        metrics['auroc_sample'] = roc_auc_score(true, prob)
    else:
        for key in ['total_sz_dura','avg_sz_dura', 'auprc_sample','auroc_sample']:
            metrics[key]=np.nan

    metrics['recall_event'] = scores.sensitivity
    metrics['recall_sample'] = scores_sample.sensitivity
    metrics['fp'] = scores.fpRate/24
    metrics['precision_event'] = scores.precision
    metrics['precision_sample'] = scores_sample.precision
    metrics['f1_event'] = scores.f1
    metrics['f1_sample'] = scores_sample.f1

    metrics['balanced_acc']=np.nanmean([metrics['recall_event'],metrics['tn']])
    metrics['mcc'] = matthews_corrcoef(true, pred)
    metrics['ece'] = compute_ece(true,prob)
    return metrics

def get_optimal_auc(true, prob, opt_method = 'yodenj'):
    """
    Potential optimal operating point methods
    yodenj, maximize tpr-fpr
    f1, maximize f1 score
    metric+number, best performance satisfying the critiera
    e.g. tpr90, performance at tpr>=90
    precision10, performance at precision >= 10 from auprc
    """
    fpr, tpr, thr = roc_curve(true, prob)
    auc_score = auc(fpr, tpr)
    prec, recall, thr_prc = precision_recall_curve(true, prob)
    prc_score = average_precision_score(true, prob)
    if opt_method == 'yodenj':
        ind = np.argmax(tpr-fpr)
        opt_thr, opt_tpr, opt_fpr = thr[ind], tpr[ind], fpr[ind]
        opt_precision = precision_score(true, prob>=opt_thr)
        opt_mcc = matthews_corrcoef(true, prob>=opt_thr)
    elif opt_method == 'f1':
        ind = np.argmax(2 * (prec * recall) / (prec + recall + 1e-12))
        opt_thr, opt_precision, opt_tpr = thr_prc[ind], prec[ind], recall[ind]
        ind2 = np.abs(thr-opt_thr).argmin()
        opt_fpr = fpr[ind2]
        opt_mcc = matthews_corrcoef(true, prob>=opt_thr)
    elif opt_method == 'mcc':
        mccs = [matthews_corrcoef(true, prob>= t) for t in thr]
        ind = np.array(mccs).argmax()
        opt_thr, opt_tpr, opt_fpr = thr[ind], tpr[ind], fpr[ind]
        opt_precision = precision_score(true, prob>=opt_thr)
        opt_mcc = mccs[ind]
    else:
        opt_metric, opt_criteria = list(opt_method.items())[0]
        if opt_metric == 'tpr' or opt_metric == 'recall':
            ind = np.argmax(tpr >= opt_criteria)
            opt_thr, opt_fpr = thr[ind], fpr[ind]
            opt_precision = precision_score(true, prob>=opt_thr)
            opt_mcc = matthews_corrcoef(true, prob>=opt_thr)
        elif opt_metric == 'precision':
            ind = np.abs(prec-opt_criteria).argmin()
            opt_thr, opt_precision, opt_tpr = thr_prc[ind], prec[ind], recall[ind]
            ind2 = np.abs(thr-opt_thr).argmin()
            opt_fpr = fpr[ind2]
            opt_mcc = matthews_corrcoef(true, prob>=opt_thr)
        elif opt_metric == 'fpr':
            ind = np.argmax(fpr >= opt_criteria)
            opt_thr, opt_tpr, opt_fpr = thr[ind], tpr[ind], fpr[ind]
            opt_precision = precision_score(true, prob>=opt_thr)
            opt_mcc = matthews_corrcoef(true, prob>=opt_thr)
    opt_f1 = f1_score(true, prob>=opt_thr)
    return {'auroc':auc_score,
            'auprc':prc_score,
            'opt_thr':opt_thr,
            'opt_tpr':opt_tpr,
            'opt_fpr':opt_fpr,
            'opt_precision':opt_precision,
            'opt_f1':opt_f1,
            'opt_mcc':opt_mcc}