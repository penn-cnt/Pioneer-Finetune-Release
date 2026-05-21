# General imports
import os, sys, json
import warnings
import argparse
import logging
import importlib

import numpy as np
import pandas as pd
import sqlite3
import torch
import mlflow
mlflow.set_tracking_uri("mlflow") 
import torch

# custom utils
from funcs.utils import *
from funcs.feat_funcs import *
from funcs.context import ctx, pred_ctx

# setup logging
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

# paths
working_dir = os.path.abspath("")
sys.path.append(working_dir)
sys.path.append(os.path.join(working_dir,'funcs'))

# set seed
def set_seed(seed: int = 42, deterministic: bool = True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    logger.info(f"Random seed set to {seed}, deterministic={deterministic}")

set_seed(666)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description="Model Finetuning")

    # Define arguments
    parser.add_argument("-m", "--model", type=str, required=True, help="Base model to fine-tune")
    parser.add_argument("-p", "--patient", type=str, required=True, help="Patient data to fine-tune on")
    parser.add_argument("-f", "--force", action = 'store_true', help="Force rerun and delete all previous checkpoints, else resume from previous checkpoints")
    parser.add_argument("-l", "--label", type=str, default='patient', help="which label to use for finetuning, potential values, databricks, patient, tech" )
    parser.add_argument("--mode", type=str, default='normal', 
                        help="""finetune mode, normal, pioneer is only obtain labels from the prediction of the previous model, if there's no prediction, even if there're seizure events, no finetuning will be done;
                                HITL, is finetuning based on the least certain samples over the whole day, batch hour for this option should be 24""")
    parser.add_argument("--setting", type=str, default='', help="Suffix to specify experiment name")
    parser.add_argument("--thres_high", type=float, default=0.5, help="threshold to get binary predictions")
    parser.add_argument("--thres_low", type=float, default=0.5, help="threshold to get binary predictions")
    # parser.add_argument("--ieeg_model", type=str, default='wvnt', help="model to use for intracranial patient")
    parser.add_argument("--thres", type=float, default=0.99, help="probability threshold for ieeg model")
    parser.add_argument("--perc", type=float, default=0.1, help="percentage of channel seizing for ieeg model and data")
    parser.add_argument("--batch_hour", type=int, default=3,help="Length of data to be used for each batch") 
    parser.add_argument("--adjthres", action = 'store_true', help="whether adjust threshold")

    # unfreeze settings
    parser.add_argument("-u", "--unfreeze", action = 'store_true', help="whether gradually unfreeze when reaching patience")
    parser.add_argument("--unfreeze_stage",type=int, default = 0, help="initial unfreeze stage")
    parser.add_argument("--unfreeze_step", type=int, default=2, help="number of epochs to unfreeze if performance not improved")

    # finetuning settings also likely won't change
    parser.add_argument("--patience", type=int, default=3, help="patience for no improvement")
    parser.add_argument("--lr", type=float, default=5e-6, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay rate" )
    parser.add_argument("--num_epoch", type=int, default=20, help="number of finetuning epochs in each batch" )
    parser.add_argument("--batch_size", type=int, default=32, help="train dataset batch size")
    parser.add_argument("--adjth", action = 'store_true', help="Use adjusted thres")
    parser.add_argument("--original", action = 'store_true', help="Finetune original model always")
    # parser.add_argument("--drop_rate", type=float, default=0.2, help="model dropout rate" )
    # parser.add_argument("--drop_fc", type=float, default=0.5, help="model fc layer dropout rate")

    # likely won't change
    parser.add_argument("-n", "--num_block", type=int, default=-1, help="Fine-tune for x rounds, default is all (till end of recording)" ) 

    # Parse the arguments
    params = vars(parser.parse_args())
    ctx.initialize(params)
    pred_params = params.copy()
    pred_params['setting'] = ''
    # pred_params_adjthres = pred_params.copy()
    # pred_params_adjthres['adjth'] = True

    # set up mlflow and logging
    mlflow.set_experiment(ctx.exp_name)
    logger.info(f"Fine-tuning base model {ctx.model} using on patients {ctx.patient_ids}. Output model {ctx.exp_name}.")

    if ctx.unfreeze:
        logger.info(f'Model will unfreeze gradually')
    logger.info(f'Model will calibrate using temperature calibration')
    logger.info(f"Finetuning parameters: learning rate {ctx.lr}, weight_decay {ctx.wd}, batch size {ctx.batch_size}, patience {ctx.patience}.")
    
    for rid in ctx.patient_ids:
        logger.info(f'Finetuning on patient {rid}')
        
        # skip if not force
        if rid in ctx.skip_patients:
            continue
        
        # batch runs
        for batch_ind in range(ctx.n_finetune_batch[rid]): # start from 1 so index 0 is base model
            logger.info(f'Processing batch {batch_ind}')
            # skip whole batch if this is not the last batch in last state
            # finetune
            ctx.set_patient_batch(rid,batch_ind)
            if batch_ind in ctx.skip_batches:
                logger.info(f'Skipping batch {batch_ind}')
                continue

            with mlflow.start_run(run_id=ctx.run_id):
                try:
                    mlflow.log_params(params)
                except:
                    pass
                if ctx.do_finetune:
                    ctx.module.finetune()
                
            # evaluate   
            pred_params['model'] = f"{rid}_{'_'.join(ctx.exp_name.split('_')[1:])}"
            pred_ctx.initialize(pred_params) 
            pred_ctx.set_patient_batch(rid,batch_ind)
            if pred_ctx.do_prob:
                pred_ctx.module.predict_proba()
            if pred_ctx.do_pred:
                pred_ctx.module.binary_prediction()
            if pred_ctx.do_metric:
                pred_ctx.module.calculate_metrics()
            if pred_ctx.do_plot:
                pred_ctx.module.generate_plots()

