# General imports
import os, sys, json
import warnings
import argparse
import logging
import torch

# paths
working_dir = os.path.abspath("")
sys.path.append(working_dir)
sys.path.append(os.path.join(working_dir,'funcs'))

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
    parser = argparse.ArgumentParser(description="Model evaluation")

    # Define arguments
    parser.add_argument("-m", "--model", type=str, required=True, help="Model to evaluate")
    parser.add_argument("-p", "--patient", type=str, required=True, help="Patient data to evaluate on")
    parser.add_argument("-f", "--force", action = 'store_true', help="Force rerun even if results already exist")
    parser.add_argument("--mode", type=str, default = 'pipeline', help="Implementation mode, default to implement whole pipeline, other potential values, prob, pred, metric, skip_prob, skip_metric")
    
    parser.add_argument("--setting", type=str, default='', help="Suffix to specify experiment name")
    parser.add_argument("--thres_high", type=float, default=0.5, help="threshold to get binary predictions")
    parser.add_argument("--thres_low", type=float, default=0.5, help="threshold to get binary predictions")
    parser.add_argument("--ieeg_model", type=str, default='wvnt', help="model to use for intracranial patient")
    parser.add_argument("--thres", type=float, default=0.9, help="probability threshold")
    parser.add_argument("--perc", type=float, default=0.15, help="percentage of channel seizing")
    parser.add_argument("--batch_hour", type=int, default=3,help="Length of data to be used for each batch") 
    parser.add_argument("--adjth", action = 'store_true', help="Use adjusted thres")
    parser.add_argument("--batch_id", type=int, default=-1,help="Which finetuned model to run") 
    
    
    # if setting is not none, search for configs/eval_configs/setting.json
    # and use threshold params there
    # this helps keep track the exact config used in each different setting
    # Parse the arguments
    params = vars(parser.parse_args())
    if params['setting']:
        try:
            additional_config = json.load(open(f"configs/eval_configs/{params['setting']}.json",'r'))
            additional_config = {k:v for k,v in additional_config if k in ['thres_high','thres_low','ieeg_model','thres','perc']}
            params.update(additional_config)
        except:
            pass
    pred_ctx.initialize(params)

    for rid in pred_ctx.patient_ids:
        logger.info(f"Evaluating model {params['model']} on patient {rid}")
        if params['batch_id'] > -1:
            pred_ctx.set_patient_batch(rid, batch_ind=params['batch_id'])
        else:
            pred_ctx.set_patient_batch(rid)
        # if pred_ctx.do_prob:
        pred_ctx.module.predict_proba()
        # if pred_ctx.do_pred:
        pred_ctx.module.binary_prediction()
        # if pred_ctx.do_metric:
        pred_ctx.module.calculate_metrics()
        if pred_ctx.do_plot:
            pred_ctx.module.generate_plots()
                

