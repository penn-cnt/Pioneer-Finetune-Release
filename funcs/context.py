import os, sys, json
from datetime import datetime, timedelta
import time
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("mlflow") 
client = MlflowClient()
import sqlite3
import pickle
import logging
import glob
import importlib
import numpy as np
from edf_reader import EdfWrapper
from sklearn.preprocessing import RobustScaler
working_dir = os.path.abspath("")
sys.path.append(working_dir)
from funcs.utils import ar_one

ieeg_list = ['P007','P013','P014']

def jsondecode_df(df):
    for col, dtype in zip(df.columns,df.dtypes):
        if dtype == object:
            try:
                df[col] = df[col].apply(json.loads)
            except:
                pass
    return df


class Context:
    """This is the class that stores all settings
     during finetuning experiment
    """
    def __init__(self):
        pass

    def initialize(self, params):
        """
        Params include:
        # Define arguments
        parser.add_argument("-m", "--model", type=str, required=True, help="Base model to fine-tune")
        parser.add_argument("-p", "--patient", type=str, required=True, help="Patient data to fine-tune on")
        parser.add_argument("-f", "--force", action = 'store_true', help="Force rerun and delete all previous checkpoints, else resume from previous checkpoints")
        parser.add_argument("-l", "--label", type=str, default='patient', help="which label to use for finetuning, potential values, databricks, patient, tech" )
        parser.add_argument("--mode", type=str, default='normal', 
                            help="finetune mode, normal, pioneer is only obtain labels from the prediction of the previous model, if there's no prediction, even if there're seizure events, no finetuning will be done;
                                    HITL, is finetuning based on the least certain samples over the whole day, batch hour for this option should be 24")
        parser.add_argument("--setting", type=str, default='', help="Suffix to specify experiment name")
        parser.add_argument("--thres_high", type=float, default=0.5, help="threshold to get binary predictions")
        parser.add_argument("--thres_low", type=float, default=0.5, help="threshold to get binary predictions")
        parser.add_argument("--ieeg_model", type=str, default='wvnt', help="model to use for intracranial patient")
        parser.add_argument("--thres", type=float, default=0.9, help="probability threshold for ieeg model")
        parser.add_argument("--perc", type=float, default=0.15, help="percentage of channel seizing for ieeg model and data")
        parser.add_argument("--batch_hour", type=int, default=3,help="Length of data to be used for each batch") 
        parser.add_argument("--adjthres", action = 'store_true', help="whether adjust threshold")

        # unfreeze settings
        parser.add_argument("-u", "--unfreeze", action = 'store_true', help="whether gradually unfreeze when reaching patience")
        parser.add_argument("--unfreeze_stage",type=int, default = 0, help="initial unfreeze stage")
        parser.add_argument("--unfreeze_step", type=int, default=2, help="number of epochs to unfreeze if performance not improved")

        # finetuning settings also likely won't change
        parser.add_argument("--patience", type=int, default=5, help="patience for no improvement")
        parser.add_argument("--lr", type=float, default=6e-5, help="learning rate")
        parser.add_argument("--wd", type=float, default=1e-3, help="weight decay rate" )
        parser.add_argument("--num_epoch", type=int, default=20, help="number of finetuning epochs in each batch" )
        parser.add_argument("--batch_size", type=int, default=32, help="train dataset batch size")

        # likely won't change
        parser.add_argument("-n", "--num_block", type=int, default=-1, help="Fine-tune for x rounds, default is all (till end of recording)" ) 

        """
        for k, v in params.items():
            setattr(self, k, v)
        if self.patient == 'all':
            self.patient_ids = os.listdir('/project/pioneer/rawdata')
        elif self.patient == 'scalp':
            self.patient_ids = os.listdir('/project/pioneer/rawdata')
            self.patient_ids = [p for p in self.patient_ids if p not in ieeg_list]
        elif self.patient == 'ieeg':
            self.patient_ids = ieeg_list
        else:
            self.patient_ids = sorted(self.patient.split(','))
        # if params['mode'] == 'pioneer':
        #     params['label'] = 'databricks'
        # set experiment name: finetuning patients_learning rate_weight decay rate_label to finetune
        exp_name = f"{params['patient']}_{params['model']}_lr{params['lr']:.0e}_wd{params['wd']:.0e}_{params['label']}"
        if params['setting']: # any custom suffix
            exp_name += f"_{params['setting']}"
        if params['mode'] != 'normal': # use pioneer mode that if there's no prediction, don't have label even if there're labels in patient comments
            exp_name += f"_{params['mode']}"
        if params['unfreeze']: # unfreeze setting and stage starting unfreezing
            exp_name += f"_unfreeze{params['unfreeze_stage']}"
        if params['batch_size'] != 32: # batch size in finetuning
            exp_name += f"_batch{params['batch_size']}"
        # if params['drop_rate'] != 0.2: # dropout rate
        #     exp_name += f"_drop{params['drop_rate']}"
        # if params['drop_fc'] != 0.5: # dropout rate in FC layer
        #     exp_name += f"_dropfc{params['drop_fc']}"
        if params['batch_hour'] != 3: # length of recording for each round of finetuning
            exp_name += f"_batchhour{params['batch_hour']}"
        self.exp_name = exp_name
        self.no_sz_settings = {'num_epoch':3, 'unfreeze_stage':0, 'unfreeze_base':False}
        # other settings
        # self.batch_dura = int(3600*params['batch_hour']) # batch duration in seconds
        self.model_dir = f"models/{exp_name}" # folder to store model
        self.sz_data_dir = f"{self.model_dir}/sz_data" # folder to store previous seizure data
        os.makedirs(self.sz_data_dir, exist_ok=True)
        self.progress_tracker = os.path.join(self.model_dir,'finetune_progress.csv') # file to store finetuning process
        # self.log_file = os.path.join(self.model_dir,'finetune.log') # file to store logs?
        # self.pred_folder = f'pred_{self.thres_high}_extend_{self.thres_low}'

        # mlflow settings
        experiment = client.get_experiment_by_name(self.exp_name) # check if any existing mlflow experiment
        if experiment is None:
            self.experiment_id = client.create_experiment(self.exp_name) # create experiment if a new run
        else:
            self.experiment_id = experiment.experiment_id # get experiment id if previously has

        # initialize or reinitialize mlflow and folder
        if not os.path.exists(self.progress_tracker) or self.force:
            # clean mlflow experiment if any previous runs in force mode
            runs = client.search_runs(experiment_ids=[self.experiment_id])
            for run in runs:
                client.delete_run(run.info.run_id)  # soft delete run
            # clean checkpoints/models if any
            if os.path.exists(self.model_dir):
                for f in os.listdir(self.sz_data_dir):
                    os.remove(os.path.join(self.sz_data_dir, f))
                for f in os.listdir(self.model_dir):
                    try:
                        os.remove(os.path.join(self.model_dir, f))
                    except:
                        continue
            # initialize states tracker
            states = pd.DataFrame(columns = ["patient","batch","sequence","step",'is_best','timestamp', 'train_loss','auroc','auprc','recall', 'fp','precision','f1','mcc', 'validation_loss', 'ece', 'brier'])
            states.to_csv(self.progress_tracker,index=False)
        
        # set states
        states = pd.read_csv(self.progress_tracker)
        # remove unsaved steps
        if not states.empty:
            while not states.iloc[-1]["is_best"]:
                states = states.iloc[:-1]
            states.to_csv(self.progress_tracker, index=False)
        self.states = states
        if not self.states.empty:
            self.last_state = states.iloc[-1]
            self.last_batch_states = states[(states['patient']==self.last_state['patient'])&(states['batch']==self.last_state['batch'])]
            self.skip_patients = [p for e, p in enumerate(self.patient_ids) if e < self.patient_ids.index(self.last_state['patient'])] # patients should be skipped
            self.skip_batches = list(range(self.last_state['batch'])) # batches should be skipped
            self.max_step = max(self.last_batch_states['sequence'])
        else:
            self.last_state = None
            self.last_batch_states = None
            self.skip_patients = []
            self.skip_batches = []
            self.max_step = -1
        sz_data_files = os.listdir(self.sz_data_dir)
        sz_data_file_inds = [int(f.split('_')[1][:-3]) for f in sz_data_files]
        for i,f in zip(sz_data_file_inds, sz_data_files):
            if i not in self.skip_batches:
                os.remove(os.path.join(self.sz_data_dir, f))
        # set up database connectors
        self.batch_starts = {}
        self.sampling_freqs = {}
        self.patient_types = {}
        self.filenames = {}
        self.n_batches = {} # k:int(np.ceil(v/self.batch_dura)) for k,v in self.n_clip.items()
        self.scalers = {}
        for patient in self.patient_ids:
            with open(f'labels/{patient}_data_labels.pkl','rb') as f:
                label_df = pickle.load(f)
                data_timestamps = label_df.index.to_numpy()
            start_time, end_time = data_timestamps[0], data_timestamps[-1]
            self.batch_starts[patient] = pd.date_range(start=start_time, end=end_time-pd.Timedelta(hours=self.batch_hour), freq=f"{self.batch_hour}h").tz_localize(None)
            self.n_batches[patient] = len(self.batch_starts[patient])
            # get meta info
            with open(f'INFO/{patient}.pkl','rb') as f:
                info = pickle.load(f)
            self.sampling_freqs[patient] = info['samplingFreq']
            self.patient_types[patient] = info['studyType']
            if info['studyType'] == 'eeg':
                filename = f'{self.thres_high}_extend_{self.thres_low}'
            elif info['studyType'] == 'ieeg':
                filename = f'{self.thres}_{self.perc}'
                # get 1 or 2 or 10 minutes data from ?
                self.scalers[patient] = RobustScaler()
                conn = sqlite3.connect(f'data/{patient}.db')
                cursor = conn.cursor()
                query = f"SELECT BIPOLAR, timestamps FROM {patient} ORDER BY start_timestamp LIMIT 600 OFFSET 3600"
                data = pd.read_sql(query,conn)
                data = jsondecode_df(data)
                data = pd.concat([pd.DataFrame(d['BIPOLAR'],index = d['timestamps']) for _, d in data.iterrows()]).sort_index()
                data.iloc[1:,:] = ar_one(data.values)
                self.scalers[patient].fit(data)
            # if self.setting:
            #     filename = f'{filename}_{self.setting}'
            # if self.batch_hour != 3:
            #     filename = f'{filename}_batchhour{self.batch_hour}'
            if self.adjth:
                filename = f'{filename}_adjth'
            self.filenames[patient] = filename
        if self.num_block == -1:
            self.n_finetune_batch = {k:v-1 for k,v in self.n_batches.items()} # number of finetuning batches, remove 1 from total
        else:
            self.n_finetune_batch = {k:min(self.num_block,v-1) for k,v in self.n_batches.items()}
        

    def set(self, **kwargs):
        for k, v in kwargs:
            setattr(self, k, v)

    def set_patient_batch(self, patient, batch_ind):
        # set patient and batch index
        self.current_patient = patient
        self.batch_ind = batch_ind
        self.batch_start = self.batch_starts[self.current_patient]
        self.n_batch = self.n_batches[self.current_patient]
        self.sampling_freq = self.sampling_freqs[self.current_patient]
        self.patient_type = self.patient_types[self.current_patient]
        self.filename = self.filenames[self.current_patient]
        if self.mode == 'pioneer' or self.label == 'databricks':
            with open(f'labels/{patient}_databricks_data_labels.pkl','rb') as f:
                label_df = pickle.load(f)
                data_timestamps = label_df.index.to_numpy()
            start_time, end_time = data_timestamps[0], data_timestamps[-1]
            if self.batch_start[batch_ind] <= start_time: #+ pd.Timedelta(hours = self.batch_hour)
                self.skip_batches.append(batch_ind)
                return
        if patient in self.scalers:
            self.scaler = self.scalers[patient]
        if self.patient_type == 'eeg':
            if self.model == 'finetune_sparcnet':
                self.module = importlib.import_module('funcs.finetune_sparcnet_funcs')
            else:
                self.module = importlib.import_module(f'funcs.sparcnet_funcs_update')
        elif self.patient_type == 'ieeg':
            self.ieeg_model = 'wvnt' if 'wvnt' in self.model else 'oncet'
            self.module = importlib.import_module(f'funcs.{self.ieeg_model}_funcs')
        # if resume, set previous max steps reached
        self.batch_max_step = -1
        if self.last_state is not None:
            if self.batch_ind == self.last_state['batch'] and self.current_patient == self.last_state['patient']:
                self.batch_max_step = self.max_step
        self.do_finetune = False if self.batch_max_step == self.num_epoch+2 else True
        self.do_epochs = True if self.batch_max_step < self.num_epoch else False
        self.do_cal = True if self.batch_max_step < self.num_epoch+1 else False
        self.do_adj_thres = True if self.batch_max_step < self.num_epoch+2 else False
        # set finetuned model name
        self.out_model = f'{self.model_dir}/finetuned_{self.batch_ind}.pt'
        # set previous model
        
        finetuned_models = sorted(glob.glob(f'{self.model_dir}/*.pt'), key = lambda x:int(x.split('_')[-1][:-3]))
        finetuned_models = [x for x in finetuned_models if int(x.split('_')[-1][:-3]) <= batch_ind]
        finetuned_models_full = [f"models/{self.model}.pt"]+finetuned_models
        self.base_model = finetuned_models_full[-1] #if int(finetuned_models[-1].split('_')[-1][:-3]) != batch_ind else finetuned_models[-2]
        
        if finetuned_models:
            if int(finetuned_models[-1].split('/')[-1].split('_')[-1][:-3]) == batch_ind:
                tmp_model_name = finetuned_models_full[-2]
            else:
                tmp_model_name = finetuned_models_full[-1]
        else:
            tmp_model_name = finetuned_models_full[-1]
        if tmp_model_name == f"models/{self.model}.pt":
        # if not finetuned_models: #or (finetuned_models and batch_ind == 0)
            tmp_filename = self.filename.replace('_'+self.setting,'') if self.setting else self.filename
            # self.base_model = f"models/{self.model}.pt"  # base model
            base_model_name = tmp_model_name.split('/')[-1][:-3] # base model
            self.pref_pred_file = f'results/{self.current_patient}/{base_model_name}/pred/{base_model_name}_{tmp_filename}.csv'
            self.prev_pred = pd.read_csv(self.pref_pred_file, parse_dates=['timestamp'])
            self.prev_pred_batch = self.prev_pred[self.prev_pred['batch_id']==self.batch_ind]
            self.prev_metric_file = f"results/{self.current_patient}/{base_model_name}/metrics/{base_model_name}_{tmp_filename}.csv"
            self.prev_metrics = pd.read_csv(self.prev_metric_file)
            self.prev_metrics_batch = self.prev_metrics[self.prev_metrics['batch_id']==self.batch_ind]
            # self.prev_pred = prev_model_data['smoothed_pred_temp'].to_numpy()
        else:# if finetuned_models and batch_ind > 0:
            base_model_name = tmp_model_name.split('/')[-2] # model folder name/experiment name
            finetuned_name = tmp_model_name.split('/')[-1] # finetuned_x.pt
            self.prev_pred_file = f"results/{self.current_patient}/{base_model_name}/pred/{finetuned_name.replace('.pt','')}_{self.filename}.csv" # prediction results for this previous model
            self.prev_pred = pd.read_csv(self.prev_pred_file, parse_dates=['timestamp'])
            self.prev_pred_batch = self.prev_pred[self.prev_pred['batch_id']==self.batch_ind]
            self.prev_metric_file = f"results/{self.current_patient}/{base_model_name}/metrics/{finetuned_name.replace('.pt','')}_{self.filename}.csv"
            self.prev_metrics = pd.read_csv(self.prev_metric_file)
            self.prev_metrics_batch = self.prev_metrics[self.prev_metrics['batch_id']==self.batch_ind]
            # self.prev_pred = prev_model_data['smoothed_pred_temp'].to_numpy()
                # self.use_thres = self.thres
        # if not finetuned_models or (finetuned_models and batch_ind is None):
        #     self.base_model = f"models/{self.model}.pt"  # base model
        # elif finetuned_models:
        #     self.base_model = finetuned_models[-1]
        
        self.finetune_progress_plot = f"results/{self.current_patient}/{self.current_patient}_{'_'.join(self.exp_name.split('_')[1:])}/figures/finetune_progress/finetuned_{self.batch_ind}"
        os.makedirs(os.path.dirname(self.finetune_progress_plot), exist_ok=True)
            # self.use_thres = self.thres
        self.get_adj_threshold()
        # set mlflow run name
        self.run_name = f"Patient_{self.current_patient}_Batch_{self.batch_ind}"
        # search for previous runs, shouldn't be any if force mode because previous ones are deleted
        runs = client.search_runs(
            [self.experiment_id],
            filter_string=f"tags.mlflow.runName = '{self.run_name}'",
            order_by=["start_time DESC"],
            max_results=1
        )
        if runs:
            self.run_id = runs[0].info.run_id  # get previous run id if not force
        else:
            # create a new run if force deleted all previous runs
            run = client.create_run(
                experiment_id=self.experiment_id,
                tags={"mlflow.runName": self.run_name}
            )
            self.run_id = run.info.run_id
        
    def get_adj_threshold(self):
        all_thres = pd.read_csv(f'models/thres.csv')
        all_thres = all_thres[all_thres['use']==True]
        all_thres = all_thres[(all_thres['patient']==self.current_patient) & 
                            (all_thres['model_name']==self.exp_name) &
                            (all_thres['batch_id']<=self.batch_ind)].sort_values(by=['timestamp'])
        if not all_thres.empty:
            self.adjusted_thres = all_thres.iloc[-1]

    def get_batch_data(self):
        if self.batch_ind is None:
            return None, None
        batch_start = self.batch_start[self.batch_ind]
        batch_end = batch_start + pd.Timedelta(hours = self.batch_hour)
        conn = sqlite3.connect(f'data/{self.current_patient}.db')
        query = f"SELECT start_timestamp, timestamps, BIPOLAR FROM {self.current_patient} WHERE start_timestamp >= '{batch_start}' AND start_timestamp < '{batch_end}'"
        df = pd.read_sql(query,conn)
        if df.empty:
            return None, None
        df = jsondecode_df(df)
        df['start_timestamp'] = pd.to_datetime(df["start_timestamp"]).dt.tz_localize(None).apply(lambda x:x.replace(microsecond=0) + timedelta(seconds=(x.microsecond >= 500000)))
        if self.label == 'tech':
            # logger.info('Technician label used for finetuning.')
            feat_labels = pd.read_pickle(f'labels/{self.current_patient}_feat_labels.pkl')
        elif self.label == 'patient':
            # logger.info('Patient label used for finetuning.')
            feat_labels = pd.read_pickle(f'labels/{self.current_patient}_patient_feat_labels.pkl')
        elif self.label == 'databricks':
            # logger.info('Databricks patient annotation label used for finetuning.')
            feat_labels = pd.read_pickle(f'labels/{self.current_patient}_databricks_feat_labels.pkl')
        feat_labels = feat_labels[(feat_labels.index >= batch_start+pd.Timedelta(seconds=self.module.feat_setting['win']))&(feat_labels.index <= batch_end)]
        return df, feat_labels

class EvaluationContext():

    def __init__(self):
        pass

    def initialize(self, params):
        '''
        Should contain model, patient, batch_hour, thres
        parser.add_argument("-m", "--model", type=str, required=True, help="Model to evaluate")
        parser.add_argument("-p", "--patient", type=str, required=True, help="Patient data to evaluate on")
        parser.add_argument("-f", "--force", action = 'store_true', help="Force rerun even if results already exist")
        parser.add_argument("--prob_only", action = 'store_true', help="Only get probability")
        parser.add_argument("--pred_only", action = 'store_true', help="Only get binary predictions, will automatically run prob if no prob files available")
        parser.add_argument("--metric_only", action = 'store_true', help="Only compute metrics, will automatically run prob and pred if no pred files available")
        parser.add_argument("--plot_only", action = 'store_true', help="Only generate necessary, will automatically run prob and pred if no pred files available")

        parser.add_argument("--setting", type=str, default='', help="Suffix to specify experiment name")
        parser.add_argument("--thres_high", type=float, default=0.5, help="threshold to get binary predictions")
        parser.add_argument("--thres_low", type=float, default=0.5, help="threshold to get binary predictions")
        parser.add_argument("--thres", type=float, default=0.9, help="probability threshold")
        parser.add_argument("--perc", type=float, default=0.15, help="percentage of channel seizing")
        parser.add_argument("--batch_hour", type=int, default=3,help="Length of data to be used for each batch") 
        add adjust_thres
        '''
        for k, v in params.items():
            setattr(self, k, v)
        self.flag_list = {'pipeline':(True, True, True), 
                          'prob':(True, False, False), 
                          'pred':(False, True, False),
                          'metric':(False, False, True),
                          'skip_prob':(False, True, True),
                          'skip_metric':(True, True, False)}
        self.default_flag = self.flag_list.get(self.mode, (True, True, True))
        if self.patient == 'all':
            self.patient_ids = os.listdir('/project/pioneer/rawdata')
        elif self.patient == 'scalp':
            self.patient_ids = os.listdir('/project/pioneer/rawdata')
            self.patient_ids = [p for p in self.patient_ids if p not in ieeg_list]
        elif self.patient == 'ieeg':
            self.patient_ids = ieeg_list
        else:
            self.patient_ids = sorted(self.patient.split(','))
        self.batch_starts = {}
        self.sampling_freqs = {}
        self.patient_types = {}
        self.filenames = {}
        self.n_batches = {} # k:int(np.ceil(v/self.batch_dura)) for k,v in self.n_clip.items()
        self.scalers = {}
        for patient in self.patient_ids:
            with open(f'labels/{patient}_data_labels.pkl','rb') as f:
                label_df = pickle.load(f)
                data_timestamps = label_df.index.to_numpy()
            start_time, end_time = data_timestamps[0], data_timestamps[-1]
            self.batch_starts[patient] = pd.date_range(start=start_time, end=end_time-pd.Timedelta(hours=self.batch_hour), freq=f"{self.batch_hour}h").tz_localize(None)
            self.n_batches[patient] = len(self.batch_starts[patient])
            # get meta info
            with open(f'INFO/{patient}.pkl','rb') as f:
                info = pickle.load(f)
            self.sampling_freqs[patient] = info['samplingFreq']
            self.patient_types[patient] = info['studyType']
            if info['studyType'] == 'eeg':
                filename = f'{self.thres_high}_extend_{self.thres_low}'
            elif info['studyType'] == 'ieeg':
                filename = f'{self.thres}_{self.perc}'
                # get 1 or 2 or 10 minutes data from ?
                self.scalers[patient] = RobustScaler()
                conn = sqlite3.connect(f'data/{patient}.db')
                cursor = conn.cursor()
                query = f"SELECT BIPOLAR, timestamps FROM {patient} ORDER BY start_timestamp LIMIT 600 OFFSET 3600"
                data = pd.read_sql(query,conn)
                data = jsondecode_df(data)
                data = pd.concat([pd.DataFrame(d['BIPOLAR'],index = d['timestamps']) for _, d in data.iterrows()]).sort_index()
                col_filter = [c for c in data.columns if 'DC' not in c ]
                data = data[col_filter]
                data.iloc[1:,:] = ar_one(data.values)
                self.scalers[patient].fit(data)
            if self.setting:
                filename = f'{filename}_{self.setting}'
            if self.adjth:
                filename = f'{filename}_adjth'
            self.filenames[patient] = filename
    
    def get_adj_threshold(self):
        all_thres = pd.read_csv(f'models/thres.csv')
        all_thres = all_thres[all_thres['use']==True]
        all_thres = all_thres[(all_thres['patient']==self.current_patient) & 
                            (all_thres['model_name']==self.model) &
                            (all_thres['batch_id']<=self.batch_ind)].sort_values(by=['timestamp'])
        if not all_thres.empty:
            self.adjusted_thres = all_thres.iloc[-1]

    def set_patient_batch(self, patient, batch_ind = None):
        self.current_patient = patient
        self.batch_ind = batch_ind
        self.batch_start = self.batch_starts[self.current_patient]
        self.n_batch = self.n_batches[self.current_patient]
        self.sampling_freq = self.sampling_freqs[self.current_patient]
        self.patient_type = self.patient_types[self.current_patient]
        self.filename = self.filenames[self.current_patient]
        if patient in self.scalers:
            self.scaler = self.scalers[patient]
        if self.patient_type == 'eeg':
            if self.model == 'finetune_sparcnet':
                self.module = importlib.import_module('funcs.finetune_sparcnet_funcs')
            elif self.model == 'morgoth':
                self.module = importlib.import_module('funcs.morgoth_funcs')
            else:
                self.module = importlib.import_module(f'funcs.sparcnet_funcs_update')
        elif self.patient_type == 'ieeg':
            self.ieeg_model = 'wvnt' if 'wvnt' in self.model else 'oncet'
            self.module = importlib.import_module(f'funcs.{self.ieeg_model}_funcs')
        self.set_pred()
        self.set_plot()
        if self.batch_ind is not None:
            self.get_adj_threshold()
        if self.adjth:
            if hasattr(self, 'adjusted_thres'):
                self.thres_high = self.adjusted_thres['thres_high']
                self.thres_low = self.adjusted_thres['thres_low']
                self.thres = self.adjusted_thres['thres']
                self.perc = self.adjusted_thres['perc']
            else:
                self.do_prob = False
                self.do_pred = False
                self.do_metric = False
                self.do_plot = False
   
    def set_pred(self):
        self.result_dir = f'results/{self.current_patient}/{self.model}'
        self.tmp_dir = os.path.join(self.result_dir,'tmp') # for storing temporary prob files
        self.prob_dir = os.path.join(self.result_dir,'prob') # for storing complete probability files, its annoying that files with the same name can't be opened at the same time
        self.pred_dir = os.path.join(self.result_dir,'pred')
        self.metric_dir = os.path.join(self.result_dir,'metrics')
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.prob_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)
        os.makedirs(self.metric_dir, exist_ok=True)
        if self.batch_ind is not None:
            self.eval_model = f'models/{self.model}/finetuned_{self.batch_ind}.pt'
            self.prob_file = os.path.join(self.prob_dir,f'finetuned_{self.batch_ind}.csv')
            self.pred_file = os.path.join(self.pred_dir,f'finetuned_{self.batch_ind}_{self.filename}.csv')
            self.metric_file = os.path.join(self.metric_dir,f'finetuned_{self.batch_ind}_{self.filename}.csv')
            # # get a thres
            # all_thres = pd.read_csv(f'models/thres.csv')
            # all_thres = all_thres[all_thres['use']==True]
            # all_thres = all_thres[(all_thres['patient']==self.current_patient) & 
            #                     (all_thres['model_name']==self.model) &
            #                     (all_thres['batch_id']<=self.batch_ind)].sort_values(by=['timestamp'])
            # if not all_thres.empty:
            #     self.adjusted_thres = all_thres['thres'].iloc[-1]
            # else:
            #     self.adjusted_thres = self.thres
        else:
            self.eval_model = f'models/{self.model}.pt'
            self.prob_file = os.path.join(self.prob_dir,f'{self.model}.csv')
            self.pred_file = os.path.join(self.pred_dir,f'{self.model}_{self.filename}.csv')
            self.metric_file = os.path.join(self.metric_dir,f'{self.model}_{self.filename}.csv')# 'test.csv')#
            # self.adjusted_thres = None
        # decide if do each step
        has_model  = os.path.exists(self.eval_model)
        has_prob   = os.path.exists(self.prob_file)
        has_pred   = os.path.exists(self.pred_file)
        has_metric = os.path.exists(self.metric_file)
        self.do_prob, self.do_pred, self.do_metric = self.default_flag
        self.do_plot = True
        if self.do_metric:
            # Ensure upstream stages exist or will run
            if not has_pred:
                self.do_pred = has_prob or has_model
                self.do_metric = self.do_pred and (self.force or not has_metric)
            else:
                self.do_metric = (self.force or not has_metric)
        if self.do_pred:
            # Ensure probs exist or will be created
            if not has_prob:
                self.do_prob = has_model
                self.do_pred = self.do_prob and (self.force or not has_pred)
            else:
                self.do_pred = (self.force or not has_pred)
        self.do_prob = self.do_prob and (has_model and (self.force or not has_prob))
        self.do_plot = (self.do_pred or has_pred) and (self.do_metric or has_metric)
        # self.skip_plot = True if not os.path.exists(self.pred_file)  else False
    
    def set_plot(self):
        self.plot_vars = ['auroc_sample','auprc_sample','recall_event','balanced_acc', 'fp','f1_event']
        # what figures I want to plot, full trajectory and overlay trajectory
        # the finetuned model's performance compared to the previous model, on testing data
        # the finetuned model's performance compared to baseline, on testing data
        self.figure_dir = f'results/{self.current_patient}/{self.model}/figures'
        self.trajectory_dir = f'{self.figure_dir}/trajectory'
        self.auroc_dir = f'{self.figure_dir}/auroc'
        self.cal_dir = f'{self.figure_dir}/calibration'
        self.paired_line_test_prev = f'{self.figure_dir}/paired_line_test_prev_{self.filename}'
        self.paired_line_test_base = f'{self.figure_dir}/paired_line_test_base_{self.filename}'
        self.agg_dir = f'results/{self.current_patient}/{self.model}/agg'
        os.makedirs(self.trajectory_dir, exist_ok=True)
        os.makedirs(self.auroc_dir, exist_ok=True)
        os.makedirs(self.cal_dir, exist_ok=True)
        os.makedirs(self.agg_dir, exist_ok=True)
        if self.batch_ind is None:
            self.plot_paired_line = False
            self.plot_overlay = False
            self.trajectory_file = os.path.join(self.trajectory_dir, self.model)
            self.cal_file = os.path.join(self.cal_dir, self.model)
            self.auroc_file = os.path.join(self.auroc_dir, self.model)
        else:
            self.plot_paired_line = True
            self.plot_overlay = True
            self.trajectory_file = os.path.join(self.trajectory_dir, f'finetuned_{self.batch_ind}_{self.filename}')
            self.overlay_file = os.path.join(self.trajectory_dir, f'overlay_finetuned_{self.batch_ind}_{self.filename}')
            self.cal_file = os.path.join(self.cal_dir, f'finetuned_{self.batch_ind}')
            self.cal_overlay_file = os.path.join(self.cal_dir, f'overlay_finetuned_{self.batch_ind}')
            self.auroc_file = os.path.join(self.auroc_dir, f'finetuned_{self.batch_ind}')
            self.auroc_overlay_file = os.path.join(self.auroc_dir, f'overlay_finetuned_{self.batch_ind}')
            tmp_filename = self.filename.replace('_'+self.setting,'') if self.setting else self.filename
            tmp_filename = tmp_filename.replace('_adjth','') if self.adjth else tmp_filename
            if self.patient_type == 'eeg':
                self.baseline_prob_file = f"results/{self.current_patient}/base_model/prob/base_model.csv" # add setting
                self.baseline_feat_file = f"results/{self.current_patient}/base_model/pred/base_model_{tmp_filename}.csv" # add setting
                self.baseline_metric_file = f"results/{self.current_patient}/base_model/metrics/base_model_{tmp_filename}.csv"
            elif self.patient_type == 'ieeg':
                self.baseline_prob_file = f"results/{self.current_patient}/{self.ieeg_model}_base_model/prob/{self.ieeg_model}_base_model.csv" # add setting
                self.baseline_feat_file = f"results/{self.current_patient}/{self.ieeg_model}_base_model/pred/{self.ieeg_model}_base_model_{tmp_filename}.csv" # add setting
                self.baseline_metric_file = f"results/{self.current_patient}/{self.ieeg_model}_base_model/metrics/{self.ieeg_model}_base_model_{tmp_filename}.csv"
            # self.thres_plot_setting = 'FC+AdjustThres'

ctx = Context()
pred_ctx = EvaluationContext()


# def get_path(key: str) -> str:
#     """Retrieve a path by key anywhere in the code."""
#     return PATHS.get(key)
