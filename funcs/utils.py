import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter,filtfilt,sosfiltfilt
import re
from os.path import join as ospj

# UTILITY FUNCTIONS
## FEATURE EXTRACTION
def num_wins(xLen, fs, winLen, winDisp):
  return int(((xLen/fs - winLen + winDisp) - ((xLen/fs - winLen + winDisp)%winDisp))/winDisp)

def MovingWinClips(x,fs,winLen,winDisp):
  # calculate number of windows and initialize receiver
  nWins = num_wins(len(x),fs,winLen,winDisp)
  samples = np.empty((nWins,int(winLen*fs)))
  # create window indices - these windows are left aligned
  idxs = np.array([(winDisp*fs*i,(winLen+winDisp*i)*fs)\
                   for i in range(nWins)]).astype(int)
  # apply feature function to each channel
  for i in range(idxs.shape[0]):
    samples[i,:] = x[idxs[i,0]:idxs[i,1]]
  
  return samples
    
## ————————————————CHANNEL PREPROCESSING————————————————


def clean_labels(channel_li) :
    """
    Clean and standardize channel labels.

    Parameters:
    - channel_li (Union[Iterable[str], str]): Either a single channel label or an iterable of channel labels.

    Returns:
    - np.ndarray: An array of cleaned and standardized channel labels.

    Example:
    >>> clean_labels('LA 01')
    array(['LA1'])
    """
    if isinstance(channel_li, str):
        channel_li = [channel_li]
    new_channels = []
    for i in range(len(channel_li)):
        # standardizes channel names
        label_num_search = re.search(r"\d", channel_li[i])
        if label_num_search is not None:
            label_num_idx = label_num_search.start()
            label_non_num = channel_li[i][:label_num_idx]
            label_num = channel_li[i][label_num_idx:]
            label_num = label_num.lstrip("0")
            label = label_non_num + label_num
        else:
            label = channel_li[i]
        label = label.replace("EEG", "")
        label = label.replace("Ref", "")
        label = label.replace(" ", "")
        label = label.replace("-", "")
        label = label.replace("CAR", "")
        label = label.replace("HIPP", "DH")
        label = label.replace("AMY", "DA")
        label = label.replace("FP", "Fp")
        label = label.replace("CZ", "Cz")
        label = label.replace("FZ", "Fz")
        label = label.replace("PZ", "Pz")
        label = label.replace("PZ", "Pz")
        label = label.replace("FPz", "Fpz")
        label = label.replace("FPZ", "Fpz")
        label = "T3" if label == "T7" else label
        label = "T4" if label == "T8" else label
        label = "T5" if label == "P7" else label
        label = "T6" if label == "P8" else label
        new_channels.append(label)
    return np.array(new_channels)


def check_channel_type(channel_li):
    """
    Find non-iEEG channel labels.

    Parameters:
    - channel_li (Union[Iterable[str], str]): Either an iterable of channel labels or a single channel label.

    Returns:
    - np.ndarray: Boolean array indicating whether each channel is non-iEEG.
    """
    scalp = [
        "O",
        "C",
        "CZ",
        "F",
        "FP",
        "FZ",
        "T",
        "P",
        "PZ",
        "FPZ",
    ]
    ekg = ["EKG","ECG"]
    emg = ["EMG"]
    eog = ['LOC','ROC']
    other = ["RATE","DC"]

    if isinstance(channel_li, str):
        channel_li = [channel_li]
    ch_df = []
    for i in channel_li:
        regex_match = re.search(r"\d", i)
        if regex_match is None:
            ch_df.append({"name": i, "lead": i, "contact": 0})
            continue
        label_num_idx = regex_match.start()
        label_non_num = i[:label_num_idx]
        label_num = i[label_num_idx:]
        ch_df.append({"name": i, "lead": label_non_num, "contact": label_num})
    ch_df = pd.DataFrame(ch_df)
    for lead, group in ch_df.groupby("lead"):
        if lead.upper() in ekg:
            ch_df.loc[group.index, "type"] = "ekg"
            continue
        if lead.upper() in scalp:
            ch_df.loc[group.index, "type"] = "eeg"
            if i == "O1" or i == "O2":
                if (
                    channel_li.count("O3") == 1 or channel_li.count("O4") == 1
                ):  # if intracranial, should have these too
                    ch_df.loc[group.index, "type"] = "ieeg"
            continue
        if lead.upper() in emg:
            ch_df.loc[group.index, "type"] = "emg"
            continue
        if lead.upper() in eog:
            ch_df.loc[group.index, "type"] = "eog"
            continue
        if lead.upper() in other:
            ch_df.loc[group.index, "type"] = "misc"
            continue
        if len(group) > 16:
            ch_df.loc[group.index.to_list(), "type"] = "ecog"
        else:
            ch_df.loc[group.index.to_list(), "type"] = "seeg"
    return ch_df["type"].to_numpy()

def detect_bad_channels_eeg(data,fs):
    values = data.copy()
    which_chs = np.arange(values.shape[1])
    ## Parameters to reject super high variance
    tile = 99
    mult = 10
    num_above = 1
    abs_thresh = 400
    abs_thresh2 = 500

    ## Parameter to reject high 60 Hz
    percent_60_hz = 0.7

    ## Parameter to reject electrodes with much higher std than most electrodes
    mult_std = 10

    bad = []
    high_ch = []
    nan_ch = []
    zero_ch = []
    flat_ch = []
    high_var_ch = []
    noisy_ch = []
    all_std = np.empty((len(which_chs),1))
    all_std[:] = np.nan
    details = {}

    for i in range(len(which_chs)):       
        ich = which_chs[i]
        eeg = values[:,ich]
        bl = np.nanmedian(eeg)
        all_std[i] = np.nanstd(eeg)
        
        ## Remove channels with nans in more than half
        if sum(np.isnan(eeg)) > 0.5*len(eeg):
            bad.append(ich)
            nan_ch.append(ich)
            continue
        
        ## Remove channels with zeros in more than half
        if sum(eeg == 0) > (0.5 * len(eeg)):
            bad.append(ich)
            zero_ch.append(ich)
            continue

        ## Remove channels with extended flat-lining
        if (sum(np.diff(eeg,1) == 0) > (0.5 * len(eeg))):
            bad.append(ich)
            flat_ch.append(ich)
            continue
        
        ## Remove channels with too many above absolute thresh
        if sum(abs(eeg - bl) > abs_thresh) + sum(abs(eeg) > abs_thresh2) > 0.1*len(eeg):
            bad.append(ich)
            high_ch.append(ich)
            continue

        ## Remove channels if there are rare cases of super high variance above baseline (disconnection, moving, popping)
        pct = np.percentile(eeg,[100-tile,tile])
        thresh = [bl - mult*(bl-pct[0]), bl + mult*(pct[1]-bl)]
        sum_outside = sum(((eeg > thresh[1]) + (eeg < thresh[0])) > 0)
        if sum_outside >= num_above:
            bad.append(ich)
            high_var_ch.append(ich)
            continue
        
        ## Remove channels with a lot of 60 Hz noise, suggesting poor impedance
        # Calculate fft
        Y = np.fft.fft(eeg-np.nanmean(eeg))
        
        # Get power
        P = abs(Y)**2
        freqs = np.linspace(0,fs,len(P)+1)
        freqs = freqs[:-1]
        
        # Take first half
        P = P[:np.ceil(len(P)/2).astype(int)]
        freqs = freqs[:np.ceil(len(freqs)/2).astype(int)]
        
        P_60Hz = sum(P[(freqs > 58) * (freqs < 62)])/sum(P)
        if P_60Hz > percent_60_hz:
            bad.append(ich)
            noisy_ch.append(ich)
            continue

    ## Remove channels for whom the std is much larger than the baseline
    median_std = np.nanmedian(all_std)
    higher_std = which_chs[(all_std > (mult_std * median_std)).squeeze()]
    bad_std = higher_std

    channel_mask = np.ones((values.shape[1],),dtype=bool)
    channel_mask[bad] = False
    details['noisy'] = noisy_ch
    details['nans'] = nan_ch
    details['zeros'] = zero_ch
    details['flat'] = flat_ch
    details['var'] = high_var_ch
    details['higher_std'] = bad_std
    details['high_voltage'] = high_ch
    
    return channel_mask,details


def detect_bad_channels(data,fs,lf_stim = False):
    '''
    data: raw EEG traces after filtering (i think)
    fs: sampling frequency
    channel_labels: string labels of channels to use
    '''
    values = data.copy()
    which_chs = np.arange(values.shape[1])
    ## Parameters to reject super high variance
    tile = 99
    mult = 10
    num_above = 1
    abs_thresh = 5e3

    ## Parameter to reject high 60 Hz
    percent_60_hz = 0.1

    ## Parameter to reject electrodes with much higher std than most electrodes
    mult_std = 10

    bad = []
    high_ch = []
    nan_ch = []
    zero_ch = []
    flat_ch = []
    high_var_ch = []
    noisy_ch = []
    all_std = np.empty((len(which_chs),1))
    all_std[:] = np.nan
    details = {}

    for i in range(len(which_chs)):       
        ich = which_chs[i]
        eeg = values[:,ich]
        bl = np.nanmedian(eeg)
        all_std[i] = np.nanstd(eeg)
        
        ## Remove channels with nans in more than half
        if sum(np.isnan(eeg)) > 0.5*len(eeg):
            bad.append(ich)
            nan_ch.append(ich)
            continue
        
        ## Remove channels with zeros in more than half
        if sum(eeg == 0) > (0.5 * len(eeg)):
            bad.append(ich)
            zero_ch.append(ich)
            continue

        # ## Remove channels with extended flat-lining
        # if (sum(np.diff(eeg,1) <= 1e-3) > (0.5 * len(eeg))):
        #     bad.append(ich)
        #     flat_ch.append(ich)
        
        ## Remove channels with too many above absolute thresh
        if sum(abs(eeg - bl) > abs_thresh) > 10:
            if not lf_stim:
                bad.append(ich)
            high_ch.append(ich)
            continue

        ## Remove channels if there are rare cases of super high variance above baseline (disconnection, moving, popping)
        pct = np.percentile(eeg,[100-tile,tile])
        thresh = [bl - mult*(bl-pct[0]), bl + mult*(pct[1]-bl)]
        sum_outside = sum(((eeg > thresh[1]) + (eeg < thresh[0])) > 0)
        if sum_outside >= num_above:
            if not lf_stim:
                bad.append(ich)
            high_var_ch.append(ich)
            continue
        
        ## Remove channels with a lot of 60 Hz noise, suggesting poor impedance
        # Calculate fft
        Y = np.fft.fft(eeg-np.nanmean(eeg))
        
        # Get power
        P = abs(Y)**2
        freqs = np.linspace(0,fs,len(P)+1)
        freqs = freqs[:-1]
        
        # Take first half
        P = P[:np.ceil(len(P)/2).astype(int)]
        freqs = freqs[:np.ceil(len(freqs)/2).astype(int)]
        
        P_60Hz = sum(P[(freqs > 58) * (freqs < 62)])/sum(P)
        if P_60Hz > percent_60_hz:
            bad.append(ich)
            noisy_ch.append(ich)
            continue

    ## Remove channels for whom the std is much larger than the baseline
    median_std = np.nanmedian(all_std)
    higher_std = which_chs[(all_std > (mult_std * median_std)).squeeze()]
    # for ch in bad_std:
    #     if ch not in bad:
    #         if ~lf_stim:
    #             bad.append(ch)
    channel_mask = np.ones((values.shape[1],),dtype=bool)
    channel_mask[bad] = False
    channel_mask[higher_std] = False
    details['noisy'] = noisy_ch
    details['nans'] = nan_ch
    details['zeros'] = zero_ch
    details['flat'] = flat_ch
    details['var'] = high_var_ch
    details['higher_std'] = np.where(higher_std)[0].tolist()
    details['high_voltage'] = high_ch
    
    return channel_mask,details


def bipolar_montage_ieeg(ch_list):
    """_summary_

    Args:
        data (np.ndarray): _description_
        ch_types (pd.DataFrame): _description_

    Returns:
        np.ndarray: _description_
    """
    bipolar_labels = []
    bipolar_idx = []
    for i, ch in enumerate(ch_list):
        ch1Ind = i
        label_num_search = re.search(r"\d", ch)
        if label_num_search is not None:
            label_num_idx = label_num_search.start()
            label_non_num = ch[:label_num_idx]
            label_num = int(ch[label_num_idx:])
            ch2_num = label_num + 1
            ch2 = label_non_num + f"{ch2_num}"
            ch2exists = np.where(ch_list == ch2)[0]
            if len(ch2exists) > 0:
                ch2Ind = ch2exists[0]
            else:
                ch2Ind = np.nan
            ch3_num = label_num + 2
            ch3 = label_non_num + f"{ch3_num}"
            ch3exists = np.where(ch_list == ch3)[0]
            if len(ch3exists) > 0:
                ch3Ind = ch3exists[0]
            else:  
                ch3Ind = np.nan
            if np.isnan(ch2Ind):
                if not np.isnan(ch3Ind):
                    bipolar_idx.append([ch1Ind,ch3Ind, np.nan])
                    bipolar_labels.append(ch + "-" + ch3)
            else:
                bipolar_idx.append([ch1Ind,ch2Ind,ch3Ind])
                bipolar_labels.append(ch + "-" + ch2)
    return np.array(bipolar_labels), np.array(bipolar_idx)

def bipolar_montage_eeg(ch_list):

    ch_dict = {ch:i for i,ch in enumerate(ch_list)}
    ch1 = ['Fp1','F7','T3','T5','Fp2','F8','T4','T6',
                    'Fp1','F3','C3','P3','Fp2','F4','C4','P4','Fz','Cz']
    ch2 = ['F7','T3','T5','O1','F8','T4','T6','O2',
                    'F3','C3','P3','O1','F4','C4','P4','O2','Cz','Pz']
    ch1_index = [ch_dict.get(ch, np.nan) for ch in ch1]
    ch2_index = [ch_dict.get(ch, np.nan) for ch in ch2]
    bipolar_index = np.array([ch1_index,ch2_index]).T
    bipolar_labels = [f"{ch1[i]}-{ch2[i]}" for i in range(len(ch1))]
    nan_mask = np.any(np.isnan(bipolar_index),axis=1)
    return np.array(bipolar_labels)[~nan_mask],bipolar_index[~nan_mask].astype(int)

def car(data):
    """
    Perform Common Average Reference (CAR) on the input iEEG data.
    """
    out_data = data - np.nanmean(data, 1)[:, np.newaxis]
    return out_data

def bipolar(data, bipolar_index):
    out_data = data[:,bipolar_index[:,0]] - data[:,bipolar_index[:,1]]
    return out_data

## ————————————SIGNAL PREPROCESSING——————————————

def downsample(data,fs,target):
    if fs != target:
        if fs % target == 0:
            div = int(fs//target)
            if isinstance(data, np.ndarray):
                data_bpd = data[::div,:]
            elif isinstance(data, pd.DataFrame):
                data_bpd = data.iloc[::div]
        else:
            signal_len = int(data.shape[0]/fs*target)
            data_bpd = scipy.signal.resample(data,signal_len,axis=0)
        return data_bpd
    else:
        return data

# def interp_resample(time,data,target):
#     signal_len = int(data.shape[0]/fs*target)
#     data_bpd = sc.signal.resample(data,signal_len,axis=0)
#     return data_bpd,target

def notch_filter(data, fs):
    """_summary_

    Args:
        data (np.ndarray): _description_
        fs (float): _description_

    Returns:
        np.array: _description_
    """
    # remove 60Hz noise
    b, a = butter(4,(58,62),'bandstop',fs=fs)
    d, c = butter(4,(118,122),'bandstop',fs=fs)

    data_filt = filtfilt(b, a, data, axis=0)
    data_filt_filt = filtfilt(d, c, data_filt, axis = 0)
    # TODO: add option for causal filter
    # TODO: add optional argument for order

    return data_filt_filt

def bandpass_filter(data, fs, order=3, lo=1, hi=120):
    """_summary_

    Args:
        data (np.ndarray): _description_
        fs (float): _description_
        order (int, optional): _description_. Defaults to 3.
        lo (int, optional): _description_. Defaults to 1.
        hi (int, optional): _description_. Defaults to 120.

    Returns:
        np.array: _description_
    """
    # TODO: add causal function argument
    # TODO: add optional argument for order
    sos = butter(order, [lo, hi], output="sos", fs=fs, btype="bandpass")
    data_filt = sosfiltfilt(sos, data, axis=0)
    return data_filt

def ar_one(data):
    """
    The ar_one function fits an AR(1) model to the data and retains the residual as
    the pre-whitened data
    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates
    Returns
    -------
        data_white: ndarray, shape (T, N)
            Whitened signal with reduced autocorrelative structure
    """
    # Retrieve data attributes
    n_samp, n_chan = data.shape
    # Apply AR(1)
    data_white = np.zeros((n_samp-1, n_chan))
    for i in range(n_chan):
        win_x = np.vstack((data[:-1, i], np.ones(n_samp-1)))
        w = np.linalg.lstsq(win_x.T, data[1:, i], rcond=None)[0]
        data_white[:, i] = data[1:, i] - (data[:-1, i]*w[0] + w[1])
    return data_white

from sklearn.linear_model import LinearRegression

def pre_whiten(data):
    """Pre-whiten the input data using linear regression.

    Args:
        data (np.ndarray): Matrix representing data. Each column is a channel, and each row is a time point.

    Returns:
        np.ndarray: Pre-whitened data matrix.
    """
    prewhite_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        vals = data[:, i].reshape(-1, 1)
        if np.sum(~np.isnan(vals)) == 0:
            continue
        model = LinearRegression().fit(vals[:-1, :], vals[1:, :])
        E = model.predict(vals[:-1, :]) - vals[1:, :]
        if len(E) < len(vals):
            E = np.concatenate([E, E[-1] * np.zeros([len(vals) - len(E), 1])])
        prewhite_data[:, i] = E.reshape(-1)

    return prewhite_data

#——————————————————————somewhat feature related——————————————————————————————-
from scipy.signal import welch
from scipy.integrate import simpson

def bandpower(
    data,
    fs,
    band,
    win_size = None,
    relative = False,
):
    """Adapted from https://raphaelvallat.com/bandpower.html
    Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array or 2d-array
        Input signal in the time-domain. (time by channels)
    fs : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    win_size : float
        Length of each window in seconds.
        If None, win_size = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : np.ndarray
        Absolute or relative band power. channels by bands
    """
    band = np.asarray(band)
    assert len(band) == 2, "CNTtools:invalidBandRange"
    assert band[0] < band[1], "CNTtools:invalidBandRange"
    if np.ndim(data) == 1:
        data = data[:,np.newaxis]
    nchan = data.shape[1]
    bp = np.nan * np.zeros(nchan)
    low, high = band

    # Define window length
    # if win_size is not None:
    #     nperseg = int(win_size * fs)
    # else:
    #     nperseg = int((2 / low) * fs)

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data.T, fs)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    if psd.ndim == 2:
        bp = simpson(psd[:, idx_band], dx=freq_res)
    elif psd.ndim == 1:
        bp = simpson(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simpson(psd, dx=freq_res)

    return bp




class Preprocessor():
    """
    Do basic preprocessing for a data batch
    1. Bandpass filter 0.5-100 Hz
    2. Notch filter 60 Hz
    3. Detect bad channels, if more than 20% of channels are bad, mark as a potential artifact
    4. Downsample to 256 Hz (optional)
    5. Rereference (CAR and bipolar), for CAR, bad channels are excluded in calculating mean, bad channel indices are provided
    6. Prewhiten (if True)
    7. Return the processed data in data frame, with other metadata saved in a dictionary

    User can retrieve either the rerefed data, or other metadata fields through get_last_packet method.
    
    Note for interface with Azure, can insert database writing after filter, two rereferencing, and prewhiten, and also for bad masks
    May need a "patient" column to specify which patient the data is from
    """
    def __init__(self, lowcut = 0.5, highcut = 100, artifact_perc = 0.2, batch = False, batch_size = 1):
        self.chs = None
        self.n_chs = None
        self.lowcut = lowcut
        self.highcut = highcut
        self.batch = batch
        self.artifact_perc = artifact_perc
        self.batch_size = batch_size
        self.bad_channels = None
        self.fitted = False
        self.last_packet = None

    def _filter_data(self, data):
        # bandpass filter
        data = bandpass_filter(data, self.fs, lo = self.lowcut, hi = self.highcut)
        data = notch_filter(data, self.fs)
        return data

    def fit(self, info):
        self.fs = info['samplingFreq']
        self.fs_raw = info['samplingFreqRaw']
        
        self.batch_sample_raw = int(self.fs_raw*self.batch_size) 
        self.batch_sample = int(self.fs*self.batch_size)
        self.sample_step = self.batch_sample_raw//self.batch_sample
        
        self.raw_chs = info['channelNames']
        self.nchs = info['numberOfChannels']
        if self.nchs != len(info['channelNames']):
            self.raw_chs = info['channelNames'][:self.nchs]
        self.chs = clean_labels(self.raw_chs)
        self.type = info['studyType']
        self.ch_type = check_channel_type(self.chs)
        self.ieeg_idx = [True if i == 'seeg' or i == 'ecog' or i == 'ieeg' else False for i in self.ch_type ]
        self.eeg_idx = [True if i == 'eeg' else False for i in self.ch_type ]
        self.ekg_idx = [True if i == 'ekg' else False for i in self.ch_type ]
        self.eog_idx = [True if i == 'eog' else False for i in self.ch_type ]
        self.sleep_idx = [True if i == 'C3' or i == 'C4' or i == 'Cz' else False for i in self.chs]
        if not self.type:
            if np.sum(self.eeg_idx) >= 15:
                self.type = 'eeg'
            elif np.sum(self.ieeg_idx) >= 10:
                self.type = 'ieeg'
                
        if self.type.lower() == 'ieeg':# not sure what's the code for intracranial
            self.nchs_eeg = np.sum(self.ieeg_idx)
            self.bipolar_labels, self.bipolar_idx = bipolar_montage_ieeg(self.chs[self.ieeg_idx])
            self.nchs_bipolar = len(self.bipolar_labels)
            self.down_fs = 512
            self.highcut = np.min([250,int(self.fs//2)])
        elif self.type.lower() == 'eeg':
            self.nchs_eeg = np.sum(self.eeg_idx)
            self.bipolar_labels, self.bipolar_idx = bipolar_montage_eeg(self.chs[self.eeg_idx])
            self.nchs_bipolar = len(self.bipolar_labels)
            self.down_fs = 256
        self.fitted = True
    
    def preprocess(self, data, last_batch_ind = 0):
        """
        Do actual preprocess. 

        Args:
            data (pd.Dataframe): A pd.dataframe with columns being channel names, and two index, the first being seconds from start, the second being stamps
            last_batch_ind (int): An index for last available batch to make indexing consecutive
            ref (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        assert self.fitted == True, "Processor not fitted yet"
        assert data.shape[1] == self.nchs, "Data has different number of channels than fitted processor"
        # if self.batch:
        #     batches = self.get_batches(data, last_batch_ind)
        # else:
        #     batches = [self.data]
        artifact = False
        artifact_perc = 0
        # if isinstance(data, pd.DataFrame):
        #     timestamps = data.index
        #     data = data.values
        # else:
        #     timestamps = np.arange(data.shape[0])/self.fs
        # packets = []
        # for batch in batches:
        # downsample
        # batch_ind = batch.index[0][2]
        # data = batch.values
        # index = batch.index
        data = data[self.raw_chs]
        index = data.index
        data = data.values

        # removed downsample temporarily, I think the SDK and the headbox type available naturally makes all EEG data 256 Hz
        # and iEEG data 512 Hz
        # if self.fs > self.down_fs:
        #     data = downsample(data, self.fs, self.down_fs)
        #     timestamps = downsample(timestamps, self.fs, self.down_fs)

        processed = self._filter_data(data)
        raw_filtered = pd.DataFrame(processed, columns=self.chs)
        if self.type == 'eeg':
            eeg_data = processed[:,self.eeg_idx]
            eeg_chs = self.chs[self.eeg_idx]
            bad_mask, details = detect_bad_channels_eeg(eeg_data, self.fs)
            artifact_perc = np.sum(bad_mask == False)/self.nchs_eeg
        elif self.type == 'ieeg':
            eeg_data = processed[:,self.ieeg_idx]
            eeg_chs = self.chs[self.ieeg_idx]
            bad_mask, details  = detect_bad_channels(eeg_data, self.fs)
            artifact_perc = np.sum(bad_mask == False)/self.nchs_eeg
            if np.any(self.sleep_idx):
                scalp_data = processed[:, self.eeg_idx]
                scalp_chs = self.chs[self.eeg_idx]
                scalp_bad_mask, scalp_details = detect_bad_channels(scalp_data, self.fs)
                sleep_scalp_idx = np.array(self.sleep_idx)[self.eeg_idx]
                sleep_chs = scalp_chs[sleep_scalp_idx]
                sleep_bad_mask = scalp_bad_mask[sleep_scalp_idx]
        if artifact_perc > self.artifact_perc:
            artifact = True

        # rereference
        # CAR
        car_data = eeg_data - np.mean(eeg_data[:,bad_mask], axis = 1)[:,np.newaxis]
        if self.type == 'ieeg' and np.any(self.sleep_idx):
            scalp_car_data = scalp_data - np.mean(scalp_data[:,scalp_bad_mask], axis = 1)[:,np.newaxis]
            sleep_car_data = scalp_car_data[:,sleep_scalp_idx]
            car_data = np.concatenate([car_data, sleep_car_data], axis = 1)
            eeg_chs = np.concatenate([eeg_chs, sleep_chs])
            bad_mask = np.concatenate([bad_mask, sleep_bad_mask])
        
        # BIPOLAR
        bad_ind = np.where(~bad_mask)[0]
        if self.type == 'ieeg':
            tmp_bipolar_index = self.bipolar_idx.copy()
            # tmp_bipolar_index[np.isin(tmp_bipolar_index,bad_ind)] = np.nan
            for i, ch in enumerate(tmp_bipolar_index):
                if np.isin(ch[1],bad_ind) and not np.isin(ch[2],bad_ind) and not np.isnan(ch[2]):
                    tmp_bipolar_index[i,1] = tmp_bipolar_index[i,2]
            tmp_bipolar_index = tmp_bipolar_index[:,:2].astype('int')
            bipolar_data = eeg_data[:,tmp_bipolar_index[:,0]] - eeg_data[:,tmp_bipolar_index[:,1]]
            bad_mask_bipolar = ~np.any(np.isin(tmp_bipolar_index[:,:1],bad_ind),axis=1)
        else:
            bipolar_data = eeg_data[:,self.bipolar_idx[:,0]] - eeg_data[:,self.bipolar_idx[:,1]]
            bad_mask_bipolar = ~np.any(np.isin(self.bipolar_idx,bad_ind),axis=1)

        car_data_df = pd.DataFrame(car_data, columns = eeg_chs)
        bipolar_data_df = pd.DataFrame(bipolar_data, columns = self.bipolar_labels)

        car_data_prewhite = pd.DataFrame(pre_whiten(car_data), columns = eeg_chs)
        bipolar_data_prewhite = pd.DataFrame(pre_whiten(bipolar_data), columns = self.bipolar_labels)

        ekg_data = pd.DataFrame(processed[:,self.ekg_idx], columns = self.chs[self.ekg_idx])
        eog_data = pd.DataFrame(processed[:,self.eog_idx], columns = self.chs[self.eog_idx])

        car_bad = pd.DataFrame(bad_mask.reshape(1,-1), columns=eeg_chs)
        bipolar_bad = pd.DataFrame(bad_mask_bipolar.reshape(1,-1), columns=self.bipolar_labels)
            
        packet = {'raw': data, 'raw_fs': self.fs, 
                    'filtered':raw_filtered, 'fs': self.fs,
                    'bad_mask': bad_mask, 'bad_details': details, 
                    'artifact': artifact, 'artifact_perc': artifact_perc,
                    'CAR': car_data_df, 'BIPOLAR': bipolar_data_df, 
                    'CAR_prewhite': car_data_prewhite, 'BIPOLAR_prewhite': bipolar_data_prewhite,
                    'CAR_bad': car_bad, 'BIPOLAR_bad': bipolar_bad,
                    'EKG':ekg_data, 'EOG': eog_data}
            
            # packets.append(packet)
            # change to database writing here
        return packet
        # else:
        #     return packets[0]
    
    # def fit_preprocess(self, data, fs, ref = None):
    #     self.fit(data,fs)
    #     processed_dict = self.preprocess(data)
    #     if ref == 'bipolar':
    #         return processed_dict['BIPOLAR']
    #     elif ref == 'car':
    #         return processed_dict['CAR']
    #     else:
    #         return processed_dict

    # def get_last_packet(self, field = None):
    #     if field:
    #         return self.last_packet[field]
    #     return self.last_packet


class ClipLoader:
    """
    A clip loader serves feature calculation. It links to processed database and provides new clips for each feature function.
    One clip loader should be created per feature per patient. 

    TODO: How to deal with artifacts?
    """
    def __init__(self, feat_config, aligned = 'right', reject_art = False):
        self.feat_win = feat_config['win']
        self.feat_stride = feat_config['stride']
        self.ref = feat_config['reref']
        self.prewhite = feat_config['prewhite']
        self.aligned = aligned
        self.reject_art = reject_art
        self._set_data_key()
        self.data_start_stamp = 1
    
    def _set_data_key(self):
        self.data_key = self.ref
        if self.ref == 'filtered':
            self.bad_key = 'CAR_bad'
        else:
            if self.prewhite:
                self.data_key += '_prewhite'
            self.bad_key = self.ref + '_bad'

    def set_info(self, info):
        self.info = info
        self.fs = info['samplingFreq']
        self.fs_raw = info['samplingFreqRaw']
        self.sample_step = self.fs_raw//self.fs
        self.win_sample = int(self.feat_win*self.fs)
        self.win_sample_raw = int(self.feat_win*self.fs_raw)
        self.stride_sample = int(self.feat_stride*self.fs)
        self.stride_sample_raw = int(self.feat_stride*self.fs_raw)

    def set_ref(self, ref):
        self.ref = ref
        self._set_data_key()

    def set_prewhite(self, pre_whiten):
        self.prewhite = pre_whiten
        self._set_data_key()

    def set_feat_win(self, feat_win):
        self.feat_win = feat_win

    def set_feat_stride(self, feat_stride):
        self.feat_stride = feat_stride

    def set_data(self, data_df, bad_df):
        self.data = data_df
        self.bad_mask = bad_df
        self.win_start = self.data.index[0]
        self.stamp_end = self.data.index[-1]

    def get_data_start_stamp(self, last_feat_stamp):
        if last_feat_stamp is None:
            return 1
        else:
            if self.aligned == 'right':
                data_start_stamp = last_feat_stamp+ self.sample_step-(self.win_sample_raw)+self.stride_sample_raw
            elif self.aligned == 'left':
                data_start_stamp = last_feat_stamp + self.stride_sample_raw
            return data_start_stamp
    
    def __iter__(self):
        return self

    def __next__(self):
        """
        Allows iteration over batches.
        """
        win_stop = self.win_start + self.win_sample_raw - 1
        if win_stop > self.stamp_end:
            raise StopIteration  # Stop when limit is reached
        clip_data = self.data.query(f'{self.win_start} <= stamp <= {win_stop}')
        bad_mask = self.bad_mask.query(f'{self.win_start} <= stamp <= {win_stop}')
        self.win_start = self.win_start + self.stride_sample_raw
        if self.aligned == 'right':
            feat_index = pd.Index([clip_data.index[-1]],name='stamp')
        elif self.aligned == 'left':
            feat_index = pd.Index([clip_data.index[0]],name='stamp')
        bad_mask = np.squeeze(((bad_mask == False).mean() <= 0.2).to_numpy())
        if np.sum(bad_mask == False) > 0.2 and self.reject_art:
            artifact = True
            return self.__next__()
        else:
            artifact = False
            return feat_index, clip_data, bad_mask

def sort_lists(a, *others, reverse=False):
    """
    Sort multiple lists based on the values in the key list `a`.

    Args:
        a (list): The list to sort by.
        *others (list): Other lists to reorder accordingly.
        reverse (bool): Whether to sort in descending order.

    Returns:
        Tuple of sorted lists: (a_sorted, *others_sorted)
    """
    # Combine all lists into a list of tuples
    zipped = zip(a, *others)

    # Sort based on the first element of each tuple (values in `a`)
    sorted_zipped = sorted(zipped, reverse=reverse)

    # Unzip the sorted result into individual sorted lists
    return tuple(map(list, zip(*sorted_zipped)))



    