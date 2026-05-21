import numpy as np
import concurrent.futures
import scipy
import pandas as pd
from scipy import signal as sig
from scipy.signal import welch
from scipy.integrate import simpson
from scipy.signal import hilbert
from scipy.ndimage import uniform_filter1d, binary_opening, binary_closing


# ————————general————————

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


# ————————adratio————————

def ad_ratio_single(data, fs):
    data = data.values
    delta_power = np.nanmean(bandpower(data, fs, [1,4]))
    alpha_power = np.nanmean(bandpower(data, fs, [8,13]))
    return alpha_power/delta_power


# ————————spike———————————


def eeg_filter(signal, fc, filttype, fs):
    """
    Filters an EEG signal using a Butterworth filter.

    Parameters:
    - signal (array-like): The EEG signal to be filtered. This should be a one-dimensional array of values.
    - fc (float): The cutoff frequency for the filter in Hz. Must be less than half the sampling rate (fs/2).
    - filttype (str): The type of the filter. Acceptable values are 'low', 'high', 'bandpass', and 'bandstop'.
    - fs (float): The sampling rate of the EEG signal in Hz.

    Returns:
    - array-like: The filtered EEG signal.

    Raises:
    - AssertionError: If the cutoff frequency is greater than or equal to half the sampling rate.

    Note:
    - This function uses the Butterworth filter from the `scipy.signal` module.
    """
    assert fc < fs / 2, "Cutoff frequency must be < one half the sampling rate"
    # Initialize constants for filter
    order = 6
    # Create and apply filter
    B, A = sig.butter(order, fc, filttype, fs=fs)
    return sig.filtfilt(B, A, signal)


def find_peaks(signal):
    """
    Finds the peaks and troughs in a given signal.

    This function identifies the peaks and troughs of a signal by computing its first derivative
    and then determining where the derivative changes sign.

    Parameters:
    - signal (array-like): A one-dimensional array of numeric values representing the signal.

    Returns:
    - tuple: A tuple containing two arrays:
        1. An array of indices indicating the locations of the troughs.
        2. An array of indices indicating the locations of the peaks.

    Notes:
    - Peaks are regions where the signal increases and then decreases, while troughs are regions where the signal decreases and then increases.
    - This function may not capture all peaks or troughs if the signal has very small fluctuations or noise.

    Examples:
    >>> signal = [1, 3, 7, 6, 4, 5, 8, 6]
    >>> find_peaks(signal)
    (array([2]), array([6]))
    """
    ds = np.diff(signal, axis=0)
    ds = np.insert(ds, 0, ds[0])  # pad diff
    mask = np.argwhere(np.abs(ds[1:]) <= 1e-3).squeeze()  # got rid of +1
    ds[mask] = ds[mask - 1]
    ds = np.sign(ds)
    ds = np.diff(ds)
    ds = np.insert(ds, 0, ds[0])
    t = np.argwhere(ds > 0)
    p = np.argwhere(ds < 0)
    return p, t


def multi_channel_requirement(gdf, nchs, fs):
    # Need to change so that it returns total spike counter to help remove duplicates
    min_chs = 2
    if nchs < 16:
        max_chs = np.inf
    else:
        max_chs = np.ceil(nchs / 2)
    min_time = 100 * 1e-3 * fs

    # Check if there is even more than 1 spiking channel. Will throw error
    try:
        if len(np.unique(gdf[:, 1])) < min_chs:
            return np.array([])
    except IndexError:
        return np.array([])

    final_spikes = []

    s = 0  # start at time negative one for 0 based indexing
    curr_seq = [s]
    last_time = gdf[s, 0]
    spike_count = 0
    while s < (gdf.shape[0] - 1):  # check to see if we are at last spike
        # move to next spike time
        new_time = gdf[s + 1, 0]  # calculate the next spike time

        # if it's within the time diff
        if (
            new_time - last_time
        ) < min_time:  # check that the spikes are within the window of time
            curr_seq.append(s + 1)  # append it to the current sequence

            if s == (
                gdf.shape[0] - 2
            ):  # see if you just added the last spike, if so, done with sequence
                # done with sequence, check if the number of involved chs is
                # appropriate
                l = len(np.unique(gdf[curr_seq, 1]))
                if l >= min_chs and l <= max_chs:
                    final_spikes.append(
                        np.hstack(
                            (
                                gdf[curr_seq, :],
                                np.ones((len(curr_seq), 1)) * spike_count,
                            )
                        )
                    )
        else:
            # done with sequence, check if the length of sequence is
            # appropriate
            l = len(np.unique(gdf[curr_seq, 1]))
            if (l >= min_chs) & (l <= max_chs):
                final_spikes.append(
                    np.hstack(
                        (gdf[curr_seq, :], np.ones((len(curr_seq), 1)) * spike_count)
                    )
                )
                spike_count += 1
            # reset sequence
            curr_seq = [s + 1]
        # increase the last time
        last_time = gdf[s + 1, 0]

        # increase the current spike
        s += 1

    if len(final_spikes) != 0:
        return np.vstack(
            final_spikes
        )  # return all final spikes with a spike sequence counter
    else:
        return np.array([])


def multi_channel_requirement_with_label(gdf, nchs, fs):
    # Need to change so that it returns total spike counter to help remove duplicates
    min_chs = 2
    if nchs < 16:
        max_chs = np.inf
    else:
        max_chs = np.ceil(nchs / 2)
    min_time = 100 * 1e-3 * fs

    # Check if there is even more than 1 spiking channel. Will throw error
    try:
        if len(np.unique(gdf[:, 1])) < min_chs:  # check unique channels
            return np.array([])
    except IndexError:
        return np.array([])

    final_spikes = []

    s = 0  # start at time negative one for 0 based indexing
    curr_seq = [s]
    last_time = gdf[s, 0]
    spike_count = 0
    while s < (gdf.shape[0] - 1):  # check to see if we are at last spike
        # move to next spike time
        new_time = gdf[s + 1, 0]  # calculate the next spike time

        # if it's within the time diff
        if (
            new_time - last_time
        ) < min_time:  # check that the spikes are within the window of time
            curr_seq.append(s + 1)  # append it to the current sequence

            if s == (
                gdf.shape[0] - 2
            ):  # see if you just added the last spike, if so, done with sequence
                # done with sequence, check if the number of involved chs is
                # appropriate
                l = len(np.unique(gdf[curr_seq, 1]))
                if l >= min_chs and l <= max_chs:
                    final_spikes.append(
                        np.hstack(
                            (
                                gdf[curr_seq, :2],
                                np.ones((len(curr_seq), 1)) * spike_count,
                                gdf[curr_seq, 2:],
                            )
                        )
                    )
        else:
            # done with sequence, check if the length of sequence is
            # appropriate
            l = len(np.unique(gdf[curr_seq, 1]))
            if (l >= min_chs) & (l <= max_chs):
                final_spikes.append(
                    np.hstack(
                        (
                            gdf[curr_seq, :2],
                            np.ones((len(curr_seq), 1)) * spike_count,
                            gdf[curr_seq, 2:],
                        )
                    )
                )
                spike_count += 1
            # reset sequence
            curr_seq = [s + 1]
        # increase the last time
        last_time = gdf[s + 1, 0]

        # increase the current spike
        s += 1

    if len(final_spikes) != 0:
        return np.vstack(
            final_spikes
        )  # return all final spikes with a spike sequence counter
    else:
        return np.array([])


def process_channel(
    signal, fs, tmul, absthresh, sur_time, too_high_abs, spkdur, lpf1, hpf
):
    out = []  # initialize preliminary spike receiver

    if np.any(np.isnan(signal)):
        return None  # if there are any nans in the signal skip the channel (worth investigating)

    # Re-adjust the mean of the signal to be zero
    signal = signal - np.mean(signal)

    # receiver initialization
    spike_times = []
    spike_durs = []
    spike_amps = []

    # low pass filter to remove artifact
    lpsignal = eeg_filter(signal, lpf1, "lowpass", fs)

    # high pass filter for the 'spike' component
    hpsignal = eeg_filter(lpsignal, hpf, "highpass", fs)

    # defining thresholds
    lthresh = np.median(np.abs(hpsignal))
    thresh = lthresh * tmul  # this is the final threshold we want to impose

    signals = [hpsignal, -hpsignal]

    for ksignal in signals:
        # apply custom peak finder
        spp, spv = find_peaks(ksignal)  # calculate peaks and troughs
        spp, spv = spp.squeeze(), spv.squeeze()  # reformat

        # find the durations less than or equal to that of a spike
        idx = np.argwhere(np.diff(spp) <= spkdur[1]).squeeze()
        startdx = spp[idx]  # indices for each spike that has a long enough duration
        startdx1 = spp[idx + 1]  # indices for each "next" spike

        # Loop over peaks
        for i in range(len(startdx)):
            spkmintic = spv[(spv > startdx[i]) & (spv < startdx1[i])]
            # find the valley that is between the two peaks
            if not any(spkmintic):
                continue
            max_height = max(
                np.abs(ksignal[startdx1[i]] - ksignal[spkmintic]),
                np.abs(ksignal[startdx[i]] - ksignal[spkmintic]),
            )[0]
            if max_height > thresh:  # see if the peaks are big enough
                spike_times.append(int(spkmintic))  # add index to the spike list
                spike_durs.append(
                    (startdx1[i] - startdx[i]) * 1000 / fs
                )  # add spike duration to list
                spike_amps.append(max_height)  # add spike amplitude to list

    # Generate spikes matrix
    spikes = np.vstack([spike_times, spike_durs, spike_amps]).T

    # initialize exclusion receivers
    toosmall = []
    toosharp = []
    toobig = []

    # now have all the info we need to decide if this thing is a spike or not.
    for i in range(spikes.shape[0]):  # for each spike
        # re-define baseline to be 2 seconds surrounding
        surround = sur_time
        istart = int(
            max(0, np.around(spikes[i, 0] - surround * fs))
        )  # find -2s index, ensuring not to exceed idx bounds
        iend = int(
            min(len(hpsignal), np.around(spikes[i, 0] + surround * fs + 1))
        )  # find +2s index, ensuring not to exceed idx bounds

        alt_thresh = (
            np.median(np.abs(hpsignal[istart:iend])) * tmul
        )  # identify threshold within this window

        if (spikes[i, 2] > alt_thresh) & (
            spikes[i, 2] > absthresh
        ):  # both parts together are bigger than thresh: so have some flexibility in relative sizes
            if (
                spikes[i, 1] > spkdur[0]
            ):  # spike wave cannot be too sharp: then it is either too small or noise
                if spikes[i, 2] < too_high_abs:
                    out.append(spikes[i, 0])  # add timestamp of spike to output list
                else:
                    toobig.append(spikes[i, 0])  # spike is above too_high_abs
            else:
                toosharp.append(spikes[i, 0])  # spike duration is too short
        else:
            toosmall.append(spikes[i, 0])  # window-relative spike height is too short

    # Spike Realignment
    if out:
        timeToPeak = np.array([-0.15, 0.15])
        fullSurround = np.array([-sur_time, sur_time]) * fs
        idxToPeak = timeToPeak * fs

        hpsignal_length = len(hpsignal)

        for i in range(len(out)):
            currIdx = out[i]
            surround_idx = np.arange(
                max(0, round(currIdx + fullSurround[0])),
                min(round(currIdx + fullSurround[1]), hpsignal_length),
            )
            idxToLook = np.arange(
                max(0, round(currIdx + idxToPeak[0])),
                min(round(currIdx + idxToPeak[1]), hpsignal_length),
            )
            snapshot = hpsignal[idxToLook] - np.median(hpsignal[surround_idx])
            I = np.argmax(np.abs(snapshot))
            out[i] = idxToLook[0] + I

    return np.array(out)


def spike_detector(data, fs, **kwargs):
    """
    Parameters
    data:           np.NDArray - iEEG recordings (m samples x n channels)
    fs:             int - sampling frequency

    kwargs
    tmul:           float - 19 - threshold multiplier
    absthresh:      float - 100 - absolute threshold for spikes
    sur_time:       float - 0.5 - time surrounding a spike to analyze in seconds
    close_to_edge:  float - 0.05 - beginning and ending buffer in seconds
    too_high_abs:   float - threshold for artifact rejection
    spkdur:         Iterable - min and max spike duration thresholds in ms (min, max)
    lpf1:           float - low pass filter cutoff frequency
    hpf:            float - high pass filter cutoff frequency

    Returns
    gdf:            np.NDArray - spike locations (m spikes x (peak index, channel))
    """

    ### Assigning KWARGS using a more efficient method ###############
    tmul = kwargs.get("tmul", 19)  # 25
    absthresh = kwargs.get("absthresh", 100)
    sur_time = kwargs.get("sur_time", 0.5)
    close_to_edge = kwargs.get("close_to_edge", 0.05)
    too_high_abs = kwargs.get("too_high_abs", 1e3)
    # spike duration must be less than this in ms. It gets converted to samples here
    spkdur = kwargs.get("spkdur", np.array([15, 260]))
    lpf1 = kwargs.get("lpf1", 30)  # low pass filter for spikey component
    hpf = kwargs.get("hpf", 7)  # high pass filter for spikey component
    ###################################

    # Assertions and assignments
    if not isinstance(data, np.ndarray):
        data = data.to_numpy()
    if not isinstance(spkdur, np.ndarray):
        spkdur = np.array(spkdur)

    # Receiver and constant variable initialization
    # all_spikes = np.ndarray((1,2),dtype=float)
    all_spikes = []
    nchs = data.shape[1]
    spkdur = (spkdur / 1000) * fs  # From milliseconds to samples

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures_to_channel = {
            executor.submit(
                process_channel,
                data[:, j],  # collect channel
                fs,
                tmul,
                absthresh,
                sur_time,
                too_high_abs,
                spkdur,
                lpf1,
                hpf,
            ): j
            for j in range(nchs)
        }

        for future in concurrent.futures.as_completed(futures_to_channel):
            j = futures_to_channel[future]
            try:
                channel_out = future.result()
                # Concatenate the list of spikes to the global spike receiver
                if channel_out is not None and channel_out.any():
                    temp = (
                        np.array(
                            [
                                np.expand_dims(channel_out, 1),
                                np.tile([j], (len(channel_out), 1)),
                            ],
                            dtype=float,
                        )
                        .squeeze()
                        .T
                    )
                    # all_spikes = np.vstack((all_spikes,temp))
                    all_spikes.append(temp)
                    # change all spikes to a list, append and then vstack all at end
            except Exception as exc:
                print(f"Channel {j} generated an exception: {exc}")

    # Final Post-Processing - sort spikes by time not np.isnan(all_spikes).all():
    if len(all_spikes) == 0:
        return np.array([])
    else:
        all_spikes = np.vstack(all_spikes)
        all_spikes = np.vstack(list({tuple(row) for row in list(all_spikes)}))
        idx = np.argsort(all_spikes[:, 0], axis=0)
        gdf = all_spikes[idx, :]

        # if there are no spikes, just give up
        # print('No spikes detected')

    # if all_spikes.any():
    #     # remove exact copies of spikes
    #     all_spikes = np.vstack(list({tuple(row) for row in list(all_spikes)}))

    #     # sort spikes
    #     idx = np.argsort(all_spikes[:,0],axis=0)
    #     gdf = all_spikes[idx,:]

    ## Remove those too close to beginning and end
    if gdf.shape[0]:
        close_idx = close_to_edge * fs
        gdf = gdf[gdf[:, 0] > close_idx, :]
        gdf = gdf[gdf[:, 0] < data.shape[0] - close_idx, :]

    # removing duplicate spikes with closeness threshold
    if gdf.any():
        # distance between spike times
        gdf_diff = np.diff(gdf, axis=0)
        # time difference is below threshold - thresh not necessary becasue of spike realignment
        mask1 = np.abs(gdf_diff[:, 0]) < 100e-3 * fs
        # ensuring they're on different channels
        mask2 = gdf_diff[:, 1] == 0
        too_close = np.argwhere(mask1 & mask2) + 1

        # coerce shape of mask to <=2 dimensions
        too_close = too_close.squeeze()
        close_mask = np.ones((gdf.shape[0],), dtype=bool)
        close_mask[too_close] = False
        gdf = gdf[close_mask, :]

    # Check that spike occurs in multiple channels
    if gdf.any() & (nchs > 1):
        gdf = multi_channel_requirement(gdf, nchs, fs)

    return gdf


def spike_ieeg_single(data, fs, bad_mask):
    """The function takes a data clip of 60 seconds, do signal preprocessing, and return spike rate and location
    Data in the format of pandas dataframe, with columns being channels.

    Args:
        data (_type_): data should be convert to volts before usage
        fs (int): sampling frequency of data
    """
    # Check if data dataframe is all NaNs
    if data.isnull().values.all():
        print(f"Empty dataframe after download, skip...")
        return np.array([]),np.nan

    chs = data.columns[bad_mask]
    ieeg_data = data.to_numpy()
    ieeg_data = ieeg_data[:,bad_mask]

    ##############################
    # Detect spikes
    ##############################

    spike_output = spike_detector(
        data=ieeg_data,
        fs=fs,
        electrode_labels=chs,
    )
    spike_output = spike_output.astype(int)
    actual_number_of_spikes = len(spike_output)

    return spike_output, actual_number_of_spikes



    # ————————————————synchrony————————————————

def calculate_synchrony(time_series):
    """
    Calculate the Kuramoto order parameter for a set of time series
    Args:
        time_series (np.array): 2D array where each row is a time series
    Returns:
        np.array: Kuramoto order parameter for each time point
    """
    # Extract the number of time series and the number of time points
    N, _ = time_series.shape
    # Apply the Hilbert Transform to get an analytical signal
    analytical_signals = hilbert(time_series)
    assert analytical_signals.shape == time_series.shape
    # Extract the instantaneous phase for each time series using np.angle
    phases = np.angle(analytical_signals, deg=False)
    assert phases.shape == time_series.shape
    # Compute the Kuramoto order parameter for each time point
    # 1j*1j == -1
    r_t = np.abs(np.sum(np.exp(1j * phases), axis=0)) / N
    R = np.mean(r_t)
    return r_t, R

def synchrony_single(data, bad_mask):
    """The function takes a data clip of 2 minutes, do signal preprocessing, and return synchrony

    Args:
        data (_type_): data should be convert to volts before usage
        fs (int): sampling frequency of data
    """
    # Check if data dataframe is all NaNs
    if data.isnull().values.all():
        print(f"Empty dataframe after download, skip...")
        return np.nan

    # chs = data.columns[bad_mask]
    ieeg_data = data.to_numpy()
    ieeg_data = ieeg_data[:,bad_mask]
    _, R = calculate_synchrony(ieeg_data.T)

    return R


# ————————————sparcnet————————————

def extract_seiz_ranges(true_data):
    # the extracted start and stop can should be used like start:stop to extract data
    if len(true_data) < 1:
        return []
    diff_data = np.diff(np.concatenate([[0],true_data.flatten(),[0]]))
    starts = np.where(diff_data == 1)[0].tolist()
    stops = np.where(diff_data == -1)[0].tolist()
    return list(zip(starts,stops))

def nan_aware_uniform_filter1d(arr, size):
    arr = np.asarray(arr, dtype=float)
    half = size // 2
    padded = np.pad(arr, pad_width=half, mode='edge')  # nearest-like padding
    result = np.full_like(arr, np.nan)

    for i in range(len(arr)):
        window = padded[i:i+size]
        if np.isfinite(window).any():
            result[i] = np.nanmean(window)  # mean ignoring NaNs
    return result


def smooth_pred(pred):
    pred = np.array(pred)
    if len(pred) == 1:
        return pred.astype(int)
    smoothed = binary_opening(pred, structure=np.ones(2))  # remove short 1s
    smoothed = binary_closing(smoothed, structure=np.ones(2))  # remove short 0s
    smoothed[0] = smoothed[1]
    smoothed[-1] = smoothed[-2]

    smoothed = binary_opening(pred, structure=np.ones(3))  # remove short 1s
    smoothed = binary_closing(smoothed, structure=np.ones(3))  # remove short 0s
    smoothed[0] = smoothed[1]
    smoothed[-1] = smoothed[-2]
    return smoothed.astype(int)

def get_events(pred, gap_num = 2, min_event_num = 10, return_pred = False):
    smoothed_pred = smooth_pred(pred)
    new_pred = np.zeros_like(smoothed_pred)
    sz_events = np.array(extract_seiz_ranges(smoothed_pred))

    if sz_events.shape[0] == 0:
        if return_pred:
            return np.array([]), new_pred
        else:
            return np.array([])

    # Merge events that are close together
    start_times = sz_events[:, 0]
    end_times = sz_events[:, 1]

    merged_start = [start_times[0]]
    merged_end = [end_times[0]]

    for i in range(1, len(start_times)):
        if start_times[i] - merged_end[-1] <= gap_num:
            # Merge the events
            merged_end[-1] = end_times[i]
        else:
            # Start a new event
            merged_start.append(start_times[i])
            merged_end.append(end_times[i])

    sz_events = np.array(list(zip(merged_start, merged_end)))

    # Filter short events
    durations = sz_events[:, 1] - sz_events[:, 0]
    sz_events = sz_events[durations >= min_event_num]

    if sz_events.shape[0] == 0:
        if return_pred:
            return np.array([]), new_pred
        else:
            return np.array([])
    
    # Merge events that are close together
    start_times = sz_events[:, 0]
    end_times = sz_events[:, 1]

    merged_start = [start_times[0]]
    merged_end = [end_times[0]]

    for i in range(1, len(start_times)):
        if start_times[i] - merged_end[-1] <= 4:
            # Merge the events
            merged_end[-1] = end_times[i]
        else:
            # Start a new event
            merged_start.append(start_times[i])
            merged_end.append(end_times[i])

    sz_events = np.array(list(zip(merged_start, merged_end)))

    for start, end in sz_events:
        new_pred[start:end] = 1
    if return_pred:
        return sz_events, new_pred
    else:
        return sz_events
    

def smooth_pred_and_extend(prob, thres_high = 0.5, thres_low = 0.5, gap_num = 2, min_event_num = 10, max_event_num = 200, auto_adjust = True, return_thres = False, return_pred = False):
    pred = (prob >= thres_high).astype(int)
    smoothed_pred = smooth_pred(pred)
    new_pred = np.zeros_like(smoothed_pred)

    sz_events = get_events(smoothed_pred, min_event_num = min_event_num, gap_num=gap_num)

    if sz_events.shape[0] == 0:
        if return_pred and return_thres:
            return sz_events, new_pred, thres_high, thres_low
        elif return_pred:
            return sz_events, new_pred
        elif return_thres:
            return sz_events, thres_high, thres_low
        else:
            return sz_events
    
    # if the patient's probs are significantly higher, update threshold
    sz_dura = np.array([sz[1]-sz[0] for sz in sz_events])
    if auto_adjust:
        while np.any(sz_dura > max_event_num):
            if thres_high == 0.9:
                break
            else:
                thres_high += 0.1
                thres_high = min(thres_high, 0.9)
            pred = (prob >= thres_high).astype(int)
            smoothed_pred = smooth_pred(pred)
            sz_events = get_events(smoothed_pred, min_event_num = min_event_num, gap_num=gap_num)
            sz_dura = np.array([sz[1]-sz[0] for sz in sz_events])

    # now extend from the event cores
    extended_start = []
    extended_end = []
    for sz in sz_events:
        done = False
        while not done:
            new_start, new_end = sz[0], sz[1]

            # --- Extend backward ---
            below_count = 0
            for i in range(sz[0]-1, -1, -1):
                if prob[i] < thres_low:
                    below_count += 1
                else:
                    below_count = 0
                if below_count >= gap_num+1:
                    break
            try:
                new_start = i+gap_num+1
            except:
                pass

            # --- Extend forward ---
            below_count = 0
            for i in range(sz[1], len(prob)):
                if prob[i] < thres_low:
                    below_count += 1
                else:
                    below_count = 0
                if below_count >= gap_num+1:
                    break
            try:
                new_end = i-gap_num
            except:
                pass
            

            # --- Check range length ---
            if auto_adjust:
                curr_len = new_end - new_start + 1
                if curr_len > max_event_num and thres_low < thres_high:
                    thres_low = min(thres_high, thres_low + 0.1)
                else:
                    done = True
                    extended_start.append(new_start)
                    extended_end.append(new_end)
            else:
                done = True
                extended_start.append(new_start)
                extended_end.append(new_end)
    # sort events again in case any events overlap
    merged_start = [extended_start[0]]
    merged_end = [extended_end[0]]

    for i in range(1, len(extended_start)):
        if extended_start[i] - merged_end[-1] <= gap_num:
            # Merge the events
            merged_end[-1] = extended_end[i]
        else:
            # Start a new event
            merged_start.append(extended_start[i])
            merged_end.append(extended_end[i])

    sz_events = np.array(list(zip(merged_start, merged_end)))

    # Filter short events
    durations = sz_events[:, 1] - sz_events[:, 0]
    sz_events = sz_events[durations >= min_event_num]
    for start, end in sz_events:
        new_pred[start:end] = 1
    if return_pred and return_thres:
        return sz_events, new_pred, thres_high, thres_low
    elif return_pred:
        return sz_events, new_pred
    elif return_thres:
        return sz_events, thres_high, thres_low
    else:
        return sz_events


def cusum_detect(probs, mul=1.2, h=3):
    # adaptive k based on data if not provided
    k = probs.mean() + mul*probs.std()
    
    S = np.zeros(len(probs))
    for i in range(1, len(probs)):
        S[i] = max(0, S[i-1] + probs[i] - k)
    
    # anywhere S exceeds h is flagged
    binary = S > h
    chunks = extract_seiz_ranges(binary)
    for c in chunks:
        binary[int(c[0]-h):c[0]] = 1
    
    return binary

def get_prediction(probs, window = 400, stride = 100, fs = 0.5, mul=1.2, h = 3):

    WINDOW = int(window*fs)
    STRIDE = int(stride*fs)
    n_win = int(np.ceil(len(probs)/STRIDE))
    all_pred = []
    for n in range(n_win):
        win_prob = probs[n*STRIDE:min(n*STRIDE+WINDOW,len(probs))]
        win_pred = cusum_detect(win_prob, mul=mul, h = h)
        if n == 0:
            all_pred.append(win_pred)
        else:
            all_pred.append(win_pred[WINDOW-STRIDE:])
    pred = np.concatenate(all_pred)
    _, pred = get_events(pred, gap_num = 1, min_event_num = 5, return_pred = True)
    return pred
    
def smooth_pred_filtering(smoothed_pred, prob, thres = 0.8):
    # # Extract seizure events
    sz_events = np.array(extract_seiz_ranges(smoothed_pred))
    pred_updated = np.zeros_like(smoothed_pred).astype(int)

    if sz_events.shape[0] == 0:
        return pred_updated

    # Filter by avg prob
    probs = np.array([np.nanmean(prob[e[0]:e[1]]) for e in sz_events])
    sz_events = sz_events[probs >= thres]

    # Update predictions
    for sz in sz_events:
        pred_updated[sz[0]:sz[1]] = 1

    return pred_updated

# def detect_high_prob_core(prob, thres = 0.8, min_event_num = 5):
#     pred = (prob >= thres).astype(int)
#     events =  np.array(extract_seiz_ranges(pred))
#     if events.shape[0] > 0:
#         durations = events[:, 1] - events[:, 0]
#         events = events[durations >= min_event_num]
#     return events

# def smooth_pred_new(pred, prob):
#     smoothing_num = 5 #int(smoothing_win / info['stride'])
#     min_event_num = 10 #int(min_event_len / info['stride'])
#     gap_num = 2 #int(gap_len / info['stride'])

#     # Smooth predictions using a uniform filter
#     smoothed = nan_aware_uniform_filter1d(pred.astype(float), smoothing_num + 1)
#     pred_new = (smoothed >= 0.6).astype(int)

    
#     # # Extract seizure events
#     sz_events = np.array(extract_seiz_ranges(pred_new))
#     pred_updated = np.zeros_like(pred_new).astype(int)

#     if sz_events.shape[0] == 0:
#         return pred_updated

#     # Merge events that are close together
#     start_times = sz_events[:, 0]
#     end_times = sz_events[:, 1]

#     merged_start = [start_times[0]]
#     merged_end = [end_times[0]]

#     for i in range(1, len(start_times)):
#         if start_times[i] - merged_end[-1] <= gap_num:
#             # Merge the events
#             merged_end[-1] = end_times[i]
#         else:
#             # Start a new event
#             merged_start.append(start_times[i])
#             merged_end.append(end_times[i])

#     sz_events = np.array(list(zip(merged_start, merged_end)))

#     # Filter short events
#     durations = sz_events[:, 1] - sz_events[:, 0]
#     sz_events = sz_events[durations >= min_event_num]

#     # Filter by avg prob
#     pred = np.array([np.any(detect_high_prob_core(prob[e[0]:e[1]])) for e in sz_events])
#     sz_events = sz_events[pred]

#     # Update predictions
#     for sz in sz_events:
#         pred_updated[sz[0]:sz[1]] = 1

#     return pred_updated


# ————————————wvnt————————————————

def get_pred(sz_prob, threshold, filter_w=5, rwin_size=5, rwin_req=4, num_chan = 1):
    feat_settings = {'win':1, 'stride':1}
    filter_w_idx = int(np.floor((filter_w - feat_settings['win']) / feat_settings['stride'])) + 1
    rwin_size_idx = int(np.floor((rwin_size - feat_settings['win']) / feat_settings['stride'])) + 1
    rwin_req_idx = int(np.floor((rwin_req - feat_settings['win']) / feat_settings['stride'])) + 1

    sz_clf = sz_prob > threshold
    sz_clf = scipy.ndimage.median_filter(sz_clf, size=filter_w_idx, mode='nearest',axes=0,origin=0)
    
    rolling_sums = np.round(np.abs(scipy.signal.fftconvolve(sz_clf, np.ones([rwin_size_idx,1]), mode='full',axes=0)))
    rolling_sums[:rwin_size_idx-1,:] = rolling_sums[rwin_size_idx-1,:]
    rolling_sums = rolling_sums[:sz_clf.shape[0],:]
    sz_spread_idxs_all = rolling_sums >= rwin_req_idx
    
    pred = (np.sum(sz_spread_idxs_all, axis=1) >= num_chan).astype(int)

    return pred, sz_spread_idxs_all
    
def ieeg_get_pred(prob, threshold=0.99, perc_chan_thres=0.1, gap_num=2, min_event_num=10):
    _, smoothed_pred = get_pred(prob, threshold)
    perc_channel = np.sum(smoothed_pred, axis=1)/smoothed_pred.shape[1]
    pred = smooth_pred((perc_channel > perc_chan_thres).astype(int))
    _, pred = get_events(pred, gap_num = gap_num, min_event_num = min_event_num, return_pred = True)#prob >= ctx.use_thresprev_pred_list.append(pred)
    return pred

def ieeg_get_prediction(prob_mat, window = 400, stride = 100, fs = 1, mul = 0.8, h = 3):
    # print(prob.shape)
    # prob_vals = np.percentile(prob, perc,axis=1).values
    pred_mat = np.zeros_like(prob_mat)
    for ch in range(pred_mat.shape[1]):
        prob_chan = prob_mat[:,ch]
        pred_chan = get_prediction(prob_chan, window=window, stride=stride, fs=fs, mul=mul, h=h)
        pred_mat[:,ch] = pred_chan
    perc_chan = pred_mat.sum(axis=1)/pred_mat.shape[1]
    pred = perc_chan >= 0.1
    _, pred = get_events(pred, gap_num = 2, min_event_num = 10, return_pred = True)
    return pred, perc_chan