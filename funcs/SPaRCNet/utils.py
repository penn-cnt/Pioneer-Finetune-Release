# General Imports, Functions, Paths
# imports
import numpy as np
import pandas as pd
import scipy
import mne


from kneed import KneeLocator

mne.set_log_level('ERROR')


def get_threshold(sz_prob):
    probabilities = sz_prob.flatten()
    probabilities = probabilities[probabilities <= np.percentile(probabilities,99.99)]
    probabilities = probabilities[probabilities <= 1000]
    # probabilities = np.log(probabilities)
    # probabilities = probabilities[~np.isnan(probabilities)]
    # probabilities = probabilities[~np.isinf(probabilities)]
    num_thresh = 3000
    thresh_sweep = np.linspace(min(probabilities),max(probabilities),num_thresh)
    peak_buff = int(0.005 * num_thresh)
    kde_model = scipy.stats.gaussian_kde(probabilities,'scott')
    kde_vals = kde_model(thresh_sweep)

    # Find KDE peaks
    kde_peaks,_ = scipy.signal.find_peaks(kde_vals)
    try:
        biggest_pk_idx = np.where(kde_vals[kde_peaks]>(np.mean(kde_vals)+np.std(kde_vals)))[0][-1]
    except:
        biggest_pk_idx = np.argmax(kde_vals[kde_peaks])
    if biggest_pk_idx == len(kde_peaks)-1:
        biggest_pk_idx = 0

    # Identify optimal threshold as knee between peaks
    if (len(kde_peaks) == 1) or (biggest_pk_idx == (len(kde_peaks)-1)):
        start, end = kde_peaks[biggest_pk_idx], len(kde_vals)-1
    else:
        start, end = kde_peaks[biggest_pk_idx], kde_peaks[biggest_pk_idx+1]

    try:
        kneedle = KneeLocator(thresh_sweep[start+peak_buff:end],kde_vals[start+peak_buff:end],
                curve='convex',direction='decreasing',interp_method='polynomial',S=0)
    except:
        kneedle = KneeLocator(thresh_sweep[start:end],kde_vals[start:end],
                curve='convex',direction='decreasing',interp_method='polynomial',S=0)
    threshold = kneedle.knee
    return threshold

def get_threshold_sparcnet(sz_prob):
    """Generate threshold for sparcnet model

    Args:
        sz_prob (_type_): _description_

    Returns:
        threshold: a single value if data is shorter than 5400 (3 hours), but an array of same size as sz_prob if longer than 3 hours
        The threshold for the first 3 hours would be same, and then changed from hour to hour?
    """
    try:
        probabilities = sz_prob.flatten()
        probabilities = probabilities[~np.isnan(probabilities)]
        num_thresh = 3000
        thresh_sweep = np.linspace(min(probabilities),max(probabilities),num_thresh)
        kde_model = scipy.stats.gaussian_kde(probabilities,'scott')
        kde_vals = kde_model(thresh_sweep)

        dx1 = np.concatenate([[np.nan],np.diff(thresh_sweep)])
        df1 = np.concatenate([[np.nan],np.diff(kde_vals)])
        df2 = np.concatenate([[np.nan],np.diff(df1)*100])
        curv = np.concatenate([[np.nan],np.abs(df1[1:] / dx1[1:] - df1[:-1] / dx1[:-1])])

        curv_peaks,_ = scipy.signal.find_peaks(curv)
        curv_peaks = curv_peaks[df2[curv_peaks] < 0]
        curv_peak_amp = curv[curv_peaks]
        curv_peaks = curv_peaks[curv_peak_amp >= 0.1]
        curv_peaks = thresh_sweep[curv_peaks]
        
        sub_peaks = curv_peaks[(curv_peaks >= 0.3) & (curv_peaks <= 0.6)]
        if sub_peaks.size <= 0:
            kde_peaks,_ = scipy.signal.find_peaks(kde_vals)
            kde_heights = kde_vals[kde_peaks]
            biggest_peak = kde_peaks[np.argmax(kde_heights)]
            if thresh_sweep[biggest_peak] <= 0.6:
                start, end = biggest_peak, len(kde_vals)-1
                try:
                    kneedle = KneeLocator(thresh_sweep[start+200:end],kde_vals[start+200:end],
                            curve='convex',direction='decreasing',interp_method='polynomial',S=0)
                except:
                    kneedle = KneeLocator(thresh_sweep[start:end],kde_vals[start:end],
                            curve='convex',direction='decreasing',interp_method='polynomial',S=0)
            else:
                start, end = 0, biggest_peak
                try:
                    kneedle = KneeLocator(thresh_sweep[start:end-200],kde_vals[start:end-200],
                            curve='convex',direction='increasing',interp_method='polynomial',S=0)
                except:
                    kneedle = KneeLocator(thresh_sweep[start:end],kde_vals[start:end],
                            curve='convex',direction='increasing',interp_method='polynomial',S=0)
            thres = kneedle.knee
        elif len(sub_peaks) > 1:
            thres = sub_peaks[0]#?
        else:
            thres = sub_peaks[0]
    except:
        thres = get_threshold(probabilities)
    return thres


from scipy.ndimage import uniform_filter1d

def extract_seiz_ranges(true_data):
    diff_data = np.diff(np.concatenate([[0],np.squeeze(true_data),[0]]))
    starts = np.where(diff_data == 1)[0]
    stops = np.where(diff_data == -1)[0]
    return list(zip(starts,stops))


def smooth_pred(pred, min_event_len = 5, gap_len = 2, smoothing_win = 5):
    info = {'win':10, 'stride':2}
    smoothing_num = int(smoothing_win / 2 / info['stride'])
    min_event_num = int(min_event_len / info['stride'])
    gap_num = int(gap_len / info['stride'])

    # Smooth predictions using a uniform filter
    smoothed = uniform_filter1d(pred.astype(float), size=2 * smoothing_num + 1, mode="nearest")
    pred[:] = (smoothed >= 0.6).astype(int)

    # Extract seizure events
    sz_events = np.array(extract_seiz_ranges(pred))
    pred_updated = np.zeros_like(pred)

    if sz_events.shape[0] == 0:
        return pred_updated

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

    # Update predictions
    for sz in sz_events:
        pred_updated[sz[0]:sz[1]] = 1

    return pred_updated