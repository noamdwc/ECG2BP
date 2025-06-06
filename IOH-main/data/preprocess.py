import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def ioh_detect(map_signal, time_interval=None, MBP_key="ART_MBP", ioh_min_duration=1):
    """
    Detects all sequences where MBP < 65 and valid
    :param map_signal: the signal to detect the sequences as a pandas dataframe containing the columns "Time" and MBP_key
    :param time_interval: manually insert the time interval, if None, the function will compute the median time interval
    :param MBP_key: the key for the MBP signal
    :return: a list of dictionaries containing the segment length, start time and end time of the sequences
    """
    # 1. Clean all rows with NaNs
    map_signal = map_signal.dropna()

    # 2. Clean all rows where map_signal[MBP_key] <= 0
    map_signal = map_signal[map_signal[MBP_key] > 0]

    # 3. Sort the dataframe by "Time"
    map_signal = map_signal.sort_values('Time').reset_index(drop=True)

    # 4. Compute the time interval
    if time_interval is None:
        time_interval = map_signal['Time'].diff().median()

    # 5. Identify sequences where map_signal[MBP_key] < 65
    below_threshold = map_signal[MBP_key] < 65

    sequences = []
    current_sequence_length = 0
    start_time = None
    in_sequence = False
    for idx, val in below_threshold.items():
        # print(map_signal.loc[idx, 'Time'])
        if val:
            if not in_sequence:
                start_time = map_signal.loc[idx, 'Time']
                in_sequence = True
        else:
            # in case the sequence is broken
            if in_sequence:
                end_time = map_signal.loc[idx - 1, 'Time']
                segment_len = end_time - start_time
                if segment_len >= ioh_min_duration:
                    sequences.append({
                        'segment_len': segment_len,
                        'Time_start': start_time,
                        'Time_end': end_time
                    })
                # reset the sequence
                in_sequence = False

    return sequences, time_interval


def ioh_negative(signal, ioh_events, time_before_sec=300, time_after_sec=None):
    """
    This method cleans all ioh positive events from the signal
    :param raw_signal: dataframe containing the signal as a pandas dataframe with the columns "Time"
    :param ioh_events: list of ioh events as a list of dictionaries containing "segment_len", "Time_start", "Time_end"
    :param time_before_sec: amount of time before the ioh event to be considered as ioh negative
    :param time_after_sec: amount of time after the previous ioh event to be considered as ioh negative
    :return: list of negative segments in signal
    """
    # Remove NaNs from the signal
    signal_no_nans = signal.dropna().reset_index(drop=True)

    # plt.plot(signal['Time'], signal['ECG_II'])
    # plt.show()

    negative_segments = []
    current_start_idx = 0
    valid_start_time = signal_no_nans['Time'][0]
    # Iterate through IOH events and find negative segments
    for event in ioh_events:
        time_start = event['Time_start']
        time_end = event['Time_end']

        # Find segments in the signal before the IOH event with the margin time_before_sec
        valid_end_time = time_start - time_before_sec

        # Find the last valid index before the IOH event
        valid_indices = signal[(signal['Time'] > valid_start_time) & (signal['Time'] < valid_end_time)].index

        if not valid_indices.empty:
            segment = signal.iloc[current_start_idx:valid_indices[-1] + 1]
            if not segment.empty:
                negative_segments.append(segment)
                # plt.plot(segment['Time'], segment['ECG_II'])
                # plt.show()
        # the next valid start time is the end of the IOH event
        if time_after_sec is not None:
            valid_start_time = time_end + time_after_sec
        else:
            valid_start_time = time_end
        next_seg = signal[signal['Time'] > valid_start_time]
        # the next valid start index is the first index after the IOH event
        if not next_seg.empty:
            current_start_idx = next_seg.index[0]

    # Add the remaining segment if there is any
    valid_indices = signal[signal['Time'] > valid_start_time].index
    if not valid_indices.empty:
        segment = signal.iloc[current_start_idx:valid_indices[-1] + 1]
        if not segment.empty:
            negative_segments.append(segment)
            # plt.plot(segment['Time'], segment['ECG_II'])
            # plt.show()

    return negative_segments


def sample_ioh_negative(negative_signal, time_interval=None, duration_sec=30):
    """
    This split a negative segment of signal into dataframe of duration_sec.
    If there are leftovers in the negative segment, it will be discarded
    :param negative_signal: the negative signal as a pandas dataframe
    :param time_interval: the time interval of the signal
    :param duration_sec: the duration of the negative segment
    :return: a sampled negative segment from the negative signal
    """
    # drop rows of NaNs only for the column "ECG_II"
    negative_signal = negative_signal.dropna(subset=['ECG_II']).reset_index(drop=True)
    # Compute the time interval
    if time_interval is None:
        time_interval = negative_signal['Time'].diff().median()
    # Compute the number of samples considering the time interval and the duration
    n_samples = round((negative_signal.shape[0]*time_interval)/duration_sec)
    duration_seg_len = round(duration_sec/time_interval)
    # Sample the signal
    sampled_signal = []
    for i in range(n_samples):
        start_idx = i * duration_seg_len
        end_idx = (i + 1) * duration_seg_len
        if end_idx > negative_signal.shape[0]:
            break
        sampled_signal.append(negative_signal.iloc[start_idx:end_idx])
        # plt.plot(sampled_signal[-1]['ECG_II'])
        # plt.show()

    return sampled_signal


def sample_ioh_positive(signal, ioh_events, time_before_sec, duration_sec=30, time_interval=None):
    """
    This method samples the ioh positive events
    :param signal: the signal to sample the ioh positive events as a pandas dataframe containing the columns "Time"
    :param ioh_events: list of ioh events as a list of dictionaries containing "segment_len", "Time_start", "Time_end"
    :param ioh_events: list of ioh events as a list of dictionaries containing "segment_len", "Time_start", "Time_end"
    :param time_before_sec: amount of time before the ioh event to be considered as ioh positive
    :param duration_sec: the duration of the ioh positive sample
    :return: a list of ioh positive samples as pandas dataframes
    """
    # drop rows of NaNs only for the column "ECG_II"
    signal = signal.dropna(subset=['ECG_II']).reset_index(drop=True)

    pos_samples = []
    for event in ioh_events:
        time_start = event['Time_start']
        valid_start_time = time_start - time_before_sec
        valid_end_time = valid_start_time + duration_sec
        valid_indices = signal[(signal['Time'] > valid_start_time) & (signal['Time'] < valid_end_time)].index
        if not valid_indices.empty:
            segment = signal.iloc[valid_indices[0]:valid_indices[-1] + 1]
            # check if the segment is not empty
            if not segment.empty:
                pos_samples.append(segment)
                # # plot the first seconds of the segment
                # first_sec_of_segment = segment[segment['Time'] < valid_start_time + 1]
                # plt.plot(first_sec_of_segment['Time'], first_sec_of_segment['ECG_II'])
                # plt.show()
                # plt.plot(segment['Time'], segment['ECG_II'])
                # plt.show()
    return pos_samples


def split_to_samples(ecg_signal):
    pass
