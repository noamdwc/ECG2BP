import vitaldb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy import stats

def clean_cardiac_cycles(signal, sampling_rate=500, min_cycle_duration=0.6, max_cycle_duration=1.0):
    # Process ECG signal to find peaks
    ecg_signal = signal['ECG_II'].dropna().values
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
    
    # Extract P-peaks
    P_peaks = signals['ECG_P_Peaks'].values
    P_peaks_indices = np.where(P_peaks == 1)[0]
    
    # Calculate P-P intervals in seconds
    pp_intervals = np.diff(P_peaks_indices) / sampling_rate
    
    # Determine indices where P-P intervals are within the normal range
    normal_intervals = (pp_intervals >= min_cycle_duration) & (pp_intervals <= max_cycle_duration)
    
    # Segment data based on normal intervals
    segments = []
    start_idx = P_peaks_indices[0]
    for i in range(1, len(P_peaks_indices)):
        if not normal_intervals[i - 1]:
            if start_idx < P_peaks_indices[i-1]:
                if signal['Time'].iloc[P_peaks_indices[i-1]+1]-signal['Time'].iloc[start_idx]>=30:
                     segments.append(signal.iloc[start_idx:P_peaks_indices[i-1] + 1])
            start_idx = P_peaks_indices[i]
    if normal_intervals[-1]:
        segments.append(signal.iloc[start_idx:])
    
    return segments

def clean_outliers(segments):
    sub_segments = []
    for segment in segments:
        current_sub_segment = []
        for index, row in segment.iterrows():
            # Check if the row meets the condition
            if -1 <= row['ECG_II'] <= 1 and 20 <= row['ART_MBP'] <= 160:
                current_sub_segment.append(row)
            else:
                # If current sub-segment has data and condition fails, save and start new
                if current_sub_segment and (current_sub_segment['Time'].iloc[-1] - current_sub_segment['Time'].iloc[0])>=30:
                    sub_segments.append(pd.DataFrame(current_sub_segment))
                    current_sub_segment = []  # Reset for new sub-segment

        # If there's data left in the current sub-segment after finishing the loop
        if current_sub_segment and (current_sub_segment[0]['Time'].iloc[-1] - current_sub_segment[-1]['Time'].iloc[0])>=30:
            sub_segments.append(pd.DataFrame(current_sub_segment))

    return sub_segments

def filter_segments_by_length(segments, min_length_sec=30):
    # Filter segments by duration using the 'Time' column
    filtered_segments = [segment for segment in segments if (segment['Time'].iloc[-1] - segment['Time'].iloc[0]) >= min_length_sec]
    return filtered_segments