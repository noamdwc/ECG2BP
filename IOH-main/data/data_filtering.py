import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.signal as signal
from pecg import Preprocessing as Pre


def filter_sig(signal, fs=100):
    pre = Pre.Preprocessing(signal, fs)
    # Notch filter the powerline:
    # filtered_signal = pre.notch(n_freq=50)  # 50 Hz for european powerline, 60 Hz for USA
    # Bandpass for baseline wander and high-frequency noise:
    filtered_signal = pre.bpfilt()
    return filtered_signal, pre


def ecg_r_peak_freq(sig: pd.DataFrame, sampling_rate, signal_name='ECG_II'):
    r_peaks, _ = nk.ecg_peaks(sig[signal_name], sampling_rate=sampling_rate)
    r_peaks_indices = np.where(r_peaks['ECG_R_Peaks'] == 1)[0]

    r_peaks_times = sig["Time"].values[r_peaks_indices]
    time_intervals_prev = np.diff(r_peaks_times, prepend=r_peaks_times[0] - (r_peaks_times[1] - r_peaks_times[0]))
    time_intervals_next = np.diff(r_peaks_times, append=r_peaks_times[-1] + (r_peaks_times[-1] - r_peaks_times[-2]))

    ecg_average_time_intervals = (time_intervals_prev + time_intervals_next) / 2.0
    ecg_instantaneous_frequency = 1 / ecg_average_time_intervals

    ecg_frequency_time = r_peaks_times[1:-1]  # Corresponding times for the calculated frequencies
    ecg_instantaneous_frequency = ecg_instantaneous_frequency[1:-1]

    # Interpolate the instantaneous frequency to match the original time points
    original_time = sig["Time"].values
    interpolator = interp1d(ecg_frequency_time, ecg_instantaneous_frequency, kind='nearest', fill_value='extrapolate')
    interpolated_frequency = interpolator(original_time)

    # plt.figure(figsize=(10, 6))
    # plt.scatter(original_time, interpolated_frequency, color='red', label='Instantaneous Frequency')
    # plt.xlabel('Time')
    # plt.ylabel('Signal / Frequency')
    # plt.legend()
    # plt.title('Binary Signal and Instantaneous Frequency')
    # plt.show()

    return original_time, interpolated_frequency


def pleth_peak_freq(sig: pd.DataFrame, signal_name='PLETH', bias_window_size=100, smooth_window_size=20):
    # smooth the signal
    smoothed = sig[signal_name].rolling(window=smooth_window_size, center=True).mean()

    smoothed = smoothed.values
    # # make the first nan values same as the first value
    # smoothed[:bias_window_size // 2] = smoothed[bias_window_size // 2]
    # # make the last nan values same as the last value
    # smoothed[-bias_window_size // 2:] = smoothed[-bias_window_size // 2]
    # detrended_signal, _ = filter_sig(smoothed)

    # moving_avg = smoothed.rolling(window=bias_window_size, center=True).mean()
    # detrended_signal = smoothed - moving_avg
    # detrended_signal = detrended_signal.values
    # # make the first nan values same as the first value
    # detrended_signal[:bias_window_size // 2] = detrended_signal[bias_window_size // 2]
    # # make the last nan values same as the last value
    # detrended_signal[-bias_window_size // 2:] = detrended_signal[-bias_window_size // 2]

    # pleth_peaks, _ = signal.find_peaks(detrended_signal)
    # take all peaks that are also defenite positive in detrended_signal
    # pleth_peaks = pleth_peaks[detrended_signal[pleth_peaks] > 1]

    pleth_peaks, _ =  signal.find_peaks(smoothed)

    pleth_peak_times = sig['Time'].iloc[pleth_peaks].values

    pleth_time_intervals_prev = np.diff(pleth_peak_times,
                                        prepend=pleth_peak_times[0] - (pleth_peak_times[1] - pleth_peak_times[0]))
    pleth_time_intervals_next = np.diff(pleth_peak_times,
                                        append=pleth_peak_times[-1] + (pleth_peak_times[-1] - pleth_peak_times[-2]))
    pleth_average_time_intervals = (pleth_time_intervals_prev + pleth_time_intervals_next) / 2.0
    pleth_frequency_time = pleth_peak_times[1:-1]
    pleth_instantaneous_frequency = 1 / pleth_average_time_intervals
    pleth_frequency_time = pleth_peak_times[1:-1]
    # pleth_instantaneous_frequency = pleth_instantaneous_frequency[1:-1]/2.0
    pleth_instantaneous_frequency = pleth_instantaneous_frequency[1:-1]

    # plt.plot(sig['Time'], sig[signal_name])
    # plt.scatter(sig['Time'].iloc[pleth_peaks], sig[signal_name].iloc[pleth_peaks], color='red', label='Instantaneous Frequency')
    # plt.show()
    #
    # plt.plot(sig['Time'], detrended_signal)
    # plt.scatter(sig['Time'].iloc[pleth_peaks], detrended_signal[pleth_peaks], color='red', label='Instantaneous Frequency')
    # plt.show()

    # Interpolate the instantaneous frequency to match the original time points
    original_time = sig["Time"].values
    interpolator = interp1d(pleth_frequency_time, pleth_instantaneous_frequency, kind='nearest', fill_value='extrapolate')
    interpolated_frequency = interpolator(original_time)

    # plt.figure(figsize=(10, 6))
    # plt.scatter(original_time, interpolated_frequency, color='red', label='Instantaneous Frequency')
    # plt.xlabel('Time')
    # plt.ylabel('Signal / Frequency')
    # plt.legend()
    # plt.title('Binary Signal and Instantaneous Frequency')
    # plt.show()

    return original_time, interpolated_frequency


def pleth_waveletes(sig: pd.DataFrame, signal_name='PLETH', fs=100):
    pleth_signal = sig[signal_name].values
    time = sig["Time"].values
    import pywt
    coefficients, frequencies = pywt.cwt(pleth_signal, np.arange(1, 128), 'cgau1', sampling_period=1 / fs)
    coefficients = coefficients[:30]
    frequencies = frequencies[:30]
    plt.imshow(np.abs(coefficients), extent=(time.min(), time.max(), frequencies.min(), frequencies.max()),
               cmap='PRGn', aspect='auto', vmax=abs(coefficients).max(), vmin=-abs(coefficients).max())
    plt.yscale('log')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Wavelet Transform')
    plt.show()

    plt.plot(sig['Time'], sig[signal_name])
    plt.show()

    return coefficients, frequencies


def pleth_spectogram(sig: pd.DataFrame, signal_name='PLETH', fs=100):
    pleth_signal = sig[signal_name].values
    time = sig["Time"].values
    f, t, Sxx = signal.spectrogram(pleth_signal, fs, nperseg=256)
    Sxx = Sxx[:fs//10]
    f = f[:fs//10]
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.show()

    plt.plot(sig['Time'], sig[signal_name])
    plt.show()

    return f, t, Sxx


def compare_frequencies(time, freq_1, frq_2, std_threshold=0.2):
    # Calculate the absolute
    # difference between the two frequency arrays
    frequency_diff = np.abs(freq_1 - frq_2)

    # Check if the difference is within the specified standard deviation threshold
    within_threshold = frequency_diff <= std_threshold

    # Identify segments where frequencies are approximately the same
    segments = []
    start_time = None

    for i in range(len(within_threshold)):
        if within_threshold[i]:
            if start_time is None:
                start_time = time[i]
        else:
            if start_time is not None:
                end_time = time[i - 1]
                segments.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                })
                start_time = None

    # If the last segment reaches the end of the array
    if start_time is not None:
        end_time = time[-1]
        segments.append({
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
        })

    return segments


def plot_good_segments(df, segments, title_suffix=""):
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot ECG_II
    axs[0].plot(df["Time"], df["ECG_II"], label="ECG_II", color='blue')
    for segment in segments:
        start_time = segment["start_time"]
        end_time = segment["end_time"]
        axs[0].axvspan(start_time, end_time, color='green', alpha=0.3)
    axs[0].set_ylabel("ECG_II")
    axs[0].legend()
    axs[0].set_title(f"PLETH with Good Segments Highlighted{title_suffix}")

    # Plot PLETH
    axs[1].plot(df["Time"], df["PLETH"], label="PLETH", color='red')
    for segment in segments:
        start_time = segment["start_time"]
        end_time = segment["end_time"]
        axs[1].axvspan(start_time, end_time, color='green', alpha=0.3)
    axs[1].set_ylabel("PLETH")
    axs[1].legend()
    axs[1].set_title(f"PLETH with Good Segments Highlighted{title_suffix}")

    # Set common x-axis label
    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()

