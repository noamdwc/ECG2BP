import argparse
import vitaldb
from pathlib import Path
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from preprocess import ioh_detect, ioh_negative, sample_ioh_negative, sample_ioh_positive
from tqdm import tqdm

from signal_analysis import wavelet_transform, emd_transform, hht_transform
from signal_analysis import spectrogram_plot_ax, wavelet_transform_plot_ax, emd_plot_ax, hht_plot_ax
import random

from neurokit2.ecg.ecg_clean import ecg_clean


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pw', type=str, default="ShahafAndFadi123", help="data password")
    parser.add_argument('--user-id', type=str, default="140389", help="data user-id")
    parser.add_argument('--output_dir', type=str, default="/strg/C/shared-data/vitaldb", help="path to output directory")
    parser.add_argument('--params', nargs='+', help='list of parameters to download from vitaldb',
                        # default=[('ECG_II', 0.002), ('PLETH', 0.01), ('NIBP_DBP', 0.01), ('NIBP_MBP', 0.01), ('NIBP_SBP', 0.01)])
                        default=[('DBP', 0.01)])  # ['ECG_II', 'PLETH', 'NIBP_DBP', 'NIBP_MBP', 'NIBP_SBP']
    return parser.parse_args()


# def download_vitaldb_data(params, output_dir, pw, user_id):
#     # create output directory to save the data
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
#     # login to vitaldb server
#     vitaldb.login(user_id, pw)
#     # find cases with specific parameters from server
#     # all_params = [p[0] for p in params]
#     # all_params = ['ECG_II', 'PLETH', 'NIMP_DBP', 'NIBP_MBP', 'NIBP_SBP']
#     all_params = ['ECG_II', 'PLETH', 'NIBP_DBP', 'NIBP_MBP', 'NIBP_SBP']
#     caseids = vitaldb.find_cases(all_params)
#     # download all cases from server
#     for caseid in tqdm.tqdm(caseids, desc="Downloading from vitaldb"):
#         for param, interval in params:
#             val = vitaldb.load_case(caseid, param, interval)
#             # np.save(f"{output_dir}/{caseid}_{param}.npy", val)
#             # val = vitaldb.load_case(caseid, params, interval)
#             plt.plot(np.arange(len(val)), val)
#             plt.plot(np.arange(1000), val[:1000])
#             plt.scatter(np.arange(len(val)), val)
#             plt.scatter(np.arange(10000), val[:10000])
#             plt.show()
#
#         plt.show()


def agg_df(func, dataframes:list):
    """
    Apply a function to a list of dataframes
    :param func: method to apply
    :param dataframes: list of dataframes
    :return: new list of dataframes
    """
    dfs = []
    for df in dataframes:
        dfs.extend(func(df))
    return dfs


def ax_annotate(ax, text):
    ax.annotate(text,
                xy=(-0.1, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0), xycoords=ax.yaxis.label,
                textcoords='offset points', size='large', ha='center', va='center', rotation=90)

def ecg_plot_ax(ax, sig):
    ecg = sig['ECG_II']
    time = sig['Time']
    ax.plot(time, ecg)
    ax.grid()

def plot_egc_samples(cases_ids,
                     ecg_save_dir="ECG_vis",
                     decomposition_save_dir="decompositions_vis",
                     max_cases=None,
                     max_samples=10,
                     cleaning_func=None,
                     title_prefix=""):
    Path(ecg_save_dir).mkdir(parents=True, exist_ok=True)
    Path(decomposition_save_dir).mkdir(parents=True, exist_ok=True)
    if max_cases is not None:
        case_ids = random.choices(cases_ids, k=max_cases)
    for case_id in tqdm(case_ids, desc="Processing cases"):
        try:
            # Load signals for the patient
            vf = vitaldb.VitalFile(case_id, required_signals)
            # convert to pandas dataframe
            signal = vf.to_pandas(required_signals, time_interval, return_timestamp=True)
            ioh_events, _ = ioh_detect(signal)
            negative_segments = ioh_negative(signal, ioh_events, time_before_sec=600, time_after_sec=600)
            sampled_negative_signal = agg_df(sample_ioh_negative, negative_segments)
            sampled_positive_signal_30 = sample_ioh_positive(signal, ioh_events, time_before_sec=30)
            sampled_positive_signal_60 = sample_ioh_positive(signal, ioh_events, time_before_sec=60)
            sampled_positive_signal_90 = sample_ioh_positive(signal, ioh_events, time_before_sec=90)
            sampled_positive_signal_120 = sample_ioh_positive(signal, ioh_events, time_before_sec=120)

            sorted_indices = sorted(range(len(ioh_events)), key=lambda k: ioh_events[k]['segment_len'], reverse=True)
            if max_samples is not None:
                sorted_indices = sorted_indices[:max_samples]
            # if max_samples is not None:
            #     sampled_negative_signal = random.choices(sampled_negative_signal, k=max_samples)
            #     sampled_positive_signal_30 = random.choices(sampled_positive_signal_30, k=max_samples)
            #     sampled_positive_signal_60 = random.choices(sampled_positive_signal_60, k=max_samples)
            #     sampled_positive_signal_120 = random.choices(sampled_positive_signal_120, k=max_samples)
            # randomly sample a sub list from sampled_negative_signal with the same length as the positive samples
            if len(sampled_negative_signal) > max_samples*3:
                sampled_negative_signal = random.choices(sampled_negative_signal, k=max_samples*3)
            neg_ind = 0
            for i in sorted_indices:
                event = ioh_events[i]
                # Add row labels
                row_labels = ["Pos (30)", "Pos (60)", "Pos (90)", "Pos (120)", "Neg", "Neg", "Neg"]
                col_labels = ["Spec", "Wav", "EMD", "HHT"]
                fig_decompose, axs_decompose = plt.subplots(len(row_labels), len(col_labels), figsize=(20, 15))
                fig_ecg, axs_ecg = plt.subplots(len(row_labels), 1, figsize=(20, 15))
                # Turn off ticks and spines
                for ax in axs_decompose.flat:
                    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                # Add column headers
                for j, col_name in enumerate(col_labels):
                    axs_decompose[0, j].set_title(col_name)
                axs_ecg[0].set_title("ECG")

                for j, row_name in enumerate(row_labels):
                    if row_name == "Neg":
                        continue
                    ax_annotate(axs_decompose[j, 0], row_name)
                    ax_annotate(axs_ecg[j], row_name)


                all_samples = [sampled_positive_signal_30, sampled_positive_signal_60, sampled_positive_signal_90, sampled_positive_signal_120]
                j = 0
                for j, samples_list in enumerate(all_samples):
                    if i < len(samples_list):
                        sampled_positive_signal = samples_list[i]  # using i to get the same event
                        if cleaning_func is not None:
                            sampled_positive_signal = cleaning_func(sampled_positive_signal)
                        # time = sampled_positive_signal['Time'].values[0]
                        # print(f"j={j} time={time}")
                        spectrogram_plot_ax(axs_decompose[j, 0], *spec_transform(sampled_positive_signal))
                        wavelet_transform_plot_ax(axs_decompose[j, 1], *wavelet_transform(sampled_positive_signal))
                        emd_plot_ax(axs_decompose[j, 2], *emd_transform(sampled_positive_signal))
                        hht_plot_ax(axs_decompose[j, 3], *hht_transform(sampled_positive_signal))
                        ecg_plot_ax(axs_ecg[j], sampled_positive_signal)

                for k in range(j + 1, axs_decompose.shape[0]):
                    if neg_ind >= len(sampled_negative_signal):
                        break
                    neg_signal = sampled_negative_signal[neg_ind]
                    if cleaning_func is not None:
                        neg_signal = cleaning_func(sampled_negative_signal[neg_ind])
                    spectrogram_plot_ax(axs_decompose[k, 0], *spec_transform(neg_signal))
                    wavelet_transform_plot_ax(axs_decompose[k, 1], *wavelet_transform(neg_signal))
                    emd_plot_ax(axs_decompose[k, 2], *emd_transform(neg_signal))
                    hht_plot_ax(axs_decompose[k, 3], *hht_transform(neg_signal))
                    time = sampled_negative_signal[neg_ind]['Time'].values[0]
                    ax_annotate(axs_decompose[k, 0], f"{row_labels[k]}_t={time:.0f}")

                    ecg_plot_ax(axs_ecg[k], neg_signal)
                    ax_annotate(axs_ecg[k], f"{row_labels[k]}_t={time:.0f}")

                    neg_ind += 1

                title = f"{title_prefix}Case {case_id} - Duration {event['segment_len']:.2f} - Start time {event['Time_start']}"
                fig_decompose.suptitle(title, fontsize=16)
                # fig_decompose.show()
                fig_decompose.tight_layout()
                fig_decompose.savefig(f"{decomposition_save_dir}/case_{case_id}_no_{i}_duration_{event['segment_len']:.2f}.png")
                plt.close(fig_decompose)

                # plot the ECG signal
                fig_ecg.suptitle(title, fontsize=16)
                fig_ecg.savefig(f"{ecg_save_dir}/ECG_case_{case_id}_no_{i}_duration_{event['segment_len']:.2f}.png")
                plt.close(fig_ecg)

        except Exception as e:
            print(f'Error processing case {case_id}: {e}')


if __name__ == "__main__":
    args = get_args()
    # download_vitaldb_data(args.params, args.output_dir, args.pw, args.user_id)
    vitaldb.login(args.user_id, args.pw)
    # required_signals = ['ECG_II', 'PLETH', 'NIBP_DBP', 'NIBP_MBP', 'NIBP_SBP']

    required_signals = ['ECG_II', 'PLETH', 'ART_MBP']
    # Main script
    # case_ids = vitaldb.find_cases(required_signals)
    # case_id = 41
    case_id = 150
    # Load signals for the patient
    vf = vitaldb.VitalFile(case_id, required_signals)
    # convert to pandas dataframe
    signal = vf.to_pandas(required_signals, 0.01, return_timestamp=True)
    print(signal)


    import neurokit2 as nk
    # take 'ECG_II', 'PLETH'
    df = signal[['Time', 'ECG_II', 'PLETH']]

    # time_start = 11200
    # time_end = 11235

    # time_start = 11100
    # time_end = 11200
    time_start = 1337
    time_end = 1368

    df = df[(df['Time'] >= time_start) & (df['Time'] <= time_end)]

    from pecg import Preprocessing as Pre
    import scipy.signal as signal


    def filter_ecg(signal, fs=500):
        pre = Pre.Preprocessing(signal, fs)
        # Notch filter the powerline:
        # filtered_signal = pre.notch(n_freq=50)  # 50 Hz for european powerline, 60 Hz for USA
        # Bandpass for baseline wander and high-frequency noise:
        filtered_signal = pre.bpfilt()
        return filtered_signal, pre

    # Remove rows with NaN values
    df = df.dropna()

    # Extract the sampling rate from your data (assuming it's constant)
    sampling_rate = 100  # Set your actual sampling rate here

    from data_filtering import ecg_r_peak_freq, pleth_peak_freq, compare_frequencies, plot_good_segments

    # pleth_waveletes(df)
    # pleth_spectogram(df)


    ecg_frequency_time, ecg_instantaneous_frequency = ecg_r_peak_freq(df, sampling_rate)
    pleth_frequency_time, pleth_instantaneous_frequency = pleth_peak_freq(df)

    segments = compare_frequencies(ecg_frequency_time, ecg_instantaneous_frequency, pleth_instantaneous_frequency, std_threshold=0.2)
    plot_good_segments(df, segments, title_suffix=" std_threshold=0.2")
    segments = compare_frequencies(ecg_frequency_time, ecg_instantaneous_frequency, pleth_instantaneous_frequency,
                                   std_threshold=0.3)
    plot_good_segments(df, segments, title_suffix=" std_threshold=0.3")
    segments = compare_frequencies(ecg_frequency_time, ecg_instantaneous_frequency, pleth_instantaneous_frequency,
                                      std_threshold=0.4)
    plot_good_segments(df, segments, title_suffix=" std_threshold=0.4")
    segments = compare_frequencies(ecg_frequency_time, ecg_instantaneous_frequency, pleth_instantaneous_frequency,
                                        std_threshold=0.5)
    plot_good_segments(df, segments, title_suffix=" std_threshold=0.5")
    segments = compare_frequencies(ecg_frequency_time, ecg_instantaneous_frequency, pleth_instantaneous_frequency,
                                        std_threshold=0.6)
    plot_good_segments(df, segments, title_suffix=" std_threshold=0.6")
    segments = compare_frequencies(ecg_frequency_time, ecg_instantaneous_frequency, pleth_instantaneous_frequency,
                                        std_threshold=0.7)
    plot_good_segments(df, segments, title_suffix=" std_threshold=0.7")
    segments = compare_frequencies(ecg_frequency_time, ecg_instantaneous_frequency, pleth_instantaneous_frequency,
                                        std_threshold=0.8)
    plot_good_segments(df, segments, title_suffix=" std_threshold=0.8")
    segments = compare_frequencies(ecg_frequency_time, ecg_instantaneous_frequency, pleth_instantaneous_frequency,
                                        std_threshold=0.9)
    plot_good_segments(df, segments, title_suffix=" std_threshold=0.9")
    segments = compare_frequencies(ecg_frequency_time, ecg_instantaneous_frequency, pleth_instantaneous_frequency,
                                        std_threshold=1.0)
    plot_good_segments(df, segments, title_suffix=" std_threshold=1.0")



    # Step 2: R-peak detection
    # Extract the ECG signal
    ecg_signal = df['ECG_II']

    # Detect R-peaks
    r_peaks, _ = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
    r_peaks_indices = np.where(r_peaks['ECG_R_Peaks'] == 1)[0]

    plt.plot(df['Time'], df['ECG_II'])
    plt.scatter(df['Time'].iloc[r_peaks_indices], df['ECG_II'].iloc[r_peaks_indices], color='red')
    plt.show()

    time_array = df["Time"].values
    binary_signal = r_peaks['ECG_R_Peaks'].values
    event_indices = np.where(binary_signal == 1)[0]
    event_times = time_array[event_indices]
    time_intervals_prev = np.diff(event_times, prepend=event_times[0] - (event_times[1] - event_times[0]))
    time_intervals_next = np.diff(event_times, append=event_times[-1] + (event_times[-1] - event_times[-2]))
    ecg_average_time_intervals = (time_intervals_prev + time_intervals_next) / 2.0

    ecg_instantaneous_frequency = 1 / ecg_average_time_intervals
    ecg_frequency_time = event_times[1:-1]  # Corresponding times for the calculated frequencies
    ecg_instantaneous_frequency = ecg_instantaneous_frequency[1:-1]
    print("ecg freq len ", len(ecg_instantaneous_frequency))
    plt.figure(figsize=(10, 6))
    plt.plot(time_array, binary_signal, label='Binary Signal', drawstyle='steps-post')
    plt.scatter(ecg_frequency_time, ecg_instantaneous_frequency, color='red', label='Instantaneous Frequency')
    plt.xlabel('Time')
    plt.ylabel('Signal / Frequency')
    plt.legend()
    plt.title('Binary Signal and Instantaneous Frequency')
    plt.show()

    window_size = 10  # Adjust the window size as needed
    # take only first 100 samples
    # df = df[:100]
    moving_avg = df['PLETH'].rolling(window=window_size, center=True).mean()
    detrended_signal = df['PLETH'] - moving_avg
    detrended_signal = detrended_signal.values

    plt.plot(df['Time'], detrended_signal)
    plt.show()

    pleth_peaks, _ = signal.find_peaks(detrended_signal, prominence=0.1)
    # take all peaks that are also defenite positive in detrended_signal
    pleth_peaks = pleth_peaks[detrended_signal[pleth_peaks] > 1]
    # pleth_peaks, _ = signal.find_peaks(df['PLETH'])
    plt.plot(df['Time'], df['PLETH'])
    plt.scatter(df['Time'].iloc[pleth_peaks], df['PLETH'].iloc[pleth_peaks], color='red')
    plt.show()

    plt.plot(df['Time'], detrended_signal)
    plt.scatter(df['Time'].iloc[pleth_peaks], detrended_signal[pleth_peaks], color='red')
    plt.show()

    pleth_peak_times = df['Time'].iloc[pleth_peaks].values
    pleth_time_intervals_prev = np.diff(pleth_peak_times, prepend=pleth_peak_times[0] - (pleth_peak_times[1] - pleth_peak_times[0]))
    pleth_time_intervals_next = np.diff(pleth_peak_times, append=pleth_peak_times[-1] + (pleth_peak_times[-1] - pleth_peak_times[-2]))
    pleth_average_time_intervals = (pleth_time_intervals_prev + pleth_time_intervals_next) / 2.0
    pleth_frequency_time = pleth_peak_times[1:-1]
    pleth_instantaneous_frequency = 1 / pleth_average_time_intervals

    pleth_event_times = df['Time'].iloc[pleth_peaks]
    pleth_frequency_time = pleth_peak_times[1:-1]
    pleth_instantaneous_frequency = pleth_instantaneous_frequency[1:-1]/2.0
    print("pleth freq len ", len(pleth_instantaneous_frequency))
    plt.figure(figsize=(10, 6))
    plt.scatter(pleth_frequency_time, pleth_instantaneous_frequency, color='red', label='Instantaneous Frequency')
    plt.xlabel('Time')
    plt.ylabel('Signal / Frequency')
    plt.legend()
    plt.title('Binary Signal and Instantaneous Frequency')
    plt.show()


    # Step 3: Calculate heart rate from PLETH signal
    pleth_signal = df['PLETH']
    pleth_sig_filt, _ = filter_ecg(df['PLETH'].to_numpy(), fs=100)
    plt.plot(pleth_sig_filt)
    plt.show()
    plt.plot(pleth_signal)
    plt.show()
    pleth_rate = nk.signal_rate(pleth_signal, sampling_rate=sampling_rate)
    from signal_analysis import spec_transform

    window_length = int(1 * sampling_rate)  # 1-second window
    overlap = int(window_length / 2)  # 50% overlap
    window = 'hann'  # Type of window, e.g., Hann window
    from scipy.signal import spectrogram

    f, t, Sxx = spectrogram(df['PLETH'].values, fs=sampling_rate, window=window, nperseg=window_length, noverlap=overlap)
    # plot spectrogram
    plt.pcolormesh(t, f, 10 * np.log10(Sxx))

    plt.show()

    f, t, Sxx = spectrogram(pleth_sig_filt, fs=sampling_rate, window=window, nperseg=window_length,
                            noverlap=overlap)
    # plot spectrogram
    plt.pcolormesh(t, f, 10 * np.log10(Sxx))

    plt.show()

    # Create a time series for the PLETH rate
    pleth_times = df['Time'].values
    pleth_rate_series = pd.Series(pleth_rate, index=pleth_times[:len(pleth_rate)])

    # Resample the PLETH rate to match the ECG signal length
    pleth_rate_resampled = np.interp(df['Time'], pleth_times[:len(pleth_rate)], pleth_rate)

    # Step 4: Filter ECG signal based on PLETH rate
    # Use the PLETH rate to create a bandpass filter around the heart rate frequency
    # Assuming heart rate from PLETH is in bpm, convert to Hz for filtering (1 bpm = 1/60 Hz)
    lowcut = pleth_rate_resampled.min() / 60.0
    highcut = pleth_rate_resampled.max() / 60.0

    filtered_ecg = nk.signal_filter(ecg_signal, sampling_rate=sampling_rate, lowcut=lowcut, highcut=highcut)

    # Adding filtered ECG to the dataframe
    df['Filtered_ECG'] = filtered_ecg

    # Save the processed data
    df.to_csv('processed_data.csv', index=False)

    # Plotting for visualization (optional)
    nk.events_plot(r_peaks_indices, ecg_signal)

    plt.plot(signal['Time'], signal['PLETH'])
    # remove nan in pleth
    sig = signal.dropna(subset=['PLETH'])
    # print pleth with Time column
    print(sig[['Time', 'PLETH']])
    plt.show()
    time_interval = 0.01
    random.seed(42)


    def cleaning_method(df, method='biosppy'):
        ecg_signal = df['ECG_II'].values
        cleaned_sig = ecg_clean(ecg_signal, sampling_rate=100, method=method)
        df['ECG_II'] = cleaned_sig
        return df

    methods = ['neurokit', 'biosppy', 'pantompkins1985', 'hamilton2002', 'elgendi2010', 'vg']
    from functools import partial
    for method_name in methods:
        cleaning = partial(cleaning_method, method=method_name)
        plot_egc_samples(vitaldb.find_cases(required_signals),
                         decomposition_save_dir=f"/strg/C/shared-data/Shahaf_and_Fadi/IOH/decomposition_vis_cleaning_{method_name}",
                         ecg_save_dir=f"/strg/C/shared-data/Shahaf_and_Fadi/IOH/ECG_vis_cleaning_{method_name}",
                         max_cases=300,
                         max_samples=10,
                         cleaning_func=cleaning,
                         title_prefix=f"Cleaning: {method_name} - ")


    # # just for testing
    # case_ids = [1]
    # # first step is to save all the data to csv in (time, value) format by patient id without any nan values
    # for case_id in tqdm(case_ids, desc="Processing cases"):
    #     try:
    #         # Load signals for the patient
    #         vf = vitaldb.VitalFile(case_id, required_signals)
    #         # convert to pandas dataframe
    #         signal = vf.to_pandas(required_signals, time_interval, return_timestamp=True)
    #         ioh_events, _ = ioh_detect(signal)
    #         negative_segments = ioh_negative(signal, ioh_events)
    #         sampled_negative_signal = agg_df(sample_ioh_negative, negative_segments)
    #         sampled_positive_signal_30 = sample_ioh_positive(signal, ioh_events, time_before_sec=30)
    #         sampled_positive_signal_60 = sample_ioh_positive(signal, ioh_events, time_before_sec=60)
    #         sampled_positive_signal_120 = sample_ioh_positive(signal, ioh_events, time_before_sec=120)
    #
    #         pos_sample = sampled_positive_signal_30[0]
    #         plot_spectrogram(pos_sample)
    #         plot_wavelet_transform(pos_sample)
    #         # plot_emd(pos_sample)
    #         # plot_hht(pos_sample)
    #         # plot_emd_hilbert_spectrum(pos_sample)
    #         # plot_hilbert_spectrum(pos_sample)
    #         plot_emd_image(pos_sample)
    #         plot_hht_image(pos_sample)
    #         if ioh_events:
    #             print(f'IOH events detected for case {case_id}: {ioh_events}')
    #         # ecg_signal = df['ECG_II'].values
    #         # show the ecg signal
    #         # plt.plot(np.arange(len(ecg_signal)), ecg_signal)
    #         # plt.show()
    #         # ecg_cleaning(ecg_signal)
    #         # signals = {sig: vf.to_numpy([sig], 1 / 100) for sig in required_signals}
    #         #
    #         # # Resample and align signals
    #         # df = resample_and_align(signals)
    #
    #
    #         # Save DataFrame to CSV
    #         # df.to_csv(f'{case_id}_data.csv', index=False)
    #         # print(f'Data for case {case_id} saved to CSV.')
    #
    #     except Exception as e:
    #         print(f'Error processing case {case_id}: {e}')
