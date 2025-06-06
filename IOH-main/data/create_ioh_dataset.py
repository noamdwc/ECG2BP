import os
import pecg
from pecg import Preprocessing as Pre
import pandas as pd
import vitaldb
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from preprocess import ioh_detect, ioh_negative, sample_ioh_negative, sample_ioh_positive
from functools import partial
import io
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_args():
    parser = argparse.ArgumentParser(description="Create IOH dataset")
    parser.add_argument('--workers', type=int, default=4, help="number of workers to process the cases")
    parser.add_argument('--signals', nargs='+', help='list of signals to download from vitaldb',
                        default=['ECG_II', 'ART_MBP'])
    parser.add_argument('--cols-to-save', nargs='+', help='list of signals to download from vitaldb',
                        default=['Time', 'ECG_II'])

    parser.add_argument('--pw', type=str, default="ShahafAndFadi123", help="data password")
    parser.add_argument('--user-id', type=str, default="140389", help="data user-id")
    parser.add_argument('--dataset_root', type=str, default="/strg/D/shared-data/Shahaf_and_Fadi/vitaldb_ioh_dataset/",
                        help="path to output directory")
    parser.add_argument('--time-interval', type=float, default=0.002, help="time interval for ECG signal")
    parser.add_argument('--sample-duration-sec', type=float, default=300, help="duration of the sample")
    parser.add_argument('--neg-time-before-sec', type=float, default=600,
                        help="time before the IOH event to be considered as negative sample")
    parser.add_argument('--neg-time-after-sec', type=float, default=600,
                        help="time after the IOH event to be considered as negative sample")
    parser.add_argument('--fs-ecg', type=int, default=500, help="sampling frequency of the ECG signal")
    parser.add_argument('--fs-downsample', type=int, default=100,
                        help="downsample the signal to this frequency")
    parser.add_argument('--ioh-min-duration', type=float, default=20, help="minimum duration of IOH event")
    return parser.parse_args()


def agg_df(func, dataframes: list, **kwargs):
    """
    Apply a function to a list of dataframes
    :param func: method to apply
    :param dataframes: list of dataframes
    :return: new list of dataframes
    """
    if kwargs:
        func = partial(func, **kwargs)
    dfs = []
    for df in dataframes:
        dfs.extend(func(df))
    return dfs


def clean_and_save_samples(dataframes, save_dir, case_id, fs=500, fs_downsample=None, cols=['Time', 'ECG_II']):
    """
    This method saves the samples to a directory
    :param dataframes: list of dataframes to save
    :param save_dir: directory to save the samples
    :param case_id: the case id of vitaldb
    :param fs: sampling frequency of the ecg signal
    :param fs_downsample: Note that fs/fs_downsample must be an integer and fs_downsample < fs
    :return:
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for sig in dataframes:
        try:
            sig = sig[cols].copy()
            with contextlib.redirect_stdout(io.StringIO()):
                pre = Pre.Preprocessing(np.array(sig["ECG_II"]), fs)
                bsqi_value = pre.bsqi()
            if bsqi_value < 0.8:
                continue
            if fs_downsample:
                # replace the signal with clean signal
                with contextlib.redirect_stdout(io.StringIO()):
                    cleaned = pre.bpfilt()
                sig.loc[:, "ECG_II"] = cleaned
                # downsample the signal
                downsample_factor = fs // fs_downsample
                sig = sig.iloc[::downsample_factor].reset_index(drop=True)
            sig.to_csv(os.path.join(save_dir, f"case_{case_id}_{sig['Time'].iloc[0]:.3f}.csv"), index=False)
        except Exception as e:
            print(f"Error in case {case_id}: {e}")

def chunkify(lst, n):
    """Split list `lst` into `n` chunks."""
    return [lst[i::n] for i in range(n)]


def process_cases(cases, args, worker_id=None):
    signals_to_download = args.signals
    dataset_root = args.dataset_root
    time_interval = args.time_interval
    duration_sec = args.sample_duration_sec
    neg_time_before_sec = args.neg_time_before_sec
    neg_time_after_sec = args.neg_time_after_sec
    fs = args.fs_ecg
    fs_downsample = args.fs_downsample
    cols = args.cols_to_save
    ioh_min_duration = args.ioh_min_duration
    print(args)
    description = "processing cases" if worker_id is None else f"worker {worker_id} processing cases"
    for case_id in tqdm(cases, desc=description):
        try:
            # Load signals for the patient
            vf = vitaldb.VitalFile(case_id, signals_to_download)
            # Convert to pandas dataframe
            signal = vf.to_pandas(signals_to_download, time_interval, return_timestamp=True)

            ioh_events, _ = ioh_detect(signal, ioh_min_duration=ioh_min_duration)
            negative_segments = ioh_negative(signal, ioh_events,
                                             time_before_sec=neg_time_before_sec, time_after_sec=neg_time_after_sec)

            sampled_negative_signal = agg_df(sample_ioh_negative, negative_segments, duration_sec=duration_sec)
            sampled_positive_signal_30 = sample_ioh_positive(signal, ioh_events,
                                                             time_before_sec=duration_sec,
                                                             duration_sec=duration_sec)
            sampled_positive_signal_60 = sample_ioh_positive(signal, ioh_events,
                                                             time_before_sec=duration_sec+30,
                                                             duration_sec=duration_sec)
            sampled_positive_signal_90 = sample_ioh_positive(signal, ioh_events,
                                                             time_before_sec=duration_sec+60,
                                                             duration_sec=duration_sec)
            sampled_positive_signal_120 = sample_ioh_positive(signal, ioh_events,
                                                              time_before_sec=duration_sec+90,
                                                              duration_sec=duration_sec)

            save_func = partial(clean_and_save_samples, case_id=case_id, fs=fs, fs_downsample=fs_downsample, cols=cols)

            save_func(sampled_positive_signal_30,
                      save_dir=os.path.join(dataset_root, "positive_30", str(case_id)))
            save_func(sampled_positive_signal_60,
                      save_dir=os.path.join(dataset_root, "positive_60", str(case_id)))
            save_func(sampled_positive_signal_90,
                      save_dir=os.path.join(dataset_root, "positive_90", str(case_id)))
            save_func(sampled_positive_signal_120,
                      save_dir=os.path.join(dataset_root, "positive_120", str(case_id)))
            save_func(sampled_negative_signal, save_dir=os.path.join(dataset_root, "negative", str(case_id)))
        except Exception as e:
            print(f"Error in case {case_id}: {e}")


if __name__ == "__main__":
    args = get_args()
    n_workers = args.workers
    # login to vitaldb
    vitaldb.login(args.user_id, args.pw)
    # get all the cases
    cases = vitaldb.find_cases(args.signals)
    # cases = [3]
    # n_workers = 0
    if n_workers:
        chunks = chunkify(cases, n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(process_cases, chunk, args, i)
                for i, chunk in enumerate(chunks)
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing chunk: {e}")
    else:
        process_cases(cases, args)

    print("Done!")
