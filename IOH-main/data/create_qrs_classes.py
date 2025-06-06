import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import neurokit2 as nk
import argparse

from ioh.dataset.ioh5min import IOH5MinDataset1D
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def show_peaks(processed_ecg_signals: pd.DataFrame, n=200):
    # Limit the data to the first n rows
    # data = processed_ecg_signals.iloc[:n]

    data = processed_ecg_signals

    # Plot ECG raw signal
    plt.figure(figsize=(15, 8))
    plt.plot(data.index, data['ECG_Raw'], label='ECG Raw', color='lightgrey', zorder=1)

    # Plot R-peaks
    r_peaks = np.where(data['ECG_R_Peaks'] > 0)[0]
    plt.scatter(data.index[r_peaks], data.loc[data.index[r_peaks], 'ECG_Raw'], color='red', label='R Peaks', zorder=2)

    # Plot T-peaks
    t_peaks = np.where(data['ECG_T_Peaks'] > 0)[0]
    plt.scatter(data.index[t_peaks], data.loc[data.index[t_peaks], 'ECG_Raw'], color='green', label='T Peaks', zorder=3)

    # Plot Q-peaks
    q_peaks = np.where(data['ECG_Q_Peaks'] > 0)[0]
    plt.scatter(data.index[q_peaks], data.loc[data.index[q_peaks], 'ECG_Raw'], color='blue', label='Q Peaks', zorder=4)

    # Plot P-peaks
    p_peaks = np.where(data['ECG_P_Peaks'] > 0)[0]
    plt.scatter(data.index[p_peaks], data.loc[data.index[p_peaks], 'ECG_Raw'], color='purple', label='P Peaks', zorder=5)

    # Add title and labels
    plt.title(f"ECG Signal with R, T, Q, and P Peaks (First {n} Samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()

    # Show the plot
    plt.show()


def process_dataset(dataset, save_dir, chunk_data=None):
    if chunk_data is not None:
        dataset.data = chunk_data
    for i in tqdm(range(len(dataset)), desc="Processing dataset"):
        signal_tensor, label_tensor, csv_file = dataset[i]
        file_parts = csv_file.split("/")[-3:]
        new_save_dir = str(os.path.join(save_dir, *file_parts[:-1]))
        Path(new_save_dir).mkdir(parents=True, exist_ok=True)
        new_file_full_path = os.path.join(new_save_dir, file_parts[-1])
        ecg_signal = signal_tensor.squeeze(0).to('cpu').numpy()
        processed_ecg_signals, info = nk.ecg_process(ecg_signal, sampling_rate=100)
        processed_ecg_signals.to_csv(new_file_full_path)
        # for debug only
        # show_peaks(processed_ecg_signals)


def chunkify_dataset(dataset: IOH5MinDataset1D, n):
    """Split list `dataset` into `n` chunks."""
    orig_data = dataset.data[:]
    return [orig_data[i::n] for i in range(n)]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=str, default="/strg/D/shared-data/Shahaf_and_Fadi/vitaldb_ioh_dataset",
                        help="path to dataset")
    parser.add_argument('--ann-file', type=str, default="anns_2min_window.json",
                        help="path to dataset")
    parser.add_argument('--signals', nargs='+', default=['ECG_II'])
    parser.add_argument('--workers', type=int, default=0, help="number of workers to process the cases")

    return parser.parse_args()


def launch_proc(dataset, n_workers, save_dir):
    chunks_data = chunkify_dataset(dataset, n_workers)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(process_dataset, dataset, save_dir, chunk_data=chunk)
            for i, chunk in enumerate(chunks_data)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing chunk: {e}")


if __name__ == "__main__":
    args = get_args()
    signals = args.signals
    anns_file = args.ann_file
    dataset_root = args.dataset_root
    workers = args.workers

    train_dataset = IOH5MinDataset1D(signals, dataset_root, anns_file, validation=False)
    val_dataset = IOH5MinDataset1D(signals, dataset_root, anns_file, validation=True)
    train_dataset.return_csv_file = True
    val_dataset.return_csv_file = True

    root_dir = os.path.join(dataset_root, "neurokit2_processed")

    if workers:
        launch_proc(train_dataset, workers, root_dir)
        launch_proc(val_dataset, workers, root_dir)
    else:
        print("Processing train dataset")
        process_dataset(train_dataset, root_dir)
        print("Processing validation dataset")
        process_dataset(val_dataset, root_dir)

    print("Done!")

