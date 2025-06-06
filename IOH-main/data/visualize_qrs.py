import os
import argparse
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qrs_root', type=str,
                        default="/strg/E/shared-data/Shahaf_and_Fadi/vitaldb_ioh_dataset/neurokit2_processed")
    return parser.parse_args()


def fill_segment_area(df_segment, onset_col, offset_col, color, label):
    """Fill the area between onset and offset for a given ECG segment."""
    for i in range(len(df_segment)):
        if df_segment[onset_col].iloc[i] == 1:
            start_index = df_segment.index[i]
            end_index = df_segment.index[df_segment[offset_col].iloc[i:].idxmax()]
            plt.fill_between(df_segment.index[start_index:end_index], df_segment['ECG_Clean'].min(), df_segment['ECG_Clean'].max(),
                             color=color, alpha=0.3, label=label)


def visualize_qrs(df, start_row, end_row):
    # Extract the segment of interest
    df_segment = df.iloc[start_row:end_row]

    # Plot the raw ECG signal
    plt.figure(figsize=(14, 6))
    plt.plot(df_segment.index, df_segment['ECG_Clean'], label='ECG Clean', color='blue')

    # Add vertical lines for P segment (Onsets and Offsets)
    plt.vlines(df_segment.index[df_segment['ECG_P_Onsets'] == 1],
               ymin=df_segment['ECG_Clean'].min(), ymax=df_segment['ECG_Clean'].max(),
               color='red', linestyle='--', label='P-Onsets')

    plt.vlines(df_segment.index[df_segment['ECG_P_Offsets'] == 1],
               ymin=df_segment['ECG_Clean'].min(), ymax=df_segment['ECG_Clean'].max(),
               color='red', linestyle='-', label='P-Offsets')

    # Add vertical lines for QRS segment (R Onsets and Offsets)
    plt.vlines(df_segment.index[df_segment['ECG_R_Onsets'] == 1],
               ymin=df_segment['ECG_Clean'].min(), ymax=df_segment['ECG_Clean'].max(),
               color='yellow', linestyle='--', label='QRS-Onsets')

    plt.vlines(df_segment.index[df_segment['ECG_R_Offsets'] == 1],
               ymin=df_segment['ECG_Clean'].min(), ymax=df_segment['ECG_Clean'].max(),
               color='yellow', linestyle='-', label='QRS-Offsets')

    # Add vertical lines for T segment (Onsets and Offsets)
    plt.vlines(df_segment.index[df_segment['ECG_T_Onsets'] == 1],
               ymin=df_segment['ECG_Clean'].min(), ymax=df_segment['ECG_Clean'].max(),
               color='magenta', linestyle='--', label='T-Onsets')

    plt.vlines(df_segment.index[df_segment['ECG_T_Offsets'] == 1],
               ymin=df_segment['ECG_Clean'].min(), ymax=df_segment['ECG_Clean'].max(),
               color='magenta', linestyle='-', label='T-Offsets')

    # # Fill areas for P segment
    # fill_segment_area(df_segment, 'ECG_P_Onsets', 'ECG_P_Offsets', 'red', 'P-Segment')
    #
    # # Fill areas for QRS segment (R Onsets and Offsets)
    # fill_segment_area(df_segment, 'ECG_R_Onsets', 'ECG_R_Offsets', 'green', 'QRS-Segment')
    #
    # # Fill areas for T segment
    # fill_segment_area(df_segment, 'ECG_T_Onsets', 'ECG_T_Offsets', 'magenta', 'T-Segment')

    # Adding labels and legend
    plt.title('ECG Signal with Colored Segments')
    plt.xlabel('Time (index)')
    plt.ylabel('ECG Signal')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    args = get_args()
    qrs_root = args.qrs_root
    files = glob(os.path.join(qrs_root, "negative",  "*", "*.csv"))
    # files = glob(os.path.join(qrs_root, "*", "*", "*.csv"))
    print(f"Total files: {len(files)}")
    good_files = []
    pbar = tqdm(files, desc="Checking files")
    for i, file in enumerate(pbar):
        df = pd.read_csv(file)
        if df['ECG_Quality'].mean() > 0.5:
            good_files.append(file)
            # Update the description during the loop
            pbar.set_postfix(item=f"{i}", status=f"{len(good_files) / (i+1) * 100:.2f}%")
            if i > 10:
                break
    print(f"Good signals percentage: {len(good_files) / len(files) * 100:.2f}%")


    for file in good_files:
        df = pd.read_csv(file)
        visualize_qrs(df, start_row=500, end_row=1000)
