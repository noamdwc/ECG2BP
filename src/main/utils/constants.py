import os

import numpy as np

TIME_COL = 'Time'
ECG_SIGNAL_COL = 'ECG_II'
ART_SIGNAL_COL = 'ART_MBP'
BINNED_ART_COL = 'binned_art'
BASE_PATH = os.environ.get('VITLAB_DATA_PATH', '/content/drive/MyDrive/data/ecg2bg')
DATA_PATH = f'{BASE_PATH}/vitlab_data'
BINNED_PATH = f'{DATA_PATH}/art_binned'
MERGED_PATH = f'{DATA_PATH}/merged'
NP_PATH = f'{DATA_PATH}/np'
WINDOWED_300_OFFSET_PT_PATH = f'{DATA_PATH}/windowed_300_offset'
NPZ_FORMAT_PATH = f'{DATA_PATH}/npz_format'
H5_FORMAT_PATH = f'{BASE_PATH}/early_dataset_segments3.h5'
caseid_to_idx_path = f'{BASE_PATH}/caseid_to_early_idx3.json'

SAMPLING_RATE = 500  # Htz
SECONDS_TO_SAMPLE = 1 / SAMPLING_RATE  # 0.002
WINDOW_SECONDS = 60
WINDOW_SIZE = int(WINDOW_SECONDS * SAMPLING_RATE)
HORIZON_SECONDS = 90
HORIZON_SIZE = int(HORIZON_SECONDS * SAMPLING_RATE)
STRIDE_SEC = 5
STRIDE_SIZE = int(STRIDE_SEC * SAMPLING_RATE)
CONTEXT_CHUNK_SIZE = 16  # 16 time WINDOWSIZE = 16 min context

CLIP_LOWER = -5.0  # or based on 1st percentile of min_ECG_II
CLIP_UPPER = 5.0  # or based on 99th percentile of max_ECG_II
TRIM_SECONDS = 5

# Split constanst
MIN_TIME_DURATION = 600  # 10 minutes
MAX_TIME_DURATION = 10 * 60 * 60  # 10 hours
EARLY_EXP_MAX_TIME_DURATION = 24000  # 10 minutes
TIME_BINS_FOR_SPLIT = [600, 12000, 24000, np.inf]
LONG_OUTLINER_CASEID = '5534'
IOH_PERCENT_BINS = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]

IS_EARLY_EXP = True
if IS_EARLY_EXP:
    split_name = 'early_exp'
else:
    split_name = 'full_exp'

    bins = [-np.inf, 55, 60, 65, 70, 100, np.inf]
    labels = ['<55', '55-60', '60-65', '65-70', '70-100', '>100']
    label_to_int = {label: i for i, label in enumerate(labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}
