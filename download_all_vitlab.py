import os

import pandas as pd
import vitaldb
from tqdm import tqdm
time_interval_ecg = 1 / 500 # 500 Htz
time_interval_pleth = 1 / 60 # 500 Htz
time_interval_mbp = 3 * 60 # once every 3 min
wave_signals = ['ECG_II', 'PLETH']
num_signals = ['ART_MBP']
# Retrieve all case IDs
caseids = vitaldb.find_cases(wave_signals + num_signals)
print('there are total of', len(caseids), 'cases')


# Specify the signals to download: ECG, Pleth waveform, and arterial BP (ART)


# Base directory where CSVs will be saved
base_dir = "vitlab_data"
os.makedirs(base_dir, exist_ok=True)

for caseid in tqdm(caseids):
    # Instantiate a VitalFile for this case and those signals

    vf = vitaldb.VitalFile(caseid, wave_signals + num_signals)

    # Convert those signals into a single pandas DataFrame, with timestamps
    # The returned DataFrame will have columns: "timestamp", "ECG", "PLETH", "ART"
    # signal_df = vf.to_pandas(signals, time_interval, return_timestamp=True)
    df_ecg = vf.to_pandas(['ECG_II'], time_interval_ecg, return_timestamp=True)
    df_ecg.to_parquet(f"{base_dir}/ecg_{caseid}.parquet", engine="pyarrow", compression="gzip", index=False)

    df_art = vf.to_pandas(['ART_MBP'], time_interval_mbp, return_timestamp=True)
    df_art.to_parquet(f"{base_dir}/art_{caseid}.parquet", engine="pyarrow", compression="gzip", index=False)

    df_pleth = vf.to_pandas(['PLETH'], time_interval_pleth, return_timestamp=True)
    df_pleth.to_parquet(f"{base_dir}/pleth_{caseid}.parquet", engine="pyarrow", compression="gzip", index=False)