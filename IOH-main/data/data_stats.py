import os
from glob import glob

if __name__ == "__main__":
    # data_root = "/strg/D/shared-data/Shahaf_and_Fadi/vitaldb_ioh_dataset/"
    data_root = "/strg/E/shared-data/Shahaf_and_Fadi/vitaldb_ioh_dataset/"



    # files_pos = glob(os.path.join(data_root, "positive*", "*", "*.csv"))
    # print(f"Positive files: {len(files_pos)}")
    files_neg = glob(os.path.join(data_root, "negative", "*", "*.csv"))
    print(f"Negative files: {len(files_neg)}")
    files_pos_30 = glob(os.path.join(data_root, "positive_30", "*", "*.csv"))
    files_pos_60 = glob(os.path.join(data_root, "positive_60", "*", "*.csv"))
    files_pos_90 = glob(os.path.join(data_root, "positive_90", "*", "*.csv"))
    files_pos_120 = glob(os.path.join(data_root, "positive_120", "*", "*.csv"))
    total_pos = len(files_pos_30) + len(files_pos_60) + len(files_pos_90) + len(files_pos_120)

    print(f"Positive files 0 sec: {len(files_pos_30)}")
    print(f"Positive files 30 sec: {len(files_pos_60)}")
    print(f"Positive files 60 sec: {len(files_pos_90)}")
    print(f"Positive files 90 sec: {len(files_pos_120)}")

    print(f"Total positive files: {total_pos}")
    val_set_ratio = 0.3

    # print(f"Validation set ratio: {val_set_ratio}")
    # val_set_size = int(len(files_pos) * val_set_ratio)
    # print(f"Validation set size: {val_set_size}")
    # train_set_size = len(files_pos) - val_set_size
    # print(f"Train set size: {train_set_size}")

