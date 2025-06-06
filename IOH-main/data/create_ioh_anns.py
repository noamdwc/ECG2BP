import os
from glob import glob
import argparse
import random
import time
import matplotlib.pyplot as plt
import json
import pandas as pd
from tqdm import tqdm

def is_csv_ok(csv_file, num_rows=30000):
    try:
        df = pd.read_csv(csv_file, nrows=num_rows)
        return len(df) == num_rows
    except Exception as e:
        print(f"Error reading file {csv_file}: {e}")
        return False

def get_args():
    parser = argparse.ArgumentParser(description="Create IOH dataset annotations")
    parser.add_argument('--dataset_root', type=str, default="/strg/D/shared-data/Shahaf_and_Fadi/vitaldb_ioh_dataset",
                        help="path to output directory")
    parser.add_argument('--output_dir', type=str,
                        default="/strg/D/shared-data/Shahaf_and_Fadi/vitaldb_ioh_dataset/annotations",
                        help="path to output directory to save the annotations")
    parser.add_argument('--val_set_ratio', type=float, default=0.3, help="validation set ratio")
    parser.add_argument('--ann-file', type=str, default="anns_2min_window_30_pos.json", help="annotation file name")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--desc', type=str,
                        default="This is 2 minute before IOH as positives, IOH negative is >10 min diff from positive",
                        help="comment to add to the annotation file")
    parser.add_argument('--positive_dirs', nargs='+',
                        default=['positive_30'])
                        # default=['positive_30', 'positive_60', 'positive_90', 'positive_120'])
    parser.add_argument('--all_positive_dirs', nargs='+',
                        default=['positive_30', 'positive_60', 'positive_90', 'positive_120'])
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
    anns = {
        "meta_data": {
            "val_set_ratio": args.val_set_ratio,
            "seed": args.seed,
            "desc": args.desc
        },
        "validation": {
            "positive": [],
            "negative": []
        },
        "train": {
            "positive": [],
            "negative": []
        }
    }
    random.seed(args.seed)
    data_root = args.dataset_root if not args.dataset_root.endswith("/") else args.dataset_root[:-1]
    files_pos = []
    pos_dirs = args.positive_dirs
    all_pos_dirs = args.all_positive_dirs
    print("Collecting positive and negative files...")
    for pos_dir in pos_dirs:
        files_pos += glob(os.path.join(data_root, pos_dir, "*", "*.csv"))
    # files_pos = glob(os.path.join(data_root, "positive*", "*", "*.csv"))
    print(f"Positive files found: {len(files_pos)}")
    files_neg = glob(os.path.join(data_root, "negative", "*", "*.csv"))
    # get the negative files that are not in the positive directories
    for pos_dir in all_pos_dirs:
        if pos_dir not in pos_dirs:
            files_neg += glob(os.path.join(data_root, pos_dir, "*", "*.csv"))
    print(f"Negative files found: {len(files_neg)}")
    all_cases_pos = set([f.split("/")[-2] for f in files_pos])
    print(f"Positive cases found: {len(all_cases_pos)}")
    # plt.hist(all_cases_pos, bins=100)
    # plt.title("Histogram of positive cases")
    # plt.show()
    all_cases_neg = set([f.split("/")[-2] for f in files_neg])
    print(f"Negative cases found: {len(all_cases_neg)}")
    # plt.hist(all_cases_neg, bins=100)
    # plt.title("Histogram of negative cases")
    # plt.show()

    all_cases_intersection = all_cases_neg.intersection(all_cases_pos)
    all_cases_intersection = sorted(list(all_cases_intersection))

    val_cases_pos = set(random.sample(all_cases_intersection, int(len(all_cases_intersection) * args.val_set_ratio)))
    all_cases_pos = set(all_cases_intersection)

    print(f"Validation positive cases: {len(val_cases_pos)}")
    train_cases_pos = all_cases_pos - val_cases_pos
    print(f"Train positive cases: {len(train_cases_pos)}")
    train_neg_cases = all_cases_neg - val_cases_pos
    print(f"Train negative cases: {len(train_neg_cases)}")
    val_neg_cases = all_cases_neg - train_neg_cases
    print(f"val neg cases: {len(val_neg_cases)}")

    root_dirs = len(data_root.split("/")) - 1
    for case in tqdm(train_cases_pos, desc="Creating train set annotations"):
        pos_case_files = []
        for pos_dir in pos_dirs:
            pos_case_files += glob(os.path.join(data_root, pos_dir, case, "*.csv"))
        # get reed of the root path
        anns["train"]["positive"] += ["/".join(f.split("/")[root_dirs+1:]) for f in pos_case_files if is_csv_ok(f)]
        neg_case_files = glob(os.path.join(data_root, "negative", case, "*.csv"))
        anns["train"]["negative"] += ["/".join(f.split("/")[root_dirs+1:]) for f in neg_case_files if is_csv_ok(f)]

    for case in tqdm(val_cases_pos, desc="Creating validation set annotations"):
        pos_case_files = glob(os.path.join(data_root, f"positive*", case, "*.csv"))
        anns["validation"]["positive"] += ["/".join(f.split("/")[root_dirs+1:]) for f in pos_case_files if is_csv_ok(f)]
        neg_case_files = glob(os.path.join(data_root, "negative", case, "*.csv"))
        anns["validation"]["negative"] += ["/".join(f.split("/")[root_dirs+1:]) for f in neg_case_files if is_csv_ok(f)]

    print("Writing the annotation file")
    with open(os.path.join(args.output_dir, args.ann_file), "w") as f:
        json.dump(anns, f)

    print(f"Total pos files train: {len(anns['train']['positive'])}")
    print(f"Total neg files train: {len(anns['train']['negative'])}")
    print(f"Total pos files val: {len(anns['validation']['positive'])}")
    print(f"Total neg files val: {len(anns['validation']['negative'])}")
