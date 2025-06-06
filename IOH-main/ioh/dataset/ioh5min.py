from .dataset import IOHDataset
from pytorch_forecasting import TimeSeriesDataSet
from .dataset import register_dataset
import pandas as pd
import torch
import os
import json
import glob
import re
import random


def balance_lists(pos_list, neg_list, seed=42):
    random.seed(seed)
    # Calculate the lengths of the positive and negative lists
    len_pos = len(pos_list)
    len_neg = len(neg_list)

    # Determine which list is shorter and the difference in size
    if len_pos < len_neg:
        smaller_list = pos_list
        larger_list = neg_list
    else:
        smaller_list = neg_list
        larger_list = pos_list

    difference = len(larger_list) - len(smaller_list)

    # Determine how many times each item in the smaller list should appear
    quotient, remainder = divmod(difference, len(smaller_list))

    # Create the balanced list by duplicating entries in the smaller list
    balanced_smaller_list = smaller_list * (quotient + 1)  # Ensure all files appear at least once
    balanced_smaller_list = balanced_smaller_list[:difference] + smaller_list

    # Shuffle to distribute duplicated files randomly
    random.shuffle(balanced_smaller_list)

    # Assign the new balanced lists back to pos_list and neg_list
    if len_pos < len_neg:
        return balanced_smaller_list, larger_list
    else:
        return larger_list, balanced_smaller_list


def load_data(dataset_root, anns_file, validation, use_balancing=False):
    data_key = 'validation' if validation else 'train'
    anns = json.load(open(os.path.join(dataset_root, "annotations", anns_file), 'r'))
    anns = anns[data_key]
    data = []
    if use_balancing and not validation:
        anns['positive'], anns['negative'] = balance_lists(anns['positive'], anns['negative'])
    for case_file in anns['positive']:
        series_id = case_file.split('/')[-1].replace('.csv', '')
        data.append({'csv_file': os.path.join(dataset_root, case_file), 'series_id': series_id, 'label': 'pos'})
    for case_file in anns['negative']:
        series_id = case_file.split('/')[-1].replace('.csv', '')
        data.append({'csv_file': os.path.join(dataset_root, case_file), 'series_id': series_id, 'label': 'neg'})
    return data


def load_data_for_time_series(dataset_root, anns_file, validation):
    data_key = 'validation' if validation else 'train'
    anns = json.load(open(os.path.join(dataset_root, "annotations", anns_file), 'r'))
    anns = anns[data_key]
    # values


@register_dataset
class IOH5MinDataset(IOHDataset):
    def __init__(self, signals, dataset_root, anns_file, validation, use_balancing=False):
        super().__init__()
        self.dataset_root = dataset_root
        self.anns_file = anns_file
        self.validation = validation
        self.signals = signals
        self.data = load_data(dataset_root, anns_file, validation, use_balancing)


    @classmethod
    def from_config(cls, cfg, validation=False):
        kwargs = dict(cfg.INPUT.DATASET_KWARGS)
        kwargs["validation"] = validation
        return kwargs

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.
        """
        sample = self.data[index]
        csv_file = sample['csv_file']
        label = sample['label']

        # Load the signal from the CSV file
        csv_path = os.path.join(self.dataset_root, csv_file)
        df = pd.read_csv(csv_path)
        signal = torch.tensor(df[self.signals].values).float()

        # Split the signal into N tokens
        signal_length = signal.size(0)
        token_length = signal_length // self.num_tokens
        if signal_length % self.num_tokens != 0:
            # Calculate the number of elements to discard from the beginning
            discard_length = signal_length % self.num_tokens
            signal = signal[discard_length:]

        tokens = signal.view(self.num_tokens, -1, signal.size(1))  # (N, token_length, C)

        # Convert label to binary format
        gt_label = 1 if label == 'pos' else 0

        return {
            "encoder_cont": tokens,
            "encoder_target": torch.tensor([gt_label], dtype=torch.long),
            "time_idx": list(range(self.num_tokens)),
            "series_id": sample["series_id"]
        }

    def __len__(self):
        return len(self.data)


@register_dataset
class IOH5MinDataset1D(IOHDataset):
    def __init__(self, signals, dataset_root, anns_file, seq_len, validation, use_balancing=False, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.dataset_root = dataset_root
        self.anns_file = anns_file
        self.seq_len = seq_len
        self.validation = validation
        self.signals = signals
        self.data = load_data(dataset_root, anns_file, validation, use_balancing)
        self.return_csv_file = False

    @classmethod
    def from_config(cls, cfg, validation=False):
        kwargs = dict(cfg.INPUT.DATASET_KWARGS)
        kwargs["validation"] = validation
        return kwargs

    def __getitem__(self, index):
        case = self.data[index]
        csv_file = case['csv_file']
        label = case['label']

        # Load the data from the CSV file
        df = pd.read_csv(csv_file)
        signal_values = df[self.signals].values.flatten()  # Assuming the signal values are in the specified columns

        # Convert the signal values to a tensor
        signal_tensor = torch.tensor(signal_values, dtype=torch.float32)
        signal_tensor = signal_tensor[-self.seq_len:]
        # Make the signal tensor 1D with 1 channel
        signal_tensor = signal_tensor.unsqueeze(0)

        # Convert the label to a tensor (binary classification)
        label_tensor = torch.tensor(1 if label == 'pos' else 0, dtype=torch.long)
        if self.return_csv_file:
            return signal_tensor, label_tensor, csv_file
        else:
            return signal_tensor, label_tensor

    def __len__(self):
        return len(self.data)
