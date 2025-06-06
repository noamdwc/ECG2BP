import os
from pytorch_forecasting import TimeSeriesDataSet
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf

DATASET_REGISTRY = {}


def register_dataset(cls):
    """
    Register a dataset class by class name. This is a method for the decorator pattern. Part of infrastructure.
    :param cls:
    :return:
    """
    if cls.__name__ in DATASET_REGISTRY:
        raise ValueError(f"Cannot register duplicate dataset ({cls.__name__})")
    DATASET_REGISTRY[cls.__name__] = cls
    return cls


def create_dataset(cfg, dataset_name, validation=False):
    """
    Create a dataset from a config. This is a method for the factory pattern. Part of infrastructure.
    :param cfg:
    :param dataset_name:
    :param validation:
    :return:
    """
    return DATASET_REGISTRY[dataset_name](**DATASET_REGISTRY[dataset_name].from_config(cfg, validation=validation))


def get_train_dataset(cfg):
    """
    Get the training dataset from the config.
    :param cfg:
    :return:
    """
    train_name = cfg.TRAIN_SET
    if train_name:
        if train_name not in DATASET_REGISTRY:
            raise ValueError(f"Train-set {train_name} not found")
        train_set = create_dataset(cfg, train_name, validation=False)
    else:
        print("No training set specified")
        train_set = None
    return train_set


def get_test_dataset(cfg):
    """
    Get the test dataset from the config.
    :param cfg:
    :return:
    """
    test_name = cfg.TEST_SET
    if test_name:
        if test_name not in DATASET_REGISTRY:
            raise ValueError(f"Test-set {test_name} not found")
        test_set = create_dataset(cfg, test_name, validation=True)
    else:
        print("No test set specified")
        test_set = None
    return test_set


def get_datasets(cfg):
    """
    API for getting a dataset. Call this method to get a dataset.
    :param cfg:
    :return: This method creates a train and test dataset from the config.
    """
    train_set = get_train_dataset(cfg)
    test_set = get_test_dataset(cfg)
    return train_set, test_set


class IOHDataset(Dataset):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        """
        A dataset for time series data. This is base class for all our time series datasets.
        :param Dataset:
        :return:
        """
        super().__init__(**kwargs)
    """
    A dataset for time series data. This is base class for all our time series datasets.
    :param Dataset:
    :return:
    """
    @classmethod
    def from_config(cls, cfg, validation=False):
        """
        Create a dataset from a config. This method must be implemented by the subclass.
        :param cfg:
        :param validation: If True, return a dataset for validation. If False, return a dataset for training.
        :return: A dict containing the arguments to the constructor of the dataset
        """
        pass


@register_dataset
class DebugDataset(IOHDataset):
    def __init__(self, **kwargs):
        import pandas as pd
        import numpy as np
        # some debug dataset as pandas dataframe
        debug_data = pd.DataFrame(
            dict(
                value=np.random.rand(30) - 0.5,
                group=np.repeat(np.arange(3), 10),
                time_idx=np.tile(np.arange(10), 3),
            )
        )
        # Note: This is just for example. data, time_idx, target, group_ids MUST be provided in the constructor.
        # data: must be pandas dataframe with columns for time_idx, target, group_ids
        super().__init__(data=debug_data,
                         # time_idx="time_idx",
                         # target="value",
                         # group_ids=["group"],
                         **kwargs)

    @classmethod
    def from_config(cls, cfg, validation=False):
        from pytorch_forecasting.data.encoders import NaNLabelEncoder
        """
        Just for example. This method must be implemented by the subclass and return initialization arguments.
        """
        kwargs = {
            "target_normalizer": NaNLabelEncoder()
        }
        # convert the config to a dictionary and update the kwargs
        kwargs.update(OmegaConf.to_container(cfg.INPUT.KWARGS, resolve=True, enum_to_str=True))
        return kwargs



