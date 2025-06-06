from omegaconf import OmegaConf
import ast
import os

def parse_value(value):
    try:
        # Attempt to evaluate the value as a Python literal
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # If evaluation fails, return the value as is (string)
        return value


def update_config(cfg, updates):
    '''
    Update the config dictionary with new values.
    :param cfg:
    :param updates: a list in which odd element is the key and even element is it's the value
    :return:
    '''
    for update in updates[::2]:
        key = update
        value = updates[updates.index(update) + 1]
        value = parse_value(value)
        OmegaConf.update(cfg, key, value, merge=True)


def load_config(cfg_path, updates=None, base_configs="../exp_configs/base_configs.yaml"):
    # load the base config to construct the base dictionary (must have dictionary structure)
    base_cfg = OmegaConf.load(base_configs)
    # update the base config with the config file for the specific experiment
    exp_cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.merge(base_cfg, exp_cfg)
    # check if there are any updates to the config from cmd line and change the config dictionary accordingly
    if updates:
        update_config(cfg, updates)
    return cfg


def save_config(cfg, path):
    OmegaConf.save(cfg, path)
