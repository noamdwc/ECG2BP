import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils.augmentaions import ECGAug, AnnealConfig, BatchAugmenter
from .utils.constants import NP_PATH, BASE_PATH
from .utils.datasets import SplitDataset
from .utils.models import CNNLSTMModel
from .utils.training_utils import train_model

# Speedups
torch.backends.cudnn.benchmark = True  # fixed-size convs get autotuned
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')  # Ampere+ (Colab T4/A100 vary)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


MODEL_NAME = "CNNLSTM_NP_T16_H90s_binary_pair_qugmenter--test--test"


def main():
    train_ds = SplitDataset(["train_segments_part1.npy",
                             "train_segments_part2.npy"],
                            "train_labels.npy")
    test_ds = SplitDataset(["test_segments_part1.npy",
                            "test_segments_part2.npy",
                            os.path.join(NP_PATH, "test_segments_part3.npy")],
                           "test_labels.npy")

    load_kwargs = {"num_workers": 2, "pin_memory": True, "worker_init_fn": seed_worker}
    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True,
        **load_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False,
        **load_kwargs
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device")
    model = CNNLSTMModel(
        cnn_out_dim=128,
        lstm_hidden=64,
        num_classes=2,
        num_blocks=(1, 1)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    augmenter = BatchAugmenter(
        use_mixup=True, mixup_alpha=0.15, mixup_p=0.15, mixup_per_sample=True,
        use_timecutmix=True, tcm_p=0.45, tcm_len_s=(0.25, 0.75), tcm_per_sample=True,
        mutually_exclusive=True, fs=500, label_smoothing=0.05
    )

    # allowed_pairs = {(2, 3), (3, 2), (1, 3), (3, 1)}
    #
    # augmenter = ConditionalPairAugmenter(
    #     fs=500,
    #     allowed_pairs=allowed_pairs,
    #     use_mixup=True, mixup_alpha=0.3, mixup_p=0.35,
    #     use_timecutmix=True, tcm_p=0.15, tcm_len_s=(0.20, 0.50),
    #     tcm_per_sample=True, mixup_per_sample=True,
    #     label_smoothing=0.03,
    #     mutually_exclusive=True
    # )

    per_sample_aug = ECGAug(
        fs=500,
        p_jitter=0.55, sigma_rel=(0.010, 0.090),  # ~1.3× your p90
        p_gain=0.55, gain_range=(0.93, 1.07),
        p_shift=0.50, max_shift_ms=10,  # keep micro-events aligned
        p_mask=0.30, n_masks=(0, 2), mask_ms=(10, 60),
        p_wander=0.40, wander_hz=(0.05, 0.5), wander_amp_rel=(0.035, 0.180),
        p_hum=0.0
    )

    anneal = AnnealConfig(
        ps_strength_scale=0.6,  # ECGAug strength
        ps_prob_scale=0.7,  # ECGAug probs
        ba_mixup_scale=0.6,  # mixup alpha & p
        ba_tcm_scale=0.6,  # tcm p
        schedule="cosine"
    )
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    model, evaluation = train_model(
        model,
        train_loader,
        test_loader,  # used as val_loader here
        optimizer,
        criterion,
        device,
        num_epochs,
        2,
        patience=17,
        save_dir=os.path.join(BASE_PATH, "checkpoints"),
        model_name=MODEL_NAME,
        scheduler=scheduler,
        scheduler_mode="max",
        use_wandb=True,
        project="ecg-ioh",
        run_name=MODEL_NAME,  # keep run name same as checkpoint name
        entity=None,  # or your W&B team/user
        class_names=["0", "1"],  # binary after your label fold
        log_cm_every=1,  # log CM each epoch
        watch_gradients=True,
        resume="allow",
        config_extra={
            "lr": 1e-4,
            "weight_decay": 1e-3,
            "scheduler": "ReduceLROnPlateau(min, factor=0.5, patience=3)",
            "criterion": "CrossEntropyLoss",
            "architecture": "CNNLSTMModel",
            "cnn_out_dim": 128,
            "lstm_hidden": 64,
            "num_blocks": (1, 1),
            "label_rule": "labels<3 -> 1 else 0",
            "seq_hypers": {"T": 16, "win_sec": 90},
        },
        augmenter=augmenter,
        per_sample_aug=per_sample_aug,
        anneal=anneal,
        sample_weight=False,
        eval_every=2
    )

    with open(os.path.join(BASE_PATH, f"{MODEL_NAME}_metrics.json"), "w") as f:
        json.dump(evaluation, f)


if __name__ == '__main__':
    main()
