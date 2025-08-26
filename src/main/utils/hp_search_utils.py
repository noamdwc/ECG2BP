import os

import optuna
import torch
import torch.nn as nn

from .augmentaions import BatchAugmenter, ECGAug
from .constants import BASE_PATH
from .models import CNNLSTMModel
from .training_utils import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_optimizer(model, opt_kind, lr, wd, momentum=None, betas=(0.9, 0.999)):
    if opt_kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
    elif opt_kind == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
    else:  # sgd
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum or 0.9, nesterov=True)


def objective(trial: optuna.Trial):
    # ---------- Multi-fidelity ----------
    max_epochs = trial.suggest_int("max_epochs", 6, 28)
    data_fraction = trial.suggest_float("data_fraction", 0.25, 1.0)
    batch_size = trial.suggest_categorical("batch_size", [8, 12, 16])
    num_workers = trial.suggest_categorical("num_workers", [2, 4])

    # ---------- Model ----------
    cnn_out_dim = trial.suggest_categorical("cnn_out_dim", [32, 64, 128])
    lstm_hidden = trial.suggest_categorical("lstm_hidden", [64, 96, 128])
    num_blocks = trial.suggest_categorical("num_blocks", [(1, 1), (2, 1), (2, 2)])
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.0, 0.5)

    # ---------- Optimization ----------
    opt_kind = trial.suggest_categorical("optimizer", ["adamw", "adam"])
    lr = trial.suggest_float("lr", 3e-5, 3e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    momentum = trial.suggest_float("momentum", 0.7, 0.98)

    # ---------- EMA ----------
    use_ema = trial.suggest_categorical("ema", [False, True])

    # ---------- ConditionalPairAugmenter (mixup + TCM) ----------
    mixup_on = trial.suggest_categorical("mixup_on", [False, True])
    tcm_on = trial.suggest_categorical("tcm_on", [False, True])
    mixup_alpha = trial.suggest_float("mixup_alpha", 0.1, 0.6) if mixup_on else 0.0
    mixup_p = trial.suggest_float("mixup_p", 0.05, 0.5) if mixup_on else 0.0
    tcm_p = trial.suggest_float("tcm_p", 0.05, 0.5) if tcm_on else 0.0
    tcm_lo = trial.suggest_float("tcm_len_lo_s", 0.15, 0.5) if tcm_on else 0.2
    tcm_hi = trial.suggest_float("tcm_len_hi_s", max(0.2, tcm_lo + 0.05), 0.8) if tcm_on else 0.5
    label_smooth = trial.suggest_float("label_smoothing", 0.0, 0.08)

    augmenter = BatchAugmenter(
        use_mixup=mixup_on, mixup_alpha=mixup_alpha, mixup_p=mixup_p, mixup_per_sample=True,
        use_timecutmix=tcm_on, tcm_p=tcm_p, tcm_len_s=(tcm_lo, tcm_hi), tcm_per_sample=True,
        mutually_exclusive=True, fs=500, label_smoothing=label_smooth
    )

    # ---------- Light-weight per-sample ECG aug (toggle a few) ----------
    p_jitter = trial.suggest_float("p_jitter", 0.3, 0.7)
    sj_lo = trial.suggest_float("sigma_rel_lo", 0.006, 0.03)
    sj_hi = trial.suggest_float("sigma_rel_hi", max(sj_lo + 1e-3, 0.02), 0.12)
    p_gain = trial.suggest_float("p_gain", 0.3, 0.7)
    g_lo = trial.suggest_float("gain_lo", 0.92, 0.99)
    g_hi = trial.suggest_float("gain_hi", max(g_lo + 0.005, 1.00), 1.10)
    p_shift = trial.suggest_float("p_shift", 0.3, 0.7)
    max_shift_ms = trial.suggest_int("max_shift_ms", 5, 15)
    p_mask = trial.suggest_float("p_mask", 0.15, 0.40)

    per_sample_aug = ECGAug(
        fs=500,
        p_jitter=p_jitter, sigma_rel=(sj_lo, sj_hi),
        p_gain=p_gain, gain_range=(g_lo, g_hi),
        p_shift=p_shift, max_shift_ms=max_shift_ms,
        p_mask=p_mask, n_masks=(0, 2), mask_ms=(10, 60),
        p_wander=0.3, wander_hz=(0.05, 0.5), wander_amp_rel=(0.02, 0.12),
        p_hum=0.0
    )

    # ---------- Loaders @ this fidelity ----------
    train_loader, val_loader = ...  # TODO: build make_loaders with train-val split
    # make_loaders(
    #     train_segments_paths, train_labels_path,
    #     val_segments_paths, val_labels_path,
    #     batch_size=batch_size, num_workers=num_workers, data_fraction=data_fraction,
    #     seed=trial.number + 123
    # )

    # ---------- Model/opt/scheduler ----------
    model = CNNLSTMModel(
        cnn_out_dim=cnn_out_dim, lstm_hidden=lstm_hidden,
        num_classes=2, num_blocks=num_blocks, lstm_dropout=lstm_dropout
    ).to(device)

    optimizer = _build_optimizer(model, opt_kind, lr, wd, momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # Don’t spam W&B during search; flip on only in final retrains
    model, evaluation = train_model(
        model, train_loader, val_loader, optimizer,
        nn.CrossEntropyLoss(), device,
        num_epochs=max_epochs, num_classes=2,
        patience=10, save_dir=os.path.join(BASE_PATH, "checkpoints"),
        model_name=f"hp_trial_{trial.number}",
        scheduler=scheduler, scheduler_mode="max",
        use_wandb=False,  # keep off for HPO
        augmenter=augmenter, anneal=None,
        sample_weight=False, is_ema=use_ema,
        eval_every=1, trial=trial
    )

    # Use the best macro-F1 your trainer already tracks
    return evaluation["best_val_macro_f1"]
