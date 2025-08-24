import os
import warnings
from copy import deepcopy

import torch
import wandb
from torch import amp
from tqdm import tqdm

from .augmentaions import BatchAugmenter, ConditionalPairAugmenter, AnnealConfig, _apply_anneal, _capture_aug_bases, \
    _phase
from .ema import EMA, save_with_ema
from .eval import evaluate_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _init_wandb(use_wandb, project, run_name, entity, resume, model,
                watch_gradients, base_cfg, augmenter, per_sample_aug):
    if not use_wandb:
        return
    wb_cfg = dict(base_cfg)
    wb_cfg["batch_augmenter"] = bool(augmenter)
    wb_cfg["per_sample_aug"] = bool(per_sample_aug)
    if augmenter is not None:
        for k in ["mutually_exclusive", "mixup_alpha", "mixup_p",
                  "mixup_per_sample", "tcm_p", "tcm_len_s", "tcm_per_sample"]:
            if hasattr(augmenter, k):
                wb_cfg[f"aug/{k}"] = getattr(augmenter, k)
    wandb.init(project=project, name=run_name, entity=entity, resume=resume, config=wb_cfg)
    if watch_gradients:
        wandb.watch(model, log="gradients", log_freq=100)


def _forward_and_loss(model, signals, labels, num_classes, criterion, augmenter,
                      subcat=None, is_sample_weight=False):
    if is_sample_weight and subcat is not None:
        sw = torch.ones_like(labels, device=device, dtype=torch.float32)
        focus_mask = (subcat == 1) | (subcat == 2)
        sw = torch.where(focus_mask, torch.full_like(sw, 2.0), sw)
    else:
        sw = None

    if augmenter is not None and num_classes != 1:
        if isinstance(augmenter, ConditionalPairAugmenter) and (subcat is not None):
            B = signals.size(0)
            subcat = subcat.view(B, -1).long()
            signals, y_a, y_b, lam, aug_name, mixed_frac = augmenter(signals, subcat)
        elif isinstance(augmenter, BatchAugmenter):
            signals, y_a, y_b, lam, aug_name, mixed_frac = augmenter(signals, labels)
        else:
            raise ValueError(f"Unknown augmenter type: {type(augmenter)}")

        logits = model(signals)
        loss = augmenter.ce_loss(logits, y_a, y_b, lam, sample_weights=sw)
        meta = {"aug_name": aug_name, "mixed_frac": mixed_frac}
    else:
        warnings.warn("No sample weight is applied")
        logits = model(signals)
        if num_classes == 1:
            loss = criterion(logits.view(-1).float(), labels.view(-1).float())
        else:
            loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
        meta = {"aug_name": "none", "mixed_frac": 0.0}
    return loss, logits, meta


def _log_epoch(use_wandb, epoch, optimizer, avg_train_loss, val_loss, val_macro_f1,
               val_per_class, class_names, y_true_np, y_pred_np,
               aug_counts, mixed_frac_sum, n_batches,
               per_sample_aug, augmenter, phase):
    if not use_wandb:
        return
    per_class_log = {f"val/acc_class_{cls}": acc for cls, acc in val_per_class.items()}
    pred_pos_pct = float((y_pred_np == 1).mean())
    payload = {
        "train/loss": avg_train_loss,
        "val/loss": val_loss,
        "val/macro_f1": val_macro_f1,
        "pred/pos_pct"" ": pred_pos_pct,
        "lr": optimizer.param_groups[0]["lr"],
        "anneal/phase": phase,
        **per_class_log
    }
    if augmenter is not None:
        payload.update({
            "aug/avg_mixed_frac": mixed_frac_sum / max(1, n_batches),
            "aug/batches_mixup": aug_counts.get("mixup", 0),
            "aug/batches_tcm": aug_counts.get("timecutmix", 0),
            "aug/batches_none": aug_counts.get("none", 0),
            "aug_ba/mixup_alpha": getattr(augmenter, "mixup_alpha", 0.0),
            "aug_ba/mixup_p": getattr(augmenter, "mixup_p", 0.0),
            "aug_ba/tcm_p": getattr(augmenter, "tcm_p", 0.0),
        })
    if per_sample_aug is not None:
        payload.update({
            "aug_ps/p_jitter": getattr(per_sample_aug, "p_jitter", 0.0),
            "aug_ps/p_wander": getattr(per_sample_aug, "p_wander", 0.0),
            "aug_ps/p_mask": getattr(per_sample_aug, "p_mask", 0.0),
            "aug_ps/p_shift": getattr(per_sample_aug, "p_shift", 0.0),
            "aug_ps/sigma_lo": getattr(per_sample_aug, "sigma_rel", (0, 0))[0],
            "aug_ps/sigma_hi": getattr(per_sample_aug, "sigma_rel", (0, 0))[1],
        })
    wandb.log(payload, step=epoch)


######################
# training function #
######################

def train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        num_epochs,
        num_classes,
        patience=20,
        save_dir="./checkpoints",
        model_name="model",
        scheduler=None,
        scheduler_mode="max",
        # wandb
        use_wandb=True,
        project="ecg-ioh",
        run_name=None,
        entity=None,
        class_names=None,
        log_cm_every=1,
        watch_gradients=True,
        resume="allow",
        config_extra=None,
        augmenter=None,
        per_sample_aug=None,
        anneal: AnnealConfig = None,
        sample_weight=False,
        is_ema=False,
        eval_every=1,
):
    os.makedirs(save_dir, exist_ok=True)

    if class_names is None:
        class_names = [str(i) for i in range(max(2, num_classes))]

    # init wandb
    base_cfg = {"epochs": num_epochs, "num_classes": num_classes,
                "patience": patience, "scheduler_mode": scheduler_mode,
                **(config_extra or {})}
    _init_wandb(use_wandb, project, run_name, entity, resume, model, watch_gradients,
                base_cfg, augmenter, per_sample_aug)

    scaler = amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    ps_base, ba_base = _capture_aug_bases(augmenter, per_sample_aug)
    best_val_macro_f1, best_epoch, best_model_state = 0.0, 0, None
    no_improve_epochs = 0
    t = 0.0
    loss_history, val_accuracies, val_loss_history, val_per_class_history, lr_hist = [], [], [], [], []

    if is_ema:
        steps_per_epoch = max(1, len(train_loader))
        half_life_epochs = 2.0
        ema_decay = 0.5 ** (1.0 / (half_life_epochs * steps_per_epoch))  # e.g., ~0.999
        ema = EMA(model, decay=ema_decay)

    for epoch in range(1, num_epochs + 1):
        model.train()
        # anneal current
        if anneal is not None:
            t = _phase(epoch, num_epochs)
            _apply_anneal(per_sample_aug, augmenter, ps_base, ba_base, anneal, t)

        aug_counts = {"mixup": 0, "timecutmix": 0, "none": 0}
        mixed_frac_sum, total_loss, n_batches = 0.0, 0.0, 0

        with tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False) as train_bar:
            for signals, labels in train_bar:
                # map labels once and move to device
                subcat = labels.clone().to(device)
                labels = torch.where(labels < 3, 1, 0).long().to(device, non_blocking=True)
                signals = signals.to(device, non_blocking=True)

                if per_sample_aug is not None:
                    with torch.no_grad():
                        signals = torch.stack([per_sample_aug(s) for s in signals], dim=0)
                        signals = signals.contiguous()

                optimizer.zero_grad(set_to_none=True)
                with amp.autocast('cuda', dtype=torch.float16):
                    loss, logits, meta = _forward_and_loss(model,
                                                           signals,
                                                           labels,
                                                           num_classes,
                                                           criterion,
                                                           augmenter,
                                                           subcat=subcat,
                                                           is_sample_weight=sample_weight)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # unscale BEFORE clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if is_ema:
                    ema.update(model)

                total_loss += loss.item()
                if augmenter is not None:
                    aug_counts[meta["aug_name"]] = aug_counts.get(meta["aug_name"], 0) + 1
                    mixed_frac_sum += float(meta["mixed_frac"])
                    n_batches += 1

                train_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / max(1, len(train_loader))
        loss_history.append(avg_train_loss)

        # ---- evaluation ----
        if (epoch % eval_every) != 0:
            continue

        model.eval()
        if is_ema:
            ema.store(model)
            ema.copy_to(model)
        val_macro_f1, val_per_class, val_loss, y_true_np, y_pred_np = evaluate_model(
            model, val_loader, num_classes, device, criterion
        )
        if is_ema:
            ema.restore(model)
        val_accuracies.append(val_macro_f1)
        val_per_class_history.append(val_per_class)
        val_loss_history.append(val_loss)

        print(
            f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, Val macro_f1 = {val_macro_f1:.4f}")
        for cls, acc in val_per_class.items():
            print(f"  Class {cls} Accuracy: {acc:.2%}")

        # wandb logging per epoch
        _log_epoch(use_wandb, epoch, optimizer, avg_train_loss, val_loss, val_macro_f1,
                   val_per_class, class_names, y_true_np, y_pred_np,
                   aug_counts, mixed_frac_sum, n_batches,
                   per_sample_aug, augmenter, t)
        if use_wandb and is_ema:
            wandb.config.update({"ema_decay": ema.decay}, allow_val_change=True)

        # Checkpoint (latest)
        latest_path = os.path.join(save_dir, f"latest_{model_name}.pt")
        if is_ema:
            save_with_ema(model, latest_path, ema)
        else:
            torch.save(model.state_dict(), latest_path)

        # Checkpoint (best) + artifact
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_model_state = deepcopy(model.state_dict())
            best_path = os.path.join(save_dir, f"best_{model_name}.pt")
            if is_ema:
                save_with_ema(model, best_path, ema)
            else:
                torch.save(model.state_dict(), best_path)
            no_improve_epochs = 0

            if use_wandb:
                wandb.run.summary["best_val_macro_f1"] = best_val_macro_f1
                wandb.run.summary["best_epoch"] = best_epoch
                art = wandb.Artifact(f"{model_name}-weights", type="model")
                art.add_file(best_path)
                wandb.log_artifact(art, aliases=["best", f"epoch-{epoch}"])
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"⏹️ Early stopping at epoch {epoch} after {patience} epochs without improvement.")
                break

        # Scheduler step
        if scheduler is not None:
            if scheduler_mode == "max":
                scheduler.step(val_macro_f1)
            elif scheduler_mode == "min":
                scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"📉 Learning Rate: {current_lr:.6f}")
        lr_hist.append(current_lr)

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    if use_wandb:
        wandb.finish()

    return model, {
        "train_loss": loss_history,
        "val_loss": val_loss_history,
        "val_accuracy": val_accuracies,
        "val_per_class": val_per_class_history,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_macro_f1,
        "lr_history": lr_hist
    }
