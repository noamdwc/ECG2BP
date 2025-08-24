import torch
import wandb
from sklearn.metrics import confusion_matrix, f1_score
from torch import amp
from tqdm import tqdm


def log_confmat_to_wandb(y_true_np, y_pred_np, class_names):
    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(len(class_names))))
    wandb.log({
        "val/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true_np,
            preds=y_pred_np,
            class_names=class_names
        )
    })
    return cm


def accuracy_per_class(preds, labels, num_classes):
    from collections import defaultdict
    correct = defaultdict(int)
    total = defaultdict(int)

    preds = preds.view(-1).cpu()
    labels = labels.view(-1).cpu()

    for p, y in zip(preds, labels):
        total[int(y)] += 1
        if int(p) == int(y):
            correct[int(y)] += 1

    acc_dict = {}
    for cls in range(num_classes):
        if total[cls] > 0:
            acc_dict[cls] = correct[cls] / total[cls]
        else:
            acc_dict[cls] = float('nan')  # no samples
    return acc_dict, total


@torch.no_grad()
def evaluate_model(model, val_loader, num_classes, device, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    # if device.type == "cuda":
    #     ac = amp.autocast("cuda", dtype=torch.float16)
    # elif device.type == "cpu":
    #     ac = amp.autocast("cpu", dtype=torch.bfloat16)
    # else:
    #     ac = nullcontext()

    for signals, labels in tqdm(val_loader, desc=f"eval stage", leave=False):
        signals = signals.to(device, non_blocking=True)
        labels = torch.where(labels < 3, 1, 0)
        labels = labels.to(device, non_blocking=True)
        with amp.autocast('cuda', dtype=torch.float16):
            logits = model(signals)  # (B, T, num_classes)
            if num_classes == 1:
                loss = criterion(logits.view(-1).float(), labels.view(-1).float())
            else:
                loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
            total_loss += loss.item()

        if num_classes == 1:
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().squeeze(-1)
        else:
            preds = torch.argmax(logits, dim=-1)
        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0).reshape(-1)
    all_labels = torch.cat(all_labels, dim=0).reshape(-1)

    per_class_acc, class_counts = accuracy_per_class(all_preds, all_labels, num_classes)

    all_preds_np = all_preds.cpu().numpy()
    all_labels_np = all_labels.cpu().numpy()
    macro_f1 = f1_score(all_labels_np, all_preds_np, average='macro')
    avg_loss = total_loss / len(val_loader)

    return macro_f1, per_class_acc, avg_loss, all_labels_np, all_preds_np


@torch.no_grad()
def plot_confusion_matrix(model, dataloader, device, num_classes, label_names=None, normalize='true',
                          return_probs=False):
    # (left as-is except we now actually return preds/labels; cm handled by helper above)
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_sub_labels = []

    for signals, labels in tqdm(dataloader, desc='cm stage', leave=False):
        signals = signals.to(device)
        all_sub_labels.append(labels.cpu())
        labels = torch.where(labels < 3, 1, 0)
        labels = labels.to(device)

        logits = model(signals)
        if num_classes == 1:
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().squeeze(-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        if return_probs:
            all_probs.append(probs.cpu())

    y_pred = torch.cat(all_preds, dim=0).numpy().reshape(-1)
    y_true = torch.cat(all_labels, dim=0).numpy().reshape(-1)
    y_sub_true = torch.cat(all_sub_labels, dim=0).numpy().reshape(-1)
    if return_probs:
        y_probs = torch.cat(all_probs, dim=0).numpy().reshape(-1, num_classes)
    cm = None
    if return_probs:
        return cm, y_pred, y_true, y_probs, y_sub_true
    return cm, y_pred, y_true
