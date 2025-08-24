import math
import random
from dataclasses import dataclass
from functools import partial
from typing import Iterable, Tuple, Set, Dict, List

import torch
import torch.nn.functional as F


class BatchAugmenter:
    """
    Probabilistic Mixup & TimeCutMix for CE classifiers.

    Usage:
      augmenter = BatchAugmenter(use_mixup=True, mixup_alpha=0.3, mixup_p=0.15,
                                 use_timecutmix=True, tcm_p=0.15, tcm_len_s=(0.5, 2.0),
                                 mutually_exclusive=True, fs=500)

      x_out, y_a, y_b, lam, aug_name, mixed_frac = augmenter(x, y)
      logits = model(x_out)
      loss = BatchAugmenter.ce_loss(logits, y_a, y_b, lam)

    Returns:
      x_out: augmented batch (same shape as x)
      y_a, y_b: paired labels (for mixed loss)
      lam: (B,) mixing weights (1 means no mix for that sample)
      aug_name: "mixup" | "timecutmix" | "none"
      mixed_frac: fraction of samples actually mixed in the batch
    """

    def __init__(self,
                 # Mixup
                 use_mixup=True, mixup_alpha=0.3, mixup_p=0.15, mixup_per_sample=False,
                 # TimeCutMix
                 use_timecutmix=True, tcm_p=0.15, tcm_len_s=(0.5, 2.0), tcm_per_sample=False,
                 # Strategy
                 mutually_exclusive=True,
                 # Signal params
                 fs=500,
                 # Loss params
                 label_smoothing=0.0):
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.mixup_p = mixup_p
        self.mixup_per_sample = mixup_per_sample

        self.use_tcm = use_timecutmix
        self.tcm_p = tcm_p
        self.tcm_len_s = tcm_len_s
        self.tcm_per_sample = tcm_per_sample

        self.mutually_exclusive = mutually_exclusive
        self.fs = fs
        self.label_smoothing = label_smoothing

    # ------------ internals ------------
    def _mixup(self, x, y, alpha, p, per_sample):
        B = x.size(0);
        device = x.device
        if B < 2:
            lam = torch.ones(B, device=device)
            return x, y, y, lam, 0.0

        idx = torch.randperm(B, device=device)
        lam = torch.distributions.Beta(alpha, alpha).sample((B,)).to(device)
        lam = torch.maximum(lam, 1 - lam)  # symmetry

        if per_sample:
            m = (torch.rand(B, device=device) < p).float()
            lam = lam * m + (1 - m)  # if not selected -> lam=1 (no-op)
        else:
            if random.random() >= p:  # batch-level gate
                lam = torch.ones(B, device=device)

        view = (B, *([1] * (x.dim() - 1)))
        x_out = lam.view(*view) * x + (1 - lam).view(*view) * x[idx]
        mixed_frac = float((lam < 1).float().mean().item())
        return x_out, y, y[idx], lam, mixed_frac

    def _timecutmix(self, x, y, p, len_s, per_sample):
        B = x.size(0);
        T = x.shape[-1];
        device = x.device
        if B < 2:
            lam = torch.ones(B, device=device)
            return x, y, y, lam, 0.0

        idx = torch.randperm(B, device=device)

        if not per_sample:
            if random.random() >= p:
                return x, y, y, torch.ones(B, device=device), 0.0
            L = int(random.uniform(*len_s) * self.fs);
            L = max(1, min(L, T - 1))
            s = random.randint(0, T - L)
            x2 = x.clone()
            x2[..., s:s + L] = x[idx, ..., s:s + L]
            lam = torch.full((B,), 1 - L / float(T), device=device)
            return x2, y, y[idx], lam, 1.0

        # per-sample gate
        m = (torch.rand(B, device=device) < p)
        if not m.any():
            return x, y, y, torch.ones(B, device=device), 0.0

        x2 = x.clone()
        lam = torch.ones(B, device=device)
        for i in torch.nonzero(m, as_tuple=False).view(-1):
            j = idx[i].item()
            L = int(random.uniform(*len_s) * self.fs);
            L = max(1, min(L, T - 1))
            s = random.randint(0, T - L)
            x2[i, ..., s:s + L] = x[j, ..., s:s + L]
            lam[i] = 1 - L / float(T)
        mixed_frac = float((lam < 1).float().mean().item())
        return x2, y, y[idx], lam, mixed_frac

    def __call__(self, x, y):
        """
        x: (B,T) or (B,C,T)
        y: (B,) class indices (long)
        """
        if self.mutually_exclusive and self.use_tcm and self.use_mixup:
            r = random.random()
            if r < self.tcm_p:
                x, y_a, y_b, lam, frac = self._timecutmix(
                    x, y, p=self.tcm_p if self.tcm_per_sample else 1.0,
                    len_s=self.tcm_len_s, per_sample=self.tcm_per_sample
                )
                aug = "timecutmix" if (lam < 1).any() else "none"
                return x, y_a, y_b, lam, aug, frac
            elif r < self.tcm_p + self.mixup_p:
                x, y_a, y_b, lam, frac = self._mixup(
                    x, y, alpha=self.mixup_alpha,
                    p=self.mixup_p if self.mixup_per_sample else 1.0,
                    per_sample=self.mixup_per_sample
                )
                aug = "mixup" if (lam < 1).any() else "none"
                return x, y_a, y_b, lam, aug, frac
            else:
                return x, y, y, torch.ones(x.size(0), device=x.device), "none", 0.0

        # independent gates (may apply none, one, or both in sequence)
        applied = "none";
        frac = 0.0
        y_a = y;
        y_b = y
        lam = torch.ones(x.size(0), device=x.device)

        if self.use_tcm:
            x, y_a, y_b, lam, frac_t = self._timecutmix(x, y, p=self.tcm_p, len_s=self.tcm_len_s,
                                                        per_sample=self.tcm_per_sample)
            if (lam < 1).any(): applied, frac = "timecutmix", max(frac, frac_t)

        if self.use_mixup:
            x, y_a, y_b, lam_mu, frac_m = self._mixup(x, y_a, alpha=self.mixup_alpha, p=self.mixup_p,
                                                      per_sample=self.mixup_per_sample)
            if (lam_mu < 1).any(): applied, frac = "mixup", max(frac, frac_m)
            lam = lam_mu  # use last non-trivial lam

        return x, y_a, y_b, lam, applied, frac

    def ce_loss(self, logits, y_a, y_b, lam, sample_weights=None):
        """
        Mixed CE loss if any lam<1 else plain CE.
        Supports logits of shape (B,C) or (B,T,C).
        """
        if logits.dim() == 3:  # (B,T,C)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            y_a = y_a.view(-1)
            y_b = y_b.view(-1)
            lam = lam.view(-1).repeat_interleave(T)
            if sample_weights is not None:
                sample_weights = sample_weights.view(-1).repeat_interleave(T)

        ce_none = partial(F.cross_entropy, label_smoothing=self.label_smoothing, reduction="none")

        if torch.all(lam == 1):
            loss = ce_none(logits, y_a)
            if sample_weights is not None: loss = loss * sample_weights
            return loss.mean()

        loss_a = ce_none(logits, y_a)  # (N,)
        loss_b = ce_none(logits, y_b)  # (N,)
        loss = lam * loss_a + (1 - lam) * loss_b
        if sample_weights is not None: loss = loss * sample_weights
        return loss.mean()


class ConditionalPairAugmenter:
    """
    Same API as BatchAugmenter, but only pairs samples whose (y[i], y[j]) is in allowed_pairs.
    Works with y of shape (B,) or (B, ...). If multi-dim, a per-sample primary label is computed
    (mode over all positions) on GPU.

    Returns:
      x_out, y_a, y_b, lam, aug_name, mixed_frac
    """

    def __init__(self,
                 allowed_pairs: Iterable[Tuple[int, int]],
                 # Mixup
                 use_mixup: bool = True, mixup_alpha: float = 0.3, mixup_p: float = 0.15,
                 mixup_per_sample: bool = False,
                 # TimeCutMix
                 use_timecutmix: bool = True, tcm_p: float = 0.15, tcm_len_s: Tuple[float, float] = (0.5, 2.0),
                 tcm_per_sample: bool = False,
                 # Strategy
                 mutually_exclusive: bool = True,
                 # Signal params
                 fs: int = 500,
                 # Loss params
                 label_smoothing: float = 0.0,
                 # partner search
                 partner_resample_tries: int = 8):
        self.allowed_pairs: Set[Tuple[int, int]] = set(allowed_pairs)

        # Precompute map: src_class -> list[target_classes]
        targets: Dict[int, List[int]] = {}
        for c1, c2 in self.allowed_pairs:
            targets.setdefault(int(c1), []).append(int(c2))
        self.allowed_targets: Dict[int, List[int]] = targets

        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.mixup_p = mixup_p
        self.mixup_per_sample = mixup_per_sample

        self.use_tcm = use_timecutmix
        self.tcm_p = tcm_p
        self.tcm_len_s = tcm_len_s
        self.tcm_per_sample = tcm_per_sample

        self.mutually_exclusive = mutually_exclusive
        self.fs = fs
        self.label_smoothing = label_smoothing
        self.partner_resample_tries = partner_resample_tries

    # ---------- helpers ----------
    def _primary_per_sample(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (B,) or (B, ...). Returns (B,) long primary label per sample.
        If multi-dim, takes per-sample mode on GPU.
        """
        if y.dim() == 1:
            return y.long()
        B = y.shape[0]
        yf = y.view(B, -1).long()
        primary = torch.empty(B, dtype=torch.long, device=y.device)
        # Loop over B (small), stays on GPU
        for i in range(B):
            vals, counts = torch.unique(yf[i], return_counts=True)
            primary[i] = vals[counts.argmax()]
        return primary

    def _build_idx_with_constraints(self, y: torch.Tensor) -> torch.Tensor:
        """
        Build partner index vector idx (B,) such that for each i,
        (primary[i], primary[idx[i]]) is in allowed_pairs if possible; else idx[i]=i.
        Torch-only, no CPU/NumPy conversion.
        """
        device = y.device
        primary = self._primary_per_sample(y)  # (B,)
        B = primary.size(0)

        # pools per class (on device)
        classes = torch.unique(primary)
        pools: Dict[int, torch.Tensor] = {int(c.item()): torch.where(primary == c)[0] for c in classes}

        idx = torch.arange(B, device=device)  # default self-pair (no-op)
        for i in range(B):
            ci = int(primary[i].item())
            targets = self.allowed_targets.get(ci, None)
            if not targets:
                continue
            # try a few times to avoid self-pair or empty pools
            ok = False
            for _ in range(self.partner_resample_tries):
                c2 = random.choice(targets)
                pool = pools.get(c2, None)
                if pool is None or pool.numel() == 0:
                    continue
                j = int(pool[torch.randint(pool.numel(), (1,), device=device)].item())
                if j != i:
                    idx[i] = j
                    ok = True
                    break
            # else keep self (no-op)
        return idx

    # --- Mixup & TCM using constrained idx ---
    def _mixup(self, x, y, alpha, p, per_sample):
        B = x.size(0);
        device = x.device
        if B < 2:
            lam = torch.ones(B, device=device)
            return x, y, y, lam, 0.0

        idx = self._build_idx_with_constraints(y)
        lam = torch.distributions.Beta(alpha, alpha).sample((B,)).to(device)
        lam = torch.maximum(lam, 1 - lam)  # symmetry

        if per_sample:
            m = (torch.rand(B, device=device) < p).float()
            lam = lam * m + (1 - m)  # not selected → lam=1
        else:
            if random.random() >= p:
                lam = torch.ones(B, device=device)

        view = (B, *([1] * (x.dim() - 1)))
        x_out = lam.view(*view) * x + (1 - lam).view(*view) * x[idx]
        mixed_frac = float((lam < 1).float().mean().item())
        return x_out, y, y[idx], lam, mixed_frac

    def _timecutmix(self, x, y, p, len_s, per_sample):
        B = x.size(0);
        T = x.shape[-1];
        device = x.device
        if B < 2:
            lam = torch.ones(B, device=device)
            return x, y, y, lam, 0.0

        idx = self._build_idx_with_constraints(y)

        if not per_sample:
            if random.random() >= p:
                return x, y, y, torch.ones(B, device=device), 0.0
            L = int(random.uniform(*len_s) * self.fs);
            L = max(1, min(L, T - 1))
            s = random.randint(0, T - L)
            x2 = x.clone()
            x2[..., s:s + L] = x[idx, ..., s:s + L]
            lam = torch.full((B,), 1 - L / float(T), device=device)
            return x2, y, y[idx], lam, 1.0

        # per-sample gate
        m = (torch.rand(B, device=device) < p)
        if not m.any():
            return x, y, y, torch.ones(B, device=device), 0.0

        x2 = x.clone()
        lam = torch.ones(B, device=device)
        for i in torch.nonzero(m, as_tuple=False).view(-1):
            j = int(idx[i])
            if j == i:  # no valid partner → skip
                continue
            L = int(random.uniform(*len_s) * self.fs);
            L = max(1, min(L, T - 1))
            s = random.randint(0, T - L)
            x2[i, ..., s:s + L] = x[j, ..., s:s + L]
            lam[i] = 1 - L / float(T)
        mixed_frac = float((lam < 1).float().mean().item())
        return x2, y, y[idx], lam, mixed_frac

    # ---------- public API (identical to BatchAugmenter) ----------
    def __call__(self, x, y):
        """
        x: (B,T) or (B,C,T)
        y: (B,) class ids or (B, ...) dense labels
        """
        if self.mutually_exclusive and self.use_tcm and self.use_mixup:
            r = random.random()
            if r < self.tcm_p:
                x, y_a, y_b, lam, frac = self._timecutmix(
                    x, y, p=self.tcm_p if self.tcm_per_sample else 1.0,
                    len_s=self.tcm_len_s, per_sample=self.tcm_per_sample
                )
                aug = "timecutmix" if (lam < 1).any() else "none"
                return x, y_a, y_b, lam, aug, frac
            elif r < self.tcm_p + self.mixup_p:
                x, y_a, y_b, lam, frac = self._mixup(
                    x, y, alpha=self.mixup_alpha,
                    p=self.mixup_p if self.mixup_per_sample else 1.0,
                    per_sample=self.mixup_per_sample
                )
                aug = "mixup" if (lam < 1).any() else "none"
                return x, y_a, y_b, lam, aug, frac
            else:
                return x, y, y, torch.ones(x.size(0), device=x.device), "none", 0.0

        # independent gates
        applied = "none";
        frac = 0.0
        y_a = y;
        y_b = y
        lam = torch.ones(x.size(0), device=x.device)

        if self.use_tcm:
            x, y_a, y_b, lam_t, frac_t = self._timecutmix(x, y, p=self.tcm_p, len_s=self.tcm_len_s,
                                                          per_sample=self.tcm_per_sample)
            if (lam_t < 1).any(): applied, frac = "timecutmix", max(frac, frac_t)
            lam = lam_t

        if self.use_mixup:
            x, y_a, y_b, lam_m, frac_m = self._mixup(x, y_a, alpha=self.mixup_alpha, p=self.mixup_p,
                                                     per_sample=self.mixup_per_sample)
            if (lam_m < 1).any(): applied, frac = "mixup", max(frac, frac_m)
            lam = lam_m

        return x, y_a, y_b, lam, applied, frac

    # Same CE mixing as BatchAugmenter
    def ce_loss(self, logits, y_a, y_b, lam, sample_weights=None):
        if logits.dim() == 3:  # (B,T,C)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            y_a = y_a.view(-1)
            y_b = y_b.view(-1)
            lam = lam.view(-1).repeat_interleave(T)
            if sample_weights is not None:
                sample_weights = sample_weights.view(-1).repeat_interleave(T)

        ce_none = partial(F.cross_entropy, label_smoothing=self.label_smoothing, reduction="none")

        if torch.all(lam == 1):
            loss = ce_none(logits, y_a)
            if sample_weights is not None: loss = loss * sample_weights
            return loss.mean()

        loss_a = ce_none(logits, y_a)
        loss_b = ce_none(logits, y_b)
        loss = lam * loss_a + (1 - lam) * loss_b
        if sample_weights is not None: loss = loss * sample_weights
        return loss.mean()


class ECGAug:
    def __init__(self,
                 p_jitter=0.4, sigma_rel=(0.005, 0.02),  # smaller noise
                 p_gain=0.5, gain_range=(0.95, 1.05),  # tighter gain
                 p_shift=0.4, max_shift_ms=10, fs=500,  # <= ±10 ms
                 p_mask=0.25, n_masks=(0, 2), mask_ms=(10, 40),
                 p_wander=0.2, wander_hz=(0.05, 0.5), wander_amp_rel=(0.005, 0.015),
                 p_hum=0.2, hum_hz=50.0, hum_amp_rel=(0.002, 0.008)):
        self.p_jitter, self.sigma_rel = p_jitter, sigma_rel
        self.p_gain, self.gain_range = p_gain, gain_range
        self.p_shift, self.fs = p_shift, fs
        self.max_shift = int(max_shift_ms * fs / 1000)
        self.p_mask, self.n_masks, self.mask_ms = p_mask, n_masks, mask_ms
        self.p_wander, self.wander_hz, self.wander_amp_rel = p_wander, wander_hz, wander_amp_rel
        self.p_hum, self.hum_hz, self.hum_amp_rel = p_hum, hum_hz, hum_amp_rel

    def _bandlimit(self, noise: torch.Tensor, lp_hz=100):
        # quick low-pass: 9-tap Hann moving average
        k = torch.hann_window(9, periodic=False, device=noise.device).unsqueeze(0).unsqueeze(0)
        k = k / k.sum()
        return F.conv1d(noise[None, None, :], k, padding=4).squeeze()

    def __call__(self, x: torch.Tensor):
        # x: (T,) or (C,T). Use SAME params across channels.
        if x.dim() == 1:
            X = x.unsqueeze(0)  # (1,T)
        else:
            X = x

        T = X.size(-1)
        y = X.clone()

        # choose all random params once (shared across channels)
        std_ref = X.std(dim=-1, keepdim=True).mean().item() + 1e-8

        # jitter (band-limited)
        if random.random() < self.p_jitter:
            sigma = random.uniform(*self.sigma_rel) * std_ref
            n = torch.randn(T, device=y.device) * sigma
            n = self._bandlimit(n)
            y = y + n

        # gain
        if random.random() < self.p_gain:
            gain = random.uniform(*self.gain_range)
            y = y * gain

        # small time shift (circular; tiny so it won’t matter at ends)
        if random.random() < self.p_shift and self.max_shift > 0:
            k = random.randint(-self.max_shift, self.max_shift)
            if k != 0:
                y = torch.roll(y, shifts=k, dims=-1)

        # time masking: short, fill with local mean + noise (no hard zeros)
        if random.random() < self.p_mask:
            m = random.randint(*self.n_masks)
            for _ in range(m):
                w = int(random.uniform(*self.mask_ms) * 1e-3 * self.fs)
                if w <= 0 or w >= T: continue
                s = random.randint(0, T - w)
                seg = y[..., max(0, s - 5):min(T, s + w + 5)]
                mu = seg.mean(dim=-1, keepdim=True)
                noise = (torch.randn_like(y[..., s:s + w]) * 0.02 * std_ref)
                y[..., s:s + w] = mu + noise

        # baseline wander (tiny, random phase)
        if random.random() < self.p_wander:
            f = random.uniform(*self.wander_hz)
            amp = random.uniform(*self.wander_amp_rel) * std_ref
            phi = random.uniform(0, 2 * math.pi)
            t = torch.arange(T, device=y.device) / self.fs
            y = y + amp * torch.sin(2 * math.pi * f * t + phi)

        # powerline hum (tiny, random phase + slight detune)
        if random.random() < self.p_hum:
            amp = random.uniform(*self.hum_amp_rel) * std_ref
            f = self.hum_hz + random.uniform(-0.3, 0.3)
            phi = random.uniform(0, 2 * math.pi)
            t = torch.arange(T, device=y.device) / self.fs
            y = y + amp * torch.sin(2 * math.pi * f * t + phi)

        return y.squeeze(0) if x.dim() == 1 else y


#######################
# Augmentation Anneal #
#######################

@dataclass
class AnnealConfig:
    # final multipliers reached at the LAST epoch (1.0 = no anneal)
    ps_strength_scale: float = 0.5  # ECGAug strengths (sigma_rel, wander_amp_rel)
    ps_prob_scale: float = 0.6  # ECGAug probs (p_jitter, p_wander, p_mask, p_shift)
    ba_mixup_scale: float = 0.6  # BatchAugmenter mixup (alpha & p)
    ba_tcm_scale: float = 0.6  # BatchAugmenter timecutmix prob
    schedule: str = "linear"  # or "cosine"


def _phase(epoch: int, num_epochs: int) -> float:
    return 0.0 if num_epochs <= 1 else (epoch - 1) / (num_epochs - 1)


def _interp(a: float, b: float, t: float, schedule: str = "linear") -> float:
    if schedule == "cosine":
        w = 0.5 * (1 - math.cos(math.pi * t))  # 0→1 smooth
        return a + (b - a) * w
    return a + (b - a) * t  # linear


def _interp_pair(tup, scale, t, schedule):
    lo, hi = tup
    return (_interp(lo, lo * scale, t, schedule), _interp(hi, hi * scale, t, schedule))


# Capture bases
def _capture_aug_bases(augmenter, per_sample_aug):
    ps_base, ba_base = None, None
    if per_sample_aug is not None:
        ps_base = {
            "sigma_rel": tuple(per_sample_aug.sigma_rel),
            "wander_amp_rel": tuple(per_sample_aug.wander_amp_rel),
            "p_jitter": float(per_sample_aug.p_jitter),
            "p_wander": float(per_sample_aug.p_wander),
            "p_mask": float(per_sample_aug.p_mask),
            "p_shift": float(per_sample_aug.p_shift),
        }
    if augmenter is not None:
        ba_base = {
            "mixup_alpha": float(getattr(augmenter, "mixup_alpha", 0.0)),
            "mixup_p": float(getattr(augmenter, "mixup_p", 0.0)),
            "tcm_p": float(getattr(augmenter, "tcm_p", 0.0)),
        }
    return ps_base, ba_base


def _apply_anneal(per_sample_aug, augmenter, ps_base, ba_base, cfg: AnnealConfig, t):
    if ps_base is not None and per_sample_aug is not None:
        per_sample_aug.sigma_rel = _interp_pair(ps_base["sigma_rel"], cfg.ps_strength_scale, t, cfg.schedule)
        per_sample_aug.wander_amp_rel = _interp_pair(ps_base["wander_amp_rel"], cfg.ps_strength_scale, t, cfg.schedule)
        per_sample_aug.p_jitter = _interp(ps_base["p_jitter"], ps_base["p_jitter"] * cfg.ps_prob_scale, t, cfg.schedule)
        per_sample_aug.p_wander = _interp(ps_base["p_wander"], ps_base["p_wander"] * cfg.ps_prob_scale, t, cfg.schedule)
        per_sample_aug.p_mask = _interp(ps_base["p_mask"], ps_base["p_mask"] * cfg.ps_prob_scale, t, cfg.schedule)
        per_sample_aug.p_shift = _interp(ps_base["p_shift"], ps_base["p_shift"] * cfg.ps_prob_scale, t, cfg.schedule)

    if ba_base is not None and augmenter is not None:
        augmenter.mixup_alpha = _interp(ba_base["mixup_alpha"], ba_base["mixup_alpha"] * cfg.ba_mixup_scale, t,
                                        cfg.schedule)
        augmenter.mixup_p = _interp(ba_base["mixup_p"], ba_base["mixup_p"] * cfg.ba_mixup_scale, t, cfg.schedule)
        augmenter.tcm_p = _interp(ba_base["tcm_p"], ba_base["tcm_p"] * cfg.ba_tcm_scale, t, cfg.schedule)
