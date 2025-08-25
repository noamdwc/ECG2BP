import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTFrontend(nn.Module):
    """
    Converts (B, T, L) time windows -> (B, T, F) frequency features.

    - rFFT with optional band-limit [fmin, fmax] Hz
    - optional notch around 50/60 Hz
    - log(1 + power) scaling
    - optional z-score per (sample, frame)
    - optional fixed-length interpolation to n_bins to stabilize input length
    """

    def __init__(self,
                 fs: float = 500.0,
                 fmin: float = 0.0,
                 fmax: float = 100.0,
                 n_bins: int | None = 1024,
                 log_power: bool = True,
                 remove_dc: bool = True,
                 notch_hz: list[float] | None = (50.0, 60.0),
                 notch_bw: float = 1.0,  # ±bw around each notch_hz
                 normalize: str | None = "zscore",  # "zscore" | None
                 eps: float = 1e-8):
        super().__init__()
        self.fs = float(fs)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.n_bins = n_bins
        self.log_power = log_power
        self.remove_dc = remove_dc
        self.notch_hz = list(notch_hz) if notch_hz else None
        self.notch_bw = float(notch_bw)
        self.normalize = normalize
        self.eps = eps

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, L) or (B, L) float
        returns: (B, T, F) or (B, F) float
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # -> (B, 1, L)
            squeeze_back = True
        else:
            squeeze_back = False

        B, T, L = x.shape
        # rFFT -> (B, T, L//2 + 1) complex
        X = torch.fft.rfft(x, dim=-1, norm="ortho")

        # Build frequency mask on the fly for this L
        freqs = torch.fft.rfftfreq(L, d=1.0 / self.fs).to(x.device)  # (L//2+1,)
        mask = (freqs >= self.fmin) & (freqs <= self.fmax)
        if self.remove_dc:
            mask &= (freqs > 1e-9)

        if self.notch_hz:
            for h in self.notch_hz:
                mask &= ~((freqs >= (h - self.notch_bw)) & (freqs <= (h + self.notch_bw)))

        X = X[..., mask]  # (B, T, F_sel)
        spec = (X.real ** 2 + X.imag ** 2)  # power
        if self.log_power:
            spec = torch.log1p(spec)

        # Optional fixed-length interpolation to stabilize input length
        if self.n_bins is not None:
            spec = F.interpolate(
                spec.reshape(B * T, 1, spec.shape[-1]),
                size=self.n_bins,
                mode="linear",
                align_corners=False
            ).reshape(B, T, self.n_bins)

        # Optional per-(B,T) z-score across frequency axis
        if self.normalize == "zscore":
            mu = spec.mean(dim=-1, keepdim=True)
            sd = spec.std(dim=-1, keepdim=True).clamp_min(self.eps)
            spec = (spec - mu) / sd

        if squeeze_back:
            spec = spec.squeeze(1)  # (B, F) for single-frame inputs
        return spec


class WaveletDWTTransform(nn.Module):
    """
    x: (T, L) torch.FloatTensor  ->  (T, n_feats) torch.FloatTensor
    Per-frame multilevel DWT using 'wavelet' (e.g., 'db4'), outputs log-power per level.
    """

    def __init__(self, fs=500.0, wavelet="db4", level=None,
                 include_approx=False, log_power=True, normalize="zscore", eps=1e-8):
        super().__init__()
        self.fs = float(fs)
        self.wavelet = wavelet
        self.level = level  # None -> pywt.dwt_max_level
        self.include_approx = include_approx
        self.log_power = log_power
        self.normalize = normalize
        self.eps = eps

    def _frame_feats(self, x_np):
        # x_np: (L,) float64/float32
        max_level = pywt.dwt_max_level(len(x_np), pywt.Wavelet(self.wavelet).dec_len)
        L = self.level if self.level is not None else max_level
        coeffs = pywt.wavedec(x_np, self.wavelet, level=L, mode="periodization")
        # coeffs = [cA_L, cD_L, cD_{L-1}, ..., cD_1]
        feats = []
        # detail bands (higher -> lower freq)
        for c in coeffs[1:]:
            p = float(np.mean(c * c))  # power
            feats.append(p)
        # optional approximation band
        if self.include_approx:
            cA = coeffs[0]
            feats.append(float(np.mean(cA * cA)))
        feats = np.asarray(feats, dtype=np.float32)  # (n_levels [+1])
        if self.log_power:
            feats = np.log1p(feats)
        return feats

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, L) torch
        T, L = x.shape
        feats = []
        x_np = x.detach().cpu().numpy()
        for t in range(T):
            feats.append(self._frame_feats(x_np[t]))
        feats = np.stack(feats, axis=0)  # (T, n_feats)

        feats = torch.from_numpy(feats)
        if self.normalize == "zscore":
            mu = feats.mean(dim=-1, keepdim=True)
            sd = feats.std(dim=-1, keepdim=True).clamp_min(self.eps)
            feats = (feats - mu) / sd
        return feats.to(torch.float32)
