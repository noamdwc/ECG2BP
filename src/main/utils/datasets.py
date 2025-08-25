import numpy as np
import torch
from torch.utils.data import Dataset


# A dataset after segmenting and resampling
class SplitDataset(Dataset):
    def __init__(self, segments_paths, labels_path, aug=None, fourier_transform=None):
        self.labels = np.load(labels_path)
        self.paths = segments_paths
        self.parts = None  # opened lazily in each worker
        self.lens = None
        self.cumsum = None
        self.total = None
        self.aug = aug
        self.fourier_transform = fourier_transform

    def _ensure_open(self):
        if self.parts is None:
            # this runs inside the worker process
            self.parts = [np.load(p, mmap_mode='r') for p in self.paths]
            self.lens = [part.shape[0] for part in self.parts]
            self.cumsum = np.cumsum([0] + self.lens)
            self.total = int(self.cumsum[-1])
            # assert len(self.labels) == self.total, "labels must match total segments"

    def __len__(self):
        self._ensure_open()
        return self.total

    def __getitem__(self, idx):
        self._ensure_open()
        part_idx = np.searchsorted(self.cumsum, idx, side='right') - 1
        local_idx = idx - self.cumsum[part_idx]
        seg = self.parts[part_idx][local_idx]
        x = torch.tensor(seg, dtype=torch.float32)
        y = torch.as_tensor(self.labels[idx], dtype=torch.long)

        if self.aug is not None:
            with torch.no_grad():
                x = self.aug(x).contiguous()

        if self.fourier_transform is not None:
            x = self.fourier_transform(x)

        return x, y
