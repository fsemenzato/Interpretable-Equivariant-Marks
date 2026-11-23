from __future__ import annotations
from pathlib import Path
from typing import List, Sequence
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SimDataset(Dataset):
    def __init__(self, paths: Sequence[Path]):
        self.paths = [str(p) for p in paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        arr = np.load(self.paths[i]).astype(np.float32, copy=False)
        return torch.from_numpy(arr).unsqueeze(0)


def make_loader(
    paths: Sequence[Path], batch_size: int, num_workers: int, device: torch.device
) -> DataLoader:
    return DataLoader(
        SimDataset(paths),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def moments(x: torch.Tensor):
    xf = x.reshape(x.shape[0], -1).double()
    mu = xf.mean(dim=1)
    c = xf - mu[:, None]
    var = (c**2).mean(dim=1)
    skew = (c**3).mean(dim=1) / var.clamp(min=1e-30) ** 1.5
    return (mu, var, skew)


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
