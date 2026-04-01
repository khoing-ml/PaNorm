from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from PIL import Image
except ImportError:
    Image = None


_CIFAR10C_URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
_CIFAR10C_DIR_CANDIDATES = ("CIFAR-10-C", "cifar10c", "cifar-10-c")


def ensure_cifar10c(data_root: str) -> Path:

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    for name in _CIFAR10C_DIR_CANDIDATES:
        candidate = root / name
        if (candidate / "labels.npy").exists():
            return candidate

    tar_path = root / "CIFAR-10-C.tar"
    tmp_path = tar_path.with_suffix(".tar.partial")

    if not tar_path.exists():
        import urllib.request

        print(f"[cifar10c] downloading to {tar_path} ...")
        urllib.request.urlretrieve(_CIFAR10C_URL, tmp_path)
        tmp_path.rename(tar_path)

    print(f"[cifar10c] extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(root)

    for name in _CIFAR10C_DIR_CANDIDATES:
        candidate = root / name
        if (candidate / "labels.npy").exists():
            return candidate

    raise RuntimeError(f"Failed to prepare CIFAR-10-C under {data_root}")


@dataclass(frozen=True)
class Cifar10CConfig:
    severity: int = 2
    corruptions: tuple[str, ...] = (
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    )


class CIFAR10CDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        corruption: str,
        severity: int,
        transform=None,
    ):
        if Image is None:
            raise ImportError("PIL is required to load CIFAR-10-C images.")

        self.root = Path(root)
        self.corruption = corruption
        self.severity = int(max(1, min(5, severity)))
        self.transform = transform

        labels_path = self.root / "labels.npy"
        images_path = self.root / f"{corruption}.npy"
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels.npy under {self.root}")
        if not images_path.exists():
            raise FileNotFoundError(
                f"Missing corruption file under {self.root}: {images_path.name}"
            )

        self._labels = np.load(labels_path, mmap_mode="r")
        self._images = np.load(images_path, mmap_mode="r")

        start = (self.severity - 1) * 10_000
        end = self.severity * 10_000
        self._slice = slice(start, end)

    def __len__(self) -> int:
        return 10_000

    def __getitem__(self, idx: int):
        idx = int(idx)
        real_idx = self._slice.start + idx
        img = self._images[real_idx]
        target = int(self._labels[real_idx])
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def _collect_gate_stats_from_model(model: nn.Module) -> dict[str, float]:

    gate_modules = [m for m in model.modules() if hasattr(m, "gate_statistics")]
    if not gate_modules:
        return {}
    stats = [m.gate_statistics() for m in gate_modules]
    keys = stats[0].keys()
    merged = {}
    for key in keys:
        merged[key] = float(np.mean([s[key] for s in stats]))
    return merged


def _reset_gate_stats(model: nn.Module) -> None:

    for m in model.modules():
        if hasattr(m, "reset_gate_statistics"):
            try:
                m.reset_gate_statistics()
            except Exception:
                pass


@torch.no_grad()
def evaluate_cifar10c(
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    data_root: str,
    batch_size: int,
    num_workers: int,
    config: Cifar10CConfig,
    collect_per_corruption_gates: bool = False,
) -> dict[str, float]:

    root = None
    for name in _CIFAR10C_DIR_CANDIDATES:
        candidate = Path(data_root) / name
        if (candidate / "labels.npy").exists():
            root = candidate
            break
    if root is None:
        raise FileNotFoundError(
            f"CIFAR-10-C not found under {data_root}. Run ensure_cifar10c(data_root) before evaluation."
        )

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    tf = transforms.Compose([transforms.ToTensor(), normalize])

    def _eval_loader(loader: DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device, dtype=torch.long)
            logits = model(x)
            _ = criterion(logits, y)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
        if total == 0:
            return 0.0
        return 100.0 * correct / total

    accs: list[float] = []
    per_corruption_gates: dict[str, dict[str, float]] = {}

    for corruption in config.corruptions:
        if collect_per_corruption_gates:
            _reset_gate_stats(model)

        ds = CIFAR10CDataset(
            root=root, corruption=corruption, severity=config.severity, transform=tf
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=num_workers > 0,
        )
        accs.append(_eval_loader(loader))

        if collect_per_corruption_gates:
            gate_stats = _collect_gate_stats_from_model(model)
            per_corruption_gates[corruption] = gate_stats

    mean_acc = float(np.mean(accs)) if accs else 0.0
    worst_acc = float(np.min(accs)) if accs else 0.0
    result = {
        "ood_acc1_cifar10c_mean": mean_acc,
        "ood_acc1_cifar10c_worst": worst_acc,
    }

    if collect_per_corruption_gates:

        for i, corruption in enumerate(config.corruptions):
            result[f"ood_acc1_{corruption}"] = accs[i]
            for gk, gv in per_corruption_gates.get(corruption, {}).items():
                result[f"ood_{corruption}_{gk}"] = gv

    return result


def summarize_corruptions(accs: Iterable[float]) -> tuple[float, float]:
    values = list(accs)
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.min(values))
