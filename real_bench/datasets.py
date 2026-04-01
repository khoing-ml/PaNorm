from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    num_classes: int
    in_channels: int
    default_image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


DATASET_INFO = {
    "cifar10": DatasetInfo(
        name="cifar10",
        num_classes=10,
        in_channels=3,
        default_image_size=32,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
    "cifar100": DatasetInfo(
        name="cifar100",
        num_classes=100,
        in_channels=3,
        default_image_size=32,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    ),
    "svhn": DatasetInfo(
        name="svhn",
        num_classes=10,
        in_channels=3,
        default_image_size=32,
        mean=(0.4377, 0.4438, 0.4728),
        std=(0.1980, 0.2010, 0.1970),
    ),
    "stl10": DatasetInfo(
        name="stl10",
        num_classes=10,
        in_channels=3,
        default_image_size=64,
        mean=(0.4467, 0.4398, 0.4066),
        std=(0.2603, 0.2566, 0.2713),
    ),
    "tinyimagenet": DatasetInfo(
        name="tinyimagenet",
        num_classes=200,
        in_channels=3,
        default_image_size=64,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    "imagenet": DatasetInfo(
        name="imagenet",
        num_classes=1000,
        in_channels=3,
        default_image_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    "food101": DatasetInfo(
        name="food101",
        num_classes=101,
        in_channels=3,
        default_image_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    "flowers102": DatasetInfo(
        name="flowers102",
        num_classes=102,
        in_channels=3,
        default_image_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    "fgvcaircraft": DatasetInfo(
        name="fgvcaircraft",
        num_classes=100,
        in_channels=3,
        default_image_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    "imagenette": DatasetInfo(
        name="imagenette",
        num_classes=10,
        in_channels=3,
        default_image_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
}


def _dataset_transforms(
    info: DatasetInfo,
    image_size: int | None = None,
    strong_augment: bool = False,
):
    target_size = int(image_size or info.default_image_size)
    normalize = transforms.Normalize(info.mean, info.std)

    if target_size <= 32:
        crop_aug = [transforms.RandomCrop(target_size, padding=4)]
        test_resize = []
    else:
        crop_aug = [transforms.RandomResizedCrop(target_size, scale=(0.7, 1.0))]
        test_resize = [transforms.Resize((target_size, target_size))]

    flip_aug = []
    if info.name not in {"svhn"}:
        flip_aug = [transforms.RandomHorizontalFlip()]

    pre_tensor_aug: list = []
    post_tensor_aug: list = []
    if strong_augment:

        pre_tensor_aug = [
            transforms.RandAugment(num_ops=2, magnitude=9),
        ]

        post_tensor_aug = [
            transforms.RandomErasing(p=0.25),
        ]

    train_tf = transforms.Compose(
        crop_aug
        + flip_aug
        + pre_tensor_aug
        + [
            transforms.ToTensor(),
            normalize,
        ]
        + post_tensor_aug
    )

    test_tf = transforms.Compose(test_resize + [transforms.ToTensor(), normalize])
    return train_tf, test_tf


class TinyImageNet(Dataset):

    def __init__(self, root: str | Path, split: str, transform=None):
        if Image is None:
            raise ImportError("PIL is required to load Tiny-ImageNet images.")

        self.root = Path(root)
        self.split = split.lower()
        self.transform = transform

        wnids_path = self.root / "wnids.txt"
        if not wnids_path.exists():
            raise FileNotFoundError(f"Missing wnids.txt under {self.root}")

        wnids = [
            line.strip() for line in wnids_path.read_text().splitlines() if line.strip()
        ]
        if not wnids:
            raise RuntimeError(f"Empty wnids.txt under {self.root}")

        self.class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}

        cache_key = "train" if self.split == "train" else "val"
        cache_path = self.root / f"_index_{cache_key}.pkl"
        cached = self._load_cache(cache_path)
        if cached is not None:
            self.samples = cached
            return

        if self.split == "train":
            samples = self._build_train_samples(wnids)
        elif self.split in {"val", "valid", "validation"}:
            samples = self._build_val_samples()
        else:
            raise ValueError(f"Unsupported split: {split}")

        self.samples = [(str(p.relative_to(self.root)), int(y)) for p, y in samples]
        self._save_cache(cache_path, self.samples)

    def _load_cache(self, cache_path: Path) -> list[tuple[str, int]] | None:
        try:
            if cache_path.exists():
                with cache_path.open("rb") as f:
                    obj = pickle.load(f)
                if (
                    isinstance(obj, list)
                    and obj
                    and isinstance(obj[0], tuple)
                    and len(obj[0]) == 2
                ):
                    return obj
        except Exception:
            return None
        return None

    def _save_cache(self, cache_path: Path, samples: list[tuple[str, int]]) -> None:
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".partial")
        try:
            with tmp_path.open("wb") as f:
                pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(cache_path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _build_train_samples(self, wnids: list[str]) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []
        train_root = self.root / "train"
        for wnid in wnids:
            img_dir = train_root / wnid / "images"
            if not img_dir.exists():
                continue
            for p in sorted(img_dir.glob("*.JPEG")):
                samples.append((p, self.class_to_idx[wnid]))
        if not samples:
            raise RuntimeError(
                f"No Tiny-ImageNet training images found under {train_root}"
            )
        return samples

    def _build_val_samples(self) -> list[tuple[Path, int]]:
        val_root = self.root / "val"
        img_dir = val_root / "images"
        anno_path = val_root / "val_annotations.txt"
        if not anno_path.exists():
            raise FileNotFoundError(f"Missing val_annotations.txt under {val_root}")

        samples: list[tuple[Path, int]] = []
        for line in anno_path.read_text().splitlines():
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            img_name, wnid = parts[0], parts[1]
            label = self.class_to_idx.get(wnid)
            if label is None:
                continue
            samples.append((img_dir / img_name, label))
        if not samples:
            raise RuntimeError(
                f"No Tiny-ImageNet validation samples found under {val_root}"
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_path, target = self.samples[idx]
        path = self.root / rel_path
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def _ensure_tinyimagenet(data_root: str) -> Path:

    root = Path(data_root) / "tiny-imagenet-200"
    if root.exists() and (root / "train").exists() and (root / "val").exists():
        return root

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = Path(data_root) / "tiny-imagenet-200.zip"
    tmp_path = zip_path.with_suffix(".zip.partial")
    Path(data_root).mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        import urllib.request

        print(f"[tinyimagenet] downloading to {zip_path} ...")
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.rename(zip_path)

    import zipfile

    if root.exists():
        import shutil

        print(f"[tinyimagenet] found incomplete extraction at {root}, removing...")
        shutil.rmtree(root)

    print(f"[tinyimagenet] extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(Path(data_root))

    if not (root.exists() and (root / "train").exists() and (root / "val").exists()):
        raise RuntimeError(
            f"Tiny-ImageNet extraction did not produce expected train/val under {root}"
        )
    return root


_IMAGENET_CANDIDATE_PATHS: list[str] = [
    "/lus/flare/projects/RobustViT/kim/pa_norm/data/imagenet",
    "/lus/flare/projects/datasets/data/imagenet",
    "/lus/flare/projects/datasets/imagenet",
    "/lus/flare/datasets/imagenet",
    "/lus/eagle/datasets/imagenet",
    "/lus/grand/datasets/imagenet",
    "/lus/gila/datasets/imagenet",
    "/home/datasets/imagenet",
    "/datasets/imagenet",
    "/data/imagenet",
]


def _imagenet_root_from_data_root(data_root: str) -> Path:

    return Path(data_root) / "imagenet"


def _is_valid_imagenet_dir(root: Path, split: str) -> bool:

    split_dir = root / split
    if not split_dir.is_dir():
        return False

    try:
        next(d for d in split_dir.iterdir() if d.is_dir())
        return True
    except StopIteration:
        return False


def _find_imagenet_root(data_root: str) -> Path:

    candidates: list[Path] = [_imagenet_root_from_data_root(data_root)] + [
        Path(p) for p in _IMAGENET_CANDIDATE_PATHS
    ]

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    for root in unique_candidates:

        if root.is_dir() and (
            _is_valid_imagenet_dir(root, "train") or _is_valid_imagenet_dir(root, "val")
        ):
            return root

    searched = "\n  ".join(str(p) for p in unique_candidates)
    raise FileNotFoundError(
        "ImageNet-1K dataset not found. Searched:\n"
        f"  {searched}\n\n"
        "To use ImageNet, either:\n"
        "  (a) Place the dataset at one of the paths above (train/ and val/ sub-directories\n"
        "      each containing 1 000 class sub-directories with JPEG images), or\n"
        "  (b) Pass data_root pointing to the parent of an 'imagenet/' directory.\n\n"
        "ImageNet-1K is not publicly downloadable; obtain it from https://image-net.org/."
    )


def _imagenet_transforms(
    info: DatasetInfo, train: bool, image_size: int | None = None
) -> transforms.Compose:

    crop_size = int(image_size or info.default_image_size)

    resize_size = round(crop_size * 256 / 224)
    normalize = transforms.Normalize(info.mean, info.std)

    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    crop_size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize,
            ]
        )


def _load_imagenet(data_root: str, train: bool, transform) -> Dataset:

    root = _find_imagenet_root(data_root)
    split = "train" if train else "val"
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"ImageNet split directory not found: {split_dir}\n"
            f"Expected sub-directories named by WordNet ID (e.g. n01440764/)."
        )
    return datasets.ImageFolder(root=str(split_dir), transform=transform)


def _load_dataset(name: str, data_root: str, train: bool, download: bool, transform):
    name = name.lower()
    if name == "cifar10":
        return datasets.CIFAR10(
            root=data_root, train=train, download=download, transform=transform
        )
    if name == "cifar100":
        return datasets.CIFAR100(
            root=data_root, train=train, download=download, transform=transform
        )
    if name == "svhn":
        split = "train" if train else "test"
        return datasets.SVHN(
            root=data_root, split=split, download=download, transform=transform
        )
    if name == "stl10":
        split = "train" if train else "test"
        return datasets.STL10(
            root=data_root, split=split, download=download, transform=transform
        )
    if name == "tinyimagenet":
        if download:
            _ensure_tinyimagenet(data_root)
        root = Path(data_root) / "tiny-imagenet-200"
        if not root.exists():
            raise FileNotFoundError(
                f"Tiny-ImageNet not found under {root}. Run prepare_datasets(download=True) first."
            )
        split = "train" if train else "val"
        return TinyImageNet(root=root, split=split, transform=transform)
    if name == "imagenet":
        if download:
            import warnings

            warnings.warn(
                "ImageNet-1K cannot be downloaded automatically (license restrictions). "
                "The dataset must be placed manually at one of the candidate paths. "
                "Attempting to load from a pre-existing location.",
                UserWarning,
                stacklevel=2,
            )
        return _load_imagenet(data_root, train=train, transform=transform)
    if name == "food101":
        split = "train" if train else "test"
        return datasets.Food101(
            root=data_root, split=split, download=download, transform=transform
        )
    if name == "flowers102":
        split = "train" if train else "test"
        return datasets.Flowers102(
            root=data_root, split=split, download=download, transform=transform
        )
    if name == "fgvcaircraft":
        split = "train" if train else "test"
        return datasets.FGVCAircraft(
            root=data_root, split=split, download=download, transform=transform
        )
    if name == "imagenette":
        split = "train" if train else "val"
        return datasets.Imagenette(
            root=data_root,
            split=split,
            size="320px",
            download=download,
            transform=transform,
        )
    raise ValueError(f"Unsupported dataset: {name}")


def prepare_datasets(
    dataset_names: list[str], data_root: str, image_size: int | None = None
) -> None:
    _imagenet_style = {
        "imagenet",
        "food101",
        "flowers102",
        "fgvcaircraft",
        "imagenette",
    }
    for name in dataset_names:
        info = DATASET_INFO[name]
        if name.lower() in _imagenet_style:
            train_tf = _imagenet_transforms(info, train=True, image_size=image_size)
            test_tf = _imagenet_transforms(info, train=False, image_size=image_size)
        else:
            train_tf, test_tf = _dataset_transforms(info, image_size=image_size)
        _load_dataset(name, data_root, train=True, download=True, transform=train_tf)
        _load_dataset(name, data_root, train=False, download=True, transform=test_tf)


def _build_subset(dataset: Dataset, max_samples: Optional[int], seed: int) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(dataset))[:max_samples]
    idx = idx.tolist()
    return Subset(dataset, idx)


def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed + worker_id)


def build_dataloaders(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    image_size: int | None = None,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    strong_augment: bool = False,
):
    info = DATASET_INFO[dataset_name]

    _imagenet_style = {
        "imagenet",
        "food101",
        "flowers102",
        "fgvcaircraft",
        "imagenette",
    }
    if dataset_name.lower() in _imagenet_style:
        train_tf = _imagenet_transforms(info, train=True, image_size=image_size)
        test_tf = _imagenet_transforms(info, train=False, image_size=image_size)
    else:
        train_tf, test_tf = _dataset_transforms(
            info,
            image_size=image_size,
            strong_augment=strong_augment,
        )

    train_set = _load_dataset(
        dataset_name, data_root, train=True, download=False, transform=train_tf
    )
    test_set = _load_dataset(
        dataset_name, data_root, train=False, download=False, transform=test_tf
    )

    train_set = _build_subset(train_set, max_train_samples, seed)
    test_set = _build_subset(test_set, max_test_samples, seed + 1)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init_fn,
        generator=generator,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, test_loader, info
