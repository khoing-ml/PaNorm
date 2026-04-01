from __future__ import annotations

import argparse
import csv
import fcntl
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

try:
    max_numexpr = int(os.environ.get("NUMEXPR_MAX_THREADS", "64"))
    if max_numexpr > 64:
        os.environ["NUMEXPR_MAX_THREADS"] = "64"
except ValueError:
    os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ.setdefault("NUMEXPR_NUM_THREADS", "64")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_bench import BenchmarkConfig, build_model, run_single_experiment
from real_bench.efficiency import profile_model

DEFAULT_DATA_ROOT = "/lus/flare/projects/RobustViT/kim/pa_norm/data"
RESULT_CSV = "exp/result.csv"


CORE_METHODS = [
    "batchnorm",
    "layernorm",
    "groupnorm",
    "rmsnorm",
    "switchnorm",
    "panorm",
    "panorm_switchnorm",
    "panorm_lite_fast",
]

EXTENDED_METHODS = CORE_METHODS + [
    "frn",
    "evonorm_s0",
    "evonorm_b0",
    "panorm_nodetach",
    "panorm_lite",
]

SGD_METHODS = [
    "batchnorm",
    "layernorm",
    "groupnorm",
    "switchnorm",
    "panorm",
    "panorm_switchnorm",
    "panorm_lite_fast",
]

SEEDS_5 = list(range(42, 47))
SEEDS_3 = list(range(42, 45))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PA-Norm comprehensive experiments")
    p.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    p.add_argument("--output", type=str, default=RESULT_CSV)
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--group",
        type=str,
        required=True,
        choices=[
            "smallcnn_32",
            "smallcnn_64",
            "resnet18_32",
            "resnet18_64",
            "resnet50",
            "tinyimagenet",
            "sgd200",
            "wideresnet",
            "ood_sev2",
            "ood_sweep",
            "detach_ablation",
            "detach_ablation_deep",
            "convergence",
            "batchsize",
            "efficiency",
            "convnext",
            "batchsize_extreme",
            "ood_per_corruption",
            "convergence_detach",
            "sgd200_extended",
            "resnet18_128",
            "vit_cifar100",
            "vit_tinyimagenet",
            "vit_cifar10",
            "imagenet_resnet50",
            "detach_dynamics_deep",
            "sgd200_vit",
        ],
    )
    return p.parse_args()


def write_csv_row(path: Path, row: dict) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            exists = path.exists() and path.stat().st_size > 0
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def already_done(
    path: Path,
    dataset: str,
    architecture: str,
    method: str,
    seed: int,
    protocol_prefix: str,
    eval_ood: bool = False,
    ood_severity: int = 2,
) -> bool:

    if not path.exists():
        return False
    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (
                    row.get("dataset") == dataset
                    and row.get("architecture") == architecture
                    and row.get("method") == method
                    and str(row.get("seed")) == str(seed)
                    and row.get("status") == "success"
                    and (
                        row.get("protocol", "").startswith(protocol_prefix)
                        or row.get("protocol", "").startswith(
                            "adamw_" + protocol_prefix
                        )
                    )
                ):

                    if eval_ood:
                        proto = row.get("protocol", "")
                        if f"_ood{ood_severity}" not in proto:
                            continue
                    else:

                        proto = row.get("protocol", "")
                        if "_ood" in proto:
                            continue
                    return True
    except Exception:
        return False
    return False


def run_group(
    group_name: str,
    datasets: list[str],
    architectures: list[str],
    methods: list[str],
    seeds: list[int],
    config: BenchmarkConfig,
    output_path: Path,
    device_id: int,
    protocol_prefix: str = "",
) -> int:

    total = len(datasets) * len(architectures) * len(methods) * len(seeds)
    done = 0
    skipped = 0

    print(f"\n{'='*60}")
    print(f"Group: {group_name} | {total} experiments")
    print(f"{'='*60}")

    for dataset in datasets:
        for architecture in architectures:
            for method in methods:
                for seed in seeds:
                    prefix = (
                        protocol_prefix or f"e{config.epochs}_bs{config.batch_size}"
                    )
                    if already_done(
                        output_path,
                        dataset,
                        architecture,
                        method,
                        seed,
                        prefix,
                        eval_ood=config.eval_ood,
                        ood_severity=config.ood_severity,
                    ):
                        skipped += 1
                        continue

                    done += 1
                    print(
                        f"\n[{done}/{total-skipped}] {dataset}/{architecture}/{method}/s={seed}"
                    )

                    try:
                        result = run_single_experiment(
                            dataset=dataset,
                            method=method,
                            seed=seed,
                            device_id=device_id,
                            config=replace(config, architecture=architecture),
                        )
                        write_csv_row(output_path, result)
                        status = result.get("status", "?")
                        acc = result.get("test_acc1", -1)
                        t = result.get("time_sec", 0)
                        print(f"  -> {status} | acc1={acc:.2f}% | {t:.1f}s")
                    except Exception as e:
                        print(f"  -> ERROR: {e}")
                        traceback.print_exc()

    print(f"\nGroup {group_name}: {done} run, {skipped} skipped")
    return done


def run_efficiency(
    methods, architecture, num_classes, image_size, device_id, output_path: Path
):

    print(f"\n{'='*60}")
    print(f"Efficiency Profiling: {architecture}")
    print(f"{'='*60}")

    device = torch.device("cpu")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device(f"xpu:{device_id}")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")

    input_shape = (64, 3, image_size, image_size)
    rows = []

    for method in methods:
        print(f"  Profiling {method}...")
        try:
            model = build_model(
                norm_type=method,
                in_channels=3,
                num_classes=num_classes,
                architecture=architecture,
                image_size=image_size,
            )
            metrics = profile_model(
                model=model,
                method=method,
                architecture=architecture,
                input_shape=input_shape,
                device=device,
            )
            row = {
                "method": metrics.method,
                "architecture": metrics.architecture,
                "num_params_m": f"{metrics.num_params_m:.4f}",
                "flops_g": f"{metrics.flops_g:.4f}",
                "train_throughput_img_s": f"{metrics.train_throughput_img_s:.1f}",
                "infer_throughput_img_s": f"{metrics.infer_throughput_img_s:.1f}",
                "train_latency_ms": f"{metrics.train_latency_ms:.2f}",
                "infer_latency_ms": f"{metrics.infer_latency_ms:.2f}",
                "peak_memory_mb": f"{metrics.peak_memory_mb:.1f}",
            }
            rows.append(row)
            print(
                f"    params={metrics.num_params_m:.2f}M | "
                f"train={metrics.train_throughput_img_s:.0f} img/s | "
                f"infer={metrics.infer_throughput_img_s:.0f} img/s"
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()

    if rows:
        eff_path = output_path.parent / "result_efficiency.csv"
        with open(eff_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved to {eff_path}")


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    group = args.group

    if group == "smallcnn_32":
        config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=20,
            batch_size=256,
            num_workers=args.num_workers,
            lr=3e-4,
            weight_decay=5e-4,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=32,
        )
        run_group(
            "SmallCNN-20ep-32x32",
            datasets=["cifar10", "cifar100", "svhn"],
            architectures=["smallcnn"],
            methods=EXTENDED_METHODS,
            seeds=SEEDS_5,
            config=config,
            output_path=output_path,
            device_id=args.device_id,
        )

    elif group == "smallcnn_64":
        config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=20,
            batch_size=256,
            num_workers=args.num_workers,
            lr=3e-4,
            weight_decay=5e-4,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=64,
        )
        run_group(
            "SmallCNN-20ep-64x64",
            datasets=["stl10"],
            architectures=["smallcnn"],
            methods=EXTENDED_METHODS,
            seeds=SEEDS_5,
            config=config,
            output_path=output_path,
            device_id=args.device_id,
        )

    elif group == "resnet18_32":
        config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=50,
            batch_size=256,
            num_workers=args.num_workers,
            lr=3e-4,
            weight_decay=5e-4,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=32,
        )
        run_group(
            "ResNet18-50ep-32x32",
            datasets=["cifar10", "cifar100", "svhn"],
            architectures=["resnet18"],
            methods=CORE_METHODS,
            seeds=SEEDS_5,
            config=config,
            output_path=output_path,
            device_id=args.device_id,
        )

    elif group == "resnet18_64":
        config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=50,
            batch_size=256,
            num_workers=args.num_workers,
            lr=3e-4,
            weight_decay=5e-4,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=64,
        )
        run_group(
            "ResNet18-50ep-64x64",
            datasets=["stl10"],
            architectures=["resnet18"],
            methods=CORE_METHODS,
            seeds=SEEDS_5,
            config=config,
            output_path=output_path,
            device_id=args.device_id,
        )

    elif group == "resnet50":
        config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=50,
            batch_size=256,
            num_workers=args.num_workers,
            lr=3e-4,
            weight_decay=5e-4,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=32,
        )
        run_group(
            "ResNet50-50ep-CIFAR100",
            datasets=["cifar100"],
            architectures=["resnet50"],
            methods=CORE_METHODS,
            seeds=SEEDS_5,
            config=config,
            output_path=output_path,
            device_id=args.device_id,
        )

    elif group == "tinyimagenet":
        config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=20,
            batch_size=256,
            num_workers=args.num_workers,
            lr=3e-4,
            weight_decay=5e-4,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=64,
        )
        run_group(
            "TinyImageNet-ResNet18-20ep",
            datasets=["tinyimagenet"],
            architectures=["resnet18"],
            methods=CORE_METHODS,
            seeds=SEEDS_3,
            config=config,
            output_path=output_path,
            device_id=args.device_id,
        )

    elif group == "sgd200":
        sgd_config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=200,
            batch_size=128,
            num_workers=args.num_workers,
            lr=0.1,
            weight_decay=1e-4,
            width=64,
            label_smoothing=0.0,
            use_amp=True,
            image_size=32,
            optimizer="sgd",
            momentum=0.9,
        )
        run_group(
            "SGD-200ep-Standard",
            datasets=["cifar10", "cifar100"],
            architectures=["resnet18"],
            methods=SGD_METHODS,
            seeds=SEEDS_5,
            config=sgd_config,
            output_path=output_path,
            device_id=args.device_id,
            protocol_prefix="sgd_e200",
        )

    elif group == "wideresnet":
        config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=50,
            batch_size=128,
            num_workers=args.num_workers,
            lr=3e-4,
            weight_decay=5e-4,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=32,
        )
        wrn_methods = [
            "batchnorm",
            "groupnorm",
            "switchnorm",
            "panorm",
            "panorm_switchnorm",
            "panorm_lite_fast",
        ]
        run_group(
            "WideResNet50-50ep-CIFAR100",
            datasets=["cifar100"],
            architectures=["wide_resnet50_2"],
            methods=wrn_methods,
            seeds=SEEDS_3,
            config=config,
            output_path=output_path,
            device_id=args.device_id,
        )

    elif group == "ood_sev2":
        config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=50,
            batch_size=256,
            num_workers=args.num_workers,
            lr=3e-4,
            weight_decay=5e-4,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=32,
            eval_ood=True,
            ood_severity=2,
        )
        run_group(
            "OOD-ResNet18-sev2",
            datasets=["cifar10"],
            architectures=["resnet18"],
            methods=CORE_METHODS,
            seeds=SEEDS_5,
            config=config,
            output_path=output_path,
            device_id=args.device_id,
        )

    elif group == "ood_sweep":
        key_methods = ["batchnorm", "switchnorm", "panorm_switchnorm", "panorm"]
        for sev in [1, 3, 4, 5]:
            config = BenchmarkConfig(
                data_root=args.data_root,
                epochs=50,
                batch_size=256,
                num_workers=args.num_workers,
                lr=3e-4,
                weight_decay=5e-4,
                width=64,
                label_smoothing=0.1,
                use_amp=True,
                image_size=32,
                eval_ood=True,
                ood_severity=sev,
            )
            run_group(
                f"OOD-Severity-{sev}",
                datasets=["cifar10"],
                architectures=["resnet18"],
                methods=key_methods,
                seeds=SEEDS_3,
                config=config,
                output_path=output_path,
                device_id=args.device_id,
            )

    elif group == "detach_ablation":
        config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=50,
            batch_size=256,
            num_workers=args.num_workers,
            lr=3e-4,
            weight_decay=5e-4,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=32,
        )
        ablation_methods = [
            "panorm",
            "panorm_nodetach",
            "panorm_switchnorm",
            "panorm_switchnorm_nodetach",
        ]
        run_group(
            "Detach-Ablation",
            datasets=["cifar10", "cifar100"],
            architectures=["resnet18"],
            methods=ablation_methods,
            seeds=SEEDS_5,
            config=config,
            output_path=output_path,
            device_id=args.device_id,
        )

    elif group == "batchsize":
        key_methods = ["batchnorm", "groupnorm", "panorm_switchnorm", "panorm"]
        for bs in [32, 64, 128, 512]:
            config = BenchmarkConfig(
                data_root=args.data_root,
                epochs=50,
                batch_size=bs,
                num_workers=args.num_workers,
                lr=3e-4,
                weight_decay=5e-4,
                width=64,
                label_smoothing=0.1,
                use_amp=True,
                image_size=32,
            )
            run_group(
                f"BatchSize-{bs}",
                datasets=["cifar100"],
                architectures=["resnet18"],
                methods=key_methods,
                seeds=SEEDS_3,
                config=config,
                output_path=output_path,
                device_id=args.device_id,
            )

    elif group == "detach_ablation_deep":
        config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=50,
            batch_size=256,
            num_workers=args.num_workers,
            lr=3e-4,
            weight_decay=5e-4,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=32,
        )
        ablation_methods = [
            "panorm",
            "panorm_nodetach",
            "panorm_switchnorm",
            "panorm_switchnorm_nodetach",
        ]
        run_group(
            "Detach-Ablation-Deep",
            datasets=["cifar10", "cifar100"],
            architectures=["resnet50"],
            methods=ablation_methods,
            seeds=SEEDS_3,
            config=config,
            output_path=output_path,
            device_id=args.device_id,
        )

    elif group == "convergence":
        config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=50,
            batch_size=256,
            num_workers=args.num_workers,
            lr=3e-4,
            weight_decay=5e-4,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=32,
            log_dynamics=True,
        )
        conv_methods = [
            "batchnorm",
            "groupnorm",
            "switchnorm",
            "panorm",
            "panorm_switchnorm",
            "panorm_lite_fast",
        ]
        run_group(
            "Convergence-Speed",
            datasets=["cifar100"],
            architectures=["resnet18"],
            methods=conv_methods,
            seeds=SEEDS_3,
            config=config,
            output_path=output_path,
            device_id=args.device_id,
        )

    elif group == "efficiency":
        all_methods = CORE_METHODS + [
            "frn",
            "evonorm_s0",
            "panorm_lite",
            "panorm_nodetach",
        ]
        run_efficiency(
            methods=all_methods,
            architecture="resnet18",
            num_classes=100,
            image_size=32,
            device_id=args.device_id,
            output_path=output_path,
        )

    elif group == "convnext":
        config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=50,
            batch_size=128,
            num_workers=args.num_workers,
            lr=3e-4,
            weight_decay=5e-4,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=32,
        )

        convnext_methods = [
            "batchnorm",
            "layernorm",
            "groupnorm",
            "switchnorm",
            "panorm",
            "panorm_switchnorm",
            "panorm_lite_fast",
        ]
        run_group(
            "ConvNeXt-Tiny-50ep-CIFAR100",
            datasets=["cifar100"],
            architectures=["convnext_tiny"],
            methods=convnext_methods,
            seeds=SEEDS_3,
            config=config,
            output_path=output_path,
            device_id=args.device_id,
        )

    elif group == "batchsize_extreme":
        key_methods = ["batchnorm", "groupnorm", "panorm_switchnorm", "panorm"]
        for bs in [4, 8, 16]:
            config = BenchmarkConfig(
                data_root=args.data_root,
                epochs=50,
                batch_size=bs,
                num_workers=args.num_workers,
                lr=3e-4,
                weight_decay=5e-4,
                width=64,
                label_smoothing=0.1,
                use_amp=True,
                image_size=32,
            )
            run_group(
                f"BatchSize-{bs}",
                datasets=["cifar100"],
                architectures=["resnet18"],
                methods=key_methods,
                seeds=SEEDS_3,
                config=config,
                output_path=output_path,
                device_id=args.device_id,
            )

    elif group == "ood_per_corruption":
        config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=50,
            batch_size=256,
            num_workers=args.num_workers,
            lr=3e-4,
            weight_decay=5e-4,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=32,
            eval_ood=True,
            ood_severity=3,
            log_dynamics=True,
        )

        ood_methods = ["batchnorm", "panorm_switchnorm", "panorm"]
        run_group(
            "OOD-Per-Corruption-Sev3",
            datasets=["cifar10"],
            architectures=["resnet18"],
            methods=ood_methods,
            seeds=SEEDS_3,
            config=config,
            output_path=output_path,
            device_id=args.device_id,
        )

    elif group == "convergence_detach":
        config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=50,
            batch_size=256,
            num_workers=args.num_workers,
            lr=3e-4,
            weight_decay=5e-4,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=32,
            log_dynamics=True,
        )

        conv_detach_methods = [
            "batchnorm",
            "panorm",
            "panorm_nodetach",
            "panorm_switchnorm",
            "panorm_switchnorm_nodetach",
        ]
        run_group(
            "Convergence-Detach",
            datasets=["cifar100"],
            architectures=["resnet18"],
            methods=conv_detach_methods,
            seeds=SEEDS_3,
            config=config,
            output_path=output_path,
            device_id=args.device_id,
        )

    elif group == "sgd200_extended":
        sgd_config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=200,
            batch_size=128,
            num_workers=args.num_workers,
            lr=0.1,
            weight_decay=1e-4,
            width=64,
            label_smoothing=0.0,
            use_amp=True,
            image_size=32,
            optimizer="sgd",
            momentum=0.9,
        )
        run_group(
            "SGD-200ep-SVHN",
            datasets=["svhn"],
            architectures=["resnet18"],
            methods=SGD_METHODS,
            seeds=SEEDS_3,
            config=sgd_config,
            output_path=output_path,
            device_id=args.device_id,
            protocol_prefix="sgd_e200",
        )

    elif group == "resnet18_128":
        config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=50,
            batch_size=128,
            num_workers=args.num_workers,
            lr=3e-4,
            weight_decay=5e-4,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=32,
        )
        run_group(
            "ResNet18-50ep-BS128",
            datasets=["cifar10", "cifar100"],
            architectures=["resnet18"],
            methods=CORE_METHODS,
            seeds=SEEDS_3,
            config=config,
            output_path=output_path,
            device_id=args.device_id,
        )

    elif group == "vit_cifar100":
        config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=50,
            batch_size=128,
            num_workers=args.num_workers,
            lr=1e-3,
            weight_decay=5e-2,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=32,
        )
        vit_methods = [
            "layernorm",
            "rmsnorm",
            "panorm",
            "panorm_nodetach",
        ]
        run_group(
            "ViT-Small-CIFAR100",
            datasets=["cifar100"],
            architectures=["vit_small"],
            methods=vit_methods,
            seeds=SEEDS_3,
            config=config,
            output_path=output_path,
            device_id=args.device_id,
        )

    elif group == "vit_cifar10":
        config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=50,
            batch_size=128,
            num_workers=args.num_workers,
            lr=1e-3,
            weight_decay=5e-2,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=32,
        )
        vit_methods = [
            "layernorm",
            "rmsnorm",
            "panorm",
            "panorm_nodetach",
        ]
        run_group(
            "ViT-Small-CIFAR10",
            datasets=["cifar10"],
            architectures=["vit_small"],
            methods=vit_methods,
            seeds=SEEDS_3,
            config=config,
            output_path=output_path,
            device_id=args.device_id,
        )

    elif group == "vit_tinyimagenet":
        config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=50,
            batch_size=128,
            num_workers=args.num_workers,
            lr=1e-3,
            weight_decay=5e-2,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=64,
        )
        vit_methods = [
            "layernorm",
            "rmsnorm",
            "panorm",
            "panorm_nodetach",
        ]
        run_group(
            "ViT-Small-TinyImageNet",
            datasets=["tinyimagenet"],
            architectures=["vit_small"],
            methods=vit_methods,
            seeds=SEEDS_3,
            config=config,
            output_path=output_path,
            device_id=args.device_id,
        )

    elif group == "imagenet_resnet50":
        config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=90,
            batch_size=256,
            num_workers=args.num_workers,
            lr=0.1,
            weight_decay=1e-4,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=224,
            optimizer="sgd",
            momentum=0.9,
        )
        imagenet_methods = [
            "batchnorm",
            "groupnorm",
            "switchnorm",
            "panorm",
            "panorm_switchnorm",
            "panorm_lite_fast",
        ]
        run_group(
            "ImageNet-ResNet50-90ep",
            datasets=["imagenet"],
            architectures=["resnet50"],
            methods=imagenet_methods,
            seeds=SEEDS_3,
            config=config,
            output_path=output_path,
            device_id=args.device_id,
            protocol_prefix="sgd_e90",
        )

    elif group == "detach_dynamics_deep":
        config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=50,
            batch_size=256,
            num_workers=args.num_workers,
            lr=3e-4,
            weight_decay=5e-4,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=32,
            log_dynamics=True,
            grad_clip=1.0,
        )
        methods = [
            "batchnorm",
            "panorm",
            "panorm_nodetach",
            "panorm_switchnorm",
            "panorm_switchnorm_nodetach",
        ]
        run_group(
            "Detach-Dynamics-Deep",
            datasets=["cifar100"],
            architectures=["resnet50"],
            methods=methods,
            seeds=SEEDS_3,
            config=config,
            output_path=output_path,
            device_id=args.device_id,
        )

    elif group == "sgd200_vit":
        sgd_config = BenchmarkConfig(
            data_root=args.data_root,
            epochs=200,
            batch_size=128,
            num_workers=args.num_workers,
            lr=1e-3,
            weight_decay=5e-2,
            width=64,
            label_smoothing=0.1,
            use_amp=True,
            image_size=32,
        )
        run_group(
            "SGD-200ep-ViT",
            datasets=["cifar100"],
            architectures=["vit_small"],
            methods=["layernorm", "rmsnorm", "panorm"],
            seeds=SEEDS_3,
            config=sgd_config,
            output_path=output_path,
            device_id=args.device_id,
            protocol_prefix="adamw_e200",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
