from __future__ import annotations

import os
import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .datasets import build_dataloaders
from .models import build_model
from .ood import Cifar10CConfig, evaluate_cifar10c


@dataclass
class BenchmarkConfig:
    data_root: str
    epochs: int = 12
    batch_size: int = 256
    num_workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 5e-4
    width: int = 64
    architecture: str = "smallcnn"
    image_size: int | None = None
    label_smoothing: float = 0.1
    use_amp: bool = True
    eval_ood: bool = False
    ood_severity: int = 2
    max_train_samples: int | None = None
    max_test_samples: int | None = None
    optimizer: str = "adamw"
    momentum: float = 0.9
    grad_clip: float = 0.0
    log_dynamics: bool = False
    strong_augment: bool = False
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    warmup_epochs: int = 0
    wandb_enabled: bool = False
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_mode: str = ""
    wandb_group: str = ""
    wandb_job_type: str = "train"
    wandb_tags: str = ""
    wandb_run_name: str = ""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "xpu"):
        try:
            torch.xpu.manual_seed(seed)
        except Exception:
            pass


def resolve_device(device_id: int) -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device(f"xpu:{device_id}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def accuracy(logits: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    maxk = min(max(topk), logits.size(1))
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    batch_size = target.size(0)
    for k in topk:
        k = min(k, logits.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append((correct_k * (100.0 / batch_size)).item())
    return results


def _move_batch(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device):
    x, y = batch
    return x.to(device), y.to(device, dtype=torch.long)


def _mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float):

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def _cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float):

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1.0 - lam)
    rH, rW = int(H * cut_ratio), int(W * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(0, cx - rW // 2)
    y1 = max(0, cy - rH // 2)
    x2 = min(W, cx + rW // 2)
    y2 = min(H, cy + rH // 2)
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed_x, y, y[idx], lam


def _mixup_criterion(
    criterion: nn.Module,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
):

    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def _parse_csv_tags(value: str) -> list[str]:
    return [tag.strip() for tag in value.split(",") if tag.strip()]


def _maybe_init_wandb(
    config: BenchmarkConfig,
    dataset: str,
    method: str,
    seed: int,
    protocol: str,
    image_size: int,
    num_params_m: float,
):
    if not config.wandb_enabled:
        return None

    try:
        import wandb
    except ImportError:
        print("wandb is enabled but not installed; continuing without logging.")
        return None

    init_kwargs: dict[str, Any] = {
        "project": config.wandb_project or os.environ.get("WANDB_PROJECT") or "panorm",
        "entity": config.wandb_entity or os.environ.get("WANDB_ENTITY") or None,
        "mode": config.wandb_mode or os.environ.get("WANDB_MODE") or "",
        "group": config.wandb_group or os.environ.get("WANDB_GROUP") or None,
        "job_type": config.wandb_job_type or os.environ.get("WANDB_JOB_TYPE") or "train",
        "name": config.wandb_run_name
        or f"{dataset}-{config.architecture}-{method}-seed{seed}",
        "tags": _parse_csv_tags(config.wandb_tags or os.environ.get("WANDB_TAGS", "")),
        "reinit": True,
        "config": {
            **asdict(config),
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "protocol": protocol,
            "image_size": image_size,
            "num_params_m": num_params_m,
        },
    }

    init_kwargs = {k: v for k, v in init_kwargs.items() if v not in (None, "", [])}
    try:
        run = wandb.init(**init_kwargs)
        try:
            wandb.define_metric("epoch")
            wandb.define_metric("*", step_metric="epoch")
        except Exception:
            pass
        return run
    except Exception as exc:
        print(f"wandb init failed; continuing without logging ({exc})")
        return None


def _wandb_log(run, data: dict[str, Any], step: int | None = None) -> None:
    if run is None:
        return
    try:
        run.log(data, step=step)
    except Exception:
        pass


def _wandb_finish(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_samples = 0

    for batch in loader:
        x, y = _move_batch(batch, device)
        logits = model(x)
        if not bool(torch.isfinite(logits).all().item()):
            return {"loss": float("nan"), "acc1": -1.0, "acc5": -1.0}
        loss = criterion(logits, y)
        if not bool(torch.isfinite(loss).item()):
            return {"loss": float("nan"), "acc1": -1.0, "acc5": -1.0}
        acc1, acc5 = accuracy(logits, y)

        bsz = y.size(0)
        total_loss += loss.item() * bsz
        total_acc1 += acc1 * bsz / 100.0
        total_acc5 += acc5 * bsz / 100.0
        total_samples += bsz

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc1": 100.0 * total_acc1 / max(total_samples, 1),
        "acc5": 100.0 * total_acc5 / max(total_samples, 1),
    }


def _collect_gate_statistics(model: nn.Module) -> dict[str, float]:
    gate_modules = [m for m in model.modules() if hasattr(m, "gate_statistics")]
    if not gate_modules:
        return {
            "gate_entropy_norm": -1.0,
            "gate_main_weight": -1.0,
            "gate_branch0": -1.0,
            "gate_branch1": -1.0,
            "gate_branch2": -1.0,
        }

    stats = [m.gate_statistics() for m in gate_modules]
    keys = stats[0].keys()
    merged = {}
    for key in keys:
        merged[key] = float(np.mean([s[key] for s in stats]))
    return merged


def _collect_grad_norm(model: nn.Module) -> float:

    norms = [
        p.grad.detach().norm(2).item() for p in model.parameters() if p.grad is not None
    ]
    return float(np.mean(norms)) if norms else 0.0


def _build_protocol(config: BenchmarkConfig, dataset: str, image_size: int) -> str:
    protocol = (
        f"{config.optimizer}"
        f"_e{config.epochs}"
        f"_bs{config.batch_size}"
        f"_im{image_size}"
        f"_lr{config.lr:g}"
        f"_wd{config.weight_decay:g}"
        f"_w{config.width}"
        f"_ls{config.label_smoothing:g}"
        f"_amp{int(config.use_amp)}"
    )
    if config.grad_clip > 0:
        protocol += f"_gc{config.grad_clip:g}"
    if config.strong_augment:
        protocol += "_sa"
    if config.mixup_alpha > 0:
        protocol += f"_mx{config.mixup_alpha:g}"
    if config.cutmix_alpha > 0:
        protocol += f"_cm{config.cutmix_alpha:g}"
    if config.warmup_epochs > 0:
        protocol += f"_wu{config.warmup_epochs}"
    if config.max_train_samples is not None:
        protocol += f"_mt{int(config.max_train_samples)}"
    if config.max_test_samples is not None:
        protocol += f"_mte{int(config.max_test_samples)}"
    if config.eval_ood and dataset == "cifar10":
        protocol += f"_ood{int(config.ood_severity)}"
    return protocol


def _build_result_dict(
    dataset: str,
    config: BenchmarkConfig,
    method: str,
    seed: int,
    device: torch.device,
    protocol: str,
    image_size: int,
    num_params_m: float,
    test_loss: float,
    test_acc1: float,
    test_acc5: float,
    elapsed: float,
    status: str,
    gate_stats: dict[str, float],
    ood_metrics: dict[str, float] | None = None,
    dynamics: list[dict] | None = None,
) -> dict[str, Any]:
    result = {
        "dataset": dataset,
        "architecture": config.architecture,
        "method": method,
        "seed": seed,
        "device": str(device),
        "protocol": protocol,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "image_size": image_size,
        "lr": float(config.lr),
        "weight_decay": float(config.weight_decay),
        "width": int(config.width),
        "label_smoothing": float(config.label_smoothing),
        "use_amp": bool(config.use_amp),
        "eval_ood": bool(config.eval_ood),
        "ood_severity": int(config.ood_severity) if config.eval_ood else -1,
        "max_train_samples": (
            int(config.max_train_samples)
            if config.max_train_samples is not None
            else -1
        ),
        "max_test_samples": (
            int(config.max_test_samples) if config.max_test_samples is not None else -1
        ),
        "num_params_m": num_params_m,
        "test_loss": test_loss,
        "test_acc1": test_acc1,
        "test_acc5": test_acc5,
        "time_sec": elapsed,
        "status": status,
    }
    result.update(gate_stats)
    result["dynamics"] = json.dumps(dynamics if dynamics is not None else [])

    ood_defaults = {
        "ood_acc1_cifar10c_mean": -1.0,
        "ood_acc1_cifar10c_worst": -1.0,
        "ood_acc1_clean": -1.0,
        "ood_drop_from_clean": -1.0,
        "ood_gate_entropy_norm": -1.0,
        "ood_gate_main_weight": -1.0,
        "ood_gate_branch0": -1.0,
        "ood_gate_branch1": -1.0,
        "ood_gate_branch2": -1.0,
    }
    if ood_metrics:
        ood_defaults.update(ood_metrics)
    result.update(ood_defaults)
    return result


def run_single_experiment(
    dataset: str,
    method: str,
    seed: int,
    device_id: int,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    set_seed(seed)
    device = resolve_device(device_id)
    use_amp = bool(config.use_amp and device.type in {"xpu", "cuda"})
    wandb_run = None

    train_loader, test_loader, info = build_dataloaders(
        dataset_name=dataset,
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=seed,
        image_size=config.image_size,
        max_train_samples=config.max_train_samples,
        max_test_samples=config.max_test_samples,
        strong_augment=config.strong_augment,
    )

    image_size = int(config.image_size or info.default_image_size)
    model = build_model(
        norm_type=method,
        in_channels=info.in_channels,
        num_classes=info.num_classes,
        width=config.width,
        architecture=config.architecture,
        image_size=image_size,
    ).to(device)

    num_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    protocol = _build_protocol(config, dataset, image_size)
    wandb_run = _maybe_init_wandb(
        config=config,
        dataset=dataset,
        method=method,
        seed=seed,
        protocol=protocol,
        image_size=image_size,
        num_params_m=num_params_m,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    if config.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=config.warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.warmup_epochs],
        )
    else:
        scheduler = cosine_scheduler

    use_mixup = config.mixup_alpha > 0
    use_cutmix = config.cutmix_alpha > 0

    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    scaler_enabled = use_amp and (amp_dtype == torch.float16)
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=scaler_enabled)
    except (TypeError, AttributeError):
        if device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=False)

    start = time.time()
    failure_reason: str | None = None
    dynamics: list[dict] = []
    for epoch_idx in range(config.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_samples = 0
        epoch_grad_norms: list[float] = []
        for batch in train_loader:
            x, y = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)

            mixed_targets = None
            if use_mixup or use_cutmix:
                if use_mixup and use_cutmix:
                    use_cut = np.random.random() > 0.5
                else:
                    use_cut = use_cutmix
                if use_cut:
                    x, y_a, y_b, lam = _cutmix_data(x, y, config.cutmix_alpha)
                else:
                    x, y_a, y_b, lam = _mixup_data(x, y, config.mixup_alpha)
                mixed_targets = (y_a, y_b, lam)

            with torch.amp.autocast(
                device_type=device.type, enabled=use_amp, dtype=amp_dtype
            ):
                logits = model(x)
                if mixed_targets is not None:
                    y_a, y_b, lam = mixed_targets
                    loss = _mixup_criterion(criterion, logits, y_a, y_b, lam)
                else:
                    loss = criterion(logits, y)
            if not bool(torch.isfinite(loss).item()):
                failure_reason = f"nonfinite_loss(epoch={epoch_idx})"
                break
            if config.log_dynamics:
                bsz = y.size(0)
                epoch_loss_sum += loss.item() * bsz
                epoch_samples += bsz
            scaler.scale(loss).backward()

            if config.log_dynamics or config.grad_clip > 0:
                scaler.unscale_(optimizer)
            if config.log_dynamics:
                epoch_grad_norms.append(_collect_grad_norm(model))
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        if failure_reason is not None:
            break
        scheduler.step()
        if wandb_run is not None:
            _wandb_log(
                wandb_run,
                {
                    "epoch": epoch_idx,
                    "train/loss": float(epoch_loss_sum / max(epoch_samples, 1)),
                    "train/grad_norm_mean": (
                        float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0
                    ),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                },
                step=epoch_idx,
            )
        if config.log_dynamics:
            epoch_test = evaluate(model, test_loader, criterion, device)
            epoch_gate = _collect_gate_statistics(model)
            dynamics.append(
                {
                    "epoch": epoch_idx,
                    "train_loss": float(epoch_loss_sum / max(epoch_samples, 1)),
                    "test_acc1": float(epoch_test["acc1"]),
                    "test_loss": float(epoch_test["loss"]),
                    "grad_norm_mean": (
                        float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0
                    ),
                    **{k: float(v) for k, v in epoch_gate.items()},
                }
            )
            if wandb_run is not None:
                _wandb_log(
                    wandb_run,
                    {
                        "epoch": epoch_idx,
                        "test/loss": float(epoch_test["loss"]),
                        "test/acc1": float(epoch_test["acc1"]),
                        "test/acc5": float(epoch_test["acc5"]),
                        **{f"gate/{k}": float(v) for k, v in epoch_gate.items()},
                    },
                    step=epoch_idx,
                )

    elapsed = time.time() - start

    def _make_error_result(reason: str) -> dict[str, Any]:
        return _build_result_dict(
            dataset=dataset,
            config=config,
            method=method,
            seed=seed,
            device=device,
            protocol=protocol,
            image_size=image_size,
            num_params_m=num_params_m,
            test_loss=float("nan"),
            test_acc1=-1.0,
            test_acc5=-1.0,
            elapsed=elapsed,
            status=f"error: {reason}",
            gate_stats=_collect_gate_statistics(model),
            dynamics=dynamics if config.log_dynamics else None,
        )

    if failure_reason is not None:
        result = _make_error_result(failure_reason)
        _wandb_log(
            wandb_run,
            {f"final/{k}": v for k, v in result.items() if k != "dynamics"},
        )
        _wandb_finish(wandb_run)
        return result

    metrics = evaluate(model, test_loader, criterion, device)
    if not float(metrics["loss"]) == float(metrics["loss"]):
        result = _make_error_result("nonfinite_test_loss")
        _wandb_log(
            wandb_run,
            {f"final/{k}": v for k, v in result.items() if k != "dynamics"},
        )
        _wandb_finish(wandb_run)
        return result
    if not bool(np.isfinite([metrics["loss"], metrics["acc1"], metrics["acc5"]]).all()):
        result = _make_error_result("nonfinite_test_metrics")
        _wandb_log(
            wandb_run,
            {f"final/{k}": v for k, v in result.items() if k != "dynamics"},
        )
        _wandb_finish(wandb_run)
        return result

    gate_stats = _collect_gate_statistics(model)

    ood_metrics = {}
    if config.eval_ood and dataset == "cifar10":
        gate_modules = [
            m for m in model.modules() if hasattr(m, "reset_gate_statistics")
        ]
        for m in gate_modules:
            try:
                m.reset_gate_statistics()
            except Exception:
                pass

        ood_metrics = evaluate_cifar10c(
            model=model,
            criterion=criterion,
            device=device,
            data_root=config.data_root,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            config=Cifar10CConfig(severity=config.ood_severity),
        )
        ood_metrics["ood_acc1_clean"] = float(metrics["acc1"])
        ood_metrics["ood_drop_from_clean"] = float(
            metrics["acc1"] - ood_metrics["ood_acc1_cifar10c_mean"]
        )
        ood_gate_stats = _collect_gate_statistics(model)
        ood_metrics.update({f"ood_{k}": v for k, v in ood_gate_stats.items()})

    result = _build_result_dict(
        dataset=dataset,
        config=config,
        method=method,
        seed=seed,
        device=device,
        protocol=protocol,
        image_size=image_size,
        num_params_m=num_params_m,
        test_loss=metrics["loss"],
        test_acc1=metrics["acc1"],
        test_acc5=metrics["acc5"],
        elapsed=elapsed,
        status="success",
        gate_stats=gate_stats,
        ood_metrics=ood_metrics,
        dynamics=dynamics if config.log_dynamics else None,
    )

    _wandb_log(
        wandb_run,
        {f"final/{k}": v for k, v in result.items() if k != "dynamics"},
    )
    if config.log_dynamics and wandb_run is not None and dynamics:
        try:
            import wandb

            columns = list(dynamics[0].keys())
            rows = [[row.get(column) for column in columns] for row in dynamics]
            wandb_run.log({"dynamics": wandb.Table(columns=columns, data=rows)})
        except Exception:
            pass
    _wandb_finish(wandb_run)
    return result
