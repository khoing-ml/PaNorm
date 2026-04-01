from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class EfficiencyMetrics:
    method: str
    architecture: str
    num_params_m: float
    flops_g: float
    train_throughput_img_s: float
    infer_throughput_img_s: float
    train_latency_ms: float
    infer_latency_ms: float
    peak_memory_mb: float


def count_flops(
    model: nn.Module, input_shape: tuple[int, ...], device: torch.device
) -> float:

    try:
        from torch.utils.flop_counter import FlopCounterMode

        inp = torch.randn(*input_shape, device=device)
        with FlopCounterMode(display=False) as counter:
            model(inp)
        return counter.get_total_flops() / 1e9
    except (ImportError, Exception):
        pass

    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1]
            total += (
                k
                * m.in_channels
                * m.out_channels
                * (input_shape[2] // max(m.stride[0], 1))
                * (input_shape[3] // max(m.stride[1], 1))
            )
        elif isinstance(m, nn.Linear):
            total += m.in_features * m.out_features
    return total / 1e9


def measure_throughput(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device,
    warmup_iters: int = 10,
    measure_iters: int = 50,
    use_amp: bool = False,
) -> tuple[float, float, float, float]:

    batch_size = input_shape[0]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.eval()
    inp = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        for _ in range(warmup_iters):
            model(inp)
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(measure_iters):
            model(inp)
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    infer_time = time.perf_counter() - start
    infer_throughput = (batch_size * measure_iters) / infer_time
    infer_latency = (infer_time / measure_iters) * 1000

    model.train()
    num_classes = (
        model.fc.out_features
        if hasattr(model, "fc")
        else model.classifier.out_features if hasattr(model, "classifier") else 10
    )
    target = torch.randint(0, num_classes, (batch_size,), device=device)

    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    for _ in range(warmup_iters):
        optimizer.zero_grad()
        with torch.amp.autocast(
            device_type=device.type, enabled=use_amp, dtype=amp_dtype
        ):
            out = model(inp)
            loss = criterion(out, target)
        loss.backward()
        optimizer.step()
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(measure_iters):
        optimizer.zero_grad()
        with torch.amp.autocast(
            device_type=device.type, enabled=use_amp, dtype=amp_dtype
        ):
            out = model(inp)
            loss = criterion(out, target)
        loss.backward()
        optimizer.step()
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    train_throughput = (batch_size * measure_iters) / train_time
    train_latency = (train_time / measure_iters) * 1000

    return train_throughput, infer_throughput, train_latency, infer_latency


def measure_peak_memory(
    model: nn.Module, input_shape: tuple[int, ...], device: torch.device
) -> float:

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        inp = torch.randn(*input_shape, device=device)
        model(inp)
        return torch.cuda.max_memory_allocated(device) / 1e6
    elif device.type == "xpu":
        try:
            torch.xpu.reset_peak_memory_stats(device)
            inp = torch.randn(*input_shape, device=device)
            model(inp)
            return torch.xpu.max_memory_allocated(device) / 1e6
        except Exception:
            return -1.0
    return -1.0


def profile_model(
    model: nn.Module,
    method: str,
    architecture: str,
    input_shape: tuple[int, ...],
    device: torch.device,
    use_amp: bool = False,
) -> EfficiencyMetrics:

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    flops = count_flops(model, input_shape, device)
    train_tp, infer_tp, train_lat, infer_lat = measure_throughput(
        model, input_shape, device, use_amp=use_amp
    )
    peak_mem = measure_peak_memory(model, input_shape, device)

    return EfficiencyMetrics(
        method=method,
        architecture=architecture,
        num_params_m=num_params,
        flops_g=flops,
        train_throughput_img_s=train_tp,
        infer_throughput_img_s=infer_tp,
        train_latency_ms=train_lat,
        infer_latency_ms=infer_lat,
        peak_memory_mb=peak_mem,
    )
