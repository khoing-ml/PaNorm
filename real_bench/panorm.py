from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_group_count(channels: int, requested_groups: int) -> int:
    g = min(channels, max(1, requested_groups))
    while channels % g != 0 and g > 1:
        g -= 1
    return g


class LayerNorm2d(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        y = F.group_norm(
            x_f, num_groups=1, weight=self.weight, bias=self.bias, eps=self.eps
        )
        return y.to(dtype=x.dtype)


class RMSNorm2d(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)
        return x * rms * self.scale + self.shift


class FRN2d(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.tau = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nu2 = x.pow(2).mean(dim=(2, 3), keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps)
        x = self.gamma * x + self.beta
        return torch.maximum(x, self.tau)


class SwitchNorm2d(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

        self.mean_weight = nn.Parameter(
            torch.tensor([2.0, 1.0, 1.0], dtype=torch.float32)
        )
        self.var_weight = nn.Parameter(
            torch.tensor([2.0, 1.0, 1.0], dtype=torch.float32)
        )

        self.register_buffer("running_mean", torch.zeros(num_channels))
        self.register_buffer("running_var", torch.ones(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        if not x_f.is_contiguous():
            x_f = x_f.contiguous()
        var_in, mean_in = torch.var_mean(x_f, dim=(2, 3), keepdim=True, unbiased=False)

        var_ln, mean_ln = torch.var_mean(
            x_f, dim=(1, 2, 3), keepdim=True, unbiased=False
        )

        if self.training:
            var_bn, mean_bn = torch.var_mean(
                x_f, dim=(0, 2, 3), keepdim=True, unbiased=False
            )
            self.running_mean.mul_(1.0 - self.momentum).add_(
                self.momentum * mean_bn.view(-1).detach()
            )
            self.running_var.mul_(1.0 - self.momentum).add_(
                self.momentum * var_bn.view(-1).detach()
            )
        else:
            mean_bn = self.running_mean.view(1, -1, 1, 1)
            var_bn = self.running_var.view(1, -1, 1, 1)

        mean_mix = torch.softmax(self.mean_weight, dim=0)
        var_mix = torch.softmax(self.var_weight, dim=0)

        mean = mean_mix[0] * mean_bn + mean_mix[1] * mean_in + mean_mix[2] * mean_ln
        var = var_mix[0] * var_bn + var_mix[1] * var_in + var_mix[2] * var_ln
        x_hat = (x_f - mean) * torch.rsqrt(var + self.eps)
        y = x_hat * self.weight + self.bias
        return y.to(dtype=x.dtype)


class _BasePANorm(nn.Module):
    def __init__(self, num_channels: int, gate_dim: int, detach_diagnostic: bool):
        super().__init__()
        self.num_channels = num_channels
        self.detach_diagnostic = detach_diagnostic
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

        self.register_buffer("gate_sum", torch.zeros(gate_dim))
        self.register_buffer("gate_count", torch.tensor(0.0))

    def _update_gate_stats(self, gate: torch.Tensor) -> None:
        with torch.no_grad():
            self.gate_sum.add_(gate.detach())
            self.gate_count.add_(1.0)

    def reset_gate_statistics(self) -> None:
        self.gate_sum.zero_()
        self.gate_count.zero_()

    def gate_statistics(self) -> dict[str, float]:
        count = max(float(self.gate_count.item()), 1.0)
        probs = self.gate_sum / count
        entropy = float(
            -(probs * torch.log(probs.clamp_min(1e-8))).sum().item()
            / math.log(len(probs))
        )
        return {
            "gate_entropy_norm": entropy,
            "gate_main_weight": float(probs.max().item()),
            "gate_branch0": float(probs[0].item()),
            "gate_branch1": float(probs[1].item()) if len(probs) > 1 else 0.0,
            "gate_branch2": float(probs[2].item()) if len(probs) > 2 else 0.0,
        }

    def _as_float(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float32:
            return x
        return x.float()

    def _bn_stats(
        self,
        x_f: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        momentum: float,
        eps: float,
    ):

        x_c = x_f.contiguous() if not x_f.is_contiguous() else x_f
        return F.batch_norm(
            x_c,
            running_mean=running_mean,
            running_var=running_var,
            weight=None,
            bias=None,
            training=self.training,
            momentum=momentum,
            eps=eps,
        )

    def _ln_stats(self, x_f: torch.Tensor, eps: float) -> torch.Tensor:
        x_c = x_f.contiguous() if not x_f.is_contiguous() else x_f
        return F.group_norm(x_c, num_groups=1, weight=None, bias=None, eps=eps)

    def _gn_stats(self, x_f: torch.Tensor, groups: int, eps: float) -> torch.Tensor:
        x_c = x_f.contiguous() if not x_f.is_contiguous() else x_f
        return F.group_norm(x_c, num_groups=groups, weight=None, bias=None, eps=eps)

    def _descriptor_from_float(self, x_f: torch.Tensor, eps: float) -> torch.Tensor:

        x_c = x_f.contiguous() if not x_f.is_contiguous() else x_f
        batch_var = x_c.var(dim=(0, 2, 3), unbiased=False).mean()
        sample_var = x_c.var(dim=(1, 2, 3), unbiased=False).mean()
        channel_var = x_c.var(dim=1, unbiased=False).mean()
        desc = torch.stack([batch_var, sample_var, channel_var])
        desc = torch.log(desc.clamp_min(eps))
        if self.detach_diagnostic:
            desc = desc.detach()
        return torch.tanh(desc)

    def _descriptor(self, x: torch.Tensor, eps: float) -> torch.Tensor:
        return self._descriptor_from_float(self._as_float(x), eps)


class PANormSwitch2d(_BasePANorm):

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        detach_diagnostic: bool = True,
    ):
        super().__init__(
            num_channels=num_channels, gate_dim=3, detach_diagnostic=detach_diagnostic
        )
        self.eps = eps
        self.momentum = momentum

        self.mean_logits = nn.Parameter(
            torch.tensor([2.0, 1.0, 1.0], dtype=torch.float32)
        )
        self.var_logits = nn.Parameter(
            torch.tensor([2.0, 1.0, 1.0], dtype=torch.float32)
        )
        self.mean_proj = nn.Parameter(torch.zeros(3, 3, dtype=torch.float32))
        self.var_proj = nn.Parameter(torch.zeros(3, 3, dtype=torch.float32))

        self.register_buffer("running_mean", torch.zeros(num_channels))
        self.register_buffer("running_var", torch.ones(num_channels))

    def _descriptor_from_vars(
        self, var_bn: torch.Tensor, var_in: torch.Tensor, var_ln: torch.Tensor
    ) -> torch.Tensor:
        a = torch.log(var_bn.mean().clamp_min(self.eps))
        b = torch.log(var_in.mean().clamp_min(self.eps))
        c = torch.log(var_ln.mean().clamp_min(self.eps))
        desc = torch.tanh(torch.stack([a, b, c]))
        if self.detach_diagnostic:
            desc = desc.detach()
        return desc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        if not x_f.is_contiguous():
            x_f = x_f.contiguous()
        var_in, mean_in = torch.var_mean(x_f, dim=(2, 3), keepdim=True, unbiased=False)

        var_ln, mean_ln = torch.var_mean(
            x_f, dim=(1, 2, 3), keepdim=True, unbiased=False
        )

        if self.training:
            var_bn, mean_bn = torch.var_mean(
                x_f, dim=(0, 2, 3), keepdim=True, unbiased=False
            )
            self.running_mean.mul_(1.0 - self.momentum).add_(
                self.momentum * mean_bn.view(-1).detach()
            )
            self.running_var.mul_(1.0 - self.momentum).add_(
                self.momentum * var_bn.view(-1).detach()
            )
        else:
            mean_bn = self.running_mean.view(1, -1, 1, 1)
            var_bn = self.running_var.view(1, -1, 1, 1)

        desc = self._descriptor_from_vars(var_bn=var_bn, var_in=var_in, var_ln=var_ln)
        mean_mix = torch.softmax(self.mean_logits + (self.mean_proj @ desc), dim=0)
        var_mix = torch.softmax(self.var_logits + (self.var_proj @ desc), dim=0)
        self._update_gate_stats(0.5 * (mean_mix + var_mix))

        mean = mean_mix[0] * mean_bn + mean_mix[1] * mean_in + mean_mix[2] * mean_ln
        var = var_mix[0] * var_bn + var_mix[1] * var_in + var_mix[2] * var_ln
        x_hat = (x_f - mean) * torch.rsqrt(var + self.eps)
        y = x_hat * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return y.to(dtype=x.dtype)


class PANorm2d(_BasePANorm):

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        groups: int = 8,
        detach_diagnostic: bool = True,
    ):
        super().__init__(
            num_channels=num_channels, gate_dim=3, detach_diagnostic=detach_diagnostic
        )
        self.eps = eps
        self.momentum = momentum
        self.groups = _valid_group_count(num_channels, groups)

        self.gate_logits = nn.Parameter(
            torch.tensor([2.8, 0.2, 1.1], dtype=torch.float32)
        )
        self.diag_proj = nn.Parameter(
            torch.tensor(
                [
                    [0.08, -0.05, 0.08],
                    [-0.03, 0.08, -0.01],
                    [-0.05, -0.04, -0.07],
                ],
                dtype=torch.float32,
            )
        )

        self.register_buffer("running_mean", torch.zeros(num_channels))
        self.register_buffer("running_var", torch.ones(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = self._as_float(x)
        if not x_f.is_contiguous():
            x_f = x_f.contiguous()
        desc = self._descriptor_from_float(x_f, self.eps)
        shift = self.diag_proj @ desc
        gate = torch.softmax(self.gate_logits + shift, dim=0)
        self._update_gate_stats(gate)

        x_bn = self._bn_stats(
            x_f, self.running_mean, self.running_var, self.momentum, self.eps
        )
        x_ln = self._ln_stats(x_f, self.eps)
        x_gn = self._gn_stats(x_f, self.groups, self.eps)

        mixed = gate[0] * x_bn + gate[1] * x_ln + gate[2] * x_gn
        y = mixed * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return y.to(dtype=x.dtype)


class PANormLite2d(_BasePANorm):

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        groups: int = 8,
        detach_diagnostic: bool = True,
        train_single_path: bool = False,
        eval_single_path: bool = True,
        single_path_threshold: float = 0.82,
        diag_update_interval: int = 1,
        diag_ema_momentum: float = 0.2,
    ):
        super().__init__(
            num_channels=num_channels, gate_dim=2, detach_diagnostic=detach_diagnostic
        )
        self.eps = eps
        self.momentum = momentum
        self.groups = _valid_group_count(num_channels, groups)
        self.train_single_path = train_single_path
        self.eval_single_path = eval_single_path
        self.single_path_threshold = single_path_threshold
        self.diag_update_interval = max(1, int(diag_update_interval))
        self.diag_ema_momentum = float(diag_ema_momentum)

        self.register_buffer("diag_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("diag_ema", torch.zeros(2))

        self.gate_logits = nn.Parameter(torch.tensor([2.4, 0.4], dtype=torch.float32))
        self.diag_proj = nn.Parameter(
            torch.tensor([[0.08, -0.03], [-0.06, -0.05]], dtype=torch.float32)
        )

        self.register_buffer("running_mean", torch.zeros(num_channels))
        self.register_buffer("running_var", torch.ones(num_channels))

    def _descriptor_lite(self, x: torch.Tensor) -> torch.Tensor:
        x_f = self._as_float(x)
        if not x_f.is_contiguous():
            x_f = x_f.contiguous()
        batch_var = x_f.var(dim=(0, 2, 3), unbiased=False).mean()
        channel_var = x_f.var(dim=1, unbiased=False).mean()
        desc = torch.stack([batch_var, channel_var])
        desc = torch.log(desc.clamp_min(self.eps))
        if self.detach_diagnostic:
            desc = desc.detach()
        return torch.tanh(desc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = self._as_float(x)
        if not x_f.is_contiguous():
            x_f = x_f.contiguous()
        if (
            (not self.training)
            or (not self.detach_diagnostic)
            or self.diag_update_interval <= 1
        ):
            desc = self._descriptor_lite(x_f)
        else:
            step = int(self.diag_step.item())
            if step % self.diag_update_interval == 0:
                desc_now = self._descriptor_lite(x_f)
                with torch.no_grad():
                    if step == 0:
                        self.diag_ema.copy_(desc_now)
                    else:
                        self.diag_ema.mul_(1.0 - self.diag_ema_momentum).add_(
                            self.diag_ema_momentum * desc_now
                        )
            desc = self.diag_ema
            with torch.no_grad():
                self.diag_step.add_(1)

        shift = self.diag_proj @ desc
        gate = torch.softmax(self.gate_logits + shift, dim=0)
        self._update_gate_stats(gate)

        use_single_path = (self.training and self.train_single_path) or (
            (not self.training) and self.eval_single_path
        )
        if use_single_path and float(gate.max().item()) >= self.single_path_threshold:
            dominant = int(gate.argmax().item())
            if dominant == 0:
                mixed = self._bn_stats(
                    x_f, self.running_mean, self.running_var, self.momentum, self.eps
                )
            else:

                if self.training:
                    self._bn_stats(
                        x_f,
                        self.running_mean,
                        self.running_var,
                        self.momentum,
                        self.eps,
                    )
                mixed = self._gn_stats(x_f, self.groups, self.eps)
        else:
            x_bn = self._bn_stats(
                x_f, self.running_mean, self.running_var, self.momentum, self.eps
            )
            x_gn = self._gn_stats(x_f, self.groups, self.eps)
            mixed = gate[0] * x_bn + gate[1] * x_gn

        y = mixed * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return y.to(dtype=x.dtype)


class PANormSelective2d(_BasePANorm):

    import threading

    _lock = threading.Lock()
    _instance_counter = 0
    _total_instances = 0

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        groups: int = 8,
        detach_fraction: float = 0.5,
    ):

        with PANormSelective2d._lock:
            PANormSelective2d._instance_counter += 1
            self._layer_idx = PANormSelective2d._instance_counter

        super().__init__(num_channels=num_channels, gate_dim=3, detach_diagnostic=True)
        self.eps = eps
        self.momentum = momentum
        self.groups = _valid_group_count(num_channels, groups)
        self.detach_fraction = detach_fraction

        self.gate_logits = nn.Parameter(
            torch.tensor([2.8, 0.2, 1.1], dtype=torch.float32)
        )
        self.diag_proj = nn.Parameter(
            torch.tensor(
                [
                    [0.08, -0.05, 0.08],
                    [-0.03, 0.08, -0.01],
                    [-0.05, -0.04, -0.07],
                ],
                dtype=torch.float32,
            )
        )

        self.register_buffer("running_mean", torch.zeros(num_channels))
        self.register_buffer("running_var", torch.ones(num_channels))

    @classmethod
    def reset_counter(cls):
        with cls._lock:
            cls._instance_counter = 0
            cls._total_instances = 0

    @classmethod
    def finalize_counter(cls):
        with cls._lock:
            cls._total_instances = cls._instance_counter

    def _should_detach(self) -> bool:
        total = max(PANormSelective2d._total_instances, self._layer_idx)
        threshold = int(total * self.detach_fraction)
        return self._layer_idx <= threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = self._as_float(x)
        if not x_f.is_contiguous():
            x_f = x_f.contiguous()

        should_detach = self._should_detach()
        desc = self._descriptor_from_float(x_f, self.eps)
        if should_detach and not desc.requires_grad:
            pass
        elif not should_detach and not desc.requires_grad:

            x_c = x_f.contiguous() if not x_f.is_contiguous() else x_f
            batch_var = x_c.var(dim=(0, 2, 3), unbiased=False).mean()
            sample_var = x_c.var(dim=(1, 2, 3), unbiased=False).mean()
            channel_var = x_c.var(dim=1, unbiased=False).mean()
            desc = torch.stack([batch_var, sample_var, channel_var])
            desc = torch.log(desc.clamp_min(self.eps))
            desc = torch.tanh(desc)

        shift = self.diag_proj @ desc
        gate = torch.softmax(self.gate_logits + shift, dim=0)
        self._update_gate_stats(gate)

        x_bn = self._bn_stats(
            x_f, self.running_mean, self.running_var, self.momentum, self.eps
        )
        x_ln = self._ln_stats(x_f, self.eps)
        x_gn = self._gn_stats(x_f, self.groups, self.eps)

        mixed = gate[0] * x_bn + gate[1] * x_ln + gate[2] * x_gn
        y = mixed * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return y.to(dtype=x.dtype)


class PANorm1d(nn.Module):

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        detach_diagnostic: bool = True,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.detach_diagnostic = detach_diagnostic

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

        self.gate_logits = nn.Parameter(
            torch.tensor([2.5, 1.0, -1.0], dtype=torch.float32)
        )
        self.diag_proj = nn.Parameter(torch.zeros(3, 3, dtype=torch.float32))

        self.register_buffer("gate_sum", torch.zeros(3))
        self.register_buffer("gate_count", torch.tensor(0.0))

    def _descriptor(self, x: torch.Tensor) -> torch.Tensor:

        x_f = x.float()

        batch_var = x_f.var(dim=0, unbiased=False).mean()

        token_var = x_f.var(dim=1, unbiased=False).mean()

        channel_var = x_f.var(dim=2, unbiased=False).mean()

        desc = torch.stack([batch_var, token_var, channel_var])
        desc = torch.log(desc.clamp_min(self.eps))
        if self.detach_diagnostic:
            desc = desc.detach()
        return torch.tanh(desc)

    def reset_gate_statistics(self) -> None:
        self.gate_sum.zero_()
        self.gate_count.zero_()

    def gate_statistics(self) -> dict[str, float]:
        count = max(float(self.gate_count.item()), 1.0)
        probs = self.gate_sum / count
        entropy = float(
            -(probs * torch.log(probs.clamp_min(1e-8))).sum().item()
            / math.log(len(probs))
        )
        return {
            "gate_entropy_norm": entropy,
            "gate_main_weight": float(probs.max().item()),
            "gate_branch0": float(probs[0].item()),
            "gate_branch1": float(probs[1].item()),
            "gate_branch2": float(probs[2].item()),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()

        squeezed = False
        if x_f.dim() == 2:
            x_f = x_f.unsqueeze(1)
            squeezed = True

        desc = self._descriptor(x_f)
        shift = self.diag_proj @ desc
        gate = torch.softmax(self.gate_logits + shift, dim=0)

        with torch.no_grad():
            self.gate_sum.add_(gate.detach())
            self.gate_count.add_(1.0)

        x_ln = F.layer_norm(x_f, [self.normalized_shape], eps=self.eps)

        rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_rms = x_f * rms

        x_id = x_f - x_f.mean(dim=-1, keepdim=True)

        mixed = gate[0] * x_ln + gate[1] * x_rms + gate[2] * x_id
        y = mixed * self.weight + self.bias
        if squeezed:
            y = y.squeeze(1)
        return y.to(dtype=x.dtype)


class ScaleNorm2d(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1) * num_channels**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        norm = torch.sqrt(x_f.pow(2).sum(dim=(1, 2, 3), keepdim=True) + self.eps)
        return (x_f / norm * self.scale).to(dtype=x.dtype)


class SubLN2d(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-5, init_alpha: float = 0.5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        x_c = x_f.contiguous() if not x_f.is_contiguous() else x_f
        ln_out = F.group_norm(
            x_c, num_groups=1, weight=self.weight, bias=self.bias, eps=self.eps
        )
        alpha = torch.sigmoid(self.alpha)
        y = alpha * ln_out + (1 - alpha) * x_f
        return y.to(dtype=x.dtype)


class PowerNorm2d(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("running_phi", torch.ones(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        mean = x_f.mean(dim=(0, 2, 3), keepdim=True)
        x_centered = x_f - mean
        if self.training:
            phi = x_centered.pow(2).mean(dim=(0, 2, 3))
            self.running_phi.mul_(1 - self.momentum).add_(self.momentum * phi.detach())
            phi_use = phi.view(1, -1, 1, 1)
        else:
            phi_use = self.running_phi.view(1, -1, 1, 1)
        y = x_centered * torch.rsqrt(phi_use + self.eps)
        return (y * self.weight + self.bias).to(dtype=x.dtype)


class EvoNormS0_2d(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-5, groups: int = 32):
        super().__init__()
        self.eps = eps
        self.groups = _valid_group_count(num_channels, groups)
        self.v = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def _group_std(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        g = self.groups
        y = x.reshape(n, g, c // g, h, w)
        var = y.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        std = std.expand(n, g, c // g, h, w).reshape(n, c, h, w)
        return std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num = x * torch.sigmoid(self.v * x)
        den = self._group_std(x)
        y = num / den
        return y * self.gamma + self.beta


class EvoNormB0_2d(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.v = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

        self.register_buffer("running_var", torch.ones(num_channels))

    def _batch_std(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            var = x.var(dim=(0, 2, 3), keepdim=False, unbiased=False)
            self.running_var.mul_(1.0 - self.momentum).add_(
                self.momentum * var.detach()
            )
        else:
            var = self.running_var
        return torch.sqrt(var.view(1, -1, 1, 1) + self.eps)

    def _instance_std(self, x: torch.Tensor) -> torch.Tensor:
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        return torch.sqrt(var + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_std = self._batch_std(x)
        inst_std = self._instance_std(x)
        den = torch.maximum(batch_std, self.v * x + inst_std)
        y = x / den
        return y * self.gamma + self.beta


class BatchRenorm2d(nn.Module):

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        r_max_init: float = 1.0,
        d_max_init: float = 0.0,
        r_max_final: float = 3.0,
        d_max_final: float = 5.0,
        warmup_steps: int = 5000,
    ):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.r_max_init = r_max_init
        self.d_max_init = d_max_init
        self.r_max_final = r_max_final
        self.d_max_final = d_max_final
        self.warmup_steps = warmup_steps

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.register_buffer("running_mean", torch.zeros(num_channels))
        self.register_buffer("running_var", torch.ones(num_channels))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        if not x_f.is_contiguous():
            x_f = x_f.contiguous()

        if self.training:
            step = self.num_batches_tracked.item()
            self.num_batches_tracked += 1

            var_b, mean_b = torch.var_mean(x_f, dim=(0, 2, 3), unbiased=False)
            std_b = torch.sqrt(var_b + self.eps)

            running_std = torch.sqrt(self.running_var + self.eps)

            r = (std_b.detach() / running_std).clamp(
                1.0 / self._r_max(step), self._r_max(step)
            )
            d = ((mean_b.detach() - self.running_mean) / running_std).clamp(
                -self._d_max(step), self._d_max(step)
            )

            x_hat = (x_f - mean_b.view(1, -1, 1, 1)) / std_b.view(1, -1, 1, 1)
            x_hat = x_hat * r.view(1, -1, 1, 1) + d.view(1, -1, 1, 1)

            self.running_mean.mul_(1.0 - self.momentum).add_(
                self.momentum * mean_b.detach()
            )
            self.running_var.mul_(1.0 - self.momentum).add_(
                self.momentum * var_b.detach()
            )
        else:
            x_hat = (x_f - self.running_mean.view(1, -1, 1, 1)) / torch.sqrt(
                self.running_var.view(1, -1, 1, 1) + self.eps
            )

        y = x_hat * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return y.to(dtype=x.dtype)

    def _r_max(self, step: int) -> float:
        if step >= self.warmup_steps:
            return self.r_max_final
        t = step / self.warmup_steps
        return self.r_max_init + (self.r_max_final - self.r_max_init) * t

    def _d_max(self, step: int) -> float:
        if step >= self.warmup_steps:
            return self.d_max_final
        t = step / self.warmup_steps
        return self.d_max_init + (self.d_max_final - self.d_max_init) * t


class WeightStdConv2d(nn.Conv2d):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight

        mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        var = weight.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        weight = (weight - mean) / torch.sqrt(var + 1e-5)
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
