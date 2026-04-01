from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models as tv_models

from .panorm import (
    BatchRenorm2d,
    EvoNormB0_2d,
    EvoNormS0_2d,
    FRN2d,
    LayerNorm2d,
    PANorm1d,
    PANorm2d,
    PANormLite2d,
    PANormSelective2d,
    PANormSwitch2d,
    PowerNorm2d,
    RMSNorm2d,
    ScaleNorm2d,
    SubLN2d,
    SwitchNorm2d,
    WeightStdConv2d,
)


def _group_count(channels: int, preferred: int = 8) -> int:
    g = min(channels, preferred)
    while channels % g != 0 and g > 1:
        g -= 1
    return g


class IdentityNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def make_norm(norm_type: str, channels: int) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type == "batchnorm":
        return nn.BatchNorm2d(channels)
    if norm_type == "layernorm":
        return LayerNorm2d(channels)
    if norm_type == "groupnorm":
        return nn.GroupNorm(_group_count(channels), channels)
    if norm_type == "rmsnorm":
        return RMSNorm2d(channels)
    if norm_type == "switchnorm":
        return SwitchNorm2d(channels)
    if norm_type == "frn":
        return FRN2d(channels)
    if norm_type == "evonorm_s0":
        return EvoNormS0_2d(channels, groups=32)
    if norm_type == "evonorm_b0":
        return EvoNormB0_2d(channels)
    if norm_type == "panorm":
        return PANorm2d(channels, groups=_group_count(channels))
    if norm_type == "panorm_switchnorm":
        return PANormSwitch2d(channels)
    if norm_type == "panorm_nodetach":
        return PANorm2d(
            channels, groups=_group_count(channels), detach_diagnostic=False
        )
    if norm_type == "panorm_lite":
        return PANormLite2d(channels, groups=_group_count(channels))
    if norm_type == "panorm_switchnorm_nodetach":
        return PANormSwitch2d(channels, detach_diagnostic=False)
    if norm_type == "panorm_lite_hard":
        return PANormLite2d(
            channels, groups=_group_count(channels), train_single_path=True
        )
    if norm_type == "panorm_lite_fast":
        return PANormLite2d(
            channels,
            groups=_group_count(channels),
            train_single_path=True,
            diag_update_interval=8,
        )
    if norm_type == "panorm_selective":
        return PANormSelective2d(
            channels, groups=_group_count(channels), detach_fraction=0.5
        )
    if norm_type == "panorm_selective_75":
        return PANormSelective2d(
            channels, groups=_group_count(channels), detach_fraction=0.75
        )
    if norm_type == "scalenorm":
        return ScaleNorm2d(channels)
    if norm_type == "subln":
        return SubLN2d(channels)
    if norm_type == "powernorm":
        return PowerNorm2d(channels)
    if norm_type == "batchrenorm":
        return BatchRenorm2d(channels)
    if norm_type == "ws_gn":

        return nn.GroupNorm(_group_count(channels), channels)
    if norm_type == "dropout":
        return IdentityNorm()
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ConvBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, stride: int, norm_type: str, dropout_p: float
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm = make_norm(norm_type, out_ch)
        if norm_type.lower() in {"frn", "evonorm_s0", "evonorm_b0"}:
            self.act = nn.Identity()
        else:
            self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return self.drop(x)


class SmallConvNet(nn.Module):

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        width: int = 64,
        norm_type: str = "batchnorm",
    ):
        super().__init__()
        dropout = 0.3 if norm_type == "dropout" else 0.1

        c1, c2, c3 = width, width * 2, width * 4
        self.features = nn.Sequential(
            ConvBlock(
                in_channels, c1, stride=1, norm_type=norm_type, dropout_p=dropout
            ),
            ConvBlock(c1, c1, stride=1, norm_type=norm_type, dropout_p=dropout),
            ConvBlock(c1, c2, stride=2, norm_type=norm_type, dropout_p=dropout),
            ConvBlock(c2, c2, stride=1, norm_type=norm_type, dropout_p=dropout),
            ConvBlock(c2, c3, stride=2, norm_type=norm_type, dropout_p=dropout),
            ConvBlock(c3, c3, stride=1, norm_type=norm_type, dropout_p=dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(c3, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


@dataclass
class ModelSpec:
    name: str
    in_channels: int
    num_classes: int
    width: int = 64


def _resnet_norm_layer(norm_type: str):
    def _make(channels: int) -> nn.Module:
        return make_norm(norm_type, channels)

    return _make


def _build_resnet(
    architecture: str,
    norm_type: str,
    in_channels: int,
    num_classes: int,
    image_size: int,
) -> nn.Module:
    kwargs = {
        "weights": None,
        "num_classes": num_classes,
        "norm_layer": _resnet_norm_layer(norm_type),
    }
    if architecture == "resnet18":
        model = tv_models.resnet18(**kwargs)
    elif architecture == "resnet50":
        model = tv_models.resnet50(**kwargs)
    else:
        raise ValueError(f"Unsupported resnet architecture: {architecture}")

    if image_size <= 64:
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()
    elif in_channels != 3:
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    elif in_channels == 3:

        pass

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_wide_resnet(
    norm_type: str,
    in_channels: int,
    num_classes: int,
    image_size: int,
) -> nn.Module:
    kwargs = {
        "weights": None,
        "num_classes": num_classes,
        "norm_layer": _resnet_norm_layer(norm_type),
    }
    model = tv_models.wide_resnet50_2(**kwargs)
    if image_size <= 64:
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _apply_weight_standardization(module: nn.Module) -> None:

    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and not isinstance(child, WeightStdConv2d):
            ws_conv = WeightStdConv2d(
                child.in_channels,
                child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
            )
            ws_conv.weight = child.weight
            if child.bias is not None:
                ws_conv.bias = child.bias
            setattr(module, name, ws_conv)
        else:
            _apply_weight_standardization(child)


def _replace_norm_in_module(
    module: nn.Module, norm_type: str, replace_1d_ln: bool = False
) -> None:

    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            setattr(module, name, make_norm(norm_type, child.num_features))
        elif isinstance(child, nn.GroupNorm):
            setattr(module, name, make_norm(norm_type, child.num_channels))
        elif (
            hasattr(child, "normalized_shape") and type(child).__name__ == "LayerNorm2d"
        ):

            setattr(module, name, make_norm(norm_type, child.normalized_shape[0]))
        elif (
            replace_1d_ln
            and isinstance(child, nn.LayerNorm)
            and len(child.normalized_shape) == 1
        ):
            setattr(module, name, make_norm(norm_type, child.normalized_shape[0]))
        else:
            _replace_norm_in_module(child, norm_type, replace_1d_ln=replace_1d_ln)


def _build_convnext(
    norm_type: str,
    in_channels: int,
    num_classes: int,
    image_size: int,
) -> nn.Module:
    model = tv_models.convnext_tiny(weights=None, num_classes=num_classes)
    if norm_type != "layernorm":
        _replace_norm_in_module(model, norm_type)
    if image_size <= 64:
        stem_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels,
            stem_conv.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
    return model


def _build_efficientnet(
    norm_type: str,
    in_channels: int,
    num_classes: int,
    image_size: int,
) -> nn.Module:
    model = tv_models.efficientnet_b0(weights=None, num_classes=num_classes)
    if norm_type != "batchnorm":
        _replace_norm_in_module(model, norm_type)
    return model


class _ScaleNorm1d(nn.Module):

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1) * normalized_shape**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        norm = torch.sqrt(x_f.pow(2).sum(dim=-1, keepdim=True) + self.eps)
        return (x_f / norm * self.scale).to(dtype=x.dtype)


class _SubLN1d(nn.Module):

    def __init__(
        self, normalized_shape: int, eps: float = 1e-5, init_alpha: float = 0.5
    ):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.alpha)
        return alpha * self.ln(x) + (1 - alpha) * x


def _make_norm_1d(norm_type: str, normalized_shape: int) -> nn.Module:

    norm_type = norm_type.lower()
    if norm_type in ("layernorm", "batchnorm", "groupnorm"):
        return nn.LayerNorm(normalized_shape)
    if norm_type == "rmsnorm":
        return _RMSNorm1d(normalized_shape)
    if norm_type == "scalenorm":
        return _ScaleNorm1d(normalized_shape)
    if norm_type == "subln":
        return _SubLN1d(normalized_shape)
    if norm_type in ("panorm", "panorm_switchnorm"):
        return PANorm1d(normalized_shape, detach_diagnostic=True)
    if norm_type in ("panorm_nodetach", "panorm_switchnorm_nodetach"):
        return PANorm1d(normalized_shape, detach_diagnostic=False)
    if norm_type == "panorm_lite_fast":
        return PANorm1d(normalized_shape, detach_diagnostic=True)

    return nn.LayerNorm(normalized_shape)


class _RMSNorm1d(nn.Module):

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * rms * self.scale).to(dtype=x.dtype)


class SimpleViT(nn.Module):

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        norm_type: str = "layernorm",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    norm_type=norm_type,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm = _make_norm_1d(norm_type, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, 0])
        return self.head(x)


class _TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, norm_type, dropout):
        super().__init__()
        self.norm1 = _make_norm_1d(norm_type, dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = _make_norm_1d(norm_type, dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


def _build_vit(
    architecture: str,
    norm_type: str,
    in_channels: int,
    num_classes: int,
    image_size: int,
) -> nn.Module:

    if architecture == "vit_tiny":
        return SimpleViT(
            image_size=image_size,
            patch_size=4 if image_size <= 64 else 16,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=192,
            depth=12,
            num_heads=3,
            norm_type=norm_type,
        )
    elif architecture == "vit_small":
        return SimpleViT(
            image_size=image_size,
            patch_size=4 if image_size <= 64 else 16,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=384,
            depth=12,
            num_heads=6,
            norm_type=norm_type,
        )
    raise ValueError(f"Unknown ViT architecture: {architecture}")


def build_model(
    norm_type: str,
    in_channels: int,
    num_classes: int,
    width: int = 64,
    architecture: str = "smallcnn",
    image_size: int = 32,
) -> nn.Module:
    use_selective = norm_type.startswith("panorm_selective")
    if use_selective:
        PANormSelective2d.reset_counter()

    architecture = architecture.lower()
    model: nn.Module
    if architecture == "smallcnn":
        model = SmallConvNet(
            in_channels=in_channels,
            num_classes=num_classes,
            width=width,
            norm_type=norm_type,
        )
    elif architecture in {"resnet18", "resnet50"}:
        model = _build_resnet(
            architecture=architecture,
            norm_type=norm_type,
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
        )
    elif architecture == "wide_resnet50_2":
        model = _build_wide_resnet(
            norm_type=norm_type,
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
        )
    elif architecture == "convnext_tiny":
        model = _build_convnext(
            norm_type=norm_type,
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
        )
    elif architecture == "efficientnet_b0":
        model = _build_efficientnet(
            norm_type=norm_type,
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
        )
    elif architecture in {"vit_small", "vit_tiny"}:
        model = _build_vit(
            architecture=architecture,
            norm_type=norm_type,
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
        )
    elif architecture == "deit_small":
        model = _build_vit(
            architecture="vit_small",
            norm_type=norm_type,
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    if use_selective:
        PANormSelective2d.finalize_counter()

    if norm_type.lower() == "ws_gn":
        _apply_weight_standardization(model)

    return model
