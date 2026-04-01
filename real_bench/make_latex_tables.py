from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

try:
    max_numexpr = int(os.environ.get("NUMEXPR_MAX_THREADS", "64"))
    if max_numexpr > 64:
        os.environ["NUMEXPR_MAX_THREADS"] = "64"
except ValueError:
    os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ.setdefault("NUMEXPR_NUM_THREADS", "64")

import pandas as pd


def _escape_latex(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


_PRETTY_DATASET = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "svhn": "SVHN",
    "stl10": "STL-10",
    "tinyimagenet": "Tiny-ImageNet",
}

_PRETTY_METHOD = {
    "batchnorm": "BatchNorm",
    "layernorm": "LayerNorm",
    "groupnorm": "GroupNorm",
    "rmsnorm": "RMSNorm",
    "switchnorm": "SwitchNorm",
    "frn": "FRN",
    "evonorm_s0": "EvoNorm-S0",
    "evonorm_b0": "EvoNorm-B0",
    "panorm": "PA-Norm (BN/LN/GN)",
    "panorm_nodetach": "PA-Norm (no detach)",
    "panorm_lite": "PA-Norm-lite (BN/GN)",
    "panorm_lite_fast": "PA-Norm-lite-fast (BN/GN)",
    "panorm_switchnorm": "PA-SwitchNorm (BN/IN/LN)",
    "dropout": "NoNorm+Dropout",
}


def _pretty_dataset(name: str) -> str:
    return _PRETTY_DATASET.get(name.lower(), name)


def _pretty_method(name: str) -> str:
    return _PRETTY_METHOD.get(name.lower(), name)


def _sanitize_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def _format_mean_std(mean: float, std: float | None, digits: int = 2) -> str:
    if std is None or pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def build_accuracy_table(
    summary: pd.DataFrame,
    out_path: Path,
    *,
    architecture: str,
    protocol: str,
    metric: str = "acc1",
    digits: int = 2,
    caption: str | None = None,
    label: str | None = None,
) -> None:
    sub = summary[
        (summary["architecture"] == architecture) & (summary["protocol"] == protocol)
    ].copy()
    if sub.empty:
        raise RuntimeError(
            f"No rows for architecture={architecture} protocol={protocol}"
        )

    datasets = sorted(sub["dataset"].unique().tolist())
    methods = sub.sort_values(["rank", "method"])["method"].unique().tolist()

    best_by_dataset = {}
    for ds in datasets:
        best_by_dataset[ds] = float(sub[sub["dataset"] == ds][f"{metric}_mean"].max())

    lines: list[str] = []
    if caption is not None:
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\small")

    lines.append("\\begin{tabular}{l" + "c" * len(datasets) + "}")
    lines.append("\\toprule")
    header = (
        "Method & "
        + " & ".join(_escape_latex(_pretty_dataset(ds)) for ds in datasets)
        + " \\\\"
    )
    lines.append(header)
    lines.append("\\midrule")

    for method in methods:
        row = [f"{_escape_latex(_pretty_method(method))}"]
        for ds in datasets:
            cell = sub[(sub["dataset"] == ds) & (sub["method"] == method)]
            if cell.empty:
                row.append("--")
                continue
            mean = float(cell[f"{metric}_mean"].iloc[0])
            std = (
                float(cell[f"{metric}_std"].iloc[0])
                if f"{metric}_std" in cell.columns
                else None
            )
            val = _format_mean_std(mean, std, digits=digits)
            if abs(mean - best_by_dataset[ds]) < 1e-9:
                val = f"\\textbf{{{val}}}"
            row.append(val)
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    if caption is not None:
        lines.append(f"\\caption{{{caption}}}")
    if label is not None:
        lines.append(f"\\label{{{label}}}")
    if caption is not None:
        lines.append("\\end{table}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate LaTeX tables from exp/result_summary_real.csv"
    )
    p.add_argument("--summary-csv", type=str, default="exp/result_summary_real.csv")
    p.add_argument("--out-dir", type=str, default="paper/tables")
    p.add_argument("--metric", type=str, default="acc1", choices=["acc1", "acc5"])
    p.add_argument("--digits", type=int, default=2)
    p.add_argument(
        "--architectures",
        type=str,
        default=None,
        help="Comma-separated architectures to export",
    )
    p.add_argument(
        "--protocols",
        type=str,
        default=None,
        help="Comma-separated protocols to export (exact match)",
    )
    p.add_argument(
        "--wrap-table",
        action="store_true",
        default=False,
        help="Include table environment + caption/label",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary = pd.read_csv(args.summary_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    architectures = sorted(summary["architecture"].unique().tolist())
    if args.architectures:
        wanted = {x.strip() for x in args.architectures.split(",") if x.strip()}
        architectures = [a for a in architectures if a in wanted]

    protocols = sorted(summary["protocol"].unique().tolist())
    if args.protocols:
        wanted = {x.strip() for x in args.protocols.split(",") if x.strip()}
        protocols = [p for p in protocols if p in wanted]

    for arch in architectures:
        for proto in protocols:
            sub = summary[
                (summary["architecture"] == arch) & (summary["protocol"] == proto)
            ]
            if sub.empty:
                continue
            fname = f"real_{_sanitize_filename(arch)}_{_sanitize_filename(proto)}_{args.metric}.tex"
            caption = None
            label = None
            if args.wrap_table:
                caption = f"{arch} results (protocol: {_escape_latex(proto)})."
                label = f"tab:{_sanitize_filename(arch)}_{_sanitize_filename(proto)}_{args.metric}"
            build_accuracy_table(
                summary,
                out_dir / fname,
                architecture=arch,
                protocol=proto,
                metric=args.metric,
                digits=args.digits,
                caption=caption,
                label=label,
            )
            print(f"Wrote {out_dir / fname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
