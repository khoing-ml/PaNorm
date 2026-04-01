from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import random
import subprocess
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

import pandas as pd
import torch

from real_bench import BenchmarkConfig, prepare_datasets, run_single_experiment
from real_bench.stats import compute_summary_tables


DEFAULT_DATA_ROOT = "/lus/flare/projects/RobustViT/kim/pa_norm/data"
DEFAULT_METHODS = (
    "batchnorm,layernorm,groupnorm,rmsnorm,switchnorm,frn,panorm,panorm_lite"
)
DEFAULT_DATASETS = "cifar10,cifar100,svhn,stl10"
DEFAULT_ARCHITECTURES = "smallcnn,resnet18"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-data PA-Norm benchmark.")

    parser.add_argument("--datasets", type=str, default=DEFAULT_DATASETS)
    parser.add_argument("--methods", type=str, default=DEFAULT_METHODS)
    parser.add_argument("--architectures", type=str, default=DEFAULT_ARCHITECTURES)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--disable-amp", action="store_true", default=False)
    parser.add_argument("--eval-ood", action="store_true", default=False)
    parser.add_argument("--ood-severity", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)

    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=str, default="exp/result.csv")
    parser.add_argument(
        "--summary-output", type=str, default="exp/result_summary_real.csv"
    )
    parser.add_argument("--stats-output", type=str, default="exp/result_stats_real.csv")
    parser.add_argument(
        "--ranking-output", type=str, default="exp/result_ranking_real.csv"
    )

    parser.add_argument(
        "--device-ids", type=str, default=None, help="Comma-separated XPU/CUDA ids"
    )
    parser.add_argument(
        "--stream-worker-logs",
        dest="stream_worker_logs",
        action="store_true",
        default=True,
        help="Stream worker log lines to stdout while workers are running (default: enabled)",
    )
    parser.add_argument(
        "--no-stream-worker-logs",
        dest="stream_worker_logs",
        action="store_false",
        help="Disable live worker log streaming; logs are still saved to worker_*.log",
    )
    parser.add_argument("--tmp-dir", type=str, default="exp/.tmp_real_bench")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--append", action="store_true", default=False)

    parser.add_argument("--worker", action="store_true", default=False)
    parser.add_argument("--task-file", type=str, default=None)
    parser.add_argument("--worker-output", type=str, default=None)
    parser.add_argument("--device-id", type=int, default=0)

    return parser.parse_args()


def parse_csv_arg(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def detect_devices(user_value: str | None) -> list[int]:
    if user_value:
        return [int(x) for x in parse_csv_arg(user_value)]

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return list(range(torch.xpu.device_count()))
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return [0]


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _acquire_output_lock(output_path: Path):
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_f = lock_path.open("w")
    fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
    return lock_f


def run_worker(args: argparse.Namespace) -> int:
    if not args.task_file or not args.worker_output:
        raise ValueError("Worker mode requires --task-file and --worker-output")

    with open(args.task_file, "r") as f:
        tasks = json.load(f)

    base_config = BenchmarkConfig(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        width=args.width,
        architecture="smallcnn",
        image_size=args.image_size,
        label_smoothing=args.label_smoothing,
        use_amp=not args.disable_amp,
        eval_ood=args.eval_ood,
        ood_severity=args.ood_severity,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )

    rows = []
    total = len(tasks)
    for i, task in enumerate(tasks, start=1):
        dataset = task["dataset"]
        architecture = task["architecture"]
        method = task["method"]
        seed = int(task["seed"])

        print(
            f"[worker device={args.device_id}] {i}/{total} "
            f"dataset={dataset} architecture={architecture} method={method} seed={seed}"
        )
        try:
            config = replace(base_config, architecture=architecture)
            result = run_single_experiment(
                dataset=dataset,
                method=method,
                seed=seed,
                device_id=args.device_id,
                config=config,
            )
        except Exception as exc:
            result = {
                "dataset": dataset,
                "architecture": architecture,
                "method": method,
                "seed": seed,
                "device": f"xpu:{args.device_id}",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_params_m": -1.0,
                "test_loss": -1.0,
                "test_acc1": -1.0,
                "test_acc5": -1.0,
                "time_sec": 0.0,
                "status": f"error: {str(exc)[:180]}",
            }
            traceback.print_exc()
        rows.append(result)

    write_csv_rows(Path(args.worker_output), rows)
    return 0


def backup_if_needed(output_path: Path, overwrite: bool) -> None:
    if not output_path.exists():
        return
    if overwrite:
        timestamp = int(time.time())
        backup = output_path.with_suffix(output_path.suffix + f".backup_{timestamp}")
        output_path.rename(backup)
        print(f"Existing output backed up to: {backup}")
    else:
        raise FileExistsError(
            f"{output_path} already exists. Use --overwrite to back it up and replace it."
        )


def split_round_robin(tasks: list[dict], n: int) -> list[list[dict]]:
    chunks = [[] for _ in range(n)]
    for i, task in enumerate(tasks):
        chunks[i % n].append(task)
    return chunks


def run_orchestrator(args: argparse.Namespace) -> int:
    dataset_names = parse_csv_arg(args.datasets)
    methods = parse_csv_arg(args.methods)
    architectures = parse_csv_arg(args.architectures)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    devices = detect_devices(args.device_ids)

    print(f"Datasets: {dataset_names}")
    print(f"Architectures: {architectures}")
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds}")
    print(f"Devices: {devices}")

    Path(args.data_root).mkdir(parents=True, exist_ok=True)
    prepare_datasets(dataset_names, args.data_root, image_size=args.image_size)
    if args.eval_ood and "cifar10" in dataset_names:
        from real_bench.ood import ensure_cifar10c

        ensure_cifar10c(args.data_root)

    tasks = [
        {
            "dataset": dataset,
            "architecture": architecture,
            "method": method,
            "seed": seed,
        }
        for dataset in dataset_names
        for architecture in architectures
        for method in methods
        for seed in seeds
    ]
    random.Random(args.seed_start).shuffle(tasks)

    tmp_dir = Path(args.tmp_dir)
    if args.tmp_dir == "exp/.tmp_real_bench":
        tmp_dir = tmp_dir / f"run_{int(time.time())}_{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if args.tmp_dir != "exp/.tmp_real_bench":
        for pattern in ("results_worker_*.csv", "tasks_worker_*.json", "worker_*.log"):
            for stale in tmp_dir.glob(pattern):
                stale.unlink()
    print(f"Temp dir: {tmp_dir}")

    chunks = split_round_robin(tasks, len(devices))

    procs = []
    shard_paths = []
    expected_rows = {}
    for worker_idx, (device_id, chunk) in enumerate(zip(devices, chunks)):
        if not chunk:
            continue

        task_path = tmp_dir / f"tasks_worker_{worker_idx}.json"
        shard_path = tmp_dir / f"results_worker_{worker_idx}.csv"
        log_path = tmp_dir / f"worker_{worker_idx}.log"

        with task_path.open("w") as f:
            json.dump(chunk, f)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--task-file",
            str(task_path),
            "--worker-output",
            str(shard_path),
            "--device-id",
            str(device_id),
            "--data-root",
            args.data_root,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--width",
            str(args.width),
            "--label-smoothing",
            str(args.label_smoothing),
            "--architectures",
            args.architectures,
        ]

        if args.image_size is not None:
            cmd += ["--image-size", str(args.image_size)]
        if args.disable_amp:
            cmd += ["--disable-amp"]
        if args.eval_ood:
            cmd += ["--eval-ood", "--ood-severity", str(args.ood_severity)]
        if args.max_train_samples is not None:
            cmd += ["--max-train-samples", str(args.max_train_samples)]
        if args.max_test_samples is not None:
            cmd += ["--max-test-samples", str(args.max_test_samples)]

        log_f = log_path.open("w")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        procs.append(
            {
                "proc": proc,
                "log_f": log_f,
                "log_path": log_path,
                "offset": 0,
                "worker_idx": worker_idx,
            }
        )
        shard_paths.append(shard_path)
        expected_rows[str(shard_path)] = len(chunk)

    failed = False

    def _flush_worker_logs(proc_items: list[dict], final: bool = False) -> None:
        for item in proc_items:
            log_path = item["log_path"]
            if not log_path.exists():
                continue
            with log_path.open("r") as rf:
                rf.seek(item["offset"])
                chunk = rf.read()
                item["offset"] = rf.tell()
            if chunk:
                for line in chunk.splitlines():
                    print(f"[worker-{item['worker_idx']}] {line}")
            if final:
                item["log_f"].close()

    if args.stream_worker_logs:
        while any(item["proc"].poll() is None for item in procs):
            _flush_worker_logs(procs)
            time.sleep(1.0)

        _flush_worker_logs(procs, final=True)
    else:
        for item in procs:
            item["proc"].wait()
            item["log_f"].close()

    for item in procs:
        ret = item["proc"].returncode
        if ret != 0:
            failed = True
            print(f"Worker failed (exit={ret}). Check log: {item['log_path']}")

    if failed:
        return 1

    frames = []
    for shard in shard_paths:
        if shard.exists():
            shard_df = pd.read_csv(shard)
            expected = expected_rows[str(shard)]
            if len(shard_df) != expected:
                raise RuntimeError(
                    f"Shard row mismatch for {shard}: expected {expected}, got {len(shard_df)}"
                )
            frames.append(shard_df)
        else:
            raise RuntimeError(f"Missing shard output: {shard}")

    if not frames:
        raise RuntimeError("No worker outputs were produced.")

    result_df = pd.concat(frames, ignore_index=True)
    sort_cols = ["dataset"]
    if "architecture" in result_df.columns:
        sort_cols.append("architecture")
    if "protocol" in result_df.columns:
        sort_cols.append("protocol")
    sort_cols += ["method", "seed"]
    result_df = result_df.sort_values(sort_cols).reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_f = _acquire_output_lock(output_path)
    try:
        if args.append and output_path.exists():
            old_df = pd.read_csv(output_path)
            result_df = pd.concat([old_df, result_df], ignore_index=True)
            dedup_keys = ["dataset"]
            if "architecture" in result_df.columns:
                dedup_keys.append("architecture")
            if "protocol" in result_df.columns:
                dedup_keys.append("protocol")
            dedup_keys += ["method", "seed"]
            result_df = (
                result_df.drop_duplicates(subset=dedup_keys, keep="last")
                .sort_values(sort_cols)
                .reset_index(drop=True)
            )
        else:
            backup_if_needed(output_path, args.overwrite)
        result_df.to_csv(output_path, index=False)
        print(f"Saved consolidated results to {output_path} ({len(result_df)} rows)")

        summary_df, stats_df, ranking_df = compute_summary_tables(result_df)

        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.summary_output, index=False)
        stats_df.to_csv(args.stats_output, index=False)
        ranking_df.to_csv(args.ranking_output, index=False)

        print(f"Saved summary table: {args.summary_output}")
        print(f"Saved statistical tests: {args.stats_output}")
        print(f"Saved ranking table: {args.ranking_output}")
    finally:
        try:
            lock_f.close()
        except Exception:
            pass

    return 0


def main() -> int:
    args = parse_args()
    if args.worker:
        return run_worker(args)
    return run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
