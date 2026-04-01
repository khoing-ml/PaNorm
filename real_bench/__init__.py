import os

try:
    max_numexpr = int(os.environ.get("NUMEXPR_MAX_THREADS", "64"))
    if max_numexpr > 64:
        os.environ["NUMEXPR_MAX_THREADS"] = "64"
except ValueError:
    os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ.setdefault("NUMEXPR_NUM_THREADS", "64")


try:
    import intel_extension_for_pytorch
except ImportError:
    pass

from .datasets import build_dataloaders, prepare_datasets
from .models import build_model
from .train import BenchmarkConfig, run_single_experiment

try:
    from .stats import compute_summary_tables
except ImportError:
    compute_summary_tables = None

__all__ = [
    "BenchmarkConfig",
    "build_dataloaders",
    "prepare_datasets",
    "build_model",
    "run_single_experiment",
    "compute_summary_tables",
]
