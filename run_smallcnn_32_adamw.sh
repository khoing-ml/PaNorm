#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

data_root="${DATA_ROOT:-/lus/flare/projects/RobustViT/kim/pa_norm/data}"
output="${OUTPUT:-exp/smallcnn_32_adamw.csv}"

python run_all_experiments.py \
  --group smallcnn_32 \
  --data-root "$data_root" \
  --output "$output"
