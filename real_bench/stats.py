from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty_like(pvals, dtype=float)

    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        val = pvals[idx] * factor
        running_max = max(running_max, val)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def _cohen_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    std = diff.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(diff.mean() / std)


def _wins_method(df: pd.DataFrame, target: str, baseline: str) -> tuple[int, int]:
    sub = df[df["method"].isin([target, baseline])]
    index_cols = ["dataset", "seed"]
    if "architecture" in sub.columns:
        index_cols.insert(1, "architecture")
    if "protocol" in sub.columns:
        insert_idx = 2 if "architecture" in sub.columns else 1
        index_cols.insert(insert_idx, "protocol")

    pivot = sub.pivot_table(index=index_cols, columns="method", values="test_acc1")
    if target not in pivot.columns or baseline not in pivot.columns:
        return 0, 0
    pivot = pivot.dropna()
    wins = int((pivot[target] > pivot[baseline]).sum())
    total = int(len(pivot))
    return wins, total


def compute_summary_tables(df: pd.DataFrame):
    ok = df[df["status"] == "success"].copy()
    group_cols = ["dataset", "method"]
    rank_cols = ["dataset"]
    if "architecture" in ok.columns:
        group_cols.insert(1, "architecture")
        rank_cols = ["dataset", "architecture"]
    if "protocol" in ok.columns:
        insert_idx = 2 if "architecture" in ok.columns else 1
        group_cols.insert(insert_idx, "protocol")
        rank_cols = rank_cols + ["protocol"]

    summary = ok.groupby(group_cols, as_index=False).agg(
        acc1_mean=("test_acc1", "mean"),
        acc1_std=("test_acc1", "std"),
        acc5_mean=("test_acc5", "mean"),
        acc5_std=("test_acc5", "std"),
        loss_mean=("test_loss", "mean"),
        time_mean=("time_sec", "mean"),
        n_runs=("seed", "count"),
    )
    for optional_col in [
        "gate_entropy_norm",
        "gate_main_weight",
        "gate_branch0",
        "gate_branch1",
        "gate_branch2",
        "ood_acc1_clean",
        "ood_acc1_cifar10c_mean",
        "ood_acc1_cifar10c_worst",
        "ood_drop_from_clean",
        "ood_gate_entropy_norm",
        "ood_gate_main_weight",
        "ood_gate_branch0",
        "ood_gate_branch1",
        "ood_gate_branch2",
    ]:
        if optional_col in ok.columns:
            summary = summary.merge(
                ok.groupby(group_cols, as_index=False).agg(
                    **{f"{optional_col}_mean": (optional_col, "mean")}
                ),
                on=group_cols,
                how="left",
            )

    summary["rank"] = summary.groupby(rank_cols)["acc1_mean"].rank(
        ascending=False, method="dense"
    )

    stat_columns = [
        "dataset",
        "architecture",
        "protocol",
        "target_method",
        "baseline",
        "n_paired",
        "target_minus_baseline_mean",
        "ci95_low",
        "ci95_high",
        "effect_size_cohen_d",
        "p_value_raw",
        "test",
        "target_win_rate",
        "p_value_holm",
        "significant_0.05",
    ]
    stat_rows = []
    rng = np.random.default_rng(1234)

    by_arch = "architecture" in ok.columns
    by_protocol = "protocol" in ok.columns
    arch_values = sorted(ok["architecture"].unique()) if by_arch else ["all"]
    protocol_values = sorted(ok["protocol"].unique()) if by_protocol else ["all"]
    for dataset in sorted(ok["dataset"].unique()):
        for architecture in arch_values:
            for protocol in protocol_values:
                subset = ok[ok["dataset"] == dataset]
                if by_arch:
                    subset = subset[subset["architecture"] == architecture]
                if by_protocol:
                    subset = subset[subset["protocol"] == protocol]
                if subset.empty:
                    continue

                available = sorted(subset["method"].unique())
                targets = [m for m in available if m.startswith("panorm")]
                baselines = [m for m in available if m not in targets]
                if not targets or not baselines:
                    continue

                for target in targets:
                    pvals = []
                    temp_rows = []
                    for baseline in baselines:
                        sub = subset[subset["method"].isin([target, baseline])]
                        pivot = sub.pivot_table(
                            index="seed", columns="method", values="test_acc1"
                        ).dropna()
                        if target not in pivot.columns or baseline not in pivot.columns:
                            continue
                        if len(pivot) < 2:
                            continue

                        x = pivot[target].to_numpy(dtype=float)
                        y = pivot[baseline].to_numpy(dtype=float)
                        diffs = x - y

                        try:
                            _, pval = stats.wilcoxon(
                                diffs, zero_method="wilcox", correction=True
                            )
                            test_name = "wilcoxon"
                        except ValueError:
                            _, pval = stats.ttest_rel(x, y)
                            test_name = "paired_t"

                        effect = _cohen_d_paired(x, y)
                        boot = np.array(
                            [
                                np.mean(
                                    rng.choice(diffs, size=len(diffs), replace=True)
                                )
                                for _ in range(1500)
                            ]
                        )
                        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
                        wins, total = _wins_method(
                            subset, target=target, baseline=baseline
                        )

                        row = {
                            "dataset": dataset,
                            "architecture": architecture,
                            "protocol": protocol,
                            "target_method": target,
                            "baseline": baseline,
                            "n_paired": int(len(pivot)),
                            "target_minus_baseline_mean": float(diffs.mean()),
                            "ci95_low": float(ci_low),
                            "ci95_high": float(ci_high),
                            "effect_size_cohen_d": effect,
                            "p_value_raw": float(pval),
                            "test": test_name,
                            "target_win_rate": float(wins / max(total, 1)),
                        }
                        temp_rows.append(row)
                        pvals.append(float(pval))

                    if temp_rows:
                        corrected = _holm_bonferroni(np.array(pvals, dtype=float))
                        for row, p_corr in zip(temp_rows, corrected):
                            row["p_value_holm"] = float(p_corr)
                            row["significant_0.05"] = bool(p_corr < 0.05)
                            stat_rows.append(row)

    stat_df = pd.DataFrame(stat_rows, columns=stat_columns)

    ranking_cols = ["method"]
    if "architecture" in summary.columns:
        ranking_cols.insert(0, "architecture")
    if "protocol" in summary.columns:
        insert_idx = 1 if "architecture" in summary.columns else 0
        ranking_cols.insert(insert_idx, "protocol")

    ranking = (
        summary.groupby(ranking_cols, as_index=False)
        .agg(mean_rank=("rank", "mean"), mean_acc1=("acc1_mean", "mean"))
        .sort_values(
            [
                c
                for c in ["protocol", "architecture", "mean_rank", "mean_acc1"]
                if c in ranking_cols + ["mean_rank", "mean_acc1"]
            ],
            ascending=[True]
            * sum(1 for c in ["protocol", "architecture"] if c in ranking_cols)
            + [True, False],
        )
    )

    sort_summary = rank_cols + ["rank"]
    return summary.sort_values(sort_summary), stat_df, ranking
