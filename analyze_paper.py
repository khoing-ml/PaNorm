import os
import sys
import warnings
import csv
import math
from collections import defaultdict
from itertools import combinations


def mean(xs):
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def std(xs, ddof=1):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - ddof))


def cohens_d(a, b):

    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    ma, mb = mean(a), mean(b)
    sa, sb = std(a), std(b)
    sp = math.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if sp == 0:
        return float("nan")
    return (ma - mb) / sp


def paired_ttest(a, b):

    n = len(a)
    if n < 2 or len(b) != n:
        return (float("nan"), float("nan"))
    diffs = [ai - bi for ai, bi in zip(a, b)]
    d_bar = mean(diffs)
    d_std = std(diffs)
    if d_std == 0:
        return (float("inf") if d_bar != 0 else 0.0, 0.0 if d_bar != 0 else 1.0)
    t = d_bar / (d_std / math.sqrt(n))

    try:
        from scipy.stats import t as tdist

        p = 2 * tdist.sf(abs(t), n - 1)
    except ImportError:

        p = 2 * (1 - _norm_cdf(abs(t)))
    return (t, p)


def _norm_cdf(x):

    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def wilcoxon_test(a, b):

    try:
        from scipy.stats import wilcoxon

        if len(a) < 6:

            return (float("nan"), float("nan"))
        stat, p = wilcoxon(a, b, alternative="two-sided")
        return (stat, p)
    except ImportError:
        return (float("nan"), float("nan"))
    except ValueError:
        return (float("nan"), float("nan"))


def _fix_csv_line(fields, n_header):

    n = len(fields)
    if n == n_header:
        return fields
    if n == n_header + 1:

        if fields[14] in ("True", "False"):

            return fields[:n_header]
        else:

            return fields[:14] + fields[15:]
    if n == n_header - 8:

        fixed = fields[:14] + fields[15:]
        fixed.extend(["-1.0"] * 9)
        return fixed
    return fields


def load_csv(path):

    import csv as csv_mod

    rows = []
    with open(path, "r") as f:
        reader = csv_mod.DictReader(f)
        header = reader.fieldnames
        n_header = len(header) if header else 0

        for csv_row in reader:
            row = dict(csv_row)

            if None in row:

                extra = row.pop(None)

                ood_mean = row.get("ood_acc1_cifar10c_mean", "")
                if isinstance(ood_mean, str) and ood_mean.startswith("[{"):

                    ood_keys = [
                        "ood_acc1_cifar10c_mean",
                        "ood_acc1_cifar10c_worst",
                        "ood_acc1_clean",
                        "ood_drop_from_clean",
                        "ood_gate_entropy_norm",
                        "ood_gate_main_weight",
                        "ood_gate_branch0",
                        "ood_gate_branch1",
                        "ood_gate_branch2",
                    ]

                    displaced = []
                    for ok in ood_keys:
                        displaced.append(row.get(ok, "-1.0"))
                    if isinstance(extra, list):
                        displaced.extend(extra)

                    real_ood = displaced[1:]
                    for i, ok in enumerate(ood_keys):
                        row[ok] = real_ood[i] if i < len(real_ood) else "-1.0"

            for k, v in row.items():
                if not isinstance(v, str):
                    continue

                try:
                    row[k] = int(v)
                except (ValueError, TypeError):
                    try:
                        row[k] = float(v)
                    except (ValueError, TypeError):
                        pass

                if v == "True":
                    row[k] = True
                elif v == "False":
                    row[k] = False

            if "protocol" in row and isinstance(row["protocol"], str):
                if row["protocol"].startswith("adamw_"):
                    row["protocol"] = row["protocol"][6:]
            rows.append(row)

    seen = set()
    deduped = []
    for row in rows:
        key = (
            row.get("dataset"),
            row.get("architecture"),
            row.get("method"),
            row.get("seed"),
            row.get("protocol"),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    if len(deduped) < len(rows):
        print(f"  [dedup] Removed {len(rows) - len(deduped)} duplicate rows")
    return deduped


METHOD_DISPLAY = {
    "batchnorm": "BatchNorm",
    "groupnorm": "GroupNorm",
    "layernorm": "LayerNorm",
    "rmsnorm": "RMSNorm",
    "switchnorm": "SwitchNorm",
    "evonorm_b0": "EvoNorm-B0",
    "evonorm_s0": "EvoNorm-S0",
    "frn": "FRN",
    "scalenorm": "ScaleNorm",
    "subln": "Sub-LN",
    "powernorm": "PowerNorm",
    "panorm": "PA-Norm",
    "panorm_lite": "PA-Norm-Lite",
    "panorm_lite_fast": "PA-Norm-LF",
    "panorm_nodetach": "PA-Norm-ND",
    "panorm_switchnorm": "PA-Norm-SN",
    "panorm_switchnorm_nodetach": "PA-SN-ND",
    "panorm_selective": "PA-Norm-Sel",
    "panorm_selective_75": "PA-Norm-Sel75",
    "batchrenorm": "BatchRenorm",
    "ws_gn": "WS+GN",
}

ARCH_DISPLAY = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "smallcnn": "SmallCNN",
    "wide_resnet50_2": "WRN-50-2",
    "convnext_tiny": "ConvNeXt-T",
    "efficientnet_b0": "EfficientNet-B0",
    "vit_small": "ViT-S",
}

DATASET_DISPLAY = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "stl10": "STL-10",
    "svhn": "SVHN",
    "tinyimagenet": "Tiny-IN",
}


METHOD_TAXONOMY = {
    "batchnorm": ("Fixed", "BN", "Yes", "No", "No", "No"),
    "groupnorm": ("Fixed", "GN", "No", "No", "No", "No"),
    "layernorm": ("Fixed", "LN", "No", "No", "No", "No"),
    "rmsnorm": ("Fixed", "RMS", "No", "No", "No", "No"),
    "scalenorm": ("Fixed", "L2-Scale", "No", "No", "No", "No"),
    "subln": ("Adaptive", "LN+Skip", "No", "No", "No", "No"),
    "powernorm": ("Adaptive", "QM-Run", "No", "No", "No", "No"),
    "switchnorm": ("Mixed", "BN+IN+LN", "Yes", "Yes", "No", "No"),
    "evonorm_b0": ("Adaptive", "BN+EvoAct", "Yes", "No", "No", "No"),
    "evonorm_s0": ("Adaptive", "GN+EvoAct", "No", "No", "No", "No"),
    "frn": ("Adaptive", "FRN+TLU", "No", "No", "No", "No"),
    "panorm": ("PA", "BN+GN+LN", "Yes", "Yes", "Yes", "No"),
    "panorm_lite": ("PA", "BN+GN+LN", "Yes", "Yes", "Yes", "No"),
    "panorm_lite_fast": ("PA", "BN+GN+LN", "Yes", "Yes", "Yes", "No"),
    "panorm_nodetach": ("PA", "BN+GN+LN", "Yes", "Yes", "Yes", "Yes"),
    "panorm_switchnorm": ("PA", "BN+IN+LN", "Yes", "Yes", "Yes", "No"),
    "panorm_switchnorm_nodetach": ("PA", "BN+IN+LN", "Yes", "Yes", "Yes", "Yes"),
    "panorm_selective": ("PA", "BN+GN+LN", "Yes", "Yes", "Yes", "Sel"),
    "panorm_selective_75": ("PA", "BN+GN+LN", "Yes", "Yes", "Yes", "Sel"),
}

PA_NORM_METHODS = [
    "panorm",
    "panorm_lite",
    "panorm_lite_fast",
    "panorm_nodetach",
    "panorm_switchnorm",
    "panorm_switchnorm_nodetach",
    "panorm_selective",
    "panorm_selective_75",
]


RANKING_METHODS = {
    "batchnorm",
    "groupnorm",
    "layernorm",
    "switchnorm",
    "batchrenorm",
    "ws_gn",
    "evonorm_s0",
    "evonorm_b0",
    "frn",
    "panorm",
    "panorm_switchnorm",
}


ABLATION_METHODS = {
    "panorm_lite",
    "panorm_lite_fast",
    "panorm_nodetach",
    "panorm_switchnorm_nodetach",
    "panorm_selective",
    "panorm_selective_75",
    "scalenorm",
    "subln",
    "powernorm",
}


def esc(s):

    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def group_rows(rows, key_fn):

    groups = defaultdict(list)
    for r in rows:
        k = key_fn(r)
        groups[k].append(r)
    return groups


def compute_mean_std_table(rows):

    filtered = [
        r for r in rows if r.get("status") == "success" and r.get("eval_ood") == False
    ]

    key_fn = lambda r: (r["architecture"], r["dataset"], r["protocol"])
    groups = group_rows(filtered, key_fn)

    result = {}
    for key, grp in groups.items():
        method_groups = group_rows(grp, lambda r: r["method"])
        method_stats = {}
        for method, mrows in method_groups.items():
            accs = [
                r["test_acc1"]
                for r in mrows
                if isinstance(r["test_acc1"], (int, float))
            ]
            if accs:
                method_stats[method] = (
                    mean(accs),
                    std(accs) if len(accs) > 1 else 0.0,
                    len(accs),
                )
        result[key] = method_stats
    return result


def compute_rankings(mean_std_table, method_filter=None):

    per_group_ranks = {}
    method_rank_lists = defaultdict(list)
    wins = defaultdict(int)

    for key, method_stats in sorted(mean_std_table.items()):

        if method_filter is not None:
            method_stats = {m: v for m, v in method_stats.items() if m in method_filter}
        if not method_stats:
            continue

        sorted_methods = sorted(method_stats.items(), key=lambda x: -x[1][0])
        ranks = {}
        for i, (method, _) in enumerate(sorted_methods):
            ranks[method] = i + 1
        per_group_ranks[key] = ranks

        if sorted_methods:
            wins[sorted_methods[0][0]] += 1

        for method, rank in ranks.items():
            method_rank_lists[method].append(rank)

    avg_ranks = {}
    for method, rank_list in method_rank_lists.items():
        avg_ranks[method] = (mean(rank_list), len(rank_list))

    return per_group_ranks, avg_ranks, wins


def compute_friedman_nemenyi(per_group_ranks, avg_ranks):

    all_groups = sorted(per_group_ranks.keys())
    if not all_groups:
        return None, None, None, []

    FRIEDMAN_CORE = [
        "batchnorm",
        "panorm",
        "panorm_switchnorm",
        "switchnorm",
        "groupnorm",
        "layernorm",
    ]

    from collections import Counter

    method_counts = Counter()
    for group_key in all_groups:
        for m in per_group_ranks[group_key]:
            if m in FRIEDMAN_CORE:
                method_counts[m] += 1

    common_methods = sorted(m for m in FRIEDMAN_CORE if method_counts.get(m, 0) >= 3)

    if len(common_methods) < 3:
        return None, None, None, []

    k = len(common_methods)
    rank_matrix = []

    for group_key in all_groups:
        group_ranks = per_group_ranks[group_key]

        if not all(m in group_ranks for m in common_methods):
            continue
        row = [float(group_ranks[m]) for m in common_methods]
        rank_matrix.append(row)

    n = len(rank_matrix)
    if n < 3:
        return None, None, None, []

    try:
        from scipy.stats import friedmanchisquare, chi2

        cols = list(zip(*rank_matrix))
        if len(cols) < 3:
            return None, None, None, []
        stat, p_val = friedmanchisquare(*cols)
    except Exception:
        return None, None, None, []

    q_alpha_table = {
        2: 1.960,
        3: 2.344,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
        11: 3.219,
        12: 3.268,
        13: 3.313,
        14: 3.354,
        15: 3.391,
        16: 3.426,
        17: 3.458,
        18: 3.489,
        19: 3.517,
        20: 3.544,
    }
    q_alpha = q_alpha_table.get(k, 2.0 + 0.1 * k)
    cd = q_alpha * math.sqrt(k * (k + 1) / (6.0 * n))

    pairwise = []
    for i in range(len(common_methods)):
        for j in range(i + 1, len(common_methods)):
            mi, mj = common_methods[i], common_methods[j]
            ri = avg_ranks.get(mi, (0, 0))[0]
            rj = avg_ranks.get(mj, (0, 0))[0]
            diff = abs(ri - rj)
            sig = diff > cd
            pairwise.append((mi, mj, ri, rj, diff, sig))

    return stat, p_val, cd, pairwise, common_methods, n


def compute_statistical_tests(rows):

    filtered = [
        r for r in rows if r.get("status") == "success" and r.get("eval_ood") == False
    ]

    key_fn = lambda r: (r["architecture"], r["dataset"], r["protocol"], r["seed"])
    groups = group_rows(filtered, key_fn)

    results = []
    for pa_method in PA_NORM_METHODS:
        pa_accs = []
        bn_accs = []
        for key, grp in groups.items():
            method_map = {r["method"]: r["test_acc1"] for r in grp}
            if pa_method in method_map and "batchnorm" in method_map:
                pa_accs.append(method_map[pa_method])
                bn_accs.append(method_map["batchnorm"])

        if len(pa_accs) >= 2:
            t_stat, t_p = paired_ttest(pa_accs, bn_accs)
            w_stat, w_p = wilcoxon_test(pa_accs, bn_accs)
            d = cohens_d(pa_accs, bn_accs)
            delta_mean = mean(pa_accs) - mean(bn_accs)
            results.append(
                {
                    "method": pa_method,
                    "n_pairs": len(pa_accs),
                    "pa_mean": mean(pa_accs),
                    "bn_mean": mean(bn_accs),
                    "delta_mean": delta_mean,
                    "t_stat": t_stat,
                    "t_pvalue": t_p,
                    "w_stat": w_stat,
                    "w_pvalue": w_p,
                    "cohens_d": d,
                }
            )
        else:
            results.append(
                {
                    "method": pa_method,
                    "n_pairs": len(pa_accs),
                    "pa_mean": mean(pa_accs) if pa_accs else float("nan"),
                    "bn_mean": mean(bn_accs) if bn_accs else float("nan"),
                    "delta_mean": float("nan"),
                    "t_stat": float("nan"),
                    "t_pvalue": float("nan"),
                    "w_stat": float("nan"),
                    "w_pvalue": float("nan"),
                    "cohens_d": float("nan"),
                }
            )

    return results


def compute_ood_analysis(rows):

    ood_rows = [
        r
        for r in rows
        if r.get("eval_ood") == True
        and r.get("status") == "success"
        and r.get("ood_severity") == 2
    ]
    if not ood_rows:
        return None

    method_groups = group_rows(ood_rows, lambda r: r["method"])
    results = {}
    for method, grp in method_groups.items():
        ood_means = [
            r["ood_acc1_cifar10c_mean"]
            for r in grp
            if isinstance(r.get("ood_acc1_cifar10c_mean"), (int, float))
        ]
        ood_worsts = [
            r["ood_acc1_cifar10c_worst"]
            for r in grp
            if isinstance(r.get("ood_acc1_cifar10c_worst"), (int, float))
        ]

        clean_accs = [
            r["test_acc1"] for r in grp if isinstance(r.get("test_acc1"), (int, float))
        ]

        drops = [
            r["test_acc1"] - r["ood_acc1_cifar10c_mean"]
            for r in grp
            if isinstance(r.get("test_acc1"), (int, float))
            and isinstance(r.get("ood_acc1_cifar10c_mean"), (int, float))
        ]

        results[method] = {
            "n": len(grp),
            "ood_mean": (mean(ood_means), std(ood_means) if len(ood_means) > 1 else 0),
            "ood_worst": (
                mean(ood_worsts),
                std(ood_worsts) if len(ood_worsts) > 1 else 0,
            ),
            "clean": (mean(clean_accs), std(clean_accs) if len(clean_accs) > 1 else 0),
            "drop": (mean(drops), std(drops) if len(drops) > 1 else 0),
        }

    ood_pa = [r for r in ood_rows if r["method"] in PA_NORM_METHODS]
    ood_gate_stats = {}
    for r in ood_pa:
        m = r["method"]
        if m not in ood_gate_stats:
            ood_gate_stats[m] = {
                "ood_entropy": [],
                "ood_main": [],
                "ood_b0": [],
                "ood_b1": [],
                "ood_b2": [],
                "clean_entropy": [],
                "clean_main": [],
                "clean_b0": [],
                "clean_b1": [],
                "clean_b2": [],
            }
        if (
            isinstance(r.get("ood_gate_entropy_norm"), (int, float))
            and r["ood_gate_entropy_norm"] != -1.0
        ):
            ood_gate_stats[m]["ood_entropy"].append(r["ood_gate_entropy_norm"])
            ood_gate_stats[m]["ood_main"].append(r["ood_gate_main_weight"])
            ood_gate_stats[m]["ood_b0"].append(r["ood_gate_branch0"])
            ood_gate_stats[m]["ood_b1"].append(r["ood_gate_branch1"])
            ood_gate_stats[m]["ood_b2"].append(r["ood_gate_branch2"])
        if (
            isinstance(r.get("gate_entropy_norm"), (int, float))
            and r["gate_entropy_norm"] != -1.0
        ):
            ood_gate_stats[m]["clean_entropy"].append(r["gate_entropy_norm"])
            ood_gate_stats[m]["clean_main"].append(r["gate_main_weight"])
            ood_gate_stats[m]["clean_b0"].append(r["gate_branch0"])
            ood_gate_stats[m]["clean_b1"].append(r["gate_branch1"])
            ood_gate_stats[m]["clean_b2"].append(r["gate_branch2"])

    return results, ood_gate_stats


def compute_gate_analysis(rows):

    pa_rows = [
        r
        for r in rows
        if r["method"] in PA_NORM_METHODS
        and r.get("status") == "success"
        and isinstance(r.get("gate_entropy_norm"), (int, float))
        and r["gate_entropy_norm"] != -1.0
    ]

    if not pa_rows:
        return None

    method_groups = group_rows(pa_rows, lambda r: r["method"])
    results = {}
    for method, grp in method_groups.items():
        entropy = [r["gate_entropy_norm"] for r in grp]
        main_w = [r["gate_main_weight"] for r in grp]
        b0 = [r["gate_branch0"] for r in grp]
        b1 = [r["gate_branch1"] for r in grp]
        b2 = [r["gate_branch2"] for r in grp]

        results[method] = {
            "n": len(grp),
            "entropy": (mean(entropy), std(entropy) if len(entropy) > 1 else 0),
            "main_weight": (mean(main_w), std(main_w) if len(main_w) > 1 else 0),
            "branch0": (mean(b0), std(b0) if len(b0) > 1 else 0),
            "branch1": (mean(b1), std(b1) if len(b1) > 1 else 0),
            "branch2": (mean(b2), std(b2) if len(b2) > 1 else 0),
        }

    detail_groups = group_rows(
        pa_rows, lambda r: (r["method"], r["architecture"], r["dataset"])
    )
    detail_results = {}
    for key, grp in detail_groups.items():
        entropy = [r["gate_entropy_norm"] for r in grp]
        main_w = [r["gate_main_weight"] for r in grp]
        b0 = [r["gate_branch0"] for r in grp]
        b1 = [r["gate_branch1"] for r in grp]
        b2 = [r["gate_branch2"] for r in grp]
        detail_results[key] = {
            "n": len(grp),
            "entropy": (mean(entropy), std(entropy) if len(entropy) > 1 else 0),
            "main_weight": (mean(main_w), std(main_w) if len(main_w) > 1 else 0),
            "branch0": (mean(b0), std(b0) if len(b0) > 1 else 0),
            "branch1": (mean(b1), std(b1) if len(b1) > 1 else 0),
            "branch2": (mean(b2), std(b2) if len(b2) > 1 else 0),
        }

    return results, detail_results


def fmt_val(v, fmt=".2f"):

    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "---"
    return f"{v:{fmt}}"


def fmt_pm(m, s, fmt=".2f"):

    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "---"
    return f"{m:{fmt}}$\\pm${s:{fmt}}"


def fmt_pvalue(p):

    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "---"
    if p == 0.0:
        return r"< 10^{-15}"
    if p < 0.001:

        exp = math.floor(math.log10(p))
        mantissa = p / (10**exp)
        return f"{mantissa:.1f}" + r"\times 10^{" + str(exp) + "}"
    elif p < 0.01:
        return f"{p:.4f}"
    elif p < 0.05:
        return f"{p:.3f}"
    else:
        return f"{p:.3f}"


def generate_consolidated_ranking(
    mean_std_table, per_group_ranks, avg_ranks, wins, output_path
):

    all_methods = sorted(
        [m for m in avg_ranks.keys() if m in RANKING_METHODS],
        key=lambda m: avg_ranks[m][0],
    )

    sorted_keys = sorted(
        k
        for k in mean_std_table.keys()
        if len(mean_std_table[k]) >= 4
        and "sgd" not in k[2]
        and k[0] != "wide_resnet50_2"
        and ("bs256" in k[2] or "bs128" in k[2])
    )

    col_labels = []
    for arch, ds, proto in sorted_keys:
        epochs = "e50" if "e50" in proto else "e20"

        import re

        bs_match = re.search(r"bs(\d+)", proto)
        bs_str = bs_match.group(1) if bs_match else ""
        bs_suffix = f"\\\\BS{bs_str}" if bs_str and bs_str != "256" else ""
        label = f"{ARCH_DISPLAY.get(arch, arch)}\\\\{DATASET_DISPLAY.get(ds, ds)}\\\\{epochs}{bs_suffix}"
        col_labels.append(label)

    n_cols = len(sorted_keys)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Consolidated ranking of normalization methods across all (architecture, dataset, protocol) settings. Rank 1 = best. Last columns: average rank and number of first-place finishes (wins).}"
    )
    lines.append(r"\label{tab:consolidated}")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    col_spec = "l" + "c" * n_cols + "|cc"
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header_parts = ["Method"]
    for label in col_labels:

        sub_parts = label.split("\\\\")
        header_parts.append(
            r"\rotatebox{70}{\parbox{2.5cm}{\centering " + r"\\".join(sub_parts) + "}}"
        )
    header_parts.append("Avg. Rank")
    header_parts.append("Wins")
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    displayed_avg_ranks = {}
    displayed_wins = defaultdict(int)
    for method in all_methods:
        rank_list = []
        for key in sorted_keys:
            ranks = per_group_ranks.get(key, {})
            if method in ranks:
                rank_list.append(ranks[method])
                if ranks[method] == 1:
                    displayed_wins[method] += 1
        if rank_list:
            displayed_avg_ranks[method] = (mean(rank_list), len(rank_list))
        else:
            displayed_avg_ranks[method] = (float("inf"), 0)

    all_methods = sorted(all_methods, key=lambda m: displayed_avg_ranks[m][0])

    for method in all_methods:
        display = METHOD_DISPLAY.get(method, method)
        row_parts = [esc(display)]

        for key in sorted_keys:
            ranks = per_group_ranks.get(key, {})
            if method in ranks:
                rank = ranks[method]

                if rank == 1:
                    row_parts.append(r"\textbf{1}")
                elif rank == 2:
                    row_parts.append(r"\underline{2}")
                else:
                    row_parts.append(str(rank))
            else:
                row_parts.append("---")

        ar, n_gr = displayed_avg_ranks[method]
        w = displayed_wins.get(method, 0)
        row_parts.append(f"{ar:.2f}")
        row_parts.append(str(w))
        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def generate_statistical_tests_table(stat_results, output_path):

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Statistical comparison of PA-Norm variants vs.\ BatchNorm. Paired $t$-test and Wilcoxon signed-rank test on matched (architecture, dataset, protocol, seed) pairs. Cohen\textquotesingle s $d$ measures effect size; $\Delta\overline{\text{Acc}}$ is the mean accuracy difference (PA $-$ BN). Significance: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}"
    )
    lines.append(r"\label{tab:statistical_tests}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lcccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"PA Variant & $N_{\text{pairs}}$ & $\overline{\text{Acc}}_{\text{PA}}$ & $\overline{\text{Acc}}_{\text{BN}}$ & $\Delta\overline{\text{Acc}}$ & $t$-stat ($p$) & Wilcoxon ($p$) & Cohen\textquotesingle s $d$ \\"
    )
    lines.append(r"\midrule")

    for r in stat_results:
        display = METHOD_DISPLAY.get(r["method"], r["method"])

        t_stars = ""
        if isinstance(r["t_pvalue"], float) and not math.isnan(r["t_pvalue"]):
            if r["t_pvalue"] < 0.001:
                t_stars = "$^{***}$"
            elif r["t_pvalue"] < 0.01:
                t_stars = "$^{**}$"
            elif r["t_pvalue"] < 0.05:
                t_stars = "$^{*}$"

        w_stars = ""
        if isinstance(r["w_pvalue"], float) and not math.isnan(r["w_pvalue"]):
            if r["w_pvalue"] < 0.001:
                w_stars = "$^{***}$"
            elif r["w_pvalue"] < 0.01:
                w_stars = "$^{**}$"
            elif r["w_pvalue"] < 0.05:
                w_stars = "$^{*}$"

        t_str = fmt_val(r["t_stat"], ".2f")
        tp_str = fmt_pvalue(r["t_pvalue"])
        w_str = fmt_val(r["w_stat"], ".1f")
        wp_str = fmt_pvalue(r["w_pvalue"])

        row = (
            f"{esc(display)} & "
            f"{r['n_pairs']} & "
            f"{fmt_val(r['pa_mean'], '.2f')} & "
            f"{fmt_val(r['bn_mean'], '.2f')} & "
            f"{fmt_val(r['delta_mean'], '.2f')} & "
            f"${t_str}$ (${tp_str}$){t_stars} & "
            f"${w_str}$ (${wp_str}$){w_stars} & "
            f"{fmt_val(r['cohens_d'], '.3f')}"
            r" \\"
        )
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def generate_friedman_table(friedman_result, avg_ranks, output_path):

    if friedman_result[0] is None:
        print("  Skipping Friedman table (insufficient data)")
        return

    f_stat, f_p, cd, pairwise, common_methods, n_groups = friedman_result

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Friedman test for overall ranking significance with Nemenyi post-hoc critical difference (CD). "
    )
    lines.append(
        f"Friedman $\\chi^2 = {f_stat:.2f}$, $p = {f_p:.2e}$, CD$_{{\\alpha=0.05}} = {cd:.3f}$ "
    )
    lines.append(f"($k={len(common_methods)}$ methods, $n={n_groups}$ settings). ")
    lines.append(
        r"Methods whose rank difference exceeds CD are statistically significantly different.}"
    )
    lines.append(r"\label{tab:friedman}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Avg.\ Rank & Significant vs.\ BN? \\")
    lines.append(r"\midrule")

    ranked = sorted(
        [(m, avg_ranks.get(m, (999, 0))[0]) for m in common_methods], key=lambda x: x[1]
    )

    bn_rank = avg_ranks.get("batchnorm", (999, 0))[0]
    for method, rank in ranked:
        display = METHOD_DISPLAY.get(method, method)
        diff_from_bn = abs(rank - bn_rank)
        sig = (
            "Yes"
            if diff_from_bn > cd and method != "batchnorm"
            else ("---" if method == "batchnorm" else "No")
        )
        bold = (
            r"\textbf{" + f"{rank:.2f}" + "}" if rank == ranked[0][1] else f"{rank:.2f}"
        )
        lines.append(f"{esc(display)} & {bold} & {sig}" + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def generate_taxonomy_table(output_path):

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Taxonomy of normalization methods. Category: Fixed (single norm), Mixed (learned mixture), Adaptive (data-dependent activation), PA (physics-aware gating). Properties: uses batch statistics (Batch), learnable mixture weights (Mix), gradient-informed gating (Grad), gradient flows through gate (GradFlow).}"
    )
    lines.append(r"\label{tab:comparison_taxonomy}")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Category & Components & Batch & Mix & Grad & GradFlow \\")
    lines.append(r"\midrule")

    category_order = {"Fixed": 0, "Mixed": 1, "Adaptive": 2, "PA": 3}
    sorted_methods = sorted(
        METHOD_TAXONOMY.items(), key=lambda x: (category_order.get(x[1][0], 99), x[0])
    )

    prev_cat = None
    for method, (cat, components, batch, mix, grad, gradflow) in sorted_methods:
        display = METHOD_DISPLAY.get(method, method)
        if prev_cat is not None and cat != prev_cat:
            lines.append(r"\midrule")
        prev_cat = cat

        def check(v):
            return r"\checkmark" if v == "Yes" else ""

        row = f"{esc(display)} & {cat} & {components} & {check(batch)} & {check(mix)} & {check(grad)} & {check(gradflow)}"
        lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def generate_ood_table(ood_results, output_path):

    if ood_results is None:
        print("  Skipping OOD table (no data)")
        return

    acc_results, gate_stats = ood_results

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Out-of-distribution robustness on CIFAR-10-C (severity 2). Mean $\pm$ std over seeds. Drop = clean accuracy $-$ corrupted mean accuracy.}"
    )
    lines.append(r"\label{tab:ood}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Clean Acc & OOD Mean Acc & OOD Worst Acc & Drop (\%) \\")
    lines.append(r"\midrule")

    for method in sorted(acc_results.keys()):
        display = METHOD_DISPLAY.get(method, method)
        r = acc_results[method]
        row = (
            f"{esc(display)} & "
            f"{fmt_pm(r['clean'][0], r['clean'][1])} & "
            f"{fmt_pm(r['ood_mean'][0], r['ood_mean'][1])} & "
            f"{fmt_pm(r['ood_worst'][0], r['ood_worst'][1])} & "
            f"{fmt_pm(r['drop'][0], r['drop'][1])}"
            r" \\"
        )
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def generate_gate_table(gate_results, output_path):

    if gate_results is None:
        print("  Skipping gate table (no data)")
        return

    agg_results, detail_results = gate_results

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Gate statistics for PA-Norm variants (aggregated across all settings). Entropy is normalized $\in [0,1]$; higher means more uniform gating. Branch weights sum to 1.}"
    )
    lines.append(r"\label{tab:gate_analysis}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Variant & $N$ & Entropy & Branch 0 & Branch 1 & Branch 2 \\")
    lines.append(r"\midrule")

    for method in sorted(agg_results.keys()):
        display = METHOD_DISPLAY.get(method, method)
        r = agg_results[method]
        row = (
            f"{esc(display)} & "
            f"{r['n']} & "
            f"{fmt_pm(r['entropy'][0], r['entropy'][1], '.3f')} & "
            f"{fmt_pm(r['branch0'][0], r['branch0'][1], '.3f')} & "
            f"{fmt_pm(r['branch1'][0], r['branch1'][1], '.3f')} & "
            f"{fmt_pm(r['branch2'][0], r['branch2'][1], '.3f')}"
            r" \\"
        )
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def generate_main_results_table(mean_std_table, output_path):

    filtered_table = {
        k: v
        for k, v in mean_std_table.items()
        if "sgd" not in k[2] and k[0] != "wide_resnet50_2" and "bs256" in k[2]
    }

    arch_proto_groups = defaultdict(set)
    for arch, ds, proto in filtered_table.keys():
        arch_proto_groups[(arch, proto)].add(ds)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Top-1 test accuracy (\%) for all methods across architectures and datasets. Mean $\pm$ std over seeds. \textbf{Bold}: best, \underline{underline}: second best per column.}"
    )
    lines.append(r"\label{tab:main_accuracy}")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    all_methods = set()
    for ms in filtered_table.values():
        all_methods.update(ms.keys())
    all_methods = sorted(all_methods, key=lambda m: METHOD_DISPLAY.get(m, m))

    sorted_keys = sorted(filtered_table.keys())
    n_cols = len(sorted_keys)

    col_spec = "l" + "c" * n_cols
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header_parts = ["Method"]
    for arch, ds, proto in sorted_keys:
        epochs = "50ep" if "e50" in proto else "20ep"
        img = "64" if "im64" in proto else "32"
        header_parts.append(
            r"\rotatebox{70}{\parbox{2.2cm}{\centering "
            + ARCH_DISPLAY.get(arch, arch)
            + r"\\"
            + DATASET_DISPLAY.get(ds, ds)
            + r"\\"
            + f"{epochs}, {img}px"
            + "}}"
        )
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    col_rankings = {}
    for key in sorted_keys:
        ms = filtered_table[key]
        sorted_by_mean = sorted(ms.items(), key=lambda x: -x[1][0])
        best = sorted_by_mean[0][0] if sorted_by_mean else None
        second = sorted_by_mean[1][0] if len(sorted_by_mean) > 1 else None
        col_rankings[key] = (best, second)

    for method in all_methods:
        display = METHOD_DISPLAY.get(method, method)
        row_parts = [esc(display)]
        for key in sorted_keys:
            ms = filtered_table[key]
            if method in ms:
                m, s, n = ms[method]
                best, second = col_rankings[key]
                cell = fmt_pm(m, s)
                if method == best:
                    cell = r"\textbf{" + cell + "}"
                elif method == second:
                    cell = r"\underline{" + cell + "}"
                row_parts.append(cell)
            else:
                row_parts.append("---")
        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def generate_sgd200_table(rows, output_path):

    sgd_rows = [
        r
        for r in rows
        if r.get("status") == "success"
        and isinstance(r.get("protocol"), str)
        and "sgd" in r.get("protocol", "").lower()
        and "e200" in r.get("protocol", "")
    ]
    if not sgd_rows:
        print("  Skipping SGD200 table (no data)")
        return

    method_ds = defaultdict(list)
    for r in sgd_rows:
        method_ds[(r["method"], r["dataset"])].append(r["test_acc1"])

    datasets = sorted({r["dataset"] for r in sgd_rows})
    methods = sorted(
        {r["method"] for r in sgd_rows}, key=lambda m: METHOD_DISPLAY.get(m, m)
    )

    ds_rankings = {}
    for ds in datasets:
        means = [(m, mean(method_ds[(m, ds)])) for m in methods if method_ds[(m, ds)]]
        means.sort(key=lambda x: -x[1])
        ds_rankings[ds] = (
            means[0][0] if means else None,
            means[1][0] if len(means) > 1 else None,
        )

    n_seeds = max(len(v) for v in method_ds.values()) if method_ds else 0
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{ResNet-18 top-1 accuracy (\%) under the standard SGD 200-epoch recipe ("
        + str(n_seeds)
        + r" seeds, batch 128, lr 0.1, no label smoothing).}"
    )
    lines.append(r"\label{tab:sgd200}")
    lines.append(r"\begin{tabular}{l" + "c" * len(datasets) + "}")
    lines.append(r"\toprule")
    header = (
        "Method & " + " & ".join(DATASET_DISPLAY.get(d, d) for d in datasets) + r" \\"
    )
    lines.append(header)
    lines.append(r"\midrule")

    pa_start = False
    for method in methods:
        if method in PA_NORM_METHODS and not pa_start:
            lines.append(r"\midrule")
            pa_start = True
        display = METHOD_DISPLAY.get(method, method)
        parts = [esc(display)]
        for ds in datasets:
            accs = method_ds[(method, ds)]
            if accs:
                m, s = mean(accs), std(accs) if len(accs) > 1 else 0
                cell = fmt_pm(m, s)
                best, second = ds_rankings[ds]
                if method == best:
                    cell = r"\textbf{" + cell + "}"
                elif method == second:
                    cell = r"\underline{" + cell + "}"
                parts.append(cell)
            else:
                parts.append("---")
        lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def generate_batchsize_table(rows, output_path):

    bs_rows = [
        r
        for r in rows
        if r.get("status") == "success"
        and r.get("dataset") == "cifar100"
        and r.get("architecture") == "resnet18"
        and r.get("eval_ood") == False
        and r.get("epochs") == 50
    ]

    method_bs = defaultdict(list)
    for r in bs_rows:
        bs = r.get("batch_size")
        if isinstance(bs, int):
            method_bs[(r["method"], bs)].append(r["test_acc1"])

    all_bs = sorted({bs for (_, bs) in method_bs.keys()})
    all_methods = sorted(
        {m for (m, _) in method_bs.keys()}, key=lambda m: METHOD_DISPLAY.get(m, m)
    )

    if len(all_bs) < 2:
        print("  Skipping batch size table (not enough batch sizes)")
        return

    bs_rankings = {}
    for bs in all_bs:
        means = [
            (m, mean(method_bs[(m, bs)])) for m in all_methods if method_bs[(m, bs)]
        ]
        means.sort(key=lambda x: -x[1])
        bs_rankings[bs] = (
            means[0][0] if means else None,
            means[1][0] if len(means) > 1 else None,
        )

    n_seeds = max(len(v) for v in method_bs.values()) if method_bs else 0
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Batch size sensitivity: ResNet-18 CIFAR-100 accuracy (\%) across batch sizes ("
        + str(n_seeds)
        + r" seeds, AdamW, 50 epochs).}"
    )
    lines.append(r"\label{tab:batchsize}")
    lines.append(r"\begin{tabular}{l" + "c" * len(all_bs) + "}")
    lines.append(r"\toprule")
    header = "Method & " + " & ".join(f"BS={bs}" for bs in all_bs) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    pa_start = False
    for method in all_methods:
        if method in PA_NORM_METHODS and not pa_start:
            lines.append(r"\midrule")
            pa_start = True
        display = METHOD_DISPLAY.get(method, method)
        parts = [esc(display)]
        for bs in all_bs:
            accs = method_bs[(method, bs)]
            if accs:
                m, s = mean(accs), std(accs) if len(accs) > 1 else 0
                cell = fmt_pm(m, s)
                best, second = bs_rankings[bs]
                if method == best:
                    cell = r"\textbf{" + cell + "}"
                elif method == second:
                    cell = r"\underline{" + cell + "}"
                parts.append(cell)
            else:
                parts.append("---")
        lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def generate_ood_sweep_table(rows, output_path):

    ood_rows = [
        r
        for r in rows
        if r.get("status") == "success"
        and r.get("eval_ood") == True
        and isinstance(r.get("ood_acc1_cifar10c_mean"), (int, float))
    ]
    if not ood_rows:
        print("  Skipping OOD sweep table (no data)")
        return

    method_sev = defaultdict(list)
    for r in ood_rows:
        sev = r.get("ood_severity", 2)
        method_sev[(r["method"], sev)].append(r["ood_acc1_cifar10c_mean"])

    all_sevs = sorted({sev for (_, sev) in method_sev.keys()})
    all_methods = sorted(
        {m for (m, _) in method_sev.keys()}, key=lambda m: METHOD_DISPLAY.get(m, m)
    )

    if len(all_sevs) < 2:
        print(f"  Skipping OOD sweep table (only {len(all_sevs)} severity levels)")
        return

    sev_rankings = {}
    for sev in all_sevs:
        means = [
            (m, mean(method_sev[(m, sev)])) for m in all_methods if method_sev[(m, sev)]
        ]
        means.sort(key=lambda x: -x[1])
        sev_rankings[sev] = (
            means[0][0] if means else None,
            means[1][0] if len(means) > 1 else None,
        )

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{CIFAR-10-C corruption mean accuracy (\%) across severity levels (ResNet-18, AdamW).}"
    )
    lines.append(r"\label{tab:ood_sweep}")
    lines.append(r"\begin{tabular}{l" + "c" * len(all_sevs) + "}")
    lines.append(r"\toprule")
    header = "Method & " + " & ".join(f"Sev.\\ {s}" for s in all_sevs) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    pa_start = False
    for method in all_methods:
        if method in PA_NORM_METHODS and not pa_start:
            lines.append(r"\midrule")
            pa_start = True
        display = METHOD_DISPLAY.get(method, method)
        parts = [esc(display)]
        for sev in all_sevs:
            accs = method_sev[(method, sev)]
            if accs:
                m = mean(accs)
                cell = fmt_val(m)
                best, second = sev_rankings[sev]
                if method == best:
                    cell = r"\textbf{" + cell + "}"
                elif method == second:
                    cell = r"\underline{" + cell + "}"
                parts.append(cell)
            else:
                parts.append("---")
        lines.append(" & ".join(parts) + r" \\")

    if "batchnorm" in all_methods:
        lines.append(r"\midrule")

        best_pa = None
        best_pa_gain = -999
        for m in all_methods:
            if m in PA_NORM_METHODS:
                gains = []
                for sev in all_sevs:
                    if method_sev[(m, sev)] and method_sev[("batchnorm", sev)]:
                        gains.append(
                            mean(method_sev[(m, sev)])
                            - mean(method_sev[("batchnorm", sev)])
                        )
                if gains and mean(gains) > best_pa_gain:
                    best_pa_gain = mean(gains)
                    best_pa = m
        if best_pa:
            display = METHOD_DISPLAY.get(best_pa, best_pa)
            parts = [f"$\\Delta$BN ({esc(display)})"]
            for sev in all_sevs:
                if method_sev[(best_pa, sev)] and method_sev[("batchnorm", sev)]:
                    delta = mean(method_sev[(best_pa, sev)]) - mean(
                        method_sev[("batchnorm", sev)]
                    )
                    parts.append(f"+{delta:.2f}")
                else:
                    parts.append("---")
            lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def generate_detach_ablation_table(rows, output_path):

    ablation_methods = [
        "panorm",
        "panorm_nodetach",
        "panorm_selective",
        "panorm_selective_75",
        "panorm_switchnorm",
        "batchnorm",
    ]
    abl_rows = [
        r
        for r in rows
        if r.get("status") == "success"
        and r.get("method") in ablation_methods
        and r.get("architecture") == "resnet18"
        and r.get("eval_ood") == False
        and "sgd" not in str(r.get("protocol", ""))
        and r.get("batch_size") == 256
        and r.get("epochs", 0) == 50
    ]

    if not abl_rows:
        print("  Skipping detach ablation table (no data)")
        return

    method_ds = defaultdict(list)
    gate_stats = defaultdict(lambda: {"entropy": [], "main": []})
    for r in abl_rows:
        method_ds[(r["method"], r["dataset"])].append(r["test_acc1"])
        if r["method"] in PA_NORM_METHODS:
            if (
                isinstance(r.get("gate_entropy_norm"), (int, float))
                and r["gate_entropy_norm"] != -1.0
            ):
                gate_stats[r["method"]]["entropy"].append(r["gate_entropy_norm"])
                gate_stats[r["method"]]["main"].append(r["gate_main_weight"])

    datasets = sorted({r["dataset"] for r in abl_rows})

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Detach ablation: PA-Norm with vs.\ without stop-gradient on the diagnostic. ResNet-18, AdamW.}"
    )
    lines.append(r"\label{tab:detach_ablation}")
    lines.append(r"\begin{tabular}{l" + "c" * len(datasets) + "cc}")
    lines.append(r"\toprule")
    header = "Method & " + " & ".join(DATASET_DISPLAY.get(d, d) for d in datasets)
    header += r" & Gate Entropy & Dom.\ Branch \\"
    lines.append(header)
    lines.append(r"\midrule")

    for method in ablation_methods:
        display = METHOD_DISPLAY.get(method, method)
        parts = [esc(display)]
        for ds in datasets:
            accs = method_ds[(method, ds)]
            if accs:
                m, s = mean(accs), std(accs) if len(accs) > 1 else 0
                parts.append(fmt_pm(m, s))
            else:
                parts.append("---")

        if method in gate_stats and gate_stats[method]["entropy"]:
            ent = mean(gate_stats[method]["entropy"])
            main_w = mean(gate_stats[method]["main"])
            parts.append(f"{ent:.3f}")
            parts.append(f"BN ({main_w:.2f})")
        else:
            parts.append("---")
            parts.append("---")
        lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def generate_efficiency_table(output_path):

    eff_path = os.path.join(os.path.dirname(__file__), "result_efficiency.csv")
    if not os.path.exists(eff_path):
        print("  Skipping efficiency table (no data)")
        return

    eff_rows = load_csv(eff_path)
    if not eff_rows:
        print("  Skipping efficiency table (empty)")
        return

    bn_train = None
    for r in eff_rows:
        if r["method"] == "batchnorm":
            bn_train = r.get("train_throughput_img_s", 0)
            break

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Efficiency comparison on ResNet-18, CIFAR-100. Train/Infer throughput in images/second. ``Rel."
        " = throughput relative to BatchNorm.}"
    )
    lines.append(r"\label{tab:efficiency}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Method & Params (M) & Train (img/s) & Rel. & Infer (img/s) & Train Lat. (ms) & Infer Lat. (ms) \\"
    )
    lines.append(r"\midrule")

    pa_start = False
    for r in eff_rows:
        method = r["method"]
        if method in PA_NORM_METHODS and not pa_start:
            lines.append(r"\midrule")
            pa_start = True
        display = METHOD_DISPLAY.get(method, method)
        train_tp = r.get("train_throughput_img_s", 0)
        rel = train_tp / bn_train if bn_train and bn_train > 0 else 0
        row = (
            f"{esc(display)} & "
            f"{r.get('num_params_m', 0):.3f} & "
            f"{train_tp:.0f} & "
            f"{rel:.2f}$\\times$ & "
            f"{r.get('infer_throughput_img_s', 0):.0f} & "
            f"{r.get('train_latency_ms', 0):.1f} & "
            f"{r.get('infer_latency_ms', 0):.1f}"
            r" \\"
        )
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def generate_wideresnet_table(rows, output_path):

    wrn_rows = [
        r
        for r in rows
        if r.get("status") == "success"
        and r.get("architecture") == "wide_resnet50_2"
        and r.get("eval_ood") == False
    ]
    if not wrn_rows:
        print("  Skipping WideResNet table (no data)")
        return

    method_ds = defaultdict(list)
    for r in wrn_rows:
        method_ds[(r["method"], r["dataset"])].append(r["test_acc1"])

    datasets = sorted({r["dataset"] for r in wrn_rows})
    methods = sorted(
        {r["method"] for r in wrn_rows}, key=lambda m: METHOD_DISPLAY.get(m, m)
    )

    ds_rankings = {}
    for ds in datasets:
        means = [(m, mean(method_ds[(m, ds)])) for m in methods if method_ds[(m, ds)]]
        means.sort(key=lambda x: -x[1])
        ds_rankings[ds] = (
            means[0][0] if means else None,
            means[1][0] if len(means) > 1 else None,
        )

    n_seeds = max(len(v) for v in method_ds.values()) if method_ds else 0
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{WideResNet-50-2 top-1 accuracy (\%) on CIFAR-100 ("
        + str(n_seeds)
        + r" seeds, 50 epochs, AdamW).}"
    )
    lines.append(r"\label{tab:wideresnet}")
    lines.append(r"\begin{tabular}{l" + "c" * len(datasets) + "}")
    lines.append(r"\toprule")
    header = (
        "Method & " + " & ".join(DATASET_DISPLAY.get(d, d) for d in datasets) + r" \\"
    )
    lines.append(header)
    lines.append(r"\midrule")

    pa_start = False
    for method in methods:
        if method in PA_NORM_METHODS and not pa_start:
            lines.append(r"\midrule")
            pa_start = True
        display = METHOD_DISPLAY.get(method, method)
        parts = [esc(display)]
        for ds in datasets:
            accs = method_ds[(method, ds)]
            if accs:
                m, s = mean(accs), std(accs) if len(accs) > 1 else 0
                cell = fmt_pm(m, s)
                best, second = ds_rankings[ds]
                if method == best:
                    cell = r"\textbf{" + cell + "}"
                elif method == second:
                    cell = r"\underline{" + cell + "}"
                parts.append(cell)
            else:
                parts.append("---")
        lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def generate_per_protocol_tables(mean_std_table, output_dir):

    arch_proto = defaultdict(dict)
    for (arch, ds, proto), ms in mean_std_table.items():
        arch_proto[(arch, proto)][ds] = ms

    for (arch, proto), ds_data in sorted(arch_proto.items()):
        datasets = sorted(ds_data.keys())

        all_methods = set()
        for ms in ds_data.values():
            all_methods.update(ms.keys())
        all_methods = sorted(
            all_methods,
            key=lambda m: (
                0 if m in PA_NORM_METHODS else 1,
                -mean_std_table.get((arch, datasets[0], proto), {}).get(m, (0, 0, 0))[
                    0
                ],
            ),
        )

        def sort_key(m):
            first_ds = datasets[0]
            ms_val = ds_data.get(first_ds, {}).get(m)
            acc = ms_val[0] if ms_val else 0
            return (0 if m not in PA_NORM_METHODS else 1, -acc)

        all_methods = sorted(all_methods, key=sort_key)

        ds_rankings = {}
        for ds in datasets:
            ms = ds_data[ds]
            sorted_m = sorted(ms.items(), key=lambda x: -x[1][0])
            ds_rankings[ds] = (
                sorted_m[0][0] if sorted_m else None,
                sorted_m[1][0] if len(sorted_m) > 1 else None,
            )

        lines = []
        lines.append(r"\begin{tabular}{l" + "c" * len(datasets) + "}")
        lines.append(r"\toprule")
        header = (
            "Method & "
            + " & ".join(DATASET_DISPLAY.get(d, d) for d in datasets)
            + r" \\"
        )
        lines.append(header)
        lines.append(r"\midrule")

        for method in all_methods:
            display = METHOD_DISPLAY.get(method, method)
            parts = [esc(display)]
            for ds in datasets:
                ms = ds_data[ds]
                if method in ms:
                    m, s, n = ms[method]
                    cell = f"{m:.2f} $\\pm$ {s:.2f}"
                    best, second = ds_rankings[ds]
                    if method == best:
                        cell = r"\textbf{" + cell + "}"
                    elif method == second:
                        cell = r"\underline{" + cell + "}"
                    parts.append(cell)
                else:
                    parts.append("---")
            lines.append(" & ".join(parts) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

        fname = f"real_{arch}_{proto}_acc1.tex"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  Written: {fpath}")


def generate_detach_ablation_deep_table(rows, output_path):

    ablation_methods = [
        "panorm",
        "panorm_nodetach",
        "panorm_selective",
        "panorm_selective_75",
        "panorm_switchnorm",
        "panorm_switchnorm_nodetach",
        "batchnorm",
    ]
    abl_rows = [
        r
        for r in rows
        if r.get("status") == "success"
        and r.get("method") in ablation_methods
        and r.get("eval_ood") == False
        and "sgd" not in str(r.get("protocol", ""))
        and r.get("batch_size") == 256
        and r.get("epochs", 0) <= 50
    ]

    if not abl_rows:
        print("  Skipping deep detach ablation table (no data)")
        return

    method_arch_ds = defaultdict(list)
    for r in abl_rows:
        method_arch_ds[(r["method"], r["architecture"], r["dataset"])].append(
            r["test_acc1"]
        )

    archs = sorted({r["architecture"] for r in abl_rows})
    datasets = sorted({r["dataset"] for r in abl_rows})

    cols = [
        (a, d)
        for a in archs
        for d in datasets
        if any(method_arch_ds.get((m, a, d)) for m in ablation_methods)
    ]

    if not cols:
        print("  Skipping deep detach ablation table (no column data)")
        return

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{PA-Norm variants across architectures and datasets. ND = no detach (without stop-gradient). ``---'' indicates experiments pending.}"
    )
    lines.append(r"\label{tab:detach_ablation_deep}")
    lines.append(r"\begin{tabular}{l" + "c" * len(cols) + "}")
    lines.append(r"\toprule")

    arch_spans = {}
    for a, d in cols:
        arch_spans.setdefault(a, []).append(d)
    header1_parts = [""]
    for a, ds_list in sorted(arch_spans.items()):
        if len(ds_list) > 1:
            header1_parts.append(
                r"\multicolumn{"
                + str(len(ds_list))
                + r"}{c}{"
                + ARCH_DISPLAY.get(a, a)
                + r"}"
            )
        else:
            header1_parts.append(ARCH_DISPLAY.get(a, a))
    lines.append(" & ".join(header1_parts) + r" \\")

    header2_parts = ["Method"]
    for a, d in cols:
        header2_parts.append(DATASET_DISPLAY.get(d, d))
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")

    for method in ablation_methods:
        display = METHOD_DISPLAY.get(method, method)
        parts = [esc(display)]
        for a, d in cols:
            accs = method_arch_ds.get((method, a, d), [])
            if accs:
                m, s = mean(accs), std(accs) if len(accs) > 1 else 0
                parts.append(fmt_pm(m, s))
            else:
                parts.append("---")
        lines.append(" & ".join(parts) + r" \\")

        if method == "panorm_switchnorm_nodetach":
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def generate_convergence_table(rows, output_path):

    import json as _json

    conv_rows = [
        r
        for r in rows
        if r.get("status") == "success"
        and r.get("architecture") == "resnet18"
        and r.get("dataset") == "cifar100"
        and isinstance(r.get("dynamics"), str)
        and r.get("dynamics", "[]") != "[]"
    ]

    if not conv_rows:
        print("  Skipping convergence table (no dynamics data)")
        return

    method_dynamics = defaultdict(list)
    for r in conv_rows:
        try:
            dyn = _json.loads(r["dynamics"])
            if dyn:
                method_dynamics[r["method"]].append(dyn)
        except (ValueError, TypeError):
            pass

    if not method_dynamics:
        print("  Skipping convergence table (no parseable dynamics)")
        return

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Convergence speed: epochs to reach target accuracy (ResNet-18, CIFAR-100). "
        r"Dashes indicate the target was not reached.}"
    )
    lines.append(r"\label{tab:convergence}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Ep.\ to 50\% & Ep.\ to 60\% & Ep.\ to 65\% & Final Acc \\")
    lines.append(r"\midrule")

    targets = [50, 60, 65]
    method_order = [
        "batchnorm",
        "groupnorm",
        "switchnorm",
        "panorm",
        "panorm_switchnorm",
        "panorm_lite_fast",
    ]

    for method in method_order:
        if method not in method_dynamics:
            continue
        display = METHOD_DISPLAY.get(method, method)
        parts = [esc(display)]

        for target in targets:
            epochs_to_target = []
            for dyn in method_dynamics[method]:
                found = False
                for entry in dyn:
                    if entry.get("test_acc1", 0) >= target:
                        epochs_to_target.append(entry["epoch"] + 1)
                        found = True
                        break
                if not found:
                    epochs_to_target.append(float("inf"))

            finite = [e for e in epochs_to_target if e != float("inf")]
            if finite:
                avg_ep = mean(finite)
                parts.append(f"{avg_ep:.1f}")
            else:
                parts.append("---")

        final_accs = []
        for dyn in method_dynamics[method]:
            if dyn:
                final_accs.append(dyn[-1].get("test_acc1", -1))
        if final_accs:
            m, s = mean(final_accs), std(final_accs) if len(final_accs) > 1 else 0
            parts.append(fmt_pm(m, s))
        else:
            parts.append("---")

        lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def print_summary(
    mean_std_table, avg_ranks, wins, stat_results, gate_results, ood_results
):

    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 80)

    print("\n--- 1. Mean +/- Std Accuracy per (Arch, Dataset, Protocol) ---")
    for key in sorted(mean_std_table.keys()):
        arch, ds, proto = key
        epochs = "50ep" if "e50" in proto else "20ep"
        img = "64px" if "im64" in proto else "32px"
        print(
            f"\n  {ARCH_DISPLAY.get(arch, arch)} / {DATASET_DISPLAY.get(ds, ds)} / {epochs},{img}:"
        )
        ms = mean_std_table[key]
        sorted_methods = sorted(ms.items(), key=lambda x: -x[1][0])
        for method, (m, s, n) in sorted_methods:
            display = METHOD_DISPLAY.get(method, method)
            print(f"    {display:22s}: {m:6.2f} +/- {s:5.2f}  (n={n})")

    print("\n--- 2. Consolidated Rankings ---")
    sorted_ranks = sorted(avg_ranks.items(), key=lambda x: x[1][0])
    for method, (ar, ng) in sorted_ranks:
        display = METHOD_DISPLAY.get(method, method)
        w = wins.get(method, 0)
        print(f"  {display:22s}: avg rank = {ar:.2f} (over {ng} settings), wins = {w}")

    print("\n--- 3. Statistical Tests (PA variants vs BatchNorm) ---")
    for r in stat_results:
        display = METHOD_DISPLAY.get(r["method"], r["method"])
        print(f"\n  {display}:")
        print(f'    N pairs: {r["n_pairs"]}')
        print(
            f'    PA mean: {fmt_val(r["pa_mean"])}, BN mean: {fmt_val(r["bn_mean"])}, Delta: {fmt_val(r["delta_mean"])}'
        )
        print(
            f'    t-test: t={fmt_val(r["t_stat"])}, p={fmt_val(r["t_pvalue"], ".4f")}'
        )
        print(
            f'    Wilcoxon: W={fmt_val(r["w_stat"], ".1f")}, p={fmt_val(r["w_pvalue"], ".4f")}'
        )
        print(f'    Cohen\'s d: {fmt_val(r["cohens_d"], ".3f")}')

    if gate_results is not None:
        agg, detail = gate_results
        print("\n--- 4. Gate Statistics (Aggregated) ---")
        for method in sorted(agg.keys()):
            display = METHOD_DISPLAY.get(method, method)
            r = agg[method]
            print(f'  {display} (n={r["n"]}):')
            print(f'    Entropy:  {r["entropy"][0]:.3f} +/- {r["entropy"][1]:.3f}')
            print(f'    Branch 0: {r["branch0"][0]:.3f} +/- {r["branch0"][1]:.3f}')
            print(f'    Branch 1: {r["branch1"][0]:.3f} +/- {r["branch1"][1]:.3f}')
            print(f'    Branch 2: {r["branch2"][0]:.3f} +/- {r["branch2"][1]:.3f}')

    if ood_results is not None:
        acc_results, ood_gates = ood_results
        print("\n--- 5. OOD Analysis (CIFAR-10-C, Severity 2) ---")
        for method in sorted(acc_results.keys()):
            display = METHOD_DISPLAY.get(method, method)
            r = acc_results[method]
            print(f'  {display} (n={r["n"]}):')
            print(f'    Clean:    {r["clean"][0]:.2f} +/- {r["clean"][1]:.2f}')
            print(f'    OOD Mean: {r["ood_mean"][0]:.2f} +/- {r["ood_mean"][1]:.2f}')
            print(f'    OOD Worst:{r["ood_worst"][0]:.2f} +/- {r["ood_worst"][1]:.2f}')
            print(f'    Drop:     {r["drop"][0]:.2f} +/- {r["drop"][1]:.2f}')

        if ood_gates:
            print("\n  OOD Gate Statistics (clean vs corrupted):")
            for method in sorted(ood_gates.keys()):
                display = METHOD_DISPLAY.get(method, method)
                gs = ood_gates[method]
                if gs["clean_entropy"] and gs["ood_entropy"]:
                    print(f"  {display}:")
                    print(
                        f'    Clean entropy: {mean(gs["clean_entropy"]):.3f}, OOD entropy: {mean(gs["ood_entropy"]):.3f}'
                    )
                    print(
                        f'    Clean main:    {mean(gs["clean_main"]):.3f}, OOD main:    {mean(gs["ood_main"]):.3f}'
                    )
                    print(
                        f'    Clean b0/b1/b2: {mean(gs["clean_b0"]):.3f}/{mean(gs["clean_b1"]):.3f}/{mean(gs["clean_b2"]):.3f}'
                    )
                    print(
                        f'    OOD   b0/b1/b2: {mean(gs["ood_b0"]):.3f}/{mean(gs["ood_b1"]):.3f}/{mean(gs["ood_b2"]):.3f}'
                    )


def generate_convnext_table(rows, path):

    cn_rows = [
        r
        for r in rows
        if r.get("architecture") == "convnext_tiny"
        and r.get("status") == "success"
        and r.get("eval_ood") == False
    ]
    if not cn_rows:
        print("  No ConvNeXt results found, skipping.")
        with open(path, "w") as f:
            f.write("% No ConvNeXt results available yet.\n")
        return

    groups = defaultdict(list)
    for r in cn_rows:
        acc = r.get("test_acc1", -1)
        if isinstance(acc, (int, float)) and acc > 0:
            groups[(r["dataset"], r["method"])].append(float(acc))

    datasets = sorted(set(k[0] for k in groups))
    methods = sorted(set(k[1] for k in groups), key=lambda m: METHOD_DISPLAY.get(m, m))

    ds_rankings = {}
    for ds in datasets:
        means_list = [(m, mean(groups[(ds, m)])) for m in methods if groups[(ds, m)]]
        means_list.sort(key=lambda x: -x[1])
        ds_rankings[ds] = (
            means_list[0][0] if means_list else None,
            means_list[1][0] if len(means_list) > 1 else None,
        )

    n_seeds = max(len(v) for v in groups.values()) if groups else 0
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(
        r"\caption{ConvNeXt-Tiny top-1 accuracy (\%) on CIFAR-100 ("
        + str(n_seeds)
        + r" seeds, 50 epochs, AdamW, BS=128). Best in \textbf{bold}, second \underline{underlined}.}"
    )
    lines.append(r"\label{tab:convnext}")
    lines.append(r"\begin{tabular}{l" + "c" * len(datasets) + "}")
    lines.append(r"\toprule")
    header = (
        "Method & " + " & ".join(DATASET_DISPLAY.get(d, d) for d in datasets) + r" \\"
    )
    lines.append(header)
    lines.append(r"\midrule")

    pa_start = False
    for method in methods:
        if method in PA_NORM_METHODS and not pa_start:
            lines.append(r"\midrule")
            pa_start = True
        display = METHOD_DISPLAY.get(method, method)
        parts = [esc(display)]
        for ds in datasets:
            accs = groups[(ds, method)]
            if accs:
                m, s = mean(accs), std(accs) if len(accs) > 1 else 0
                cell = fmt_pm(m, s)
                best, second = ds_rankings[ds]
                if method == best:
                    cell = r"\textbf{" + cell + "}"
                elif method == second:
                    cell = r"\underline{" + cell + "}"
                parts.append(cell)
            else:
                parts.append("---")
        lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {path}")


def generate_vit_table(rows, path):

    vit_rows = [
        r
        for r in rows
        if r.get("architecture", "").startswith("vit_")
        and r.get("status") == "success"
        and r.get("epochs", 0) <= 50
    ]
    if not vit_rows:
        print("  No ViT results found, skipping.")
        with open(path, "w") as f:
            f.write("% No ViT results available yet.\n")
        return

    from collections import defaultdict

    groups = defaultdict(list)
    for r in vit_rows:
        key = (r["dataset"], r["method"])
        acc = r.get("test_acc1", -1)
        if isinstance(acc, (int, float)) and acc > 0:
            groups[key].append(float(acc))

    datasets = sorted(set(k[0] for k in groups))
    methods = sorted(set(k[1] for k in groups))

    method_names = dict(METHOD_DISPLAY)

    dataset_names = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "tinyimagenet": "TinyImgNet",
    }

    epochs_seen = set()
    for r in vit_rows:
        proto = str(r.get("protocol", ""))
        import re

        ep_match = re.search(r"e(\d+)", proto)
        if ep_match:
            epochs_seen.add(int(ep_match.group(1)))
    max_epochs = max(epochs_seen) if epochs_seen else 50
    ep_str = str(max_epochs)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(
        f"\\caption{{ViT-Small top-1 accuracy (\\%) on various datasets (3 seeds, {ep_str} epochs, AdamW). "
        "Best in \\textbf{bold}.}"
    )
    lines.append("\\label{tab:vit_results}")
    cols = "l" + "c" * len(datasets)
    lines.append("\\begin{tabular}{" + cols + "}")
    lines.append("\\toprule")
    header = (
        "Method & " + " & ".join(dataset_names.get(d, d) for d in datasets) + " \\\\"
    )
    lines.append(header)
    lines.append("\\midrule")

    best = {}
    for d in datasets:
        best_val = -1
        for m in methods:
            vals = groups.get((d, m), [])
            if vals and mean(vals) > best_val:
                best_val = mean(vals)
                best[d] = m

    for m in methods:
        name = method_names.get(m, m)
        cells = [name]
        for d in datasets:
            vals = groups.get((d, m), [])
            if vals:
                m_val = mean(vals)
                s_val = std(vals) if len(vals) > 1 else 0.0
                text = f"{m_val:.2f}$\\pm${s_val:.2f}"
                if best.get(d) == m:
                    text = "\\textbf{" + text + "}"
                cells.append(text)
            else:
                cells.append("---")
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote ViT table: {path}")


def generate_selective_detach_table(rows, output_path):

    sel_methods = [
        "batchnorm",
        "panorm",
        "panorm_nodetach",
        "panorm_selective",
        "panorm_selective_75",
    ]
    sel_rows = [
        r
        for r in rows
        if r.get("status") == "success"
        and r.get("method") in sel_methods
        and r.get("eval_ood") == False
        and r.get("architecture") in ("resnet18", "resnet50")
        and "sgd" not in str(r.get("protocol", ""))
        and r.get("batch_size") == 256
    ]

    if not sel_rows:
        print("  Skipping selective detach table (no data)")
        with open(output_path, "w") as f:
            f.write("% No selective detach results available yet.\n")
        return

    method_arch_ds = defaultdict(list)
    for r in sel_rows:
        method_arch_ds[(r["method"], r["architecture"], r["dataset"])].append(
            r["test_acc1"]
        )

    archs = sorted({r["architecture"] for r in sel_rows})
    datasets = sorted({r["dataset"] for r in sel_rows})
    cols = [
        (a, d)
        for a in archs
        for d in datasets
        if any(method_arch_ds.get((m, a, d)) for m in sel_methods)
    ]

    if not cols:
        print("  Skipping selective detach table (no column data)")
        return

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Selective detach analysis: depth-dependent stop-gradient strategies. "
        r"PA-Norm-Sel detaches early layers only (50\%), PA-Norm-Sel75 detaches 75\% of layers. "
        r"Best in \textbf{bold}, second \underline{underlined}.}"
    )
    lines.append(r"\label{tab:selective_detach}")
    lines.append(r"\begin{tabular}{l" + "c" * len(cols) + "}")
    lines.append(r"\toprule")

    arch_spans = {}
    for a, d in cols:
        arch_spans.setdefault(a, []).append(d)
    header1_parts = [""]
    for a, ds_list in sorted(arch_spans.items()):
        aname = ARCH_DISPLAY.get(a, a)
        if len(ds_list) > 1:
            header1_parts.append(
                r"\multicolumn{" + str(len(ds_list)) + r"}{c}{" + aname + "}"
            )
        else:
            header1_parts.append(aname)
    lines.append(" & ".join(header1_parts) + r" \\")

    header2_parts = ["Method"]
    for a, d in cols:
        header2_parts.append(DATASET_DISPLAY.get(d, d))
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")

    col_vals = {}
    for a, d in cols:
        method_means = []
        for m in sel_methods:
            accs = method_arch_ds.get((m, a, d), [])
            if accs:
                method_means.append((m, mean(accs)))
        method_means.sort(key=lambda x: -x[1])
        col_vals[(a, d)] = {
            "best": method_means[0][0] if method_means else None,
            "second": method_means[1][0] if len(method_means) > 1 else None,
        }

    for method in sel_methods:
        display = METHOD_DISPLAY.get(method, method)
        parts = [esc(display)]
        for a, d in cols:
            accs = method_arch_ds.get((method, a, d), [])
            if accs:
                m_val, s_val = mean(accs), std(accs) if len(accs) > 1 else 0
                cell = fmt_pm(m_val, s_val)
                if method == col_vals[(a, d)]["best"]:
                    cell = r"\textbf{" + cell + "}"
                elif method == col_vals[(a, d)]["second"]:
                    cell = r"\underline{" + cell + "}"
                parts.append(cell)
            else:
                parts.append("---")
        lines.append(" & ".join(parts) + r" \\")
        if method == "batchnorm":
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def generate_flops_matched_table(rows, output_path):

    import re

    pa_methods = ["panorm", "panorm_lite_fast", "panorm_switchnorm", "panorm_nodetach"]
    all_methods_needed = pa_methods + ["batchnorm"]

    flops_rows = [
        r
        for r in rows
        if r.get("status") == "success"
        and r.get("method") in all_methods_needed
        and r.get("eval_ood") == False
        and r.get("architecture") == "resnet18"
        and "sgd" not in str(r.get("protocol", ""))
        and r.get("batch_size") == 256
    ]

    if not flops_rows:
        print("  Skipping FLOPs-matched table (no data)")
        with open(output_path, "w") as f:
            f.write("% No FLOPs-matched results available yet.\n")
        return

    bn_by_epoch = defaultdict(lambda: defaultdict(list))
    pa_results = defaultdict(lambda: defaultdict(list))

    for r in flops_rows:
        proto = str(r.get("protocol", ""))
        ep_match = re.search(r"e(\d+)", proto)
        epochs = int(ep_match.group(1)) if ep_match else 50
        key = (r["architecture"], r["dataset"])
        if r["method"] == "batchnorm":
            bn_by_epoch[epochs][key].append(r["test_acc1"])
        elif r["method"] in pa_methods:
            pa_results[r["method"]][key].append(r["test_acc1"])

    settings = set()
    for m in pa_methods:
        settings.update(pa_results[m].keys())

    if not settings:
        print("  Skipping FLOPs-matched table (no settings)")
        return

    bn_epochs = sorted(bn_by_epoch.keys())

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{FLOPs-matched comparison: BatchNorm with proportionally more training epochs "
        r"vs.\ PA-Norm at standard epochs. PA-Norm overhead: $\sim$1.7$\times$ (PA-Norm) and "
        r"$\sim$1.45$\times$ (PA-Norm-LF) vs.\ BN. Best in \textbf{bold}.}"
    )
    lines.append(r"\label{tab:flops_matched}")

    sorted_settings = sorted(settings)
    ncols = len(sorted_settings)
    lines.append(r"\begin{tabular}{l" + "c" * ncols + "}")
    lines.append(r"\toprule")

    header_parts = ["Method (epochs)"]
    for arch, ds in sorted_settings:
        header_parts.append(
            ARCH_DISPLAY.get(arch, arch) + " " + DATASET_DISPLAY.get(ds, ds)
        )
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    all_row_data = []

    for ep in bn_epochs:
        display = f"BN ({ep}ep)"
        parts = [esc(display)]
        row_means = {}
        for s in sorted_settings:
            accs = bn_by_epoch[ep].get(s, [])
            if accs:
                m_val = mean(accs)
                s_val = std(accs) if len(accs) > 1 else 0
                parts.append(fmt_pm(m_val, s_val))
                row_means[s] = m_val
            else:
                parts.append("---")
        all_row_data.append(("bn_" + str(ep), display, parts, row_means))

    lines.append(r"\midrule")

    for pa_m in pa_methods:
        display = METHOD_DISPLAY.get(pa_m, pa_m) + " (50ep)"
        parts = [esc(display)]
        row_means = {}
        for s in sorted_settings:
            accs = pa_results[pa_m].get(s, [])
            if accs:
                m_val = mean(accs)
                s_val = std(accs) if len(accs) > 1 else 0
                parts.append(fmt_pm(m_val, s_val))
                row_means[s] = m_val
            else:
                parts.append("---")
        all_row_data.append((pa_m, display, parts, row_means))

    best_per_setting = {}
    for s in sorted_settings:
        best_val = -1
        best_id = None
        for row_id, _, _, row_means in all_row_data:
            if s in row_means and row_means[s] > best_val:
                best_val = row_means[s]
                best_id = row_id
        best_per_setting[s] = best_id

    for row_id, display, parts, row_means in all_row_data:
        formatted_parts = [parts[0]]
        for i, s in enumerate(sorted_settings):
            cell = parts[i + 1]
            if cell != "---" and best_per_setting.get(s) == row_id:
                cell = r"\textbf{" + cell + "}"
            formatted_parts.append(cell)
        lines.append(" & ".join(formatted_parts) + r" \\")

        if row_id.startswith("bn_") and row_id == all_row_data[len(bn_epochs) - 1][0]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def generate_vit_deit_table(rows, path):

    import re

    vit_rows = [
        r
        for r in rows
        if r.get("architecture", "").startswith("vit_")
        and r.get("status") == "success"
        and r.get("eval_ood") == False
    ]

    deit_rows = []
    std_rows = []
    for r in vit_rows:
        proto = str(r.get("protocol", ""))
        ep_match = re.search(r"e(\d+)", proto)
        epochs = int(ep_match.group(1)) if ep_match else 50
        if epochs >= 200:
            deit_rows.append(r)
        else:
            std_rows.append(r)

    if not deit_rows and not std_rows:
        print("  No ViT results found, skipping DeiT table.")
        with open(path, "w") as f:
            f.write("% No ViT DeiT results available yet.\n")
        return

    use_rows = deit_rows if deit_rows else std_rows
    ep_label = "300" if deit_rows else "50"

    groups = defaultdict(list)
    for r in use_rows:
        key = (r["dataset"], r["method"])
        acc = r.get("test_acc1", -1)
        if isinstance(acc, (int, float)) and acc > 0:
            groups[key].append(float(acc))

    datasets = sorted(set(k[0] for k in groups))
    methods = sorted(set(k[1] for k in groups))

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(
        f"\\caption{{ViT-Small with DeiT-style augmentation ({ep_label} epochs, RandAugment, "
        f"Mixup, CutMix). Best in \\textbf{{bold}}, second \\underline{{underlined}}.}}"
    )
    lines.append("\\label{tab:vit_deit_results}")
    cols = "l" + "c" * len(datasets)
    lines.append("\\begin{tabular}{" + cols + "}")
    lines.append("\\toprule")
    header = (
        "Method & " + " & ".join(DATASET_DISPLAY.get(d, d) for d in datasets) + " \\\\"
    )
    lines.append(header)
    lines.append("\\midrule")

    best = {}
    second = {}
    for d in datasets:
        method_means = []
        for m in methods:
            vals = groups.get((d, m), [])
            if vals:
                method_means.append((m, mean(vals)))
        method_means.sort(key=lambda x: -x[1])
        if method_means:
            best[d] = method_means[0][0]
        if len(method_means) > 1:
            second[d] = method_means[1][0]

    for m in methods:
        name = METHOD_DISPLAY.get(m, m)
        cells = [esc(name)]
        for d in datasets:
            vals = groups.get((d, m), [])
            if vals:
                m_val = mean(vals)
                s_val = std(vals) if len(vals) > 1 else 0.0
                text = fmt_pm(m_val, s_val)
                if best.get(d) == m:
                    text = "\\textbf{" + text + "}"
                elif second.get(d) == m:
                    text = "\\underline{" + text + "}"
                cells.append(text)
            else:
                cells.append("---")
        lines.append(" & ".join(cells) + " \\\\")

    if deit_rows and std_rows:
        lines.append("\\midrule")
        lines.append(
            "\\multicolumn{"
            + str(len(datasets) + 1)
            + "}{l}{\\textit{Standard training (50ep, no augmentation):}} \\\\"
        )
        std_groups = defaultdict(list)
        for r in std_rows:
            key = (r["dataset"], r["method"])
            acc = r.get("test_acc1", -1)
            if isinstance(acc, (int, float)) and acc > 0:
                std_groups[key].append(float(acc))
        std_methods = sorted(set(k[1] for k in std_groups))
        for m in std_methods:
            name = METHOD_DISPLAY.get(m, m)
            cells = [esc(name)]
            for d in datasets:
                vals = std_groups.get((d, m), [])
                if vals:
                    m_val = mean(vals)
                    s_val = std(vals) if len(vals) > 1 else 0.0
                    cells.append(fmt_pm(m_val, s_val))
                else:
                    cells.append("---")
            lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written ViT DeiT table: {path}")


def generate_resolution_scaling_table(rows, output_path):

    ti_rows = [
        r
        for r in rows
        if r.get("status") == "success" and r.get("dataset") == "tinyimagenet"
    ]

    grouped = defaultdict(list)
    for r in ti_rows:
        im_size = r.get("image_size", 64)
        if isinstance(im_size, str):
            im_size = int(im_size) if im_size.isdigit() else 64
        grouped[(r["method"], r["architecture"], im_size)].append(r["test_acc1"])

    all_sizes = sorted({s for (_, _, s) in grouped.keys()})
    if len(all_sizes) < 2:
        print(
            f"  Skipping resolution table (only {len(all_sizes)} resolution(s) found)"
        )
        return

    target_archs = ["resnet18", "resnet50"]
    target_methods = [
        "batchnorm",
        "layernorm",
        "groupnorm",
        "switchnorm",
        "panorm",
        "panorm_switchnorm",
        "panorm_nodetach",
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Resolution scaling on Tiny-ImageNet: accuracy (\%) at different input resolutions. Higher resolution amplifies the advantage of adaptive normalization.}"
    )
    lines.append(r"\label{tab:resolution_scaling}")

    for arch in target_archs:
        arch_sizes = sorted({s for (_, a, s) in grouped.keys() if a == arch})
        if not arch_sizes:
            continue

        lines.append(r"\begin{tabular}{l" + "c" * len(arch_sizes) + "}")
        lines.append(r"\toprule")
        arch_disp = arch.replace("resnet", "ResNet-")
        header = (
            f"Method ({arch_disp}) & "
            + " & ".join(f"{s}$\\times${s}" for s in arch_sizes)
            + r" \\"
        )
        lines.append(header)
        lines.append(r"\midrule")

        size_best = {}
        for s in arch_sizes:
            means = [
                (m, mean(grouped[(m, arch, s)]))
                for m in target_methods
                if grouped[(m, arch, s)]
            ]
            means.sort(key=lambda x: -x[1])
            size_best[s] = means[0][0] if means else None

        pa_start = False
        for method in target_methods:
            if method in PA_NORM_METHODS and not pa_start:
                lines.append(r"\midrule")
                pa_start = True
            display = METHOD_DISPLAY.get(method, method)
            parts = [esc(display)]
            for s in arch_sizes:
                accs = grouped[(method, arch, s)]
                if accs:
                    m, sd = mean(accs), std(accs) if len(accs) > 1 else 0
                    cell = fmt_pm(m, sd)
                    if method == size_best[s]:
                        cell = r"\textbf{" + cell + "}"
                    parts.append(cell)
                else:
                    parts.append("---")
            lines.append(" & ".join(parts) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        if arch != target_archs[-1]:
            lines.append(r"\vspace{0.5em}")

    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def generate_small_batch_table(rows, output_path):

    sb_rows = [
        r
        for r in rows
        if r.get("status") == "success"
        and r.get("dataset") == "cifar100"
        and r.get("architecture") == "resnet18"
        and r.get("eval_ood") == False
    ]

    grouped = defaultdict(list)
    for r in sb_rows:
        bs = r.get("batch_size")
        if isinstance(bs, str):
            bs = int(bs) if bs.isdigit() else 256
        if bs <= 32:
            grouped[(r["method"], bs)].append(r["test_acc1"])

    small_bs = sorted({bs for (_, bs) in grouped.keys()})
    if len(small_bs) < 2:
        print(
            f"  Skipping small batch table (only {len(small_bs)} batch size(s) <= 32)"
        )
        return

    target_methods = [
        "batchnorm",
        "layernorm",
        "groupnorm",
        "switchnorm",
        "panorm",
        "panorm_switchnorm",
    ]

    bs_best = {}
    for bs in small_bs:
        means = [
            (m, mean(grouped[(m, bs)])) for m in target_methods if grouped[(m, bs)]
        ]
        means.sort(key=lambda x: -x[1])
        bs_best[bs] = means[0][0] if means else None

    bn_ref = {}
    for bs in small_bs:
        if grouped[("batchnorm", bs)]:
            bn_ref[bs] = mean(grouped[("batchnorm", bs)])

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Small-batch regime: ResNet-18 CIFAR-100 accuracy (\%) with batch sizes 2--32. PA-Norm adaptively shifts from BN to GN/LN, avoiding the degradation that fixed BN suffers under small batches.}"
    )
    lines.append(r"\label{tab:small_batch}")
    lines.append(r"\begin{tabular}{l" + "c" * len(small_bs) + "}")
    lines.append(r"\toprule")
    header = "Method & " + " & ".join(f"BS={bs}" for bs in small_bs) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    pa_start = False
    for method in target_methods:
        if method in PA_NORM_METHODS and not pa_start:
            lines.append(r"\midrule")
            pa_start = True
        display = METHOD_DISPLAY.get(method, method)
        parts = [esc(display)]
        for bs in small_bs:
            accs = grouped[(method, bs)]
            if accs:
                m, sd = mean(accs), std(accs) if len(accs) > 1 else 0
                cell = fmt_pm(m, sd)
                if method == bs_best[bs]:
                    cell = r"\textbf{" + cell + "}"

                if method in PA_NORM_METHODS and bs in bn_ref:
                    delta = m - bn_ref[bs]
                    if delta > 0:
                        cell += f" \\textcolor{{green!60!black}}{{(+{delta:.1f})}}"
                    elif delta < 0:
                        cell += f" \\textcolor{{red}}{{({delta:.1f})}}"
                parts.append(cell)
            else:
                parts.append("---")
        lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def generate_sgd200_r50_table(rows, output_path):

    sgd_rows = [
        r
        for r in rows
        if r.get("status") == "success"
        and r.get("architecture") == "resnet50"
        and str(r.get("protocol", "")).startswith("sgd_e200")
    ]

    if not sgd_rows:
        print("  Skipping ResNet-50 SGD 200ep table (no data)")
        return

    grouped = defaultdict(list)
    for r in sgd_rows:
        grouped[(r["method"], r["dataset"])].append(r["test_acc1"])

    datasets = sorted({d for (_, d) in grouped.keys()})
    methods_present = sorted(
        {m for (m, _) in grouped.keys()}, key=lambda m: METHOD_DISPLAY.get(m, m)
    )

    ds_best = {}
    for ds in datasets:
        means = [
            (m, mean(grouped[(m, ds)])) for m in methods_present if grouped[(m, ds)]
        ]
        means.sort(key=lambda x: -x[1])
        ds_best[ds] = means[0][0] if means else None

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{ResNet-50 with SGD 200-epoch training (standard recipe): top-1 accuracy (\%).}"
    )
    lines.append(r"\label{tab:sgd200_r50}")
    lines.append(r"\begin{tabular}{l" + "c" * len(datasets) + "}")
    lines.append(r"\toprule")

    ds_display = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "svhn": "SVHN",
        "stl10": "STL-10",
        "tinyimagenet": "Tiny-IN",
    }
    header = "Method & " + " & ".join(ds_display.get(d, d) for d in datasets) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    pa_start = False
    for method in methods_present:
        if method in PA_NORM_METHODS and not pa_start:
            lines.append(r"\midrule")
            pa_start = True
        display = METHOD_DISPLAY.get(method, method)
        parts = [esc(display)]
        for ds in datasets:
            accs = grouped[(method, ds)]
            if accs:
                m, sd = mean(accs), std(accs) if len(accs) > 1 else 0
                cell = fmt_pm(m, sd)
                if method == ds_best[ds]:
                    cell = r"\textbf{" + cell + "}"
                parts.append(cell)
            else:
                parts.append("---")
        lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def main():
    csv_path = os.path.join(os.path.dirname(__file__), "result.csv")
    if not os.path.exists(csv_path):
        csv_path = "exp/result.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: Cannot find result.csv")
        sys.exit(1)

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "tables"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {csv_path}...")
    rows = load_csv(csv_path)
    print(f"  Loaded {len(rows)} rows")

    success_rows = [r for r in rows if r.get("status") == "success"]
    print(f"  Successful: {len(success_rows)}")

    print("\n[1/7] Computing mean +/- std accuracy...")
    mean_std_table = compute_mean_std_table(rows)
    print(f"  Found {len(mean_std_table)} (arch, dataset, protocol) groups")

    print("\n[2/7] Computing rankings (all methods)...")
    per_group_ranks, avg_ranks, wins = compute_rankings(mean_std_table)

    print("\n[2b] Computing fair rankings (core methods only, P4)...")
    fair_ranks, fair_avg_ranks, fair_wins = compute_rankings(
        mean_std_table, method_filter=RANKING_METHODS
    )
    print(
        f"  Fair ranking methods: {sorted(RANKING_METHODS & set(fair_avg_ranks.keys()))}"
    )
    for m in sorted(fair_avg_ranks.keys(), key=lambda x: fair_avg_ranks[x][0]):
        ar, n = fair_avg_ranks[m]
        w = fair_wins.get(m, 0)
        print(
            f"    {METHOD_DISPLAY.get(m, m):20s}: rank {ar:.2f} ({n} settings, {w} wins)"
        )

    print("\n[3/7] Computing statistical tests (PA vs BatchNorm)...")
    stat_results = compute_statistical_tests(rows)

    print("\n[3b] Computing Friedman test + Nemenyi post-hoc...")
    friedman_result = compute_friedman_nemenyi(fair_ranks, fair_avg_ranks)
    if friedman_result[0] is not None:
        f_stat, f_p, cd, pairwise, common_methods, n_groups = friedman_result
        print(
            f"  Friedman chi-sq = {f_stat:.2f}, p = {f_p:.2e} (k={len(common_methods)} methods, n={n_groups} settings)"
        )
        print(f"  Nemenyi CD (alpha=0.05) = {cd:.3f}")

        for mi, mj, ri, rj, diff, sig in pairwise:
            if sig and (
                "panorm" in mi
                or "panorm" in mj
                or "batchnorm" in mi
                or "batchnorm" in mj
            ):
                m_i_disp = METHOD_DISPLAY.get(mi, mi)
                m_j_disp = METHOD_DISPLAY.get(mj, mj)
                print(
                    f"    {m_i_disp} (rank {ri:.2f}) vs {m_j_disp} (rank {rj:.2f}): diff={diff:.2f} > CD={cd:.2f} -> SIGNIFICANT"
                )
    else:
        print("  Insufficient data for Friedman test")
        friedman_result = (None, None, None, [], [], 0)

    print("\n[5/7] Computing OOD analysis...")
    ood_results = compute_ood_analysis(rows)

    print("\n[6/7] Computing gate analysis...")
    gate_results = compute_gate_analysis(rows)

    print("\n[7/7] Generating LaTeX tables...")

    generate_consolidated_ranking(
        mean_std_table,
        fair_ranks,
        fair_avg_ranks,
        fair_wins,
        os.path.join(output_dir, "consolidated_ranking.tex"),
    )

    generate_statistical_tests_table(
        stat_results, os.path.join(output_dir, "statistical_tests.tex")
    )

    generate_friedman_table(
        friedman_result, fair_avg_ranks, os.path.join(output_dir, "friedman_test.tex")
    )

    generate_taxonomy_table(os.path.join(output_dir, "comparison_taxonomy.tex"))

    generate_ood_table(ood_results, os.path.join(output_dir, "ood_analysis.tex"))

    generate_gate_table(gate_results, os.path.join(output_dir, "gate_analysis.tex"))

    generate_main_results_table(
        mean_std_table, os.path.join(output_dir, "main_accuracy.tex")
    )

    print("\n[7b] Generating SGD 200ep table...")
    generate_sgd200_table(rows, os.path.join(output_dir, "sgd200_results.tex"))

    print("\n[7c] Generating batch size sensitivity table...")
    generate_batchsize_table(
        rows, os.path.join(output_dir, "batchsize_sensitivity.tex")
    )

    print("\n[7d] Generating OOD severity sweep table...")
    generate_ood_sweep_table(rows, os.path.join(output_dir, "ood_severity_sweep.tex"))

    print("\n[7e] Generating detach ablation table...")
    generate_detach_ablation_table(
        rows, os.path.join(output_dir, "detach_ablation.tex")
    )

    print("\n[7f] Generating efficiency table...")
    generate_efficiency_table(os.path.join(output_dir, "efficiency_resnet18_gen.tex"))

    print("\n[7g] Generating WideResNet table...")
    generate_wideresnet_table(rows, os.path.join(output_dir, "wideresnet_results.tex"))

    print("\n[7i] Generating deep detach ablation table...")
    generate_detach_ablation_deep_table(
        rows, os.path.join(output_dir, "detach_ablation_deep.tex")
    )

    print("\n[7j] Generating convergence table...")
    generate_convergence_table(rows, os.path.join(output_dir, "convergence.tex"))

    print("\n[7h] Generating per-protocol tables...")
    generate_per_protocol_tables(mean_std_table, output_dir)

    print("\n[7k] Generating ConvNeXt results table...")
    generate_convnext_table(rows, os.path.join(output_dir, "convnext_results.tex"))

    print("\n[7l] Generating ViT results table...")
    generate_vit_table(rows, os.path.join(output_dir, "vit_results.tex"))

    print("\n[7m] Generating selective detach table...")
    generate_selective_detach_table(
        rows, os.path.join(output_dir, "selective_detach.tex")
    )

    print("\n[7n] Generating FLOPs-matched table...")
    generate_flops_matched_table(rows, os.path.join(output_dir, "flops_matched.tex"))

    print("\n[7o] Generating ViT DeiT table...")
    generate_vit_deit_table(rows, os.path.join(output_dir, "vit_deit_results.tex"))

    print("\n[7p] Generating resolution scaling table (TinyImageNet 64/128/224)...")
    generate_resolution_scaling_table(
        rows, os.path.join(output_dir, "resolution_scaling.tex")
    )

    print("\n[7q] Generating small-batch regime table (BS=2,4,8)...")
    generate_small_batch_table(rows, os.path.join(output_dir, "small_batch.tex"))

    print("\n[7r] Generating ResNet-50 SGD 200ep table...")
    generate_sgd200_r50_table(rows, os.path.join(output_dir, "sgd200_r50.tex"))

    print_summary(
        mean_std_table, avg_ranks, wins, stat_results, gate_results, ood_results
    )

    print("\n" + "=" * 80)
    print("DONE. All tables written to:", output_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
