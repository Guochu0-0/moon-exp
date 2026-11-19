"""Summarize E52's pre-match detector repeatability on completed Test inputs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from .repeatability import METHODS, RADII_PX, TEST_COUNTS, evaluate_geometry

BUDGETS = (300, 1024)
DISPLAY = {
    "superpoint_original": "SuperPoint + LightGlue",
    "spsg": "SuperPoint + SuperGlue",
    "minima_splg": "MINIMA SuperPoint + LightGlue",
    "redfeat": "ReDFeat",
    "rift2": "RIFT2 (MATLAB)",
    "rift2_portable": "RIFT2 (SciPy / FAST-9)",
    "comir": "CoMIR + SIFT",
}
DATASETS = {"moon": "Moon Test", "whu": "WHU-OPT-SAR Test"}
FIELDS = ("dataset", "pair", "group", "method", "top_k", "status",
          "optical_raw", "sar_raw", "optical_overlap", "sar_overlap",
          "optical_evaluated", "sar_evaluated", "reference_loo_px",
          *(f"repeatability_{r}px" for r in RADII_PX))


def result_rows(root, domain, methods):
    directory = root / domain
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for sample in manifest:
        pair, group, reference = (sample[key] for key in ("pair", "group", "reference"))
        key = pair.replace("/", "__")
        for method in methods:
            suffix = ".mat" if method == "rift2" else ".npz"
            path = directory / method / (key + suffix)
            missing = not path.is_file()
            points = {}
            if not missing:
                if method == "rift2":
                    data = loadmat(path, variable_names=("optical", "sar"))
                    points = {f"{key}__{m}": data[m] for m in ("optical", "sar")}
                else:
                    with np.load(path, allow_pickle=False) as data:
                        points = {f"{key}__{m}": data[m] for m in ("optical", "sar")}
            for budget in BUDGETS:
                row = {"dataset": domain, "pair": pair, "group": group, "method": method, "top_k": budget}
                if missing:
                    yield {**row, "status": "missing_detection"}
                    continue
                row.update(optical_raw=len(points[f"{key}__optical"]), sar_raw=len(points[f"{key}__sar"]))
                if reference is None:
                    yield {**row, "status": "missing_reference"}
                    continue
                matrix = np.asarray(reference["affine"]).T
                result = evaluate_geometry(sample, points, matrix[:2], matrix[2], top_k=budget)
                row.update(status="scored", optical_overlap=result["optical_overlap_points"],
                           sar_overlap=result["sar_overlap_points"], optical_evaluated=result["optical_top_points"],
                           sar_evaluated=result["sar_top_points"], reference_loo_px=reference.get("leave_one_out_mean_px"))
                scores = {item["radius_px"]: item["repeatability"] for item in result["thresholds"]}
                yield {**row, **{f"repeatability_{r}px": scores.get(r, 0.0) for r in RADII_PX}}


def write_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, domains, methods):
    for domain in domains:
        for budget in BUDGETS:
            by_method = {method: [row for row in rows if (row["dataset"], row["top_k"], row["method"])
                                  == (domain, budget, method)] for method in methods}
            comparable = set.intersection(*({row["pair"] for row in items if row["status"] == "scored"}
                                            for items in by_method.values()))
            if not comparable:
                raise ValueError(f"No jointly scored pairs for {domain} / {budget}")
            groups = sorted({row["group"] for row in rows
                             if row["dataset"] == domain and row["pair"] in comparable})
            draws = np.random.default_rng(52).integers(0, len(groups), size=(5000, len(groups)))
            for method, items in by_method.items():
                scored = [row for row in items if row["pair"] in comparable]
                group_index = np.array([groups.index(row["group"]) for row in scored])
                counts = np.bincount(group_index, minlength=len(groups))
                for radius in RADII_PX:
                    values = np.array([row[f"repeatability_{radius}px"] for row in scored])
                    sums = np.bincount(group_index, weights=values, minlength=len(groups))
                    samples = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
                    n_scored = sum(row["status"] == "scored" for row in items)
                    yield {
                        "dataset": domain, "method": method, "top_k": budget, "radius_px": radius,
                        "total_pairs": len(items), "scored_pairs_method": n_scored,
                        "detected_pairs": sum(row["status"] != "missing_detection" for row in items),
                        "missing_reference": sum(row["status"] == "missing_reference" for row in items),
                        "joint_scored_pairs": len(scored), "groups": len(groups),
                        "mean": values.mean(), "median": np.median(values),
                        "p25": np.quantile(values, 0.25), "p75": np.quantile(values, 0.75),
                        "cluster_mean": (sums / counts).mean(),
                        "ci95_low": np.quantile(samples, 0.025), "ci95_high": np.quantile(samples, 0.975),
                        "median_optical_points": np.median([row["optical_evaluated"] for row in scored]),
                        "median_sar_points": np.median([row["sar_evaluated"] for row in scored]),
                        "zero_point_pairs": sum(not (row["optical_evaluated"] and row["sar_evaluated"]) for row in scored),
                    }


def plots(output, summary, rows, domains, methods):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    colors = dict(zip(DISPLAY, plt.get_cmap("tab10").colors))
    fig, axes = plt.subplots(len(domains), len(BUDGETS), figsize=(10.5, 3.5 * len(domains)), squeeze=False,
                             sharex=True, sharey=True, layout="constrained")
    for domain, panels in zip(domains, axes):
        for budget, ax in zip(BUDGETS, panels):
            for method in methods:
                series = [row for row in summary if (row["dataset"], row["top_k"], row["method"])
                          == (domain, budget, method)]
                mean = np.array([row["mean"] for row in series]) * 100
                ax.plot(RADII_PX, mean, marker="o", ms=4, color=colors[method], label=DISPLAY[method])
                ax.fill_between(RADII_PX, [100 * row["ci95_low"] for row in series],
                                [100 * row["ci95_high"] for row in series], color=colors[method], alpha=0.12)
            n = series[0]["joint_scored_pairs"]
            ax.set_title(f"{DATASETS[domain]} | top {budget:,} per image | n={n:,}", loc="left")
            ax.set_xticks(RADII_PX)
            ax.set_xlabel("Tolerance on SAR grid (px)")
            ax.set_ylabel("Mean repeatability (%)")
            ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(output / "repeatability_curves.png", dpi=220)
    fig.savefig(output / "repeatability_curves.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(len(domains), len(BUDGETS), figsize=(10.5, 3.4 * len(domains)), squeeze=False,
                             sharex=True, layout="constrained")
    for domain, panels in zip(domains, axes):
        for budget, ax in zip(BUDGETS, panels):
            scored = {method: {row["pair"]: row["repeatability_3px"] * 100 for row in rows
                               if (row["dataset"], row["top_k"], row["method"], row["status"])
                               == (domain, budget, method, "scored")} for method in methods}
            comparable = sorted(set.intersection(*(set(points) for points in scored.values())))
            boxes = ax.boxplot([[scored[m][pair] for pair in comparable] for m in methods], vert=False,
                               patch_artist=True, showfliers=False)
            for box, method in zip(boxes["boxes"], methods):
                box.set_facecolor(colors[method])
                box.set_alpha(0.35)
            ax.set_yticks(range(1, len(methods) + 1), [DISPLAY[m] for m in methods])
            ax.invert_yaxis()
            ax.set_title(f"{DATASETS[domain]} | top {budget:,}", loc="left")
            ax.set_xlabel("Per-pair repeatability at 3 px (%)")
            ax.grid(axis="x", alpha=0.2)
    fig.savefig(output / "repeatability_distributions.png", dpi=220)
    fig.savefig(output / "repeatability_distributions.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detections", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=tuple(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=DISPLAY, default=METHODS)
    parser.add_argument("--allow-partial", action="store_true", help="Report jointly scored pairs of an interim batch")
    args = parser.parse_args()
    rows = [row for domain in args.datasets for row in result_rows(args.detections, domain, args.methods)]
    for domain in args.datasets:
        subset = [row for row in rows if row["dataset"] == domain and row["method"] == args.methods[0]
                  and row["top_k"] == BUDGETS[0]]
        if len(subset) != TEST_COUNTS[domain][0] and not args.allow_partial:
            raise ValueError(f"Incomplete {domain} manifest: {len(subset)} test pairs")
    missing = [row for row in rows if row["status"] == "missing_detection"]
    if missing and not args.allow_partial:
        first = missing[0]
        raise ValueError(f"{len(missing) // len(BUDGETS)} missing pair detections, first: {first['pair']}/{first['method']}")
    summary = list(summarize(rows, args.datasets, args.methods))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "per_pair.csv", rows, FIELDS)
    write_csv(args.output / "summary.csv", summary)
    plots(args.output, summary, rows, args.datasets, args.methods)


if __name__ == "__main__":
    main()
