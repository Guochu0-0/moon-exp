"""Export E61's evaluation grid and aligned training curves."""

from __future__ import annotations

import csv
from io import StringIO
import math
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, StrMethodFormatter
import numpy as np

from evaluate_grid import CACHE, COUNTS, RUN, STEPS, TRAIN, demand, read, recorded, rows, sha256, slot, write


EXTRA = RUN / "extra"
SERIES = ("val_auc5", "test_auc5", "val_inlier_mean", "val_inlier_median")
REWARDS = ("rew", "rew_1")
WINDOW = 250


def export_csv(path, table):
    with StringIO(newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
        path.write_text(stream.getvalue(), encoding="utf-8", newline="\n")


def collect(plan):
    table, evidence = [], {}
    for step in STEPS:
        row = {"step": step, **dict.fromkeys(SERIES)}
        weight = plan["weights"][str(step)]
        missing = plan["missing"].get(str(step))
        row.update(weight_sha256=weight["sha256"], missing_reason=missing)
        entry = evidence[str(step)] = {"weights": weight, "missing_reason": missing, "splits": {}}
        for split, expected in COUNTS.items():
            summary = {} if missing else read(slot(step, split))
            if summary:
                demand(summary["covered"] == expected, f"{step} {split}: incomplete coverage")
                demand(summary["match"]["weights_sha256"] == weight["sha256"], "evaluated weight differs")
                if summary["mode"] == "inference":
                    run = read(RUN / f"launch/{split}_{step}.json")[-1]
                    demand(run["end"] and run["status"] == "ok", f"{split}_{step}: launch not complete")
                    demand(run["prediction_sha256"] == summary["prediction_sha256"], "launch hash differs")
            entry["splits"][split] = summary or None
            row[f"{split}_auc5"] = summary.get("auc5")
            row[f"{split}_covered"] = summary.get("covered", 0)
            row[f"{split}_fit_failures"] = summary.get("fit_failures")
            row[f"{split}_source"] = summary.get("mode", "missing")
        val = entry["splits"]["val"] or {}
        row["val_inlier_mean"] = val.get("inlier_mean")
        row["val_inlier_median"] = val.get("inlier_median")
        row["val_zero_matches"] = val.get("zero_matches")
        table.append(row)
    return table, evidence


def training():
    log = TRAIN / "log.jsonl"
    sequence = rows(log)
    demand([r["step"] for r in sequence] == list(range(1, STEPS[-1] + 1)),
           "training log has gaps or repeated steps")
    curve = [{"step": r["step"], **{key: r[key] for key in REWARDS}} for r in sequence]
    for key in REWARDS:
        values = np.array([r[key] for r in curve], dtype=float)
        demand(np.isfinite(values).all(), f"non-finite training scalar {key}")
        sums = np.r_[0.0, np.cumsum(values)]
        for i, row in enumerate(curve, start=1):
            left = max(0, i - WINDOW)
            row[f"{key}_trailing_{WINDOW}"] = (sums[i] - sums[left]) / (i - left)
    return curve, {"path": str(log), "sha256": sha256(log), "rows": len(sequence),
                   "reward_columns": list(REWARDS), "trailing_mean_steps": WINDOW}


def plot(table, curve):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#788693", "axes.labelcolor": "#263747",
        "text.color": "#263747", "xtick.color": "#495968", "ytick.color": "#495968",
        "axes.axisbelow": True, "savefig.dpi": 210,
    })
    navy, orange, teal, purple = "#225b95", "#b64a20", "#147670", "#75499a"
    fig, (accuracy, inliers, rewards) = plt.subplots(3, 1, figsize=(8.35, 7.2), sharex=True)
    fig.subplots_adjust(left=.12, right=.975, bottom=.09, top=.97, hspace=.62)
    steps = [r["step"] for r in table]
    for ax in (accuracy, inliers, rewards):
        ax.set_xlim(-110, 8115)
        ax.xaxis.set_major_locator(MultipleLocator(1000))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", color="#d9e2ea", linewidth=.8)
        ax.tick_params(axis="both", length=3)
        ax.axvline(0, color="#aab6c2", linewidth=.8, zorder=0)
    for ax, series in (
        (accuracy, (("val_auc5", "Original validation", navy, "o"),
                    ("test_auc5", "Original test", orange, "s"))),
        (inliers, (("val_inlier_mean", "Mean", teal, "o"),
                   ("val_inlier_median", "Median", purple, "s"))),
    ):
        for key, label, color, marker in series:
            values = [100 * r[key] if r[key] is not None else math.nan for r in table]
            ax.plot(steps, values, color=color, linewidth=1.45, marker=marker, markersize=3.3, label=label)
    accuracy.set_title("A  Registration accuracy", loc="left", fontweight="bold")
    accuracy.set_ylabel("AUC@5 (%)")
    inliers.set_title("B  Validation inlier rate per patch pair", loc="left", fontweight="bold")
    inliers.set_ylabel("Inlier rate (%)")
    rewards.set_title("C  Training position rewards (trailing 250-step means)", loc="left", fontweight="bold")
    rewards.set_ylabel("Mean reward")
    rewards.set_xlabel("Training step")
    training_steps = [r["step"] for r in curve]
    for key, label, color in zip(REWARDS, ("Coarse matching", "Finest refinement (scale 1)"), (navy, teal)):
        rewards.plot(training_steps, [r[f"{key}_trailing_{WINDOW}"] for r in curve],
                     color=color, linewidth=1.45, label=label)
    for ax in (accuracy, inliers, rewards):
        ax.legend(frameon=False, ncol=2, loc="upper right", handletextpad=.45, columnspacing=1.1)
        ax.margins(y=.18)
    fig.savefig(EXTRA / "training_progress.png", facecolor="white")
    plt.close(fig)


def describe(table, curve):
    """Report descriptive summaries used to write Notes."""
    for key in SERIES:
        by_step = {r["step"]: r[key] for r in table if r[key] is not None}
        print(key, "peak", max(by_step.items(), key=lambda item: item[1]))
        for low, high in ((0, 0), (250, 250), (500, 2000), (2250, 4000), (4250, 6000), (6250, 7907), (7907, 7907)):
            values = [v for step, v in by_step.items() if low <= step <= high]
            if values:
                print(f"  {low:4d}–{high:4d} ({len(values):2d})",
                      *(round(v * 100, 4) for v in (min(values), statistics.mean(values), max(values))))
    for key in REWARDS:
        print(key, "first/last 250", *(statistics.mean(r[key] for r in block)
                                     for block in (curve[:WINDOW], curve[-WINDOW:])))
    complete = [r for r in table if r["step"] >= 250 and all(r[key] is not None for key in SERIES)]
    if len(complete) >= 3:
        values = np.array([[r[key] for key in SERIES] for r in complete]).T
        for label, v in (("levels", values), ("successive differences", np.diff(values))):
            print(label, np.corrcoef(v)[:2, 2:].tolist())


def main():
    plan = read(CACHE / "plan.json")
    table, evidence = collect(plan)
    curve, training_log = training()
    EXTRA.mkdir(exist_ok=True)
    export_csv(EXTRA / "checkpoint_metrics.csv", table)
    export_csv(EXTRA / "training_scalars.csv", curve)
    write(EXTRA / "evaluation_sources.json", {"training_commit": plan["training_commit"],
                                             "training_log": training_log, "checkpoints": evidence})
    plot(table, curve)
    describe(table, curve)
    return {"evaluated_steps": sum(r["val_auc5"] is not None and r["test_auc5"] is not None for r in table),
            "scheduled_steps": len(table), "training_steps": len(curve)}


if __name__ == "__main__":
    recorded("main", main)
