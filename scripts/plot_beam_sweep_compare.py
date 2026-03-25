#!/usr/bin/env python3
import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def read_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["beam"] = int(row["beam"])
            for key in ("qps", "recall", "avg_dc", "avg_iter", "kernel_ms"):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def mode_name(mode):
    return {
        "stable": "off",
        "routea": "rabitq",
    }.get(mode, mode)


def detect_dataset_label(rows, summary_csv):
    haystacks = [summary_csv]
    for row in rows[:4]:
        haystacks.append(str(row.get("summary_path", "")))
        haystacks.append(str(row.get("source_root", "")))
    text = " ".join(haystacks).lower()
    if "sift1m" in text:
        return "SIFT1M"
    if "sift10m" in text:
        return "SIFT10M"
    return "Dataset"


def plot_metric(rows, out_dir, metric, ylabel, filename, title, connect_by_mode=False):
    beams = sorted({row["beam"] for row in rows})
    colors = plt.cm.turbo([i / max(1, len(beams) - 1) for i in range(len(beams))])
    color_map = {beam: colors[i] for i, beam in enumerate(beams)}
    marker_map = {"stable": "o", "routea": "s"}
    linestyle_map = {"stable": "--", "routea": "-"}

    grouped = defaultdict(dict)
    by_mode = defaultdict(list)
    for row in rows:
        grouped[row["beam"]][row["mode"]] = row
        by_mode[row["mode"]].append(row)

    fig, ax = plt.subplots(figsize=(9, 6))

    if connect_by_mode:
        for mode in ("stable", "routea"):
            mode_rows = sorted(by_mode.get(mode, []), key=lambda r: r["beam"])
            if not mode_rows:
                continue
            ax.plot(
                [row["recall"] for row in mode_rows],
                [row[metric] for row in mode_rows],
                color="black",
                linewidth=1.8,
                linestyle=linestyle_map[mode],
                alpha=0.8,
                zorder=1,
            )

    for beam in beams:
        pair = grouped[beam]
        xs = []
        ys = []
        modes = []
        for mode in ("stable", "routea"):
            if mode in pair:
                xs.append(pair[mode]["recall"])
                ys.append(pair[mode][metric])
                modes.append(mode)

        if (not connect_by_mode) and len(xs) >= 2:
            ax.plot(xs, ys, color=color_map[beam], linewidth=1.6, alpha=0.9)

        for mode in modes:
            row = pair[mode]
            ax.scatter(
                row["recall"],
                row[metric],
                color=color_map[beam],
                marker=marker_map[mode],
                s=72,
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
            ax.annotate(
                str(beam),
                (row["recall"], row[metric]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7,
                color=color_map[beam],
            )

    ax.set_xlabel("Recall@100")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    beam_handles = [
        Line2D([0], [0], color=color_map[beam], lw=2, label=f"beam={beam}")
        for beam in beams
    ]
    mode_handles = [
        Line2D([0], [0], color="black", marker=marker_map[mode], linestyle=linestyle_map[mode],
               markersize=7, label=mode_name(mode))
        for mode in ("stable", "routea")
    ]

    legend1 = ax.legend(handles=beam_handles, title="Beam Size", loc="center left",
                        bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.add_artist(legend1)
    ax.legend(handles=mode_handles, title="Mode", loc="upper left", frameon=False)

    fig.tight_layout()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_markdown(rows, out_dir, dataset_label):
    by_mode = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)
    for rows_mode in by_mode.values():
        rows_mode.sort(key=lambda r: r["beam"])

    md_path = os.path.join(out_dir, "beam_sweep_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Beam Sweep Summary\n\n")
        f.write(f"Dataset: `{dataset_label}`\n\n")
        f.write("`avg_iter` equals logged `AVG_HOPS`; both code paths accumulate `s_iter_count` into `d_total_hops`.\n\n")
        f.write("| mode | beam | qps | recall | avg_dc | avg_iter | kernel_ms |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for mode in ("stable", "routea"):
            for row in by_mode.get(mode, []):
                f.write(
                    f"| {mode_name(mode)} | {row['beam']} | {row['qps']:.4f} | {row['recall']:.5f} | "
                    f"{row['avg_dc']:.0f} | {row['avg_iter']:.0f} | {row['kernel_ms']:.3f} |\n"
                )


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: plot_beam_sweep_compare.py <summary_csv> <out_dir>")
    summary_csv = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    rows = read_rows(summary_csv)
    if not rows:
        raise SystemExit("empty summary csv")
    dataset_label = detect_dataset_label(rows, summary_csv)

    plot_metric(
        rows,
        out_dir,
        "qps",
        "QPS",
        "qps_vs_recall.png",
        f"{dataset_label}: QPS vs Recall@100",
        connect_by_mode=True,
    )
    plot_metric(
        rows,
        out_dir,
        "avg_dc",
        "Average Distance Computations",
        "avg_dc_vs_recall.png",
        f"{dataset_label}: Average Distance Computations vs Recall@100",
    )
    plot_metric(
        rows,
        out_dir,
        "avg_iter",
        "Average Iterations",
        "avg_iter_vs_recall.png",
        f"{dataset_label}: Average Iterations vs Recall@100",
    )
    write_markdown(rows, out_dir, dataset_label)


if __name__ == "__main__":
    main()
