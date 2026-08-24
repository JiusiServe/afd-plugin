#!/usr/bin/env python3
"""Merge AFD and baseline throughput curves into one figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    args = parse_args()
    afd_rows = read_curve(args.afd_csv)
    baseline_rows = read_curve(args.baseline_csv)
    plot_curves(afd_rows, baseline_rows, args.output)
    print(f"wrote: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--afd-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("afd_vs_baselien_throughput.png"),
    )
    return parser.parse_args()


def read_curve(path: Path) -> list[tuple[float, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[tuple[float, float]] = []
        for row in reader:
            x = first_float(row, ("elapsed_s", "second", "time", "time_s"))
            y = first_float(row, ("tokens/s/die", "tokens_per_s_per_die"))
            if x is None or y is None:
                continue
            rows.append((x, y))
    if not rows:
        raise SystemExit(f"No elapsed_s + tokens/s/die rows found in {path}")
    return rows


def first_float(row: dict[str, str], keys: tuple[str, ...]) -> float | None:
    normalized = {key.strip().lower(): value for key, value in row.items()}
    for key in keys:
        value = normalized.get(key.lower())
        if value in (None, ""):
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def plot_curves(
    afd_rows: list[tuple[float, float]],
    baseline_rows: list[tuple[float, float]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        [x for x, _ in afd_rows],
        [y for _, y in afd_rows],
        linewidth=1.6,
        label="48a16f",
        color="#1f77b4",
    )
    ax.plot(
        [x for x, _ in baseline_rows],
        [y for _, y in baseline_rows],
        linewidth=1.6,
        label="ep64",
        color="#d62728",
    )
    ax.set_title("afd vs baselien throughput")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("tokens/s/die")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
