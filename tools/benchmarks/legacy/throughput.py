#!/usr/bin/env python3
"""Build full-process throughput CSV/PNG files from one AISBench result dir."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SEARCH_ROOTS = (
    "outputs/default",
    "outputs",
    "benchmark/outputs/default",
    "benchmark/outputs",
    "/a3_inference/itask/workdir/wb02363348/cyj_afd/code/outputs/default",
    "/a3_inference/itask/workdir/wb02363348/cyj_afd/code/benchmark/outputs/default",
)

TIME_KEYS = (
    "end_time",
    "finish_time",
    "finished_time",
    "response_time",
    "completion_time",
    "request_end_time",
    "request_finished_time",
    "timestamp_end",
    "end_timestamp",
)
START_TIME_KEYS = (
    "start_time",
    "request_start_time",
    "send_time",
    "sent_time",
    "begin_time",
    "timestamp_start",
    "start_timestamp",
)
LATENCY_KEYS = (
    "latency",
    "total_latency",
    "request_latency",
    "e2e_latency",
    "duration",
)
OUTPUT_TOKEN_KEYS = (
    "output_tokens",
    "output_token",
    "output_token_num",
    "output_token_len",
    "output_token_length",
    "output_len",
    "output_length",
    "generated_tokens",
    "generated_token_num",
    "generated_token_len",
    "completion_tokens",
    "num_output_tokens",
    "decode_tokens",
)
TOKEN_TIME_KEYS = (
    "token_timestamps",
    "output_token_timestamps",
    "decode_token_timestamps",
    "generated_token_timestamps",
)


@dataclass(frozen=True)
class RequestRecord:
    start_time: float
    end_time: float
    output_tokens: int
    token_timestamps: tuple[float, ...] = ()


def main() -> None:
    args = parse_args()
    if args.result_file is not None:
        result_file = args.result_file
        records = load_records(result_file)
    else:
        ais_bench_dir = resolve_ais_bench_dir(args.ais_bench_dir, args.input_root)
        result_file = None
        records = []
        for candidate in find_candidate_results(ais_bench_dir):
            candidate_records = load_records(candidate)
            if candidate_records:
                result_file = candidate
                records = candidate_records
                break

    if result_file is None:
        raise SystemExit(
            "No AISBench detail/db/json/csv result with request records found "
            f"under: {ais_bench_dir}"
        )

    if not records:
        raise SystemExit(f"No request records with time and output tokens in {result_file}")

    rows = build_throughput_rows(records, die_count=args.die_count)
    output_prefix = Path(args.output_dir) / f"throughput_{sanitize_name(args.name)}"
    csv_path = output_prefix.with_name(f"{output_prefix.name}_full_process.csv")
    png_path = output_prefix.with_name(f"{output_prefix.name}_full_process.png")
    per_die_png_path = output_prefix.with_name(
        f"{output_prefix.name}_full_process_per_die.png"
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(csv_path, rows)
    plot_png(
        png_path,
        rows,
        y_key="tokens/s",
        y_label="tokens/s",
        title=f"{args.name} full-process throughput",
    )
    plot_png(
        per_die_png_path,
        rows,
        y_key="tokens/s/die",
        y_label="tokens/s/die",
        title=f"{args.name} full-process throughput per die",
    )

    total_tokens = sum(record.output_tokens for record in records)
    duration = rows[-1]["elapsed_s"] + 1 if rows else 0
    peak = max(row["tokens/s"] for row in rows) if rows else 0.0
    print(f"input: {result_file}")
    print(f"requests: {len(records)}")
    print(f"output_tokens: {total_tokens}")
    print(f"duration_s: {duration:.0f}")
    print(f"peak_tokens/s: {peak:.2f}")
    print(f"csv: {csv_path}")
    print(f"png: {png_path}")
    print(f"per_die_png: {per_die_png_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate throughput_<name>_full_process.csv/png and "
            "throughput_<name>_full_process_per_die.png from AISBench details."
        )
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Name used in output files, for example 16k_ep64.",
    )
    parser.add_argument(
        "--ais_bench_dir",
        "--ais-bench-dir",
        required=True,
        help=(
            "AISBench run directory name or path, for example 20260704_155051. "
            "Only this directory is parsed."
        ),
    )
    parser.add_argument(
        "--input-root",
        action="append",
        default=[],
        help="AISBench output root to search. Can be specified multiple times.",
    )
    parser.add_argument(
        "--result-file",
        type=Path,
        help="Exact detail DB/JSON/JSONL/CSV file to parse.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for generated CSV/PNG files.",
    )
    parser.add_argument(
        "--die-count",
        type=int,
        default=64,
        help="Number of dies used for tokens/s/die. Default: 64.",
    )
    return parser.parse_args()


def resolve_ais_bench_dir(value: str, input_roots: list[str]) -> Path:
    normalized = value.strip().rstrip(",")
    direct = Path(normalized)
    if direct.exists():
        return direct

    roots = candidate_roots(input_roots)
    for root in roots:
        candidate = root / normalized
        if candidate.exists():
            return candidate

    matches: list[Path] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        try:
            matches.extend(path for path in root.rglob(normalized) if path.is_dir())
        except OSError:
            continue

    if matches:
        return max(matches, key=lambda path: path.stat().st_mtime)

    roots_text = ", ".join(str(root) for root in roots)
    raise SystemExit(
        f"AISBench directory {normalized!r} was not found under: {roots_text}"
    )


def find_candidate_results(ais_bench_dir: Path) -> list[Path]:
    files: list[Path] = []
    if ais_bench_dir.is_file():
        files.append(ais_bench_dir)
    else:
        for pattern in (
            "**/*.db",
            "**/*.sqlite",
            "**/*.sqlite3",
            "**/*.jsonl",
            "**/*.json",
            "**/*.csv",
        ):
            files.extend(ais_bench_dir.glob(pattern))

    candidates = [
        path
        for path in files
        if path.is_file()
        and "throughput_" not in path.name
        and path.stat().st_size > 0
        and is_probable_detail_file(path)
    ]
    fallback = [
        path
        for path in files
        if path.is_file() and "throughput_" not in path.name and path.stat().st_size > 0
    ]
    if candidates:
        preferred = sorted(
            candidates,
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        rest = sorted(
            [path for path in fallback if path not in set(preferred)],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return preferred + rest
    return sorted(
        [
            path
            for path in files
            if path.is_file() and "throughput_" not in path.name and path.stat().st_size > 0
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def candidate_roots(input_roots: list[str]) -> list[Path]:
    roots = [Path(root) for root in input_roots]
    roots.extend(Path(root) for root in DEFAULT_SEARCH_ROOTS)
    roots.append(Path.cwd())
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            deduped.append(root)
            seen.add(key)
    return deduped


def is_probable_detail_file(path: Path) -> bool:
    lowered = str(path).lower()
    return any(
        marker in lowered
        for marker in (
            "detail",
            "detailed",
            "request",
            "perf",
            "benchmark",
            "result",
            "db",
        )
    )


def load_records(path: Path) -> list[RequestRecord]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl" and "detail" in path.name.lower():
        records = load_aisbench_detail_records(path)
        if records:
            return records
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        return load_sqlite_records(path)
    if suffix == ".jsonl":
        return load_jsonl_records(path)
    if suffix == ".json":
        return load_json_records(path)
    if suffix == ".csv":
        return load_csv_records(path)
    raise ValueError(f"Unsupported result file type: {path}")


def load_aisbench_detail_records(path: Path) -> list[RequestRecord]:
    """Load AISBench detail JSONL files that store time_points in db_data."""

    db_cache: dict[str, dict[int, tuple[float, ...]]] = {}
    records: list[RequestRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, Mapping) or not item.get("success", True):
                continue

            output_tokens = parse_int(item.get("output_tokens"))
            db_name = item.get("db_name")
            time_points = item.get("time_points")
            if (
                output_tokens is None
                or output_tokens <= 0
                or not isinstance(db_name, str)
                or not isinstance(time_points, Mapping)
            ):
                continue
            db_ref = parse_int(time_points.get("__db_ref__"))
            if db_ref is None:
                continue

            arrays = db_cache.get(db_name)
            if arrays is None:
                arrays = load_numpy_store(path.parent / "db_data" / db_name)
                db_cache[db_name] = arrays
            timestamps = arrays.get(db_ref)
            if not timestamps:
                continue

            token_timestamps: tuple[float, ...] = ()
            if len(timestamps) >= output_tokens:
                token_timestamps = tuple(timestamps[-output_tokens:])
            records.append(
                RequestRecord(
                    start_time=timestamps[0],
                    end_time=timestamps[-1],
                    output_tokens=output_tokens,
                    token_timestamps=token_timestamps,
                )
            )
    return dedupe_records(records)


def load_numpy_store(path: Path) -> dict[int, tuple[float, ...]]:
    arrays: dict[int, tuple[float, ...]] = {}
    if not path.exists():
        return arrays

    import numpy as np

    conn = sqlite3.connect(path)
    try:
        for row_id, arr_blob in conn.execute("select id, arr_blob from numpy_store"):
            try:
                arr = np.load(io.BytesIO(arr_blob), allow_pickle=False)
            except Exception:
                continue
            values = tuple(float(x) for x in arr.tolist())
            if values:
                arrays[int(row_id)] = values
    finally:
        conn.close()
    return arrays


def load_sqlite_records(path: Path) -> list[RequestRecord]:
    records: list[RequestRecord] = []
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "select name from sqlite_master where type='table'"
            ).fetchall()
        ]
        for table in tables:
            try:
                rows = conn.execute(f'select * from "{table}"').fetchall()
            except sqlite3.DatabaseError:
                continue
            for row in rows:
                record = record_from_mapping(dict(row))
                if record is not None:
                    records.append(record)
    finally:
        conn.close()
    return dedupe_records(records)


def load_jsonl_records(path: Path) -> list[RequestRecord]:
    records: list[RequestRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.extend(records_from_json_item(item))
    return dedupe_records(records)


def load_json_records(path: Path) -> list[RequestRecord]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return dedupe_records(list(records_from_json_item(data)))


def load_csv_records(path: Path) -> list[RequestRecord]:
    records: list[RequestRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            record = record_from_mapping(row)
            if record is not None:
                records.append(record)
    return dedupe_records(records)


def records_from_json_item(item: Any) -> Iterator[RequestRecord]:
    if isinstance(item, Mapping):
        record = record_from_mapping(item)
        if record is not None:
            yield record
        for value in item.values():
            if isinstance(value, list):
                for child in value:
                    yield from records_from_json_item(child)
            elif isinstance(value, Mapping):
                yield from records_from_json_item(value)
    elif isinstance(item, list):
        for child in item:
            yield from records_from_json_item(child)


def record_from_mapping(mapping: Mapping[str, Any]) -> RequestRecord | None:
    flat = flatten_mapping(mapping)
    output_tokens = extract_output_tokens(flat)
    if output_tokens is None or output_tokens <= 0:
        return None

    start_time = first_float(flat, START_TIME_KEYS)
    end_time = first_float(flat, TIME_KEYS)
    latency = first_float(flat, LATENCY_KEYS)

    if end_time is None and start_time is not None and latency is not None:
        end_time = start_time + normalize_duration_seconds(latency)
    if start_time is None and end_time is not None and latency is not None:
        start_time = end_time - normalize_duration_seconds(latency)
    if start_time is None or end_time is None:
        return None

    start_time = normalize_timestamp_seconds(start_time)
    end_time = normalize_timestamp_seconds(end_time)
    if end_time < start_time:
        start_time, end_time = end_time, start_time
    if math.isclose(end_time, start_time):
        end_time = start_time + 1e-6

    token_timestamps = extract_token_timestamps(flat)
    return RequestRecord(
        start_time=start_time,
        end_time=end_time,
        output_tokens=output_tokens,
        token_timestamps=token_timestamps,
    )


def flatten_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        key = normalize_key(prefix)
        if key:
            flat[key] = value
            flat.setdefault(key.split(".")[-1], value)
        if isinstance(value, str):
            parsed = try_json(value)
            if parsed is not value:
                visit(prefix, parsed)
        elif isinstance(value, Mapping):
            for child_key, child_value in value.items():
                visit(f"{prefix}.{child_key}" if prefix else str(child_key), child_value)

    for k, v in mapping.items():
        visit(str(k), v)
    return flat


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9_./-]+", "_", key.strip().lower())


def try_json(value: str) -> Any:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def first_float(flat: Mapping[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = value_for_key(flat, key)
        parsed = parse_float(value)
        if parsed is not None:
            return parsed
    return None


def extract_output_tokens(flat: Mapping[str, Any]) -> int | None:
    for key in OUTPUT_TOKEN_KEYS:
        parsed = parse_int(value_for_key(flat, key))
        if parsed is not None:
            return parsed

    usage = value_for_key(flat, "usage")
    if isinstance(usage, Mapping):
        parsed = parse_int(usage.get("completion_tokens"))
        if parsed is not None:
            return parsed

    token_ids = value_for_key(flat, "output_token_ids")
    if token_ids is None:
        token_ids = value_for_key(flat, "generated_token_ids")
    parsed_ids = parse_sequence(token_ids)
    if parsed_ids is not None:
        return len(parsed_ids)
    return None


def extract_token_timestamps(flat: Mapping[str, Any]) -> tuple[float, ...]:
    for key in TOKEN_TIME_KEYS:
        values = parse_sequence(value_for_key(flat, key))
        if values:
            parsed = [parse_float(value) for value in values]
            timestamps = [
                normalize_timestamp_seconds(value)
                for value in parsed
                if value is not None
            ]
            if timestamps:
                return tuple(timestamps)
    return ()


def value_for_key(flat: Mapping[str, Any], key: str) -> Any:
    normalized = normalize_key(key)
    if normalized in flat:
        return flat[normalized]
    suffix = f".{normalized}"
    for candidate_key, value in flat.items():
        if candidate_key.endswith(suffix):
            return value
    return None


def parse_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", value.strip())
        if not match:
            return None
        try:
            number = float(match.group(0))
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def parse_int(value: Any) -> int | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def parse_sequence(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        parsed = try_json(value)
        if isinstance(parsed, list):
            return parsed
        numbers = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", value)
        if numbers:
            return numbers
    return None


def normalize_timestamp_seconds(value: float) -> float:
    absolute = abs(value)
    if absolute > 1e17:
        return value / 1e9
    if absolute > 1e14:
        return value / 1e6
    if absolute > 1e11:
        return value / 1e3
    return value


def normalize_duration_seconds(value: float) -> float:
    if value > 10000:
        return value / 1000
    return value


def dedupe_records(records: list[RequestRecord]) -> list[RequestRecord]:
    seen: set[tuple[float, float, int]] = set()
    deduped: list[RequestRecord] = []
    for record in records:
        key = (
            round(record.start_time, 6),
            round(record.end_time, 6),
            record.output_tokens,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return sorted(deduped, key=lambda record: (record.start_time, record.end_time))


def build_throughput_rows(
    records: list[RequestRecord],
    *,
    die_count: int,
) -> list[dict[str, float]]:
    min_time = min(record.start_time for record in records)
    max_time = max(record.end_time for record in records)
    num_seconds = max(1, int(math.ceil(max_time - min_time)))
    buckets = [0.0 for _ in range(num_seconds)]

    for record in records:
        if len(record.token_timestamps) >= record.output_tokens:
            for timestamp in record.token_timestamps[-record.output_tokens :]:
                idx = int(math.floor(timestamp - min_time))
                if 0 <= idx < len(buckets):
                    buckets[idx] += 1.0
            continue

        duration = max(record.end_time - record.start_time, 1e-6)
        rate = record.output_tokens / duration
        start_idx = int(math.floor(record.start_time - min_time))
        end_idx = int(math.ceil(record.end_time - min_time))
        for idx in range(max(0, start_idx), min(len(buckets), end_idx)):
            bucket_start = min_time + idx
            bucket_end = bucket_start + 1
            overlap = max(
                0.0,
                min(record.end_time, bucket_end) - max(record.start_time, bucket_start),
            )
            buckets[idx] += rate * overlap

    rows: list[dict[str, float]] = []
    for idx, tokens in enumerate(buckets):
        rows.append(
            {
                "second": float(idx),
                "elapsed_s": float(idx),
                "tokens/s": tokens,
                "tokens/s/die": tokens / die_count,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["second", "elapsed_s", "tokens/s", "tokens/s/die"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "second": int(row["second"]),
                    "elapsed_s": int(row["elapsed_s"]),
                    "tokens/s": f"{row['tokens/s']:.6f}",
                    "tokens/s/die": f"{row['tokens/s/die']:.6f}",
                }
            )


def plot_png(
    path: Path,
    rows: list[dict[str, float]],
    *,
    y_key: str,
    y_label: str,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [row["elapsed_s"] for row in rows]
    y = [row[y_key] for row in rows]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, y, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def sanitize_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return sanitized.strip("_") or "benchmark"


if __name__ == "__main__":
    main()
