"""Built-in prompt-length datasets for simulator workloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from simulator.workload import read_csv_requests

DATA_ROOT = Path(__file__).with_name("data")
MOONCONV_REPOSITORY = "ShwStone/moonconv-wildchat-v4-flash-prefill"
MOONCONV_REVISION = "284a2326bbef3d5107995f52e38eeee9d0ccdb45"
P95_QUANTILE = 0.95


@dataclass(frozen=True)
class LengthDataset:
    dataset_id: str
    label: str
    csv_path: Path
    source_repository: str
    source_revision: str
    source_splits: tuple[str, ...]

    def csv_bytes(self) -> bytes:
        return self.csv_path.read_bytes()

    def summary(self) -> dict[str, str | int | float | list[str]]:
        requests = read_csv_requests(csv_text=self.csv_bytes().decode("utf-8"))
        lengths = sorted(request.input_length for request in requests)
        arrival_times = [
            request.arrival_time_ms
            for request in requests
            if request.arrival_time_ms is not None
        ]
        if len(arrival_times) != len(requests):
            raise ValueError(f"built-in dataset {self.dataset_id} lacks timestamps")
        source_span_ms = arrival_times[-1] - arrival_times[0]
        if source_span_ms <= 0:
            raise ValueError(
                f"built-in dataset {self.dataset_id} has no positive time span"
            )
        zero_gap_count = sum(
            current == previous
            for previous, current in zip(
                arrival_times[:-1],
                arrival_times[1:],
                strict=True,
            )
        )
        total_input_tokens = sum(lengths)
        p95_index = int((len(lengths) - 1) * P95_QUANTILE)
        return {
            "id": self.dataset_id,
            "label": self.label,
            "request_count": len(lengths),
            "total_input_tokens": total_input_tokens,
            "min_input_length": lengths[0],
            "p50_input_length": lengths[(len(lengths) - 1) // 2],
            "p95_input_length": lengths[p95_index],
            "max_input_length": lengths[-1],
            "mean_input_length": total_input_tokens / len(lengths),
            "source_duration_ms": source_span_ms,
            "source_mean_qps": (len(arrival_times) - 1) * 1_000.0 / source_span_ms,
            "zero_gap_count": zero_gap_count,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "source_splits": list(self.source_splits),
        }


LENGTH_DATASETS = (
    LengthDataset(
        dataset_id="moonconv-v4-flash-formal-0",
        label="MoonConv V4 Flash · formal 0",
        csv_path=DATA_ROOT / "moonconv-v4-flash-formal-0-trace.csv",
        source_repository=MOONCONV_REPOSITORY,
        source_revision=MOONCONV_REVISION,
        source_splits=("formal_0",),
    ),
    LengthDataset(
        dataset_id="moonconv-v4-flash-formal-1",
        label="MoonConv V4 Flash · formal 1",
        csv_path=DATA_ROOT / "moonconv-v4-flash-formal-1-trace.csv",
        source_repository=MOONCONV_REPOSITORY,
        source_revision=MOONCONV_REVISION,
        source_splits=("formal_1",),
    ),
    LengthDataset(
        dataset_id="moonconv-v4-flash-formal-2",
        label="MoonConv V4 Flash · formal 2",
        csv_path=DATA_ROOT / "moonconv-v4-flash-formal-2-trace.csv",
        source_repository=MOONCONV_REPOSITORY,
        source_revision=MOONCONV_REVISION,
        source_splits=("formal_2",),
    ),
    LengthDataset(
        dataset_id="moonconv-v4-flash-screening",
        label="MoonConv V4 Flash · screening",
        csv_path=DATA_ROOT / "moonconv-v4-flash-screening-trace.csv",
        source_repository=MOONCONV_REPOSITORY,
        source_revision=MOONCONV_REVISION,
        source_splits=("screening",),
    ),
)


def length_dataset_catalog() -> dict[str, LengthDataset]:
    return {dataset.dataset_id: dataset for dataset in LENGTH_DATASETS}


__all__ = ["LENGTH_DATASETS", "LengthDataset", "length_dataset_catalog"]
