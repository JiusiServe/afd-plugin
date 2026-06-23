# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CSV schedule parsing for offline inference."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OfflineCsvSchedule:
    rank_steps: tuple[tuple[int, ...], ...]
    index_to_step: dict[int, int]
    total_requests: int
    request_source_lines: tuple[int, ...]
    request_token_counts: tuple[int, ...]

    @classmethod
    def from_csv(cls, path: str, dp_size: int, dp_rank: int) -> "OfflineCsvSchedule":
        if dp_size < 1:
            raise ValueError("offline scheduler dp_size must be >= 1")
        if dp_rank < 0 or dp_rank >= dp_size:
            raise ValueError(
                f"offline scheduler dp_rank={dp_rank} must be in [0, {dp_size})"
            )

        groups: list[list[int]] = []
        current_group: list[int] = []
        request_source_lines: list[int] = []
        request_token_counts: list[int] = []
        request_index = 0

        csv_path = Path(path)
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for csv_line, row in enumerate(reader, start=1):
                if not row:
                    continue
                token_count = int(row[0])
                if token_count == 0:
                    if current_group:
                        groups.append(current_group)
                        current_group = []
                    continue

                current_group.append(request_index)
                request_source_lines.append(csv_line)
                request_token_counts.append(token_count)
                request_index += 1

        if current_group:
            groups.append(current_group)

        rank_steps = tuple(
            tuple(group)
            for group_index, group in enumerate(groups)
            if group_index % dp_size == dp_rank
        )
        index_to_step = {
            request_idx: step
            for step, group in enumerate(rank_steps)
            for request_idx in group
        }
        return cls(
            rank_steps=rank_steps,
            index_to_step=index_to_step,
            total_requests=request_index,
            request_source_lines=tuple(request_source_lines),
            request_token_counts=tuple(request_token_counts),
        )

    def validate_requests(self, requests: list[object]) -> None:
        if len(requests) != self.total_requests:
            raise ValueError(
                f"Schedule CSV consumed {self.total_requests} requests, "
                f"but prompt file contains {len(requests)}."
            )

        for request_index, request in enumerate(requests):
            source_line = getattr(request, "source_line", None)
            expected_source_line = self.request_source_lines[request_index]
            if source_line is not None and source_line != expected_source_line:
                raise ValueError(
                    f"Prompt request {request_index} metadata source_line="
                    f"{source_line} does not match CSV line "
                    f"{expected_source_line}."
                )

            target_prompt_tokens = getattr(request, "target_prompt_tokens", None)
            expected_token_count = self.request_token_counts[request_index]
            if (
                target_prompt_tokens is not None
                and target_prompt_tokens != expected_token_count
            ):
                raise ValueError(
                    f"Prompt request {request_index} target_prompt_tokens="
                    f"{target_prompt_tokens} does not match CSV token count "
                    f"{expected_token_count}."
                )

    def rank_request_indices(self) -> list[int]:
        return [request_index for step in self.rank_steps for request_index in step]


__all__ = ["OfflineCsvSchedule"]
