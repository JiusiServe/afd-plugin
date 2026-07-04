# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from afd_plugin.offline_schedule import OfflineCsvSchedule


class _Request:
    def __init__(self, source_line: int, target_prompt_tokens: int) -> None:
        self.source_line = source_line
        self.target_prompt_tokens = target_prompt_tokens


def test_offline_csv_schedule_maps_groups_to_dp_ranks(tmp_path):
    schedule_csv = tmp_path / "schedule.csv"
    schedule_csv.write_text(
        "\n".join(
            [
                "16547",
                "0",
                "20718",
                "8496",
                "98",
                "0",
                "111",
                "222",
                "0",
                "333",
                "444",
            ]
        ),
        encoding="utf-8",
    )

    dp0 = OfflineCsvSchedule.from_csv(str(schedule_csv), dp_size=2, dp_rank=0)
    dp1 = OfflineCsvSchedule.from_csv(str(schedule_csv), dp_size=2, dp_rank=1)

    assert dp0.rank_steps == ((0,), (4, 5))
    assert dp1.rank_steps == ((1, 2, 3), (6, 7))
    assert dp0.index_to_step == {0: 0, 4: 1, 5: 1}
    assert dp1.index_to_step == {1: 0, 2: 0, 3: 0, 6: 1, 7: 1}
    assert dp0.rank_request_indices() == [0, 4, 5]
    assert dp1.rank_request_indices() == [1, 2, 3, 6, 7]
    assert dp0.request_source_lines == (1, 3, 4, 5, 7, 8, 10, 11)
    assert dp0.request_token_counts == (16547, 20718, 8496, 98, 111, 222, 333, 444)
    assert dp0.total_requests == 8
    assert dp1.total_requests == 8


def test_offline_csv_schedule_allows_merged_group_source_lines(tmp_path):
    schedule_csv = tmp_path / "schedule.csv"
    schedule_csv.write_text(
        "\n".join(
            [
                "16547",
                "15069",
                "6218",
                "73",
                "0",
                "14597",
                "12699",
            ]
        ),
        encoding="utf-8",
    )
    requests = [
        _Request(source_line=1, target_prompt_tokens=16547),
        _Request(source_line=3, target_prompt_tokens=15069),
        _Request(source_line=4, target_prompt_tokens=6218),
        _Request(source_line=5, target_prompt_tokens=73),
        _Request(source_line=7, target_prompt_tokens=14597),
        _Request(source_line=8, target_prompt_tokens=12699),
    ]

    schedule = OfflineCsvSchedule.from_csv(str(schedule_csv), dp_size=3, dp_rank=0)

    schedule.validate_requests(requests)
