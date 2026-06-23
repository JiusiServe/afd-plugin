# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from afd_plugin.offline_schedule import OfflineCsvSchedule


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
