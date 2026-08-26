from __future__ import annotations

import unittest

from simulator.length_datasets import LENGTH_DATASETS, length_dataset_catalog
from simulator.workload import read_csv_requests


class LengthDatasetTests(unittest.TestCase):
    def test_moonconv_assets_contain_only_expected_lengths(self) -> None:
        expected = {
            "moonconv-v4-flash-formal-0": {
                "request_count": 512,
                "total_input_tokens": 5_331_020,
                "min_input_length": 893,
                "p50_input_length": 6_529,
                "p95_input_length": 32_238,
                "max_input_length": 63_007,
                "zero_gap_count": 459,
            },
            "moonconv-v4-flash-formal-1": {
                "request_count": 512,
                "total_input_tokens": 5_260_666,
                "min_input_length": 892,
                "p50_input_length": 6_596,
                "p95_input_length": 31_362,
                "max_input_length": 63_778,
                "zero_gap_count": 461,
            },
            "moonconv-v4-flash-formal-2": {
                "request_count": 512,
                "total_input_tokens": 5_211_377,
                "min_input_length": 891,
                "p50_input_length": 5_951,
                "p95_input_length": 32_204,
                "max_input_length": 63_649,
                "zero_gap_count": 461,
            },
            "moonconv-v4-flash-screening": {
                "request_count": 128,
                "total_input_tokens": 1_340_915,
                "min_input_length": 893,
                "p50_input_length": 6_638,
                "p95_input_length": 32_219,
                "max_input_length": 63_417,
                "zero_gap_count": 116,
            },
        }

        self.assertEqual(set(length_dataset_catalog()), set(expected))
        for dataset in LENGTH_DATASETS:
            csv_text = dataset.csv_bytes().decode("utf-8")
            self.assertEqual(
                csv_text.splitlines()[0],
                "arrival_time_ms,input_length",
            )
            self.assertNotIn("prompt_token_ids", csv_text)
            requests = read_csv_requests(csv_text=csv_text)
            self.assertTrue(
                all(request.arrival_time_ms is not None for request in requests)
            )
            self.assertTrue(
                any(
                    current.arrival_time_ms == previous.arrival_time_ms
                    for previous, current in zip(
                        requests[:-1],
                        requests[1:],
                        strict=True,
                    )
                )
            )
            summary = dataset.summary()
            self.assertEqual(
                len(requests),
                expected[dataset.dataset_id]["request_count"],
            )
            for key, value in expected[dataset.dataset_id].items():
                self.assertEqual(summary[key], value)


if __name__ == "__main__":
    unittest.main()
