from __future__ import annotations

import json
import threading
import unittest
import urllib.request

from simulator.server import SimulatorServer
from simulator.tests.helpers import make_profile


class ServerTests(unittest.TestCase):
    def test_requested_equal_die_topologies_are_cataloged(self) -> None:
        server = SimulatorServer(
            ("127.0.0.1", 0),
            {
                "twenty_four_die": make_profile(
                    layer_count=1,
                    afd_dp_size=4,
                    afd_tp_size=4,
                    merged_dp_size=6,
                    merged_tp_size=4,
                ),
                "thirty_two_die": make_profile(
                    layer_count=1,
                    afd_dp_size=6,
                    afd_tp_size=4,
                    merged_dp_size=8,
                    merged_tp_size=4,
                ),
                "forty_die": make_profile(
                    layer_count=1,
                    afd_dp_size=4,
                    afd_tp_size=8,
                    merged_dp_size=5,
                    merged_tp_size=8,
                ),
            },
        )
        try:
            expected_pairs = {
                "afd-attn-dp4tp4ep8-ffn-dp8tp1ep8": (
                    24,
                    "merged-dp6tp4ep24",
                ),
                "afd-attn-dp6tp4ep8-ffn-dp8tp1ep8": (
                    32,
                    "merged-dp8tp4ep32",
                ),
                "afd-attn-dp4tp8ep8-ffn-dp8tp1ep8": (
                    40,
                    "merged-dp5tp8ep40",
                ),
            }
            for afd_topology_id, (num_devices, merged_topology_id) in (
                expected_pairs.items()
            ):
                self.assertEqual(
                    server.afd_topologies[afd_topology_id].num_devices,
                    num_devices,
                )
                self.assertEqual(
                    server.default_merged_by_afd[afd_topology_id],
                    merged_topology_id,
                )
                comparison = server.comparison_profile(
                    afd_topology_id,
                    merged_topology_id,
                )
                self.assertEqual(
                    comparison.device_budget(),
                    {
                        "afd_attention": num_devices - 8,
                        "afd_ffn": 8,
                        "afd_total": num_devices,
                        "merged": num_devices,
                    },
                )
        finally:
            server.server_close()

    def test_defaults_page_and_simulation_api(self) -> None:
        server = SimulatorServer(
            ("127.0.0.1", 0),
            {
                "default": make_profile(layer_count=1),
                "alternate_merged": make_profile(
                    layer_count=1,
                    merged_dp_size=2,
                    merged_tp_size=8,
                ),
                "alternate": make_profile(
                    layer_count=1,
                    afd_dp_size=3,
                    afd_tp_size=8,
                    merged_dp_size=4,
                    merged_tp_size=8,
                ),
            },
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        try:
            with urllib.request.urlopen(
                f"http://{host}:{port}/api/defaults"
            ) as response:
                defaults = json.load(response)
            default_afd_id = "afd-attn-dp2tp4ep8-ffn-dp8tp1ep8"
            alternate_afd_id = "afd-attn-dp3tp8ep8-ffn-dp8tp1ep8"
            default_merged_id = "merged-dp4tp4ep16"
            alternate_merged_id = "merged-dp2tp8ep16"
            merged_32_id = "merged-dp4tp8ep32"
            self.assertEqual(defaults["default_afd_topology_id"], default_afd_id)
            self.assertEqual(
                [item["id"] for item in defaults["afd_topologies"]],
                [default_afd_id, alternate_afd_id],
            )
            self.assertEqual(
                [item["id"] for item in defaults["merged_topologies"]],
                [alternate_merged_id, default_merged_id, merged_32_id],
            )
            self.assertEqual(defaults["afd_topologies"][0]["num_devices"], 16)
            self.assertEqual(defaults["afd_topologies"][1]["num_devices"], 32)
            self.assertEqual(
                defaults["afd_topologies"][0]["default_merged_topology_id"],
                default_merged_id,
            )
            self.assertEqual(
                defaults["afd_topologies"][1]["default_merged_topology_id"],
                merged_32_id,
            )
            self.assertEqual(defaults["afd_topologies"][0]["layer_count"], 1)
            self.assertEqual(
                defaults["config"]["scheduler"]["afd_policy"], "current_runtime"
            )
            self.assertEqual(
                defaults["config"]["scheduler"]["merged_policy"],
                "current_runtime",
            )
            self.assertEqual(
                defaults["config"]["scheduler"]["afd"],
                {"max_num_batched_tokens": 8192, "chunk_size": 8192},
            )
            self.assertEqual(
                defaults["config"]["scheduler"]["merged"],
                {"max_num_batched_tokens": 8192, "chunk_size": 8192},
            )
            self.assertEqual(
                [item["id"] for item in defaults["length_datasets"]],
                [
                    "moonconv-v4-flash-formal-0",
                    "moonconv-v4-flash-formal-1",
                    "moonconv-v4-flash-formal-2",
                    "moonconv-v4-flash-screening",
                ],
            )
            self.assertEqual(
                defaults["length_datasets"][0]["request_count"],
                512,
            )
            self.assertEqual(
                defaults["length_datasets"][0]["zero_gap_count"],
                459,
            )

            with urllib.request.urlopen(
                f"http://{host}:{port}/api/length-datasets/"
                "moonconv-v4-flash-formal-0"
            ) as response:
                length_csv = response.read().decode()
            self.assertEqual(
                length_csv.splitlines()[0],
                "arrival_time_ms,input_length",
            )
            self.assertEqual(len(length_csv.splitlines()), 513)
            self.assertNotIn("prompt_token_ids", length_csv)

            with urllib.request.urlopen(f"http://{host}:{port}/") as response:
                page = response.read().decode()
            self.assertIn('value="current_runtime"', page)
            self.assertIn('id="afd-topology-select"', page)
            self.assertIn('id="merged-topology-select"', page)
            self.assertIn('id="length-dataset"', page)
            self.assertIn("data.afd_topologies.forEach", page)
            self.assertIn("data.merged_topologies.forEach", page)
            self.assertIn("data.length_datasets.forEach", page)
            self.assertIn("/api/length-datasets/", page)
            self.assertIn('value="scaled_trace"', page)
            self.assertIn("item.num_devices===afd.num_devices", page)
            self.assertIn('value="prefill_token_greedy"', page)
            self.assertIn('value="prefill_token_square_greedy"', page)
            self.assertIn('value="afd_wave_token_sum"', page)
            self.assertIn('value="afd_wave_token_square_sum"', page)
            self.assertIn('value="merged_wave_token_sum"', page)
            self.assertIn('value="merged_wave_token_square_sum"', page)
            self.assertIn('value="vllm_queue_aware"', page)
            self.assertIn("TopK 数", page)
            self.assertIn("Expert GMM 实采 Shape", page)
            self.assertIn("Profile Query", page)
            self.assertIn("EP 数", page)
            self.assertIn('<canvas id="timeline"', page)
            self.assertIn("timelineDrawEvents", page)
            self.assertIn("timeline_max_events: null", page)
            self.assertIn("data.afd.timeline.concat(data.merged.timeline)", page)
            self.assertIn("/^Global EP\\d+$/.test(event.resource)", page)
            self.assertNotIn("Math.min(...events.map", page)
            self.assertNotIn("Math.max(...events.map", page)
            self.assertIn("afd_policy: $('afd-policy').value", page)
            self.assertIn("merged_policy: $('merged-policy').value", page)
            self.assertIn("afd: {max_num_batched_tokens", page)
            self.assertIn("merged: {max_num_batched_tokens", page)
            self.assertIn("s.scheduler_policy", page)
            self.assertIn("mergedTopology.dp_size", page)
            self.assertNotIn('<svg id="timeline"', page)

            payload = json.dumps(
                {
                    "afd_topology_id": alternate_afd_id,
                    "merged_topology_id": merged_32_id,
                    "mode": "fixed",
                    "fixed_lengths": [128, 128],
                    "scheduler": {
                        "afd_policy": "current_runtime",
                        "merged_policy": "current_runtime",
                        "afd": {
                            "max_num_batched_tokens": 1024,
                            "chunk_size": 1024,
                        },
                        "merged": {
                            "max_num_batched_tokens": 2048,
                            "chunk_size": 512,
                        },
                    },
                }
            ).encode()
            request = urllib.request.Request(
                f"http://{host}:{port}/api/simulate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request) as response:
                result = json.load(response)
            self.assertIn("afd", result)
            self.assertIn("merged", result)
            self.assertEqual(
                result["topology_selection"],
                {"afd": alternate_afd_id, "merged": merged_32_id},
            )
            self.assertEqual(
                result["merged"]["summary"]["topology"],
                {"dp_size": 4, "tp_size": 8, "ep_size": 32},
            )
            self.assertEqual(result["afd"]["summary"]["topology"]["dp_size"], 3)
            self.assertEqual(
                result["afd"]["summary"]["scheduler_policy"], "round_robin"
            )
            self.assertEqual(
                result["merged"]["summary"]["scheduler_policy"],
                "vllm_queue_aware",
            )
            self.assertEqual(
                result["config"]["scheduler"]["afd"]["max_num_batched_tokens"],
                1024,
            )
            self.assertEqual(
                result["config"]["scheduler"]["merged"]["chunk_size"],
                512,
            )
            with self.assertRaisesRegex(
                ValueError,
                "AFD uses 16 dies, but merged uses 32 dies",
            ):
                server.comparison_profile(default_afd_id, merged_32_id)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
