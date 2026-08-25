from __future__ import annotations

import json
import threading
import unittest
import urllib.request

from simulator.server import SimulatorServer
from simulator.tests.helpers import make_profile


class ServerTests(unittest.TestCase):
    def test_defaults_page_and_simulation_api(self) -> None:
        server = SimulatorServer(("127.0.0.1", 0), make_profile(layer_count=1))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        try:
            with urllib.request.urlopen(
                f"http://{host}:{port}/api/defaults"
            ) as response:
                defaults = json.load(response)
            self.assertEqual(defaults["profile"]["layer_count"], 1)

            with urllib.request.urlopen(f"http://{host}:{port}/") as response:
                page = response.read().decode()
            self.assertIn('value="vllm_queue_aware"', page)
            self.assertIn("TopK 数", page)
            self.assertIn("Expert GMM 实采 Shape", page)
            self.assertIn("Profile Query", page)
            self.assertIn("EP 数", page)

            payload = json.dumps(
                {
                    "mode": "fixed",
                    "fixed_lengths": [128, 128],
                    "scheduler": {
                        "policy": "vllm_queue_aware",
                        "max_num_batched_tokens": 1024,
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
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
