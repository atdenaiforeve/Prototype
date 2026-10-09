import json
import os
import unittest
from unittest.mock import patch

from prototype_remote_memory import PrototypeRemoteMemory


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, limit=-1):
        return self.payload[:limit]


class PrototypeRemoteMemoryTests(unittest.TestCase):
    def test_bridge_stays_disabled_without_a_key(self):
        with patch.dict(os.environ, {
            "PROTOTYPE_REMOTE_NETWORK_AGENT_KEY": "",
            "PROTOTYPE_NETWORK_KEYS": "{}",
            "PROTOTYPE_DEVICE_ID": "unit-test-device",
        }, clear=False):
            with patch("prototype_remote_memory.urlopen") as mocked:
                bridge = PrototypeRemoteMemory()
        self.assertFalse(bridge.status()["enabled"])
        self.assertIsNone(bridge.branch_id)
        mocked.assert_not_called()

    def test_creates_branch_reads_memory_and_saves_branch_memory(self):
        responses = [
            FakeResponse({"created": True, "branch": {"branch_id": "br_test"}}),
            FakeResponse({
                "branch_id": "br_test",
                "shared_core_memory": [{"content": "Shared principle"}],
                "branch_memory": [{"content": "Device note"}],
            }),
            FakeResponse({
                "branch_id": "br_test",
                "shared_core_memory": [{"content": "Shared principle"}],
                "branch_memory": [{"content": "Device note"}],
            }),
            FakeResponse({"saved": True}),
        ]
        with patch.dict(os.environ, {
            "PROTOTYPE_REMOTE_NETWORK_AGENT_ID": "prototype",
            "PROTOTYPE_REMOTE_NETWORK_AGENT_KEY": "test-key",
            "PROTOTYPE_REMOTE_NETWORK_URL": "https://example.test",
            "PROTOTYPE_DEVICE_ID": "unit-test-device",
        }, clear=False):
            with patch("prototype_remote_memory.urlopen", side_effect=responses) as mocked:
                bridge = PrototypeRemoteMemory()
                context = bridge.context()
                saved = bridge.remember("Test memory")

        self.assertEqual(bridge.branch_id, "br_test")
        self.assertTrue(bridge.status()["connected"])
        self.assertIn("Shared principle", context)
        self.assertIn("Device note", context)
        self.assertTrue(saved)
        self.assertEqual(mocked.call_count, 4)
        first_request = mocked.call_args_list[0].args[0]
        self.assertEqual(first_request.method, "POST")
        self.assertNotIn("test-key", str(first_request.headers))


if __name__ == "__main__":
    unittest.main()
