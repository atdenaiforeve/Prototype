import json
import os
import unittest

from network import PrototypeNetwork


class PrototypeNetworkTests(unittest.TestCase):
    def setUp(self):
        self.previous = os.environ.get("PROTOTYPE_NETWORK_KEYS")
        os.environ["PROTOTYPE_NETWORK_KEYS"] = json.dumps({
            "chatgpt": "test-chatgpt-key",
            "gemini": "test-gemini-key",
            "prototype-001": "test-prototype-key",
        })
        self.network = PrototypeNetwork()
        self.network.reload_keys()
        self.network.register_agent("chatgpt", "ChatGPT", "ai")
        self.network.register_agent("gemini", "Gemini", "ai")
        self.network.register_agent("prototype-001", "Prototype-001", "prototype")

    def tearDown(self):
        if self.previous is None:
            os.environ.pop("PROTOTYPE_NETWORK_KEYS", None)
        else:
            os.environ["PROTOTYPE_NETWORK_KEYS"] = self.previous

    def test_each_agent_has_its_own_key(self):
        self.assertTrue(self.network.authenticate("chatgpt", "test-chatgpt-key"))
        self.assertFalse(self.network.authenticate("chatgpt", "test-gemini-key"))
        self.assertTrue(self.network.authenticate("gemini", "test-gemini-key"))

    def test_agents_can_share_a_room(self):
        sent = self.network.send("chatgpt", "main", "Hello everyone.")
        self.assertEqual(sent["sender"], "ChatGPT")
        self.assertEqual(
            self.network.send("gemini", "main", "Hello ChatGPT.")["room_id"],
            "main",
        )
        history = self.network.history("prototype-001", "main")
        self.assertEqual(len(history), 2)

    def test_message_limit_is_enforced(self):
        with self.assertRaises(ValueError):
            self.network.send("chatgpt", "main", "x" * 8001)


if __name__ == "__main__":
    unittest.main()
